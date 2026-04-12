"""Core TicketFlowEnv environment implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from grader import grade_task
from policy import (
    can_close_ticket,
    classification_matches,
    expected_next_actions,
    is_harmful_action,
    is_policy_compliant,
    is_redundant_action,
    is_unnecessary_escalation,
    is_valid_action_type,
    resolution_quality,
)
from reward import compute_step_reward
from tasks import AVAILABLE_ACTIONS, MAX_STEPS, TASK_SEQUENCE, get_task
from ticketflow_env.models import TicketFlowAction, TicketFlowObservation, TicketFlowState, TicketFlowStepResult

_logger = logging.getLogger(__name__)


class TicketFlowEnvironment:
    """Deterministic customer-support operations environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state: Optional[TicketFlowState] = None
        self._task = None
        self._conversation_history: List[str] = []
        self._action_history: List[TicketFlowAction] = []
        self._reset_index = 0

    @property
    def state(self) -> TicketFlowState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")
        return self._state

    @property
    def action_history(self) -> List[TicketFlowAction]:
        return list(self._action_history)

    def reset(self, task_id: Optional[str] = None) -> TicketFlowObservation:
        selected_task_id = task_id or TASK_SEQUENCE[self._reset_index % len(TASK_SEQUENCE)]
        self._reset_index += 1
        task = get_task(selected_task_id)
        self._task = task
        ticket_id = f"{task.task_id}-001"
        self._conversation_history = [f"customer: {task.customer_message}"]
        self._action_history = []
        self._state = TicketFlowState(
            ticket_id=ticket_id,
            issue_type=task.issue_type,
            correct_resolution=task.correct_resolution,
            refund_allowed=task.refund_allowed,
            replacement_allowed=task.replacement_allowed,
            escalation_required=task.escalation_required,
            max_steps=MAX_STEPS,
            task_id=task.task_id,
            current_status="open",
            customer_tier=task.customer_tier,
            order_value=task.order_value,
            order_age_days=task.order_age_days,
            last_action_result="Ticket loaded and ready for handling.",
        )
        return self._build_observation()

    def step(self, action: TicketFlowAction | Dict[str, Any]) -> TicketFlowStepResult:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment must be reset before stepping.")

        parsed_action = self._coerce_action(action)
        pre_state = self._state.model_copy(deep=True)
        self._action_history.append(parsed_action)
        self._state.step_count += 1

        valid_action = is_valid_action_type(parsed_action.action_type)
        correct_classification = valid_action and classification_matches(parsed_action, pre_state)
        correct_workflow = valid_action and parsed_action.action_type in expected_next_actions(pre_state)
        policy_compliant = valid_action and is_policy_compliant(parsed_action.action_type, pre_state)
        harmful = valid_action and is_harmful_action(parsed_action.action_type, pre_state)
        redundant = valid_action and is_redundant_action(parsed_action.action_type, pre_state)
        unnecessary_escalation = valid_action and is_unnecessary_escalation(parsed_action.action_type, pre_state)
        premature_close = parsed_action.action_type == "close_ticket" and not can_close_ticket(pre_state)
        resolution_credit = resolution_quality(parsed_action.action_type, pre_state)
        helpful_reply = (
            parsed_action.action_type == "send_customer_reply"
            and bool(parsed_action.response_text and len(parsed_action.response_text.strip()) >= 12)
            and (pre_state.request_more_info_done or pre_state.resolution_action is not None)
        )

        error: Optional[str] = None
        if not valid_action:
            self._state.last_action_result = f"Invalid action_type: {parsed_action.action_type}"
            error = self._state.last_action_result
        else:
            self._apply_transition(parsed_action, pre_state)
            if harmful:
                self._state.failed = True
                self._state.current_status = "policy_violation"
                self._state.policy_violations += 1
                self._state.last_action_result = (
                    self._state.last_action_result or "Action caused a policy-violating outcome."
                )
                error = "harmful_action"
            elif premature_close:
                self._state.failed = True
                self._state.closed = True
                self._state.current_status = "closed_unresolved"
                self._state.last_action_result = "Ticket was closed before it was properly resolved."
                error = "premature_close"

        # Anti-reward-hacking: detect consecutive identical actions
        # Note: parsed_action is already appended at _action_history[-1],
        # so [-2] is the previous action. len >= 2 guards first-step edge case.
        repeated_action = (
            len(self._action_history) >= 2
            and self._action_history[-1].action_type == self._action_history[-2].action_type
        )

        # Build context dict for reward computation (no signature breakage)
        reward_context = {
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
            "repeated_action": repeated_action,
        }

        reward, reward_details = compute_step_reward(
            valid_action=valid_action,
            correct_classification=correct_classification,
            correct_workflow=correct_workflow,
            policy_compliant=policy_compliant,
            resolution_credit=resolution_credit,
            helpful_reply=helpful_reply,
            harmful=harmful,
            redundant=redundant,
            unnecessary_escalation=unnecessary_escalation,
            premature_close=premature_close,
            context=reward_context,
        )
        reward = max(-1.0, min(1.0, reward))  # defense-in-depth clamp
        self._state.cumulative_reward += reward
        self._state.helpful_reply = self._state.helpful_reply or helpful_reply

        if self._state.step_count >= self._state.max_steps and not self._state.resolved and not self._state.failed:
            self._state.failed = True
            self._state.current_status = "max_steps_reached"
            self._state.last_action_result = "Episode ended because the maximum number of steps was reached."
            if error is None:
                error = "max_steps_reached"

        done = bool(self._state.resolved or self._state.failed or self._state.closed)

        # Determine policy rule applied (deterministic decision trace)
        if not valid_action:
            policy_applied = "invalid_action"
        elif harmful:
            if parsed_action.action_type == "approve_refund":
                if pre_state.customer_tier == "suspicious":
                    policy_applied = "fraud_blocked"
                elif pre_state.order_value > 1000:
                    policy_applied = "high_value_escalation_required"
                else:
                    policy_applied = "refund_denied_outside_window"
            elif parsed_action.action_type == "deny_refund":
                policy_applied = "wrongful_denial_blocked"
            else:
                policy_applied = "harmful_action_blocked"
        elif premature_close:
            policy_applied = "premature_closure_penalty"
        elif not policy_compliant:
            policy_applied = "policy_non_compliant"
        elif parsed_action.action_type == "classify_issue":
            policy_applied = "issue_classification"
        elif parsed_action.action_type == "request_more_info":
            policy_applied = "information_gathering"
        elif parsed_action.action_type == "approve_refund":
            if pre_state.customer_tier == "vip":
                policy_applied = "vip_refund_allowed"
            else:
                policy_applied = "standard_refund_allowed"
        elif parsed_action.action_type == "deny_refund":
            policy_applied = "refund_denied_policy"
        elif parsed_action.action_type == "offer_store_credit":
            policy_applied = "store_credit_alternative"
        elif parsed_action.action_type == "offer_replacement":
            policy_applied = "replacement_offered"
        elif parsed_action.action_type == "escalate_to_human":
            if pre_state.escalation_required or pre_state.order_value > 1000:
                policy_applied = "escalation_required"
            elif pre_state.customer_tier == "suspicious":
                policy_applied = "fraud_escalation"
            else:
                policy_applied = "optional_escalation"
        elif parsed_action.action_type == "send_customer_reply":
            policy_applied = "customer_communication"
        elif parsed_action.action_type == "close_ticket":
            policy_applied = "ticket_closure"
        else:
            policy_applied = "general_workflow"

        # Determine failure reason (if applicable)
        failure_reason = None
        if error == "harmful_action":
            if parsed_action.action_type == "approve_refund":
                if pre_state.customer_tier == "suspicious":
                    failure_reason = "refund_not_allowed_for_suspicious_account"
                elif pre_state.order_value > 1000:
                    failure_reason = "refund_not_allowed_for_high_value_order"
                elif pre_state.order_age_days > (60 if pre_state.customer_tier == "vip" else 30):
                    failure_reason = "refund_outside_allowed_window"
                else:
                    failure_reason = "refund_policy_violation"
            elif parsed_action.action_type == "deny_refund":
                failure_reason = "wrongful_refund_denial"
            else:
                failure_reason = "harmful_action_policy_violation"
        elif error == "premature_close":
            failure_reason = "ticket_closed_before_resolution"
        elif error and "Invalid action_type" in str(error):
            failure_reason = f"invalid_action_{parsed_action.action_type}"
        elif error == "max_steps_reached":
            failure_reason = "exceeded_maximum_steps"

        info = {
            "task_id": self._state.task_id,
            "reward_details": reward_details,
            "grading_preview": grade_task(self._state.task_id, self._state, self._action_history),
            "reward_breakdown": {
                "classification": reward_details["classification"],
                "policy_compliance": reward_details["policy_compliance"],
                "response_quality": reward_details["response_quality"],
                "efficiency_bonus": reward_details["efficiency_bonus"],
                "penalty": reward_details["penalty"],
            },
            "policy_violations": self._state.policy_violations,
            "decision_trace": {
                "last_action": parsed_action.action_type,
                "policy_applied": policy_applied,
            },
            "failure_reason": failure_reason,
        }

        _logger.debug(
            "step action=%s reward=%.4f done=%s",
            parsed_action.action_type,
            reward,
            done,
        )

        return TicketFlowStepResult(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info=info,
            error=error,
        )

    def _coerce_action(self, action: TicketFlowAction | Dict[str, Any]) -> TicketFlowAction:
        if isinstance(action, TicketFlowAction):
            return action
        if isinstance(action, dict):
            candidate = {
                "action_type": str(action.get("action_type", "__invalid__")),
                "response_text": action.get("response_text"),
                "metadata": action.get("metadata", {}) or {},
            }
            return TicketFlowAction.model_validate(candidate)
        raise TypeError("Action must be a TicketFlowAction or dict payload.")

    def _apply_transition(self, action: TicketFlowAction, pre_state: TicketFlowState) -> None:
        assert self._state is not None

        action_type = action.action_type
        if action_type == "classify_issue":
            label = str(action.metadata.get("label", "")).strip().lower() or "unclassified"
            self._state.classification_done = True
            self._state.classification_label = label
            self._state.current_status = "classified"
            self._state.last_action_result = f"Issue classified as {label}."
            return

        if action_type == "request_more_info":
            self._state.request_more_info_done = True
            self._state.current_status = "awaiting_agent_reply"
            self._state.last_action_result = "Marked ticket as requiring more customer information."
            if getattr(self._task, "followup_response", None):
                self._conversation_history.append(f"customer: {self._task.followup_response}")
            return

        if action_type in {"approve_refund", "deny_refund", "offer_replacement", "offer_store_credit", "escalate_to_human"}:
            if pre_state.resolution_action is None:
                self._state.resolution_action = action_type
            if action_type == "approve_refund":
                self._state.current_status = "refund_approved"
                self._state.last_action_result = "Refund approved for the order."
            elif action_type == "deny_refund":
                self._state.current_status = "refund_denied"
                self._state.last_action_result = "Refund request denied under policy."
            elif action_type == "offer_replacement":
                self._state.current_status = "replacement_offered"
                self._state.last_action_result = "Replacement offer recorded."
            elif action_type == "offer_store_credit":
                self._state.current_status = "store_credit_offered"
                self._state.last_action_result = "Store credit offered as an alternative."
            else:
                self._state.escalated = True
                self._state.current_status = "escalated"
                self._state.last_action_result = "Ticket escalated to a human support specialist."
            return

        if action_type == "send_customer_reply":
            reply_text = (action.response_text or "").strip()
            if reply_text:
                self._conversation_history.append(f"agent: {reply_text}")
            self._state.reply_sent = True
            if (self._state.request_more_info_done or self._state.escalated) and self._state.task_id in {"account_access_ambiguity", "incomplete_info"}:
                self._state.resolved = True
                self._state.current_status = "waiting_for_customer"
                self._state.last_action_result = "Reply sent and ticket moved to waiting-for-customer state."
            else:
                self._state.current_status = "reply_sent"
                self._state.last_action_result = "Customer reply sent."
            return

        if action_type == "close_ticket":
            self._state.closed = True
            self._state.current_status = "closed"
            if can_close_ticket(pre_state):
                self._state.resolved = True
                self._state.last_action_result = "Ticket closed after a complete resolution."
            return

    def _build_observation(self) -> TicketFlowObservation:
        assert self._state is not None
        assert self._task is not None
        return TicketFlowObservation(
            ticket_id=self._state.ticket_id,
            customer_message=self._task.customer_message,
            customer_tier=self._task.customer_tier,
            order_status=self._task.order_status,
            order_value=self._task.order_value,
            order_age_days=self._task.order_age_days,
            conversation_history=list(self._conversation_history),
            available_actions=list(AVAILABLE_ACTIONS),
            current_status=self._state.current_status,
            last_action_result=self._state.last_action_result,
            payment_method=getattr(self._task, "payment_method", None),
            delivery_partner=getattr(self._task, "delivery_partner", None),
        )
