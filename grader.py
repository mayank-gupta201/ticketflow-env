"""Deterministic task graders for TicketFlowEnv."""

from __future__ import annotations

from typing import Dict, Iterable, List

from ticketflow_env.models import TicketFlowAction, TicketFlowState


def _action_types(action_history: Iterable[TicketFlowAction]) -> List[str]:
    return [action.action_type for action in action_history]


def _policy_ok(state: TicketFlowState) -> bool:
    return state.policy_violations == 0 and not (
        state.issue_type == "refund_request" and state.resolution_action == "approve_refund"
    )


def grade_easy_refund(state: TicketFlowState, action_history: List[TicketFlowAction]) -> Dict[str, object]:
    actions = _action_types(action_history)
    details = {
        "correct_classification": state.classification_label == "damaged_item",
        "followed_policy": _policy_ok(state),
        "used_valid_resolution": state.resolution_action == "approve_refund",
        "reply_sent": state.reply_sent,
        "closed_ticket": state.closed and state.resolved,
    }
    score = (
        0.25 * float(details["correct_classification"])
        + 0.25 * float(details["followed_policy"])
        + 0.25 * float(details["used_valid_resolution"])
        + 0.15 * float(details["reply_sent"])
        + 0.10 * float(details["closed_ticket"])
    )
    if "deny_refund" in actions:
        score -= 0.15
    return {"score": max(0.0, min(1.0, round(score, 4))), "details": details}


def grade_account_access_ambiguity(
    state: TicketFlowState,
    action_history: List[TicketFlowAction],
) -> Dict[str, object]:
    actions = _action_types(action_history)
    details = {
        "correct_classification": state.classification_label == "account_access_issue",
        "followed_policy": _policy_ok(state) and "approve_refund" not in actions,
        "used_valid_resolution": state.request_more_info_done,
        "reply_sent": state.reply_sent,
        "closed_ticket": state.closed,
    }
    score = (
        0.25 * float(details["correct_classification"])
        + 0.25 * float(details["followed_policy"])
        + 0.25 * float(details["used_valid_resolution"])
        + 0.15 * float(details["reply_sent"])
        + 0.10 * float("close_ticket" not in actions)
    )
    if "approve_refund" in actions or "offer_replacement" in actions:
        score -= 0.20
    return {"score": max(0.0, min(1.0, round(score, 4))), "details": details}


def grade_out_of_policy_refund(
    state: TicketFlowState,
    action_history: List[TicketFlowAction],
) -> Dict[str, object]:
    details = {
        "correct_classification": state.classification_label == "refund_request",
        "followed_policy": _policy_ok(state),
        "used_valid_resolution": state.resolution_action in {
            "deny_refund",
            "offer_store_credit",
            "escalate_to_human",
        },
        "reply_sent": state.reply_sent,
        "closed_ticket": state.closed and state.resolved,
    }
    resolution_score = 0.0
    if state.resolution_action in {"deny_refund", "offer_store_credit"}:
        resolution_score = 1.0
    elif state.resolution_action == "escalate_to_human":
        resolution_score = 0.7

    score = (
        0.20 * float(details["correct_classification"])
        + 0.30 * float(details["followed_policy"])
        + 0.25 * resolution_score
        + 0.15 * float(details["reply_sent"])
        + 0.10 * float(details["closed_ticket"])
    )
    return {"score": max(0.0, min(1.0, round(score, 4))), "details": details}


def grade_task(task_id: str, state: TicketFlowState, action_history: List[TicketFlowAction]) -> Dict[str, object]:
    if task_id == "easy_refund":
        return grade_easy_refund(state, action_history)
    if task_id == "account_access_ambiguity":
        return grade_account_access_ambiguity(state, action_history)
    if task_id == "out_of_policy_refund":
        return grade_out_of_policy_refund(state, action_history)
    return {"score": 0.0, "reason": f"No grading logic defined for task: {task_id}"}
