"""Deterministic policy and workflow helpers for TicketFlowEnv."""

from __future__ import annotations

from typing import Set

from ticketflow_env.models import TicketFlowAction, TicketFlowState
from tasks import AVAILABLE_ACTIONS


RESOLUTION_ACTIONS = {
    "request_more_info",
    "approve_refund",
    "deny_refund",
    "offer_replacement",
    "offer_store_credit",
    "escalate_to_human",
}


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def is_valid_action_type(action_type: str) -> bool:
    return action_type in AVAILABLE_ACTIONS


def classification_matches(action: TicketFlowAction, state: TicketFlowState) -> bool:
    if action.action_type != "classify_issue":
        return False
    return _normalize(str(action.metadata.get("label"))) == _normalize(state.issue_type)


def expected_next_actions(state: TicketFlowState) -> Set[str]:
    if not state.classification_done:
        return {"classify_issue"}
    if state.issue_type == "damaged_item":
        if state.resolution_action is None:
            return {"approve_refund"}
        if not state.reply_sent:
            return {"send_customer_reply"}
        if not state.closed:
            return {"close_ticket"}
        return set()
    if state.issue_type == "account_access_issue":
        if not state.request_more_info_done:
            return {"request_more_info"}
        if not state.reply_sent:
            return {"send_customer_reply"}
        return set()
    if state.issue_type == "refund_request":
        if state.resolution_action is None:
            return {"deny_refund", "offer_store_credit", "escalate_to_human"}
        if not state.reply_sent:
            return {"send_customer_reply"}
        if not state.closed:
            return {"close_ticket"}
        return set()
    return set()


def is_policy_compliant(action_type: str, state: TicketFlowState) -> bool:
    if action_type == "classify_issue":
        return True
    if action_type == "request_more_info":
        return state.issue_type == "account_access_issue"
    if action_type == "approve_refund":
        return state.refund_allowed
    if action_type == "deny_refund":
        return state.issue_type == "refund_request" and not state.refund_allowed
    if action_type == "offer_replacement":
        return state.issue_type == "damaged_item" and state.replacement_allowed
    if action_type == "offer_store_credit":
        return state.issue_type == "refund_request" and not state.refund_allowed
    if action_type == "escalate_to_human":
        if state.escalation_required:
            return True
        if state.issue_type == "refund_request":
            return True
        if state.issue_type == "account_access_issue" and state.request_more_info_done:
            return True
        return False
    if action_type == "send_customer_reply":
        return state.classification_done and (state.request_more_info_done or state.resolution_action is not None)
    if action_type == "close_ticket":
        return can_close_ticket(state)
    return False


def is_harmful_action(action_type: str, state: TicketFlowState) -> bool:
    if action_type == "approve_refund" and not state.refund_allowed:
        return True
    if action_type == "deny_refund" and state.issue_type == "damaged_item" and state.refund_allowed:
        return True
    if action_type in {"offer_replacement", "offer_store_credit"} and state.issue_type == "account_access_issue":
        return True
    return False


def is_redundant_action(action_type: str, state: TicketFlowState) -> bool:
    if action_type == "classify_issue":
        return state.classification_done
    if action_type == "request_more_info":
        return state.request_more_info_done
    if action_type == "send_customer_reply":
        return state.reply_sent
    if action_type == "escalate_to_human":
        return state.escalated
    if action_type == "close_ticket":
        return state.closed
    if action_type in RESOLUTION_ACTIONS - {"request_more_info", "escalate_to_human"}:
        return state.resolution_action is not None
    return False


def is_unnecessary_escalation(action_type: str, state: TicketFlowState) -> bool:
    if action_type != "escalate_to_human":
        return False
    if state.issue_type == "damaged_item":
        return True
    if state.issue_type == "account_access_issue" and not state.request_more_info_done:
        return True
    return False


def can_close_ticket(state: TicketFlowState) -> bool:
    if state.issue_type == "account_access_issue":
        return False
    return bool(state.reply_sent and state.resolution_action is not None)


def resolution_quality(action_type: str, state: TicketFlowState) -> float:
    if state.issue_type == "damaged_item":
        return 1.0 if action_type == "approve_refund" else 0.0
    if state.issue_type == "account_access_issue":
        return 1.0 if action_type == "request_more_info" else 0.0
    if state.issue_type == "refund_request":
        if action_type in {"deny_refund", "offer_store_credit"}:
            return 1.0
        if action_type == "escalate_to_human":
            return 0.7
        return 0.0
    return 0.0
