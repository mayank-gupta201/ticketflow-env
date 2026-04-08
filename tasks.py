"""Deterministic benchmark task definitions for TicketFlowEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


AVAILABLE_ACTIONS = [
    "classify_issue",
    "request_more_info",
    "approve_refund",
    "deny_refund",
    "offer_replacement",
    "offer_store_credit",
    "escalate_to_human",
    "send_customer_reply",
    "close_ticket",
]

MAX_STEPS = 6


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    description: str
    customer_message: str
    customer_tier: str
    order_status: str
    order_value: int
    order_age_days: int
    issue_type: str
    correct_resolution: str
    refund_allowed: bool
    replacement_allowed: bool
    escalation_required: bool
    expected_flow: Tuple[str, ...]


TASKS: Dict[str, TaskSpec] = {
    "easy_refund": TaskSpec(
        task_id="easy_refund",
        difficulty="easy",
        description="Damaged delivered order where policy clearly allows a refund.",
        customer_message="My package arrived broken and I want a refund.",
        customer_tier="silver",
        order_status="delivered",
        order_value=1499,
        order_age_days=3,
        issue_type="damaged_item",
        correct_resolution="approve_refund",
        refund_allowed=True,
        replacement_allowed=True,
        escalation_required=False,
        expected_flow=(
            "classify_issue",
            "approve_refund",
            "send_customer_reply",
            "close_ticket",
        ),
    ),
    "account_access_ambiguity": TaskSpec(
        task_id="account_access_ambiguity",
        difficulty="medium",
        description="Account access problem where the correct move is to gather more information.",
        customer_message="I can't access my account and can't see my order.",
        customer_tier="gold",
        order_status="unknown",
        order_value=0,
        order_age_days=0,
        issue_type="account_access_issue",
        correct_resolution="request_more_info",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=False,
        expected_flow=(
            "classify_issue",
            "request_more_info",
            "send_customer_reply",
        ),
    ),
    "out_of_policy_refund": TaskSpec(
        task_id="out_of_policy_refund",
        difficulty="hard",
        description="Change-of-mind refund request outside the refund window.",
        customer_message="It's been 35 days and I want a refund because I changed my mind.",
        customer_tier="gold",
        order_status="delivered",
        order_value=2999,
        order_age_days=35,
        issue_type="refund_request",
        correct_resolution="deny_refund_or_offer_alternative",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=False,
        expected_flow=(
            "classify_issue",
            "deny_refund",
            "send_customer_reply",
            "close_ticket",
        ),
    ),
}

TASK_SEQUENCE = tuple(TASKS.keys())


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc
