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
    followup_response: Optional[str] = None
    payment_method: Optional[str] = "credit_card"
    delivery_partner: Optional[str] = "standard_shipping"


TASKS: Dict[str, TaskSpec] = {
    "easy_refund": TaskSpec(
        task_id="easy_refund",
        difficulty="easy",
        description="Damaged delivered order where policy clearly allows a refund.",
        customer_message="My package arrived broken and I want a refund.",
        customer_tier="silver",
        order_status="delivered",
        order_value=149,
        order_age_days=3,
        issue_type="damaged_item",
        correct_resolution="approve_refund",
        refund_allowed=True,
        replacement_allowed=True,
        escalation_required=False,
        expected_flow=("classify_issue", "approve_refund", "send_customer_reply", "close_ticket"),
        payment_method="credit_card",
        delivery_partner="federal_express",
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
        expected_flow=("classify_issue", "request_more_info", "send_customer_reply"),
        followup_response="My email is john@example.com and the error says 'account locked'.",
        payment_method="none",
        delivery_partner="none",
    ),
    "out_of_policy_refund": TaskSpec(
        task_id="out_of_policy_refund",
        difficulty="hard",
        description="Change-of-mind refund request outside the refund window.",
        customer_message="It's been 35 days and I want a refund because I changed my mind.",
        customer_tier="gold",
        order_status="delivered",
        order_value=299,
        order_age_days=35,
        issue_type="refund_request",
        correct_resolution="deny_refund_or_offer_alternative",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=False,
        expected_flow=("classify_issue", "offer_store_credit", "send_customer_reply", "close_ticket"),
        payment_method="paypal",
        delivery_partner="standard_shipping",
    ),
    "angry_customer": TaskSpec(
        task_id="angry_customer",
        difficulty="medium",
        description="Customer is angry but clearly requesting a refund within policy window.",
        customer_message="THIS IS OUTRAGEOUS! The product is completely damaged! Give me my money back NOW!",
        customer_tier="basic",
        order_status="delivered",
        order_value=45,
        order_age_days=2,
        issue_type="damaged_item",
        correct_resolution="approve_refund",
        refund_allowed=True,
        replacement_allowed=True,
        escalation_required=False,
        expected_flow=("classify_issue", "approve_refund", "send_customer_reply", "close_ticket"),
        payment_method="credit_card",
        delivery_partner="quick_ship",
    ),
    "vip_edge_case": TaskSpec(
        task_id="vip_edge_case",
        difficulty="hard",
        description="VIP customer requesting a refund at 45 days (allowed under VIP 60-day rule).",
        customer_message="Hi, I bought this 45 days ago but just opened it and it's broken. Can I still get a refund?",
        customer_tier="vip",
        order_status="delivered",
        order_value=550,
        order_age_days=45,
        issue_type="damaged_item",
        correct_resolution="approve_refund",
        refund_allowed=True,
        replacement_allowed=True,
        escalation_required=False,
        expected_flow=("classify_issue", "approve_refund", "send_customer_reply", "close_ticket"),
        payment_method="apple_pay",
        delivery_partner="federal_express",
    ),
    "fraud_suspicion": TaskSpec(
        task_id="fraud_suspicion",
        difficulty="hard",
        description="Customer marked as suspicious tier demands a refund.",
        customer_message="Refund me right now for my 5 iphones. They never arrived.",
        customer_tier="suspicious",
        order_status="delivered",
        order_value=5000,
        order_age_days=5,
        issue_type="refund_request",
        correct_resolution="escalate_to_human",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=True,
        expected_flow=("classify_issue", "escalate_to_human", "send_customer_reply", "close_ticket"),
        payment_method="crypto",
        delivery_partner="standard_shipping",
    ),
    "incomplete_info": TaskSpec(
        task_id="incomplete_info",
        difficulty="medium",
        description="Missing context for the request, requires agent to investigate first.",
        customer_message="I have an issue.",
        customer_tier="silver",
        order_status="shipped",
        order_value=120,
        order_age_days=10,
        issue_type="account_access_issue", # treating generic missing info as an investigate loop
        correct_resolution="request_more_info",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=False,
        expected_flow=("classify_issue", "request_more_info", "send_customer_reply"),
        followup_response="Oh, I can't log into my account to see my order tracking.",
        payment_method="credit_card",
        delivery_partner="standard_shipping",
    ),
    "high_value_order": TaskSpec(
        task_id="high_value_order",
        difficulty="hard",
        description="Order >$1000 requires escalation before granting any refunds.",
        customer_message="The $1,500 laptop I bought has a shattered screen.",
        customer_tier="gold",
        order_status="delivered",
        order_value=1500,
        order_age_days=1,
        issue_type="damaged_item",
        correct_resolution="escalate_to_human",
        refund_allowed=False, # Wait, >1000 means standard refund not allowed, must escalate
        replacement_allowed=False,
        escalation_required=True,
        expected_flow=("classify_issue", "escalate_to_human", "send_customer_reply", "close_ticket"),
        payment_method="wire_transfer",
        delivery_partner="secure_transit_inc",
    ),
    "conflicting_signals": TaskSpec(
        task_id="conflicting_signals",
        difficulty="hard",
        description="Customer asks for a refund but says they already fixed it, just want compensation for hassle. Must deny.",
        customer_message="The item arrived slightly scratched, but I managed to buff it out. I still want a full refund for the hassle though.",
        customer_tier="silver",
        order_status="delivered",
        order_value=200,
        order_age_days=10,
        issue_type="refund_request",
        correct_resolution="deny_refund",
        refund_allowed=False,
        replacement_allowed=False,
        escalation_required=False,
        expected_flow=("classify_issue", "deny_refund", "send_customer_reply", "close_ticket"),
        payment_method="credit_card",
        delivery_partner="standard_shipping",
    ),
    "missing_delivery": TaskSpec(
        task_id="missing_delivery",
        difficulty="easy",
        description="Standard missing delivery requesting a refund.",
        customer_message="It says delivered but I never got it.",
        customer_tier="basic",
        order_status="delivered",
        order_value=80,
        order_age_days=5,
        issue_type="refund_request",
        correct_resolution="approve_refund",
        refund_allowed=True,
        replacement_allowed=True,
        escalation_required=False,
        expected_flow=("classify_issue", "approve_refund", "send_customer_reply", "close_ticket"),
        payment_method="debit_card",
        delivery_partner="local_courier",
    ),
}

TASK_SEQUENCE = tuple(TASKS.keys())


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc
