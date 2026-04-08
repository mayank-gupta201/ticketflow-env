"""Dense deterministic reward helpers for TicketFlowEnv."""

from __future__ import annotations

from typing import Dict, Tuple


def clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, round(value, 4)))


def compute_step_reward(
    *,
    valid_action: bool,
    correct_classification: bool,
    correct_workflow: bool,
    policy_compliant: bool,
    resolution_credit: float,
    helpful_reply: bool,
    harmful: bool,
    redundant: bool,
    unnecessary_escalation: bool,
    premature_close: bool,
) -> Tuple[float, Dict[str, float | bool]]:
    reward = 0.0
    details: Dict[str, float | bool] = {
        "valid_action": valid_action,
        "correct_classification": correct_classification,
        "correct_workflow": correct_workflow,
        "policy_compliant": policy_compliant,
        "resolution_credit": resolution_credit,
        "helpful_reply": helpful_reply,
        "harmful": harmful,
        "redundant": redundant,
        "unnecessary_escalation": unnecessary_escalation,
        "premature_close": premature_close,
    }

    if correct_classification:
        reward += 0.20
    if correct_workflow:
        reward += 0.25
    if policy_compliant:
        reward += 0.20
    if resolution_credit > 0.0:
        reward += 0.25 * resolution_credit
    if helpful_reply:
        reward += 0.10

    if not valid_action:
        reward -= 0.30
    if harmful:
        reward -= 0.40
    if redundant:
        reward -= 0.10
    if unnecessary_escalation:
        reward -= 0.15
    if premature_close:
        reward -= 0.25

    reward = clamp_reward(reward)
    details["reward"] = reward
    return reward, details
