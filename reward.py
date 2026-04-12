"""Dense deterministic reward helpers for TicketFlowEnv."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float | bool]]:
    classification = 0.3 if correct_classification else 0.0
    policy_compliance = 0.3 if policy_compliant else 0.0
    response_quality = (0.2 * resolution_credit) + (0.1 if helpful_reply else 0.0)
    efficiency = 0.1 if correct_workflow else 0.0
    
    penalty = 0.0
    if not valid_action:
        penalty -= 0.30
    if harmful:
        penalty -= 0.40
    if redundant:
        penalty -= 0.10
    if unnecessary_escalation:
        penalty -= 0.15
    if premature_close:
        penalty -= 0.30

    # Anti-reward-hacking: penalize consecutive identical actions
    repeated_action = bool(context and context.get("repeated_action"))
    if repeated_action:
        penalty -= 0.20

    base_reward = classification + policy_compliance + response_quality + efficiency + penalty

    # Efficiency bonus: reward agents that complete tasks in fewer steps
    efficiency_bonus = 0.0
    if context and "step_count" in context and "max_steps" in context:
        step_efficiency = max(0.0, 1.0 - (context["step_count"] / context["max_steps"]))
        efficiency_bonus = round(0.05 * step_efficiency, 4)

    total_reward = clamp_reward(base_reward + efficiency_bonus)

    details: Dict[str, float | bool] = {
        "classification": classification,
        "policy_compliance": policy_compliance,
        "response_quality": round(response_quality, 4),
        "efficiency": efficiency,
        "penalty": round(penalty, 4),
        "efficiency_bonus": efficiency_bonus,
        "repeated_action": repeated_action,
        "reward": total_reward,
    }
    return total_reward, details
