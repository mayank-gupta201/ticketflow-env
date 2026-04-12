"""Typed models for TicketFlowEnv."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field


def _resolve_openenv_bases() -> Tuple[Type[BaseModel], Type[BaseModel], Type[BaseModel]]:
    """Resolve OpenEnv base models across known package layouts."""

    candidates = [
        ("openenv.types", "Action", "Observation", "State"),
        ("openenv.models", "Action", "Observation", "State"),
        ("openenv.core.types", "Action", "Observation", "State"),
        ("openenv_core.types", "Action", "Observation", "State"),
    ]

    for module_name, action_name, observation_name, state_name in candidates:
        try:
            module = import_module(module_name)
            return (
                getattr(module, action_name),
                getattr(module, observation_name),
                getattr(module, state_name),
            )
        except Exception:
            continue

    class _CompatModel(BaseModel):
        model_config = ConfigDict(extra="ignore")

    return _CompatModel, _CompatModel, _CompatModel


ActionBase, ObservationBase, StateBase = _resolve_openenv_bases()


class TicketFlowAction(ActionBase):
    """Structured agent action."""

    model_config = ConfigDict(extra="forbid")

    action_type: str
    response_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TicketFlowObservation(ObservationBase):
    """Visible ticket context returned to the agent."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    customer_message: str
    customer_tier: str
    order_status: str
    order_value: int
    order_age_days: int
    conversation_history: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    current_status: str
    last_action_result: Optional[str] = None
    payment_method: Optional[str] = None
    delivery_partner: Optional[str] = None


class TicketFlowState(StateBase):
    """Hidden internal environment state used for transitions and grading."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    issue_type: str
    correct_resolution: str
    refund_allowed: bool
    replacement_allowed: bool
    escalation_required: bool
    classification_done: bool = False
    reply_sent: bool = False
    resolved: bool = False
    failed: bool = False
    max_steps: int = 6
    task_id: str
    step_count: int = 0
    current_status: str = "open"
    customer_tier: str = "basic"
    order_value: float = 0.0
    order_age_days: int = 0
    last_action_result: Optional[str] = None
    classification_label: Optional[str] = None
    resolution_action: Optional[str] = None
    request_more_info_done: bool = False
    escalated: bool = False
    closed: bool = False
    helpful_reply: bool = False
    cumulative_reward: float = 0.0
    policy_violations: int = 0


class TicketFlowStepResult(BaseModel):
    """Step transition payload."""

    model_config = ConfigDict(extra="forbid")

    observation: TicketFlowObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class TicketFlowResetRequest(BaseModel):
    """Reset request body."""

    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = None
