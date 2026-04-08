"""FastAPI app wrapper for TicketFlowEnv."""

from __future__ import annotations

from fastapi import FastAPI

from ticketflow_env.models import (
    TicketFlowAction,
    TicketFlowObservation,
    TicketFlowResetRequest,
    TicketFlowState,
    TicketFlowStepResult,
)
from ticketflow_env.server.environment import TicketFlowEnvironment


def create_app() -> FastAPI:
    environment = TicketFlowEnvironment()
    app = FastAPI(
        title="TicketFlowEnv",
        version="0.1.0",
        description=(
            "A realistic customer support ticket resolution environment for OpenEnv-compatible "
            "agent evaluation."
        ),
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "TicketFlowEnv",
            "status": "ok",
            "description": "Customer support ticket resolution benchmark environment.",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metadata")
    def metadata() -> dict[str, str]:
        return {
            "name": "TicketFlowEnv",
            "description": (
                "A realistic customer support ticket resolution environment for "
                "OpenEnv-compatible agent evaluation. Benchmarks AI agents on "
                "classification, policy compliance, and resolution quality."
            ),
        }

    @app.get("/schema")
    def schema() -> dict[str, object]:
        return {
            "action": TicketFlowAction.model_json_schema(),
            "observation": TicketFlowObservation.model_json_schema(),
            "state": TicketFlowState.model_json_schema(),
        }

    @app.post("/reset", response_model=TicketFlowObservation)
    @app.post("/env/reset", response_model=TicketFlowObservation)
    def reset(payload: TicketFlowResetRequest | None = None) -> TicketFlowObservation:
        task_id = payload.task_id if payload is not None else None
        return environment.reset(task_id=task_id)

    @app.post("/step", response_model=TicketFlowStepResult)
    @app.post("/env/step", response_model=TicketFlowStepResult)
    def step(action: TicketFlowAction) -> TicketFlowStepResult:
        return environment.step(action)

    @app.get("/state", response_model=TicketFlowState)
    @app.get("/env/state", response_model=TicketFlowState)
    def state() -> TicketFlowState:
        return environment.state

    return app


app = create_app()
