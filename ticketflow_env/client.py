"""Minimal typed HTTP client for TicketFlowEnv."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import error, request

from ticketflow_env.models import (
    TicketFlowAction,
    TicketFlowObservation,
    TicketFlowState,
    TicketFlowStepResult,
)


class TicketFlowEnvClient:
    """Lightweight client aligned with OpenEnv-style HTTP usage."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8")
            raise RuntimeError(f"HTTP {exc.code} calling {path}: {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach TicketFlowEnv at {self.base_url}") from exc

    def reset(self, task_id: Optional[str] = None) -> TicketFlowObservation:
        payload: Dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        data = self._request("POST", "/reset", payload)
        return TicketFlowObservation.model_validate(data)

    def step(self, action: TicketFlowAction | Dict[str, Any]) -> TicketFlowStepResult:
        payload = action.model_dump() if isinstance(action, TicketFlowAction) else action
        data = self._request("POST", "/step", payload)
        return TicketFlowStepResult.model_validate(data)

    def state(self) -> TicketFlowState:
        data = self._request("GET", "/state")
        return TicketFlowState.model_validate(data)

    def close(self) -> None:
        """Provided for API symmetry with OpenEnv examples."""

        return None
