"""Basic sanity tests for TicketFlowEnv."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grader import grade_task
from ticketflow_env.models import TicketFlowAction
from ticketflow_env.server.environment import TicketFlowEnvironment


class TicketFlowEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TicketFlowEnvironment()

    def test_reset_returns_observation(self) -> None:
        observation = self.env.reset(task_id="easy_refund")
        self.assertEqual(observation.ticket_id, "easy_refund-001")
        self.assertIn("classify_issue", observation.available_actions)
        self.assertEqual(observation.current_status, "open")

    def test_step_progresses_easy_refund(self) -> None:
        self.env.reset(task_id="easy_refund")
        first = self.env.step(TicketFlowAction(action_type="classify_issue", metadata={"label": "damaged_item"}))
        self.assertGreater(first.reward, 0.0)
        second = self.env.step(TicketFlowAction(action_type="approve_refund", metadata={}))
        self.assertGreaterEqual(second.reward, 0.0)
        self.assertEqual(self.env.state.resolution_action, "approve_refund")

    def test_invalid_action_is_handled(self) -> None:
        self.env.reset(task_id="easy_refund")
        result = self.env.step(TicketFlowAction(action_type="do_magic", metadata={}))
        self.assertLess(result.reward, 0.0)
        self.assertFalse(result.done)
        self.assertIn("Invalid action_type", result.observation.last_action_result or "")

    def test_task_specific_policy_failure(self) -> None:
        self.env.reset(task_id="out_of_policy_refund")
        self.env.step(TicketFlowAction(action_type="classify_issue", metadata={"label": "refund_request"}))
        result = self.env.step(TicketFlowAction(action_type="approve_refund", metadata={}))
        self.assertTrue(result.done)
        self.assertTrue(self.env.state.failed)
        self.assertEqual(self.env.state.current_status, "policy_violation")

    def test_grader_score_range(self) -> None:
        self.env.reset(task_id="easy_refund")
        self.env.step(TicketFlowAction(action_type="classify_issue", metadata={"label": "damaged_item"}))
        self.env.step(TicketFlowAction(action_type="approve_refund", metadata={}))
        self.env.step(
            TicketFlowAction(
                action_type="send_customer_reply",
                response_text="I have approved your refund for the damaged order and processed the case.",
                metadata={},
            )
        )
        self.env.step(TicketFlowAction(action_type="close_ticket", metadata={}))
        graded = grade_task("easy_refund", self.env.state, self.env.action_history)
        self.assertGreaterEqual(float(graded["score"]), 0.0)
        self.assertLessEqual(float(graded["score"]), 1.0)


if __name__ == "__main__":
    unittest.main()
