from __future__ import annotations

import unittest

from agent.agent_loop import BoundedLoopController, LoopObservation
from agent.models import AnalysisIntent


class BoundedLoopControllerTests(unittest.TestCase):
    def test_metric_query_starts_with_query_action_and_finishes_after_validation(self) -> None:
        controller = BoundedLoopController(max_steps=3)
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            confidence=0.95,
        )
        first = controller.initial(intent)
        self.assertEqual(first.action, "query_metric")

        controller.mark_action_executed()
        final = controller.after_observation(LoopObservation(stage="validation", passed=True))
        self.assertEqual(final.action, "finish")
        self.assertEqual(final.completion_reason, "validated")

    def test_execution_error_retrieves_schema_before_retry(self) -> None:
        controller = BoundedLoopController(max_steps=3)
        controller.mark_action_executed()
        decision = controller.after_observation(
            LoopObservation(
                stage="execution",
                passed=False,
                retryable=True,
                error_summary="missing column",
            )
        )
        self.assertEqual(decision.action, "inspect_schema")

        controller.mark_action_executed()
        retry = controller.after_schema_inspection()
        self.assertEqual(retry.action, "query_metric")

    def test_step_budget_stops_repeated_failures(self) -> None:
        controller = BoundedLoopController(max_steps=1)
        controller.mark_action_executed()
        decision = controller.after_observation(
            LoopObservation(stage="guard", passed=False, retryable=True, error_summary="unsafe SQL")
        )
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.completion_reason, "step_budget_exhausted")

    def test_gmv_diagnosis_selects_recipe(self) -> None:
        controller = BoundedLoopController(max_steps=3)
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="gmv_attribution",
            metric="gmv",
            metrics=["gmv"],
            confidence=0.95,
        )
        self.assertEqual(controller.initial(intent).action, "run_gmv_recipe")

    def test_fulfillment_diagnosis_selects_recipe(self) -> None:
        controller = BoundedLoopController(max_steps=3)
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="fulfillment_diagnosis",
            metric="on_time_delivery_rate",
            metrics=["on_time_delivery_rate"],
            confidence=0.95,
        )
        self.assertEqual(controller.initial(intent).action, "run_fulfillment_recipe")


if __name__ == "__main__":
    unittest.main()
