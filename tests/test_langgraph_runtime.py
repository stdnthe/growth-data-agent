from __future__ import annotations

import unittest

from agent.agent_loop import BoundedLoopController, LoopObservation
from agent.langgraph_runtime import run_bounded_agent_graph
from agent.models import AnalysisIntent


class LangGraphRuntimeTests(unittest.TestCase):
    def test_conditional_edges_execute_schema_recovery_trajectory(self) -> None:
        controller = BoundedLoopController(max_steps=3)
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            confidence=0.95,
        )
        initial = controller.initial(intent)
        actions: list[str] = []
        decisions = [initial.action]
        query_count = 0

        def execute(decision):
            nonlocal query_count
            actions.append(decision.action)
            if decision.action == "inspect_schema":
                return LoopObservation(stage="schema", passed=True, schema_inspected=True)
            query_count += 1
            if query_count == 1:
                return LoopObservation(
                    stage="execution",
                    passed=False,
                    retryable=True,
                    error_summary="missing column",
                )
            return LoopObservation(stage="validation", passed=True, schema_inspected=True)

        final = run_bounded_agent_graph(
            initial_decision=initial,
            controller=controller,
            execute_action=execute,
            record_decision=lambda item: decisions.append(item.action),
        )

        self.assertEqual(actions, ["query_metric", "inspect_schema", "query_metric"])
        self.assertEqual(decisions, ["query_metric", "inspect_schema", "query_metric", "finish"])
        self.assertEqual(final["decision"].action, "finish")


if __name__ == "__main__":
    unittest.main()
