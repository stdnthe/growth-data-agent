from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from agent.agent_loop import BoundedLoopController, LoopObservation
from agent.models import AnalysisIntent
from agent.planner import AgentController, AgentPlanner, AgentPolicyGate, ToolRegistry


def _response(action: str) -> SimpleNamespace:
    payload = {
        "action": action,
        "reason_summary": "选择最合适的受控分析工具。",
        "parameters": {},
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric_intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            confidence=0.95,
        )

    def test_llm_action_is_approved_inside_tool_policy(self) -> None:
        planner = AgentPlanner(
            use_llm=True,
            completion_create=lambda **_: _response("query_metric"),
        )
        decision = planner.decide(
            question="近30天GMV走势",
            intent=self.metric_intent,
            observation=None,
            controller=BoundedLoopController(max_steps=3),
        )
        self.assertEqual(decision.action, "query_metric")
        self.assertEqual(decision.decision_source, "llm")
        self.assertEqual(decision.policy_status, "approved")

    def test_legacy_planner_name_points_to_single_controller(self) -> None:
        self.assertIs(AgentPlanner, AgentController)

    def test_policy_gate_corrects_invented_or_disallowed_action(self) -> None:
        planner = AgentPlanner(
            use_llm=True,
            completion_create=lambda **_: _response("run_gmv_recipe"),
        )
        decision = planner.decide(
            question="近30天GMV走势",
            intent=self.metric_intent,
            observation=None,
            controller=BoundedLoopController(max_steps=3),
        )
        self.assertEqual(decision.action, "query_metric")
        self.assertEqual(decision.decision_source, "fallback_policy")
        self.assertEqual(decision.policy_status, "corrected")

    def test_validated_observation_can_only_finish(self) -> None:
        planner = AgentPlanner(use_llm=True, completion_create=lambda **_: self.fail("no call"))
        decision = planner.decide(
            question="近30天GMV走势",
            intent=self.metric_intent,
            observation=LoopObservation(stage="validation", passed=True),
            controller=BoundedLoopController(max_steps=3),
        )
        self.assertEqual(decision.action, "finish")
        self.assertEqual(decision.decision_source, "policy")

    def test_valid_trend_is_not_sufficient_for_a_diagnosis(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="gmv_attribution",
            metric="gmv",
            metrics=["gmv"],
            confidence=0.95,
        )
        planner = AgentPlanner(use_llm=False)
        decision = planner.decide(
            question="为什么最近GMV下降？",
            intent=intent,
            observation=LoopObservation(
                stage="validation",
                passed=True,
                evidence_sufficient=False,
            ),
            controller=BoundedLoopController(max_steps=3),
        )
        self.assertEqual(decision.action, "run_gmv_recipe")

    def test_registry_exposes_fulfillment_recipe_only_for_supported_diagnosis(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="fulfillment_diagnosis",
            metric="on_time_delivery_rate",
            metrics=["on_time_delivery_rate"],
        )
        actions = ToolRegistry().allowed_actions(intent, None)
        self.assertIn("run_fulfillment_recipe", actions)
        self.assertNotIn("run_gmv_recipe", actions)

    def test_policy_gate_rejects_tool_when_budget_is_exhausted(self) -> None:
        action, error = AgentPolicyGate().validate(
            "query_metric",
            allowed_actions=("query_metric", "stop"),
            remaining_steps=0,
        )
        self.assertIsNone(action)
        self.assertIn("budget", str(error))


if __name__ == "__main__":
    unittest.main()
