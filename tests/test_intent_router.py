from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.intent_router import IntentRoutingError, MissingIntentAPIKeyError, route
from agent.metrics_store import MetricsStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_type": "metric_query",
        "metric": "gmv",
        "metrics": ["gmv"],
        "time_range": "最近30天",
        "window_days": 30,
        "comparison": None,
        "grain": "day",
        "dimensions": [],
        "filters": {},
        "business_goal": "retrieve_metric",
        "ambiguities": [],
        "assumptions": [],
        "confidence": 0.95,
        "clarification_question": None,
    }
    payload.update(overrides)
    return payload


def _response(payload: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
    )


class IntentRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MetricsStore(PROJECT_ROOT / "metrics" / "metrics.yml")
        self.runtime = {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_name": "DEEPSEEK_API_KEY",
        }

    def _route(self, question: str, payload: object):
        with patch("agent.intent_router.resolve_llm_config", return_value=self.runtime):
            return route(question, self.store, completion_create=lambda **_: _response(payload))

    def test_structured_payload_is_validated_and_routed(self) -> None:
        intent = self._route("近30天GMV走势（按天）", _payload())
        self.assertEqual(intent.metric, "gmv")
        self.assertEqual(intent.workflow, "retrieval")
        self.assertEqual(intent.action, "execute")

    def test_gmv_diagnosis_is_forced_to_existing_attribution_workflow(self) -> None:
        intent = self._route(
            "为什么最近GMV下降？",
            _payload(
                task_type="metric_diagnosis",
                comparison="previous_period",
                grain=None,
                business_goal="identify_change_drivers",
            ),
        )
        self.assertEqual(intent.workflow, "gmv_attribution")
        self.assertEqual(intent.metric, "gmv")
        self.assertEqual(intent.action, "execute")

    def test_fulfillment_diagnosis_selects_supported_recipe(self) -> None:
        intent = self._route(
            "为什么最近准时送达率下降？",
            _payload(
                task_type="metric_diagnosis",
                metric="on_time_delivery_rate",
                metrics=["on_time_delivery_rate"],
                comparison="previous_period",
                grain=None,
                business_goal="identify_fulfillment_change_drivers",
            ),
        )
        self.assertEqual(intent.workflow, "fulfillment_diagnosis")
        self.assertEqual(intent.action, "execute")

    def test_json_object_output_mode_is_requested(self) -> None:
        captured: dict[str, object] = {}

        def create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _response(_payload())

        with patch("agent.intent_router.resolve_llm_config", return_value=self.runtime):
            route("近30天GMV", self.store, completion_create=create)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertIn("Metric semantic layer", str(captured["messages"]))

    def test_missing_key_fails_without_keyword_fallback(self) -> None:
        runtime = {**self.runtime, "api_key": None}
        with patch("agent.intent_router.resolve_llm_config", return_value=runtime):
            with self.assertRaises(MissingIntentAPIKeyError):
                route("近30天GMV走势（按天）", self.store)

    def test_invalid_json_fails_without_keyword_fallback(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        with patch("agent.intent_router.resolve_llm_config", return_value=self.runtime):
            with self.assertRaisesRegex(IntentRoutingError, "invalid JSON"):
                route("近30天GMV走势（按天）", self.store, completion_create=lambda **_: response)

    def test_unknown_metric_fails_validation(self) -> None:
        with self.assertRaisesRegex(IntentRoutingError, "outside the semantic layer"):
            self._route("查询利润", _payload(metric="profit", metrics=["profit"], grain=None))

    def test_empty_question_is_deterministically_rejected_before_model_call(self) -> None:
        intent = route("", self.store, completion_create=lambda **_: self.fail("model should not be called"))
        self.assertEqual(intent.action, "clarify")
        self.assertIsNone(intent.metric)


if __name__ == "__main__":
    unittest.main()
