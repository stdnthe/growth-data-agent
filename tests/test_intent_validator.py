from __future__ import annotations

import unittest
from pathlib import Path

from agent.intent_validator import IntentValidationError, IntentValidator
from agent.metrics_store import MetricsStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_type": "metric_query",
        "metric": "gmv",
        "metrics": ["gmv"],
        "time_range": None,
        "window_days": None,
        "comparison": None,
        "grain": None,
        "dimensions": [],
        "filters": {},
        "business_goal": None,
        "ambiguities": [],
        "assumptions": [],
        "confidence": 0.9,
        "clarification_question": None,
    }
    payload.update(overrides)
    return payload


class IntentValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store = MetricsStore(PROJECT_ROOT / "metrics" / "metrics.yml")
        cls.validator = IntentValidator(store.load())

    def test_deduplicates_metrics_and_keeps_primary_consistent(self) -> None:
        intent = self.validator.validate(
            _payload(metric="paid_orders", metrics=["gmv", "paid_orders", "gmv"])
        )
        self.assertEqual(intent.metrics, ["paid_orders", "gmv"])
        self.assertEqual(intent.metric, "paid_orders")

    def test_missing_metric_requires_clarification(self) -> None:
        intent = self.validator.validate(
            _payload(metric=None, metrics=[], confidence=0.8, clarification_question=None)
        )
        self.assertEqual(intent.action, "clarify")
        self.assertTrue(intent.clarification_question)

    def test_fulfillment_diagnosis_selects_recipe(self) -> None:
        intent = self.validator.validate(
            _payload(
                task_type="metric_diagnosis",
                metric="on_time_delivery_rate",
                metrics=["on_time_delivery_rate"],
            )
        )
        self.assertEqual(intent.workflow, "fulfillment_diagnosis")
        self.assertEqual(intent.action, "execute")

    def test_unsupported_diagnosis_requires_confirmation(self) -> None:
        intent = self.validator.validate(
            _payload(
                task_type="metric_diagnosis",
                metric="average_review_score",
                metrics=["average_review_score"],
            )
        )
        self.assertEqual(intent.workflow, "retrieval")
        self.assertEqual(intent.action, "confirm")

    def test_gmv_diagnosis_caps_attribution_window(self) -> None:
        intent = self.validator.validate(
            _payload(task_type="metric_diagnosis", window_days=180)
        )
        self.assertEqual(intent.workflow, "gmv_attribution")
        self.assertEqual(intent.window_days, 120)

    def test_invalid_dimension_is_rejected(self) -> None:
        with self.assertRaises(IntentValidationError):
            self.validator.validate(_payload(dimensions=["campaign"]))

    def test_unsupported_grain_requires_clarification(self) -> None:
        intent = self.validator.validate(
            _payload(metric="p90_delivery_days", metrics=["p90_delivery_days"], grain="day")
        )
        self.assertEqual(intent.action, "clarify")

    def test_unknown_or_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(IntentValidationError):
            self.validator.validate(_payload(workflow="gmv_attribution"))


if __name__ == "__main__":
    unittest.main()
