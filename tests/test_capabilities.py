from __future__ import annotations

import unittest

from agent.capabilities import CapabilityRegistry
from agent.models import AnalysisIntent


class CapabilityRegistryTests(unittest.TestCase):
    def test_metric_query_exposes_only_query_capability(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
        )
        registry = CapabilityRegistry()

        self.assertEqual([item.id for item in registry.select(intent)], ["metric_query"])
        self.assertEqual(registry.actions_for(intent), ("query_metric",))

    def test_gmv_diagnosis_prefers_deterministic_recipe(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="gmv_attribution",
            metric="gmv",
            metrics=["gmv"],
        )
        registry = CapabilityRegistry()

        self.assertEqual([item.id for item in registry.select(intent)], ["gmv_diagnosis"])
        self.assertEqual(registry.preferred_action(intent), "run_gmv_recipe")
        self.assertEqual(registry.actions_for(intent), ("run_gmv_recipe", "query_metric"))

    def test_future_capabilities_are_not_exposed_before_implementation(self) -> None:
        registered = {item["id"] for item in CapabilityRegistry().describe(
            AnalysisIntent(
                task_type="metric_query",
                workflow="retrieval",
                metric="orders",
                metrics=["orders"],
            )
        )}

        self.assertNotIn("funnel_analysis", registered)
        self.assertNotIn("time_series_forecast", registered)

    def test_diagnosis_capability_requires_a_compatible_metric(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="gmv_attribution",
            metric="orders",
            metrics=["orders"],
        )

        self.assertEqual(CapabilityRegistry().actions_for(intent), ())


if __name__ == "__main__":
    unittest.main()
