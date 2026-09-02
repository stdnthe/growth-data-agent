from __future__ import annotations

import unittest
from pathlib import Path

from agent.context_builder import ContextBuilder
from agent.metrics_store import MetricsStore
from agent.models import AnalysisIntent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        store = MetricsStore(PROJECT_ROOT / "metrics" / "metrics.yml")
        store.load()
        self.builder = ContextBuilder(store)

    def test_bundle_scopes_schema_and_output_contract(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            grain="day",
            dimensions=["category"],
            confidence=0.95,
        )
        bundle = self.builder.build(intent)
        self.assertEqual(bundle.metric_ids, ("gmv",))
        self.assertIn("vw_fact_items", bundle.allowed_tables)
        self.assertIn("products", bundle.allowed_tables)
        self.assertIn("metric:gmv", bundle.output_contract.required_columns)
        self.assertIn("dimension:category", bundle.output_contract.required_columns)
        self.assertIn("time_grain:day", bundle.output_contract.required_columns)

    def test_delivery_bundle_includes_geography_join_rule(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="on_time_delivery_rate",
            metrics=["on_time_delivery_rate"],
            dimensions=["state"],
            confidence=0.95,
        )
        bundle = self.builder.build(intent)
        self.assertIn("vw_delivered_orders", bundle.allowed_tables)
        self.assertIn("customers", bundle.allowed_tables)
        self.assertTrue(any("customer_id" in rule for rule in bundle.join_rules))


if __name__ == "__main__":
    unittest.main()
