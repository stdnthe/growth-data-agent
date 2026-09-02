from __future__ import annotations

import unittest
from pathlib import Path

from agent.fulfillment import analyze_fulfillment_change_drivers, build_fulfillment_summary
from agent.validators import validate_fulfillment_diagnosis


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FulfillmentDiagnosisTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT_ROOT / "olist.duckdb").exists(), "demo database is missing")
    def test_real_demo_recipe_is_additive_and_evidence_backed(self) -> None:
        report = analyze_fulfillment_change_drivers(PROJECT_ROOT / "olist.duckdb", 30)
        self.assertGreater(report.delivered_orders_current, 0)
        self.assertGreater(report.delivered_orders_previous, 0)
        self.assertFalse(report.state_contributions.empty)
        self.assertAlmostEqual(
            report.mix_effect + report.within_effect,
            report.on_time_rate_delta,
            places=10,
        )
        self.assertTrue(all(item.passed for item in validate_fulfillment_diagnosis(report)))
        self.assertIn("不代表已经证明因果关系", build_fulfillment_summary(report))


if __name__ == "__main__":
    unittest.main()
