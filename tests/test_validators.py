from __future__ import annotations

import unittest

import pandas as pd

from agent.attribution import GmvAttributionReport
from agent.fulfillment import FulfillmentDiagnosisReport
from agent.models import AnalysisIntent
from agent.validators import (
    validate_fulfillment_diagnosis,
    validate_gmv_attribution,
    validate_retrieval_result,
)


def _report(order_effect: float = 20.0, aov_effect: float = 30.0) -> GmvAttributionReport:
    columns = ["dimension", "dimension_value", "gmv_current", "gmv_previous", "contribution"]
    empty = pd.DataFrame(columns=columns)
    return GmvAttributionReport(
        window_days=30,
        anchor_date="2024-01-31",
        current_start="2024-01-02",
        current_end="2024-01-31",
        previous_start="2023-12-03",
        previous_end="2024-01-01",
        gmv_current=150.0,
        gmv_previous=100.0,
        gmv_delta=50.0,
        gmv_change_rate=0.5,
        orders_current=15,
        orders_previous=10,
        buyers_current=12,
        buyers_previous=8,
        aov_current=10.0,
        aov_previous=10.0,
        order_effect=order_effect,
        aov_effect=aov_effect,
        state_drops=empty.copy(),
        category_drops=empty.copy(),
        seller_drops=empty.copy(),
        state_gains=empty.copy(),
        category_gains=empty.copy(),
        seller_gains=empty.copy(),
        all_drops=empty.copy(),
        all_gains=empty.copy(),
    )


def _fulfillment_report(contribution: float = -0.05) -> FulfillmentDiagnosisReport:
    contributions = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "orders_current": [100.0],
            "orders_previous": [100.0],
            "rate_current": [0.85],
            "rate_previous": [0.90],
            "weight_current": [1.0],
            "weight_previous": [1.0],
            "mix_effect": [0.0],
            "within_effect": [contribution],
            "contribution": [contribution],
        }
    )
    return FulfillmentDiagnosisReport(
        window_days=30,
        anchor_date="2024-01-31",
        current_start="2024-01-02",
        current_end="2024-01-31",
        previous_start="2023-12-03",
        previous_end="2024-01-01",
        delivered_orders_current=100,
        delivered_orders_previous=100,
        on_time_rate_current=0.85,
        on_time_rate_previous=0.90,
        on_time_rate_delta=-0.05,
        avg_delivery_days_current=8.0,
        avg_delivery_days_previous=7.5,
        mix_effect=0.0,
        within_effect=-0.05,
        state_contributions=contributions,
    )


class ValidatorTests(unittest.TestCase):
    def test_retrieval_validates_metric_and_time_column(self) -> None:
        frame = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01"]), "gmv": [100.0]})
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            grain="day",
        )
        results = validate_retrieval_result(frame, intent)
        self.assertTrue(all(item.passed for item in results))

    def test_ratio_out_of_range_fails(self) -> None:
        frame = pd.DataFrame({"repeat_purchase_rate": [1.2]})
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="repeat_purchase_rate",
            metrics=["repeat_purchase_rate"],
        )
        results = validate_retrieval_result(frame, intent)
        ratio = next(item for item in results if item.name == "ratio_range")
        self.assertFalse(ratio.passed)

    def test_attribution_decomposition_is_checked(self) -> None:
        valid = validate_gmv_attribution(_report())
        self.assertTrue(next(item for item in valid if item.name == "gmv_decomposition_additive").passed)
        invalid = validate_gmv_attribution(_report(order_effect=10.0, aov_effect=10.0))
        self.assertFalse(next(item for item in invalid if item.name == "gmv_decomposition_additive").passed)

    def test_fulfillment_decomposition_is_checked(self) -> None:
        valid = validate_fulfillment_diagnosis(_fulfillment_report())
        self.assertTrue(all(item.passed for item in valid))
        invalid = validate_fulfillment_diagnosis(_fulfillment_report(contribution=-0.01))
        contribution_check = next(
            item for item in invalid if item.name == "fulfillment_state_contributions_additive"
        )
        self.assertFalse(contribution_check.passed)


if __name__ == "__main__":
    unittest.main()
