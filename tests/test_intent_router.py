from __future__ import annotations

import unittest

from agent.intent_router import route


class IntentRouterTests(unittest.TestCase):
    def test_gmv_trend_is_executable_retrieval(self) -> None:
        intent = route("近30天GMV走势（按天）")
        self.assertEqual(intent.metric, "gmv")
        self.assertEqual(intent.workflow, "retrieval")
        self.assertEqual(intent.action, "execute")
        self.assertEqual(intent.window_days, 30)
        self.assertEqual(intent.grain, "day")

    def test_gmv_why_question_routes_to_attribution(self) -> None:
        intent = route("为什么最近GMV下降？")
        self.assertEqual(intent.workflow, "gmv_attribution")
        self.assertEqual(intent.task_type, "metric_diagnosis")
        self.assertEqual(intent.comparison, "previous_period")

    def test_vague_sales_question_requires_clarification(self) -> None:
        intent = route("最近销售表现怎么样？")
        self.assertEqual(intent.action, "clarify")
        self.assertIsNone(intent.metric)
        self.assertTrue(intent.clarification_question)

    def test_unsupported_diagnosis_requires_confirmation(self) -> None:
        intent = route("为什么准时送达率下降？")
        self.assertEqual(intent.action, "confirm")
        self.assertEqual(intent.metric, "on_time_delivery_rate")

    def test_multi_metric_question_preserves_all_requested_metrics(self) -> None:
        intent = route("上个月订单数、GMV、客单价分别是多少？")
        self.assertEqual(intent.metrics, ["paid_orders", "gmv", "aov"])

    def test_specific_metric_suppresses_overlapping_generic_alias(self) -> None:
        intent = route("Top10州GMV占比是多少？")
        self.assertEqual(intent.metrics, ["gmv_top10_state_share"])

    def test_multi_metric_diagnosis_prefers_supported_gmv_workflow(self) -> None:
        intent = route("为什么订单数和GMV都下降？")
        self.assertEqual(intent.metric, "gmv")
        self.assertEqual(intent.workflow, "gmv_attribution")
        self.assertEqual(intent.action, "execute")


if __name__ == "__main__":
    unittest.main()
