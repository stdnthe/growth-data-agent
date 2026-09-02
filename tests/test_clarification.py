from __future__ import annotations

import unittest

from agent.clarification import resolve_clarification
from agent.models import AnalysisIntent


class ClarificationTests(unittest.TestCase):
    def test_missing_metric_answer_becomes_executable_question(self) -> None:
        question = "最近销售表现怎么样？"
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            action="clarify",
            clarification_question="你希望分析哪个指标？",
        )
        resolved = resolve_clarification(question, intent, "GMV")
        self.assertEqual(resolved, "近30天GMV趋势（按天）")

    def test_unsupported_diagnosis_can_continue_as_trend(self) -> None:
        question = "为什么准时送达率下降？"
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="retrieval",
            metric="on_time_delivery_rate",
            metrics=["on_time_delivery_rate"],
            time_range="最近30天",
            action="confirm",
        )
        resolved = resolve_clarification(question, intent, "先查看该指标趋势")
        self.assertEqual(resolved, "最近30天准时送达率趋势")

    def test_user_can_cancel_clarification(self) -> None:
        question = "最近销售表现怎么样？"
        intent = AnalysisIntent(task_type="metric_query", workflow="retrieval", action="clarify")
        self.assertIsNone(resolve_clarification(question, intent, "暂不分析"))


if __name__ == "__main__":
    unittest.main()
