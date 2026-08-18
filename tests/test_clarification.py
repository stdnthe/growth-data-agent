from __future__ import annotations

import unittest

from agent.clarification import resolve_clarification
from agent.intent_router import route


class ClarificationTests(unittest.TestCase):
    def test_missing_metric_answer_becomes_executable_question(self) -> None:
        question = "最近销售表现怎么样？"
        intent = route(question)
        resolved = resolve_clarification(question, intent, "GMV")
        self.assertIsNotNone(resolved)
        rerouted = route(resolved or "")
        self.assertEqual(rerouted.action, "execute")
        self.assertEqual(rerouted.metric, "gmv")

    def test_unsupported_diagnosis_can_continue_as_trend(self) -> None:
        question = "为什么准时送达率下降？"
        intent = route(question)
        resolved = resolve_clarification(question, intent, "先查看该指标趋势")
        self.assertEqual(resolved, "最近30天准时送达率趋势")
        rerouted = route(resolved or "")
        self.assertEqual(rerouted.action, "execute")
        self.assertEqual(rerouted.workflow, "retrieval")

    def test_user_can_cancel_clarification(self) -> None:
        question = "最近销售表现怎么样？"
        self.assertIsNone(resolve_clarification(question, route(question), "暂不分析"))


if __name__ == "__main__":
    unittest.main()
