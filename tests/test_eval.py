from __future__ import annotations

import unittest

import pandas as pd

from eval.run_eval import QuestionResult, build_summary, check_semantic
from eval.langsmith_experiment import build_examples


class EvaluationTests(unittest.TestCase):
    def test_full_result_match_is_order_insensitive(self) -> None:
        golden = pd.DataFrame({"state": ["SP", "RJ"], "gmv": [100.0, 50.0]})
        actual = pd.DataFrame({"state": ["RJ", "SP"], "gmv": [50.0, 100.0]})
        passed, reason = check_semantic(golden, actual, "result_match")
        self.assertTrue(passed, reason)

    def test_full_result_match_rejects_wrong_value(self) -> None:
        golden = pd.DataFrame({"gmv": [100.0]})
        actual = pd.DataFrame({"gmv": [80.0]})
        passed, _ = check_semantic(golden, actual, "result_match")
        self.assertFalse(passed)

    def test_semantic_denominator_includes_execution_failures(self) -> None:
        passed = QuestionResult(1, "q1", "easy", "gmv", True, True, True, True, True, True, True)
        failed = QuestionResult(2, "q2", "easy", "gmv", True, True, False, True, False, True, True)
        summary = build_summary([passed, failed])
        self.assertEqual(summary["semantic_denominator"], 2)
        self.assertEqual(summary["semantic_pass_rate_pct"], 50.0)
        self.assertEqual(summary["end_to_end_task_success_rate_pct"], 50.0)

    def test_langsmith_examples_contain_reference_outputs(self) -> None:
        examples = build_examples()
        self.assertEqual(len(examples), 40)
        self.assertIn("golden_sql", examples[0]["outputs"])
        self.assertIn("difficulty", examples[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
