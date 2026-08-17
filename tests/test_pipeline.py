from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.feedback import FeedbackRecord, FeedbackStore
from agent.pipeline import PipelineConfig, execute_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig(
            db_path=PROJECT_ROOT / "olist.duckdb",
            metrics_path=PROJECT_ROOT / "metrics" / "metrics.yml",
            use_llm=False,
            max_retries=0,
            tracing_enabled=False,
        )

    def test_clarification_does_not_query_database(self) -> None:
        run = execute_analysis("最近销售表现怎么样？", self.config)
        self.assertEqual(run.status, "needs_clarification")
        self.assertIsNone(run.generated_sql)

    @unittest.skipUnless((PROJECT_ROOT / "olist.duckdb").exists(), "demo database is missing")
    def test_retrieval_returns_auditable_run(self) -> None:
        run = execute_analysis("近30天GMV走势（按天）", self.config)
        self.assertEqual(run.status, "success")
        self.assertTrue(run.run_id)
        self.assertTrue(run.validation_passed)
        self.assertIsNotNone(run.generated_sql)
        self.assertGreater(run.result_summary()["row_count"], 0)

    def test_feedback_store_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            FeedbackStore(path).append(
                FeedbackRecord(run_id="run-1", feedback_type="结果正确", helpful=True)
            )
            self.assertIn('"run_id": "run-1"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
