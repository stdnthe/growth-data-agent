from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from agent.feedback import FeedbackRecord, FeedbackStore
from agent.models import AnalysisIntent
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

    def test_pipeline_config_repr_hides_api_key(self) -> None:
        config = PipelineConfig(
            db_path=PROJECT_ROOT / "olist.duckdb",
            metrics_path=PROJECT_ROOT / "metrics" / "metrics.yml",
            provider="openai",
            model="gpt-test",
            api_key="visitor-secret-key",
        )
        self.assertNotIn("visitor-secret-key", repr(config))

    def test_clarification_does_not_query_database(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            action="clarify",
            confidence=0.4,
            ambiguities=["未识别明确的业务指标"],
            clarification_question="你希望分析哪个指标？",
        )
        with patch("agent.pipeline.route", return_value=intent):
            run = execute_analysis("最近销售表现怎么样？", self.config)
        self.assertEqual(run.status, "needs_clarification")
        self.assertIsNone(run.generated_sql)
        self.assertEqual(run.steps[-1].key, "human_confirmation")

    def test_controller_decision_is_captured_as_a_trace_span(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            action="clarify",
            confidence=0.4,
            ambiguities=["未识别明确的业务指标"],
            clarification_question="你希望分析哪个指标？",
        )
        span_names: list[str] = []

        class RecordingTracer:
            available = True

            @contextmanager
            def span(self, name, **_):
                span_names.append(name)
                yield SimpleNamespace(set_outputs=lambda **__: None)

        with (
            patch("agent.pipeline.route", return_value=intent),
            patch("agent.pipeline.TraceRecorder", return_value=RecordingTracer()),
        ):
            run = execute_analysis("最近销售表现怎么样？", self.config)

        self.assertIn("select_next_action", span_names)
        self.assertEqual(run.decisions[0].action, "clarify")

    @unittest.skipUnless((PROJECT_ROOT / "olist.duckdb").exists(), "demo database is missing")
    def test_retrieval_returns_auditable_run(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            time_range="最近30天",
            window_days=30,
            grain="day",
            confidence=0.95,
        )
        with patch("agent.pipeline.route", return_value=intent):
            run = execute_analysis("近30天GMV走势（按天）", self.config)
        self.assertEqual(run.status, "success")
        self.assertTrue(run.run_id)
        self.assertTrue(run.validation_passed)
        self.assertIsNotNone(run.generated_sql)
        self.assertGreater(run.result_summary()["row_count"], 0)
        step_keys = {step.key for step in run.steps}
        self.assertIn("parse_intent", step_keys)
        self.assertIn("validate_intent", step_keys)
        self.assertIn("validate_sql_1", step_keys)
        self.assertIn("execute_duckdb_1", step_keys)
        self.assertIn("validate_result_1", step_keys)
        self.assertIn("assemble_context", step_keys)
        self.assertEqual([item.action for item in run.decisions], ["query_metric", "finish"])
        self.assertIn("output_contract", {item.name for item in run.validations})
        self.assertTrue(run.context_bundle["schemas"])
        self.assertEqual(run.active_capabilities, ["metric_query"])
        self.assertEqual(run.to_trace_dict()["active_capabilities"], ["metric_query"])

    @unittest.skipUnless((PROJECT_ROOT / "olist.duckdb").exists(), "demo database is missing")
    def test_gmv_diagnosis_reuses_deterministic_attribution_workflow(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="gmv_attribution",
            metric="gmv",
            metrics=["gmv"],
            time_range="最近30天",
            window_days=30,
            comparison="previous_period",
            confidence=0.95,
        )
        with patch("agent.pipeline.route", return_value=intent):
            run = execute_analysis("为什么最近GMV下降？", self.config)
        self.assertIn(run.status, {"success", "success_with_warnings"})
        self.assertIsNotNone(run.attribution_report)
        self.assertIsNone(run.generated_sql)
        self.assertIn("calculate_gmv_attribution", {step.key for step in run.steps})
        self.assertEqual([item.action for item in run.decisions], ["run_gmv_recipe", "finish"])

    @unittest.skipUnless((PROJECT_ROOT / "olist.duckdb").exists(), "demo database is missing")
    def test_fulfillment_diagnosis_runs_additive_recipe(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_diagnosis",
            workflow="fulfillment_diagnosis",
            metric="on_time_delivery_rate",
            metrics=["on_time_delivery_rate"],
            time_range="最近30天",
            window_days=30,
            comparison="previous_period",
            confidence=0.95,
        )
        with patch("agent.pipeline.route", return_value=intent):
            run = execute_analysis("为什么最近准时送达率下降？", self.config)
        self.assertIn(run.status, {"success", "success_with_warnings"})
        self.assertIsNotNone(run.fulfillment_report)
        self.assertIsNone(run.generated_sql)
        self.assertEqual(
            [item.action for item in run.decisions],
            ["run_fulfillment_recipe", "finish"],
        )
        self.assertIn(
            "fulfillment_mix_within_additive",
            {item.name for item in run.validations},
        )

    def test_execution_failure_inspects_schema_then_repairs_query(self) -> None:
        intent = AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            metric="gmv",
            metrics=["gmv"],
            time_range="最近30天",
            window_days=30,
            grain="day",
            confidence=0.95,
        )
        config = PipelineConfig(
            db_path=PROJECT_ROOT / "olist.duckdb",
            metrics_path=PROJECT_ROOT / "metrics" / "metrics.yml",
            use_llm=True,
            max_retries=1,
            max_steps=3,
            tracing_enabled=False,
        )
        valid_result = pd.DataFrame({"dt": ["2018-08-01"], "gmv": [100.0]})
        with (
            patch("agent.pipeline.route", return_value=intent),
            patch(
                "agent.pipeline.generate_sql",
                side_effect=[("SELECT missing_column FROM vw_fact_items", "llm"), ("SELECT 1", "llm")],
            ) as generate,
            patch("agent.pipeline._run_query", side_effect=[RuntimeError("missing column"), valid_result]),
        ):
            run = execute_analysis("近30天GMV走势（按天）", config)

        self.assertEqual(run.status, "success")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [item.action for item in run.decisions],
            ["query_metric", "inspect_schema", "query_metric", "finish"],
        )
        self.assertIn("inspect_schema_2", {step.key for step in run.steps})
        self.assertIn("Relevant schema retrieved", generate.call_args_list[1].kwargs["retry_context"])

    def test_feedback_store_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            FeedbackStore(path).append(
                FeedbackRecord(run_id="run-1", feedback_type="结果正确", helpful=True)
            )
            self.assertIn('"run_id": "run-1"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
