"""Optional LangSmith dataset upload and offline experiment runner.

The module is import-safe without LangSmith. Use --upload-dataset once, then
--run-experiment to compare a model/prompt configuration on the 40-question set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from agent.pipeline import PipelineConfig, execute_analysis
from eval.run_eval import check_compliance, check_semantic, load_questions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
DB_PATH = PROJECT_ROOT / "olist.duckdb"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
DEFAULT_DATASET = "olist-growth-agent-golden-v2"


def _require_langsmith() -> Any:
    try:
        from langsmith import Client
    except ImportError as exc:
        raise SystemExit(
            "LangSmith is optional. Install it with:\n"
            "  pip install -r requirements-observability.txt"
        ) from exc
    return Client


def build_examples() -> list[dict[str, Any]]:
    examples = []
    for row in load_questions(QUESTIONS_PATH):
        examples.append(
            {
                "inputs": {"question": row["question"]},
                "outputs": {
                    "golden_sql": row["golden_sql"],
                    "check_type": row.get("check_type", "result_match"),
                    "required_tables": row.get("required_tables", []),
                    "required_patterns": row.get("required_patterns", []),
                },
                "metadata": {
                    "case_id": row["id"],
                    "difficulty": row.get("difficulty", "unknown"),
                    "category": row.get("category", "unknown"),
                },
            }
        )
    return examples


def upload_dataset(dataset_name: str) -> None:
    Client = _require_langsmith()
    client = Client()
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Olist Growth Agent golden Text-to-SQL regression set v2",
    )
    examples = build_examples()
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"uploaded {len(examples)} examples to {dataset_name}")


def _serialize_frame(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None:
        return {"columns": [], "data": []}
    return json.loads(frame.to_json(orient="split", date_format="iso"))


def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    run = execute_analysis(
        str(inputs["question"]),
        PipelineConfig(
            db_path=DB_PATH,
            metrics_path=METRICS_PATH,
            use_llm=True,
            max_retries=1,
            tracing_enabled=True,
            prompt_version=os.getenv("AGENT_PROMPT_VERSION", "growth-sql-v2"),
        ),
    )
    return {
        "status": run.status,
        "workflow": run.workflow,
        "metric": run.intent.metric,
        "generated_sql": run.generated_sql or "",
        "validation_passed": run.validation_passed,
        "result": _serialize_frame(run.result),
        "failure_stage": run.failure_stage,
    }


def _semantic_evaluator(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    actual = outputs.get("result", {})
    actual_frame = pd.DataFrame(data=actual.get("data", []), columns=actual.get("columns", []))
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        golden_frame = connection.execute(reference_outputs["golden_sql"]).fetchdf()
    finally:
        connection.close()
    passed, reason = check_semantic(
        golden_frame,
        actual_frame,
        str(reference_outputs.get("check_type", "result_match")),
    )
    return {"key": "result_correctness", "score": int(passed), "comment": reason}


def _compliance_evaluator(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    passed, reason = check_compliance(
        str(outputs.get("generated_sql", "")),
        list(reference_outputs.get("required_tables", [])),
        list(reference_outputs.get("required_patterns", [])),
    )
    return {"key": "metric_compliance", "score": int(passed), "comment": reason}


def _end_to_end_evaluator(outputs: dict[str, Any]) -> dict[str, Any]:
    passed = outputs.get("status") in {"success", "success_with_warnings"} and bool(
        outputs.get("validation_passed")
    )
    return {
        "key": "pipeline_success",
        "score": int(passed),
        "comment": str(outputs.get("failure_stage") or "ok"),
    }


def run_experiment(dataset_name: str, experiment_prefix: str) -> None:
    Client = _require_langsmith()
    client = Client()
    result = client.evaluate(
        _target,
        data=dataset_name,
        evaluators=[_semantic_evaluator, _compliance_evaluator, _end_to_end_evaluator],
        experiment_prefix=experiment_prefix,
        metadata={
            "prompts": [os.getenv("AGENT_PROMPT_VERSION", "growth-sql-v2")],
            "models": [os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"],
        },
    )
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--upload-dataset", action="store_true")
    parser.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--experiment-prefix", default="growth-agent-v2")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.preview:
        print(json.dumps(build_examples()[:2], ensure_ascii=False, indent=2))
    if args.upload_dataset:
        upload_dataset(args.dataset)
    if args.run_experiment:
        run_experiment(args.dataset, args.experiment_prefix)
    if not any((args.preview, args.upload_dataset, args.run_experiment)):
        parser.print_help()


if __name__ == "__main__":
    main()
