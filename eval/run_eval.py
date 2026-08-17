"""Offline evaluation for Text-to-SQL correctness and end-to-end task success."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from agent.llm_sql import generate_sql, resolve_llm_config
from agent.metrics_store import MetricsStore
from agent.sql_guard import SQLGuard, SQLGuardError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
DB_PATH = PROJECT_ROOT / "olist.duckdb"
NUMERIC_TOLERANCE = 0.05


@dataclass
class QuestionResult:
    id: int
    question: str
    difficulty: str
    category: str
    generation_success: bool = False
    guard_pass: bool = False
    execution_success: bool = False
    has_golden: bool = False
    semantic_pass: bool | None = None
    has_compliance: bool = False
    compliance_pass: bool | None = None
    generated_sql: str = ""
    sql_source: str = ""
    failure_stage: str = ""
    failure_detail: str = ""

    @property
    def end_to_end_pass(self) -> bool:
        semantic_ok = self.semantic_pass is True if self.has_golden else True
        compliance_ok = self.compliance_pass is True if self.has_compliance else True
        return all(
            (
                self.generation_success,
                self.guard_pass,
                self.execution_success,
                semantic_ok,
                compliance_ok,
            )
        )


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = pd.to_datetime(normalized[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if not normalized.empty:
        sort_keys = list(normalized.columns)
        try:
            normalized = normalized.sort_values(sort_keys, kind="stable", na_position="last")
        except TypeError:
            normalized = normalized.astype(str).sort_values(sort_keys, kind="stable", na_position="last")
    return normalized.reset_index(drop=True)


def _compare_full_result(golden_df: pd.DataFrame, result_df: pd.DataFrame) -> tuple[bool, str]:
    if result_df.empty and not golden_df.empty:
        return False, "result is empty"

    golden = _normalize_frame(golden_df)
    result = _normalize_frame(result_df)

    if list(result.columns) != list(golden.columns):
        return False, f"column mismatch: got {list(result.columns)}, expected {list(golden.columns)}"
    if len(result) != len(golden):
        return False, f"row count mismatch: got {len(result)}, expected {len(golden)}"

    for col in golden.columns:
        golden_series = golden[col]
        result_series = result[col]
        if pd.api.types.is_numeric_dtype(golden_series) and pd.api.types.is_numeric_dtype(result_series):
            left = pd.to_numeric(golden_series, errors="coerce")
            right = pd.to_numeric(result_series, errors="coerce")
            both_na = left.isna() & right.isna()
            denominator = left.abs().clip(lower=1e-9)
            relative_diff = (right - left).abs() / denominator
            matches = both_na | (relative_diff <= NUMERIC_TOLERANCE) | ((right - left).abs() <= 1e-6)
            if not bool(matches.all()):
                bad_index = int(matches[~matches].index[0])
                return (
                    False,
                    f"numeric mismatch at row {bad_index}, column {col}: "
                    f"got {right.iloc[bad_index]}, expected {left.iloc[bad_index]}",
                )
        else:
            left = golden_series.fillna("<NULL>").astype(str)
            right = result_series.fillna("<NULL>").astype(str)
            matches = left == right
            if not bool(matches.all()):
                bad_index = int(matches[~matches].index[0])
                return (
                    False,
                    f"value mismatch at row {bad_index}, column {col}: "
                    f"got {right.iloc[bad_index]!r}, expected {left.iloc[bad_index]!r}",
                )

    return True, "ok"


def check_semantic(
    golden_df: pd.DataFrame,
    result_df: pd.DataFrame,
    check_type: str,
) -> tuple[bool, str]:
    if check_type in {"result_match", "row_count_gte_1"}:
        return _compare_full_result(golden_df, result_df)

    if check_type == "row_count_match":
        if len(result_df) != len(golden_df):
            return False, f"row count mismatch: got {len(result_df)}, expected {len(golden_df)}"
        return _compare_full_result(golden_df, result_df)

    if check_type == "numeric_close":
        golden = _normalize_frame(golden_df)
        result = _normalize_frame(result_df)
        if len(result) != len(golden):
            return False, f"row count mismatch: got {len(result)}, expected {len(golden)}"
        golden_numbers = golden.select_dtypes(include="number")
        result_numbers = result.select_dtypes(include="number")
        if list(golden_numbers.columns) != list(result_numbers.columns):
            return False, "numeric columns mismatch"
        return _compare_full_result(golden_numbers, result_numbers)

    return False, f"unsupported check_type: {check_type}"


def check_compliance(
    sql: str,
    required_tables: list[str],
    required_patterns: list[str],
) -> tuple[bool, str]:
    sql_lower = sql.lower()
    if required_tables and not any(table.lower() in sql_lower for table in required_tables):
        return False, f"none of required tables found: {required_tables}"
    for pattern in required_patterns:
        if not re.search(pattern, sql, re.IGNORECASE):
            return False, f"required pattern not found: {pattern}"
    return True, "ok"


def _pct(num: int, den: int) -> float:
    return round(num / den * 100, 2) if den else 0.0


def build_summary(results: list[QuestionResult]) -> dict:
    total = len(results)
    semantic_cases = [result for result in results if result.has_golden]
    compliance_cases = [result for result in results if result.has_compliance]
    return {
        "total": total,
        "generation_success_rate_pct": _pct(sum(r.generation_success for r in results), total),
        "guard_pass_rate_pct": _pct(sum(r.guard_pass for r in results), total),
        "execution_success_rate_pct": _pct(sum(r.execution_success for r in results), total),
        "semantic_pass_rate_pct": _pct(sum(r.semantic_pass is True for r in semantic_cases), len(semantic_cases)),
        "semantic_denominator": len(semantic_cases),
        "compliance_pass_rate_pct": _pct(
            sum(r.compliance_pass is True for r in compliance_cases), len(compliance_cases)
        ),
        "compliance_denominator": len(compliance_cases),
        "end_to_end_task_success_rate_pct": _pct(sum(r.end_to_end_pass for r in results), total),
        "end_to_end_success": sum(r.end_to_end_pass for r in results),
    }


def _build_breakdown(results: list[QuestionResult], key: str) -> dict:
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        groups.setdefault(str(getattr(result, key)), []).append(result)
    return {
        value: {
            "total": len(group),
            "execution_success_rate_pct": _pct(sum(r.execution_success for r in group), len(group)),
            "semantic_pass_rate_pct": _pct(sum(r.semantic_pass is True for r in group), len(group)),
            "end_to_end_task_success_rate_pct": _pct(sum(r.end_to_end_pass for r in group), len(group)),
        }
        for value, group in sorted(groups.items())
    }


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_evaluation(questions_path: Path, use_llm: bool) -> dict:
    questions = load_questions(questions_path)
    llm_config = resolve_llm_config()
    metrics_context = MetricsStore(METRICS_PATH).compressed_context(max_items=30)
    guard = SQLGuard(default_limit=500)
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    results: list[QuestionResult] = []

    try:
        for row in questions:
            result = QuestionResult(
                id=int(row.get("id", 0)),
                question=str(row.get("question", "")).strip(),
                difficulty=str(row.get("difficulty", "unknown")),
                category=str(row.get("category", "unknown")),
            )
            golden_sql = str(row.get("golden_sql", "")).strip()
            check_type = str(row.get("check_type", "result_match"))
            required_tables = list(row.get("required_tables", []))
            required_patterns = list(row.get("required_patterns", []))
            result.has_golden = bool(golden_sql)
            result.has_compliance = bool(required_tables or required_patterns)

            try:
                generated_sql, source = generate_sql(
                    result.question,
                    metrics_context,
                    use_llm=use_llm,
                )
                result.generated_sql = generated_sql
                result.sql_source = source
                result.generation_success = True
            except Exception as exc:  # noqa: BLE001
                result.failure_stage = "generation"
                result.failure_detail = str(exc)
                results.append(result)
                continue

            try:
                guarded_sql = guard.validate_and_rewrite(generated_sql)
                result.guard_pass = True
            except SQLGuardError as exc:
                result.failure_stage = "guard"
                result.failure_detail = str(exc)
                results.append(result)
                continue

            if result.has_compliance:
                result.compliance_pass, reason = check_compliance(
                    guarded_sql, required_tables, required_patterns
                )
                if not result.compliance_pass:
                    result.failure_stage = "compliance"
                    result.failure_detail = reason

            try:
                actual_df = connection.execute(guarded_sql).fetchdf()
                result.execution_success = True
            except Exception as exc:  # noqa: BLE001
                result.failure_stage = "execution"
                result.failure_detail = str(exc)
                results.append(result)
                continue

            if result.has_golden:
                try:
                    golden_df = connection.execute(golden_sql).fetchdf()
                    result.semantic_pass, reason = check_semantic(golden_df, actual_df, check_type)
                    if not result.semantic_pass:
                        result.failure_stage = "semantic"
                        result.failure_detail = reason
                except Exception as exc:  # noqa: BLE001
                    result.semantic_pass = False
                    result.failure_stage = "semantic_golden_error"
                    result.failure_detail = str(exc)

            results.append(result)
    finally:
        connection.close()

    failures = [
        {
            "id": result.id,
            "question": result.question,
            "difficulty": result.difficulty,
            "category": result.category,
            "failure_stage": result.failure_stage,
            "failure_detail": result.failure_detail,
            "sql_source": result.sql_source,
            "generated_sql": result.generated_sql[:500],
        }
        for result in results
        if not result.end_to_end_pass
    ]
    return {
        "evaluation_metadata": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "questions_file": str(questions_path.relative_to(PROJECT_ROOT)),
            "provider": llm_config["provider"] if use_llm else "deterministic_fallback",
            "model": llm_config["model"] if use_llm else "rule_fallback",
            "strict_result_contract": "exact columns + complete result equivalence",
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
        "requested_mode": "llm" if use_llm else "deterministic_fallback",
        "sql_source_counts": dict(Counter(result.sql_source or "generation_failed" for result in results)),
        "summary": build_summary(results),
        "by_difficulty": _build_breakdown(results, "difficulty"),
        "by_category": _build_breakdown(results, "category"),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--use-llm", action="store_true", help="Call the configured remote model.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    for path in (args.questions, METRICS_PATH, DB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")

    report = run_evaluation(args.questions, use_llm=args.use_llm)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
