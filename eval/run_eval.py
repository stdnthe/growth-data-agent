from __future__ import annotations

import json
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from agent.llm_sql import generate_sql
from agent.metrics_store import MetricsStore
from agent.sql_guard import SQLGuard, SQLGuardError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
DB_PATH = PROJECT_ROOT / "olist.duckdb"


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pct(num: int, den: int) -> float:
    return round((num / den) * 100, 2) if den else 0.0


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions not found: {QUESTIONS_PATH}")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics not found: {METRICS_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"duckdb not found: {DB_PATH}")

    questions = load_questions(QUESTIONS_PATH)
    metrics_context = MetricsStore(METRICS_PATH).compressed_context(max_items=30)
    guard = SQLGuard(default_limit=500)

    report = {
        "total": len(questions),
        "generation_success": 0,
        "guard_pass": 0,
        "execution_success": 0,
    }

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for row in questions:
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            try:
                sql, _ = generate_sql(question, metrics_context, use_llm=True)
                report["generation_success"] += 1
            except Exception:
                continue

            try:
                guarded_sql = guard.validate_and_rewrite(sql)
                report["guard_pass"] += 1
            except SQLGuardError:
                continue

            try:
                conn.execute(guarded_sql).fetchdf()
                report["execution_success"] += 1
            except Exception:
                continue
    finally:
        conn.close()

    report["generation_success_rate_pct"] = pct(report["generation_success"], report["total"])
    report["guard_pass_rate_pct"] = pct(report["guard_pass"], report["total"])
    report["execution_success_rate_pct"] = pct(report["execution_success"], report["total"])

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
