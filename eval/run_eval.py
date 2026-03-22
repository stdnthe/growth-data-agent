"""
三层 Eval 框架：
  Layer 1 - 管道健康度：SQL 生成 → Guard 通过 → 执行成功
  Layer 2 - 语义正确性：与 golden_sql 结果对比（行数 / 数值近似）
  Layer 3 - 指标合规性：静态分析 SQL 是否引用了正确的表和关键模式
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from agent.llm_sql import generate_sql
from agent.metrics_store import MetricsStore
from agent.sql_guard import SQLGuard, SQLGuardError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
DB_PATH = PROJECT_ROOT / "olist.duckdb"

NUMERIC_TOLERANCE = 0.05  # 数值对比允许 5% 误差


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: int
    question: str
    difficulty: str
    category: str
    # Layer 1
    generation_success: bool = False
    guard_pass: bool = False
    execution_success: bool = False
    # Layer 2
    has_golden: bool = False
    semantic_pass: bool | None = None  # None = 未检查
    # Layer 3
    has_compliance: bool = False
    compliance_pass: bool | None = None  # None = 未检查
    # 调试信息
    generated_sql: str = ""
    failure_stage: str = ""
    failure_detail: str = ""


# ---------------------------------------------------------------------------
# Layer 2：语义正确性对比
# ---------------------------------------------------------------------------

def _check_semantic(
    golden_df: pd.DataFrame,
    result_df: pd.DataFrame,
    check_type: str,
) -> tuple[bool, str]:
    """返回 (pass, reason)"""
    if result_df.empty:
        return False, "result is empty"

    if check_type == "row_count_gte_1":
        return True, "ok"

    if check_type == "row_count_match":
        if len(result_df) != len(golden_df):
            return False, f"row count mismatch: got {len(result_df)}, expected {len(golden_df)}"
        return True, "ok"

    if check_type == "numeric_close":
        g_nums = golden_df.select_dtypes(include="number")
        r_nums = result_df.select_dtypes(include="number")
        if g_nums.empty or r_nums.empty:
            return True, "no numeric cols to compare"
        g_total = float(g_nums.iloc[:, 0].sum())
        r_total = float(r_nums.iloc[:, 0].sum())
        if g_total == 0:
            return r_total == 0, f"golden total=0, result total={r_total}"
        diff = abs(r_total - g_total) / abs(g_total)
        if diff > NUMERIC_TOLERANCE:
            return False, f"numeric diff {diff:.2%} > tolerance {NUMERIC_TOLERANCE:.0%}"
        return True, "ok"

    # 默认：columns_match - 结果至少包含 golden 的数值列数量
    g_numeric_count = len(golden_df.select_dtypes(include="number").columns)
    r_numeric_count = len(result_df.select_dtypes(include="number").columns)
    if r_numeric_count < g_numeric_count:
        return False, f"too few numeric cols: got {r_numeric_count}, expected >= {g_numeric_count}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Layer 3：指标合规性（静态分析）
# ---------------------------------------------------------------------------

def _check_compliance(
    sql: str,
    required_tables: list[str],
    required_patterns: list[str],
) -> tuple[bool, str]:
    """返回 (pass, reason)"""
    sql_lower = sql.lower()

    for table in required_tables:
        # 至少有一个必须出现（OR 逻辑）
        if not any(t.lower() in sql_lower for t in required_tables):
            return False, f"none of required tables found: {required_tables}"
        break  # 只需有一个命中即可

    for pattern in required_patterns:
        if not re.search(pattern, sql, re.IGNORECASE):
            return False, f"required pattern not found: {pattern}"

    return True, "ok"


# ---------------------------------------------------------------------------
# 统计汇总
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> float:
    return round(num / den * 100, 2) if den else 0.0


def _build_summary(results: list[QuestionResult]) -> dict:
    total = len(results)
    gen_ok = sum(1 for r in results if r.generation_success)
    guard_ok = sum(1 for r in results if r.guard_pass)
    exec_ok = sum(1 for r in results if r.execution_success)

    semantic_eligible = [r for r in results if r.has_golden and r.execution_success]
    semantic_ok = sum(1 for r in semantic_eligible if r.semantic_pass)

    compliance_eligible = [r for r in results if r.has_compliance and r.guard_pass]
    compliance_ok = sum(1 for r in compliance_eligible if r.compliance_pass)

    return {
        "total": total,
        "generation_success": gen_ok,
        "guard_pass": guard_ok,
        "execution_success": exec_ok,
        "semantic_eligible": len(semantic_eligible),
        "semantic_pass": semantic_ok,
        "compliance_eligible": len(compliance_eligible),
        "compliance_pass": compliance_ok,
        "generation_success_rate_pct": _pct(gen_ok, total),
        "guard_pass_rate_pct": _pct(guard_ok, total),
        "execution_success_rate_pct": _pct(exec_ok, total),
        "semantic_pass_rate_pct": _pct(semantic_ok, len(semantic_eligible)),
        "compliance_pass_rate_pct": _pct(compliance_ok, len(compliance_eligible)),
    }


def _build_breakdown(results: list[QuestionResult], key: str) -> dict:
    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        val = getattr(r, key)
        groups.setdefault(val, []).append(r)

    breakdown = {}
    for val, group in sorted(groups.items()):
        total = len(group)
        exec_ok = sum(1 for r in group if r.execution_success)
        sem_eligible = [r for r in group if r.has_golden and r.execution_success]
        sem_ok = sum(1 for r in sem_eligible if r.semantic_pass)
        breakdown[val] = {
            "total": total,
            "execution_success_rate_pct": _pct(exec_ok, total),
            "semantic_pass_rate_pct": _pct(sem_ok, len(sem_eligible)) if sem_eligible else None,
        }
    return breakdown


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    for p in (QUESTIONS_PATH, METRICS_PATH, DB_PATH):
        if not p.exists():
            raise FileNotFoundError(f"not found: {p}")

    questions = load_questions(QUESTIONS_PATH)
    metrics_context = MetricsStore(METRICS_PATH).compressed_context(max_items=30)
    guard = SQLGuard(default_limit=500)
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    results: list[QuestionResult] = []

    try:
        for row in questions:
            qid = int(row.get("id", 0))
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            res = QuestionResult(
                id=qid,
                question=question,
                difficulty=str(row.get("difficulty", "unknown")),
                category=str(row.get("category", "unknown")),
            )

            golden_sql = str(row.get("golden_sql", "")).strip()
            check_type = str(row.get("check_type", "row_count_gte_1"))
            required_tables: list[str] = row.get("required_tables", [])
            required_patterns: list[str] = row.get("required_patterns", [])
            res.has_golden = bool(golden_sql)
            res.has_compliance = bool(required_tables or required_patterns)

            # --- Layer 1a: 生成 SQL ---
            try:
                sql, _ = generate_sql(question, metrics_context, use_llm=True)
                res.generation_success = True
                res.generated_sql = sql
            except Exception as e:
                res.failure_stage = "generation"
                res.failure_detail = str(e)
                results.append(res)
                continue

            # --- Layer 1b: Guard ---
            try:
                guarded_sql = guard.validate_and_rewrite(sql)
                res.guard_pass = True
            except SQLGuardError as e:
                res.failure_stage = "guard"
                res.failure_detail = str(e)
                results.append(res)
                continue

            # --- Layer 3: 合规性静态检查（Guard 通过后做）---
            if res.has_compliance:
                comp_pass, comp_reason = _check_compliance(
                    guarded_sql, required_tables, required_patterns
                )
                res.compliance_pass = comp_pass
                if not comp_pass and not res.failure_detail:
                    res.failure_detail = f"compliance: {comp_reason}"

            # --- Layer 1c: 执行 ---
            try:
                result_df = conn.execute(guarded_sql).fetchdf()
                res.execution_success = True
            except Exception as e:
                res.failure_stage = "execution"
                res.failure_detail = str(e)
                results.append(res)
                continue

            # --- Layer 2: 语义正确性对比 ---
            if res.has_golden:
                try:
                    golden_df = conn.execute(golden_sql).fetchdf()
                    sem_pass, sem_reason = _check_semantic(golden_df, result_df, check_type)
                    res.semantic_pass = sem_pass
                    if not sem_pass and not res.failure_stage:
                        res.failure_stage = "semantic"
                        res.failure_detail = sem_reason
                except Exception as e:
                    res.semantic_pass = False
                    res.failure_stage = "semantic_golden_error"
                    res.failure_detail = str(e)

            results.append(res)

    finally:
        conn.close()

    # --- 输出报告 ---
    summary = _build_summary(results)
    by_difficulty = _build_breakdown(results, "difficulty")
    by_category = _build_breakdown(results, "category")

    failures = [
        {
            "id": r.id,
            "question": r.question,
            "difficulty": r.difficulty,
            "category": r.category,
            "failure_stage": r.failure_stage,
            "failure_detail": r.failure_detail,
            "generated_sql": r.generated_sql[:300] if r.generated_sql else "",
        }
        for r in results
        if r.failure_stage or r.semantic_pass is False or r.compliance_pass is False
    ]

    report = {
        "summary": summary,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "failures": failures,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
