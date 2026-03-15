from __future__ import annotations

import pandas as pd


def _fmt_num(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v:,.2f}"  # keep full value with commas
    if abs(v) >= 1_000:
        return f"{v:,.0f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def generate_insight(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "查询已执行，但结果为空。可尝试扩大时间范围或调整筛选条件。"

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        return f"查询成功，共返回 {len(df)} 行，请查看结果表格。"

    gmv_candidates = [c for c in numeric_cols if "gmv" in c.lower()]
    if gmv_candidates:
        col = gmv_candidates[0]
        total = float(df[col].fillna(0).sum())
        return f"查询成功：结果共 {len(df)} 行，{col} 汇总为 {_fmt_num(total)}。"

    col = numeric_cols[0]
    total = float(df[col].fillna(0).sum())
    return f"查询成功：结果共 {len(df)} 行，数值列 {col} 汇总为 {_fmt_num(total)}。"
