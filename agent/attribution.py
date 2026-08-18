from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from openai import OpenAI


@dataclass
class GmvAttributionReport:
    window_days: int
    anchor_date: str
    current_start: str
    current_end: str
    previous_start: str
    previous_end: str
    gmv_current: float
    gmv_previous: float
    gmv_delta: float
    gmv_change_rate: float | None
    orders_current: int
    orders_previous: int
    buyers_current: int
    buyers_previous: int
    aov_current: float | None
    aov_previous: float | None
    order_effect: float
    aov_effect: float
    state_drops: pd.DataFrame
    category_drops: pd.DataFrame
    seller_drops: pd.DataFrame
    state_gains: pd.DataFrame
    category_gains: pd.DataFrame
    seller_gains: pd.DataFrame
    all_drops: pd.DataFrame
    all_gains: pd.DataFrame
    evidence_sql: dict[str, str] = field(default_factory=dict)


def _fmt_money(v: float) -> str:
    return f"{v:,.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.2f}%"


def _safe_pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _dominant_structural_driver(report: GmvAttributionReport) -> tuple[str, float, float] | None:
    if report.gmv_delta == 0:
        return None

    sign = -1 if report.gmv_delta < 0 else 1
    candidates: list[tuple[str, float]] = []

    if report.order_effect * sign > 0:
        label = "订单量下降" if sign < 0 else "订单量增长"
        candidates.append((label, report.order_effect))
    if report.aov_effect * sign > 0:
        label = "客单价下降" if sign < 0 else "客单价增长"
        candidates.append((label, report.aov_effect))

    if not candidates:
        return None

    label, value = max(candidates, key=lambda x: abs(x[1]))
    share = abs(value) / abs(report.gmv_delta) if report.gmv_delta != 0 else 0.0
    return label, value, share


def _top_dimension_point(report: GmvAttributionReport) -> tuple[str, str, float, float] | None:
    df = report.all_drops if report.gmv_delta < 0 else report.all_gains
    if df.empty:
        return None

    row = df.iloc[0]
    dim_name = str(row["dimension"])
    dim_value = str(row["dimension_value"])
    contribution = float(row["contribution"])
    share = abs(contribution) / abs(report.gmv_delta) if report.gmv_delta != 0 else 0.0
    return dim_name, dim_value, contribution, share


def _merge_dimension_tables(
    state_df: pd.DataFrame,
    category_df: pd.DataFrame,
    seller_df: pd.DataFrame,
    trend: str,
    top_n: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dim_name, df in (
        ("州", state_df),
        ("品类", category_df),
        ("商家", seller_df),
    ):
        if df.empty:
            continue
        part = df.copy()
        part["dimension"] = dim_name
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["dimension", "dimension_value", "gmv_current", "gmv_previous", "contribution"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[["dimension", "dimension_value", "gmv_current", "gmv_previous", "contribution"]]
    if trend == "drop":
        merged = merged.sort_values("contribution", ascending=True)
    else:
        merged = merged.sort_values("contribution", ascending=False)
    return merged.head(top_n).reset_index(drop=True)


def build_gmv_summary(report: GmvAttributionReport) -> str:
    if report.gmv_delta == 0:
        return (
            f"本期 GMV 与前期基本持平（{_fmt_money(report.gmv_current)} vs {_fmt_money(report.gmv_previous)}），"
            "当前未识别到明显单一驱动因素。"
        )

    trend = "下滑" if report.gmv_delta < 0 else "增长"
    rate = report.gmv_change_rate
    if rate is not None and report.gmv_delta < 0:
        rate = abs(rate)
    base = f"本期 GMV {trend} {_fmt_pct(rate)}（变动额 {_fmt_money(report.gmv_delta)}）。"

    structural = _dominant_structural_driver(report)
    if structural is not None:
        label, effect, share = structural
        base += f"主要由{label}驱动，贡献 {_fmt_money(effect)}，占本期变动 {share * 100:.1f}%。"

    top_dim = _top_dimension_point(report)
    if top_dim is not None:
        dim_name, dim_value, contribution, share = top_dim
        dim_value = dim_value.replace("_", " ")
        base += (
            f"{dim_name}维度中，{dim_value} 贡献 {_fmt_money(contribution)}，"
            f"占总变动 {share * 100:.1f}%。"
        )
    return base


def _build_summary_sql(window_days: int) -> str:
    days = int(window_days)
    return f"""
WITH bounds AS (
  SELECT CAST(MAX(order_purchase_ts) AS DATE) AS anchor_dt
  FROM vw_fact_items
),
labeled AS (
  SELECT
    CASE
      WHEN CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {days - 1} DAY AND b.anchor_dt
        THEN 'current'
      WHEN CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt - INTERVAL {days} DAY
        THEN 'previous'
      ELSE NULL
    END AS period,
    f.order_id,
    f.customer_unique_id,
    f.line_gmv,
    b.anchor_dt
  FROM vw_fact_items f
  CROSS JOIN bounds b
  WHERE CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt
)
SELECT
  CAST(MAX(anchor_dt) AS VARCHAR) AS anchor_date,
  CAST(MAX(anchor_dt - INTERVAL {days - 1} DAY) AS VARCHAR) AS current_start,
  CAST(MAX(anchor_dt) AS VARCHAR) AS current_end,
  CAST(MAX(anchor_dt - INTERVAL {2 * days - 1} DAY) AS VARCHAR) AS previous_start,
  CAST(MAX(anchor_dt - INTERVAL {days} DAY) AS VARCHAR) AS previous_end,
  COALESCE(SUM(CASE WHEN period = 'current' THEN line_gmv ELSE 0 END), 0) AS gmv_current,
  COALESCE(SUM(CASE WHEN period = 'previous' THEN line_gmv ELSE 0 END), 0) AS gmv_previous,
  COUNT(DISTINCT CASE WHEN period = 'current' THEN order_id END) AS orders_current,
  COUNT(DISTINCT CASE WHEN period = 'previous' THEN order_id END) AS orders_previous,
  COUNT(DISTINCT CASE WHEN period = 'current' THEN customer_unique_id END) AS buyers_current,
  COUNT(DISTINCT CASE WHEN period = 'previous' THEN customer_unique_id END) AS buyers_previous
FROM labeled
"""


def _get_summary(conn: duckdb.DuckDBPyConnection, window_days: int) -> dict:
    query = _build_summary_sql(window_days)
    row = conn.execute(query).fetchone()
    if row is None:
        raise ValueError("无法计算归因：未读取到 summary 结果。")

    keys = [
        "anchor_date",
        "current_start",
        "current_end",
        "previous_start",
        "previous_end",
        "gmv_current",
        "gmv_previous",
        "orders_current",
        "orders_previous",
        "buyers_current",
        "buyers_previous",
    ]
    return dict(zip(keys, row))


def _build_dimension_driver_sql(
    window_days: int,
    top_n: int,
    dimension_sql: str,
    join_sql: str = "",
    trend: str = "drop",
) -> str:
    days = int(window_days)
    n = int(top_n)

    if trend not in {"drop", "gain"}:
        raise ValueError("trend must be 'drop' or 'gain'")

    condition = "contribution < 0" if trend == "drop" else "contribution > 0"
    order_by = "contribution ASC" if trend == "drop" else "contribution DESC"

    return f"""
WITH bounds AS (
  SELECT CAST(MAX(order_purchase_ts) AS DATE) AS anchor_dt
  FROM vw_fact_items
),
labeled AS (
  SELECT
    CASE
      WHEN CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {days - 1} DAY AND b.anchor_dt
        THEN 'current'
      WHEN CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt - INTERVAL {days} DAY
        THEN 'previous'
      ELSE NULL
    END AS period,
    {dimension_sql} AS dimension_value,
    f.line_gmv
  FROM vw_fact_items f
  {join_sql}
  CROSS JOIN bounds b
  WHERE CAST(f.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt
),
agg AS (
  SELECT
    dimension_value,
    COALESCE(SUM(CASE WHEN period = 'current' THEN line_gmv ELSE 0 END), 0) AS gmv_current,
    COALESCE(SUM(CASE WHEN period = 'previous' THEN line_gmv ELSE 0 END), 0) AS gmv_previous
  FROM labeled
  WHERE period IS NOT NULL
  GROUP BY 1
),
delta AS (
  SELECT
    dimension_value,
    gmv_current,
    gmv_previous,
    gmv_current - gmv_previous AS contribution
  FROM agg
)
SELECT
  dimension_value,
  gmv_current,
  gmv_previous,
  contribution
FROM delta
WHERE {condition}
ORDER BY {order_by}
LIMIT {n}
"""


def _get_dimension_driver_df(
    conn: duckdb.DuckDBPyConnection,
    window_days: int,
    top_n: int,
    dimension_sql: str,
    join_sql: str = "",
    trend: str = "drop",
) -> pd.DataFrame:
    query = _build_dimension_driver_sql(
        window_days=window_days,
        top_n=top_n,
        dimension_sql=dimension_sql,
        join_sql=join_sql,
        trend=trend,
    )
    return conn.execute(query).fetchdf()


def generate_attribution_narrative(
    report: GmvAttributionReport,
    question: str,
    llm_config: dict,
) -> str:
    api_key = str(llm_config.get("api_key") or "")
    if not api_key:
        return build_gmv_summary(report)

    trend = "下滑" if report.gmv_delta < 0 else "增长"
    key_df = report.all_drops if report.gmv_delta < 0 else report.all_gains
    top_rows = (
        key_df[["dimension", "dimension_value", "contribution"]]
        .head(3)
        .to_string(index=False)
        if not key_df.empty
        else "无"
    )

    prompt = f"""你是一名增长数据分析师，请根据以下归因数据用中文回答用户问题。

用户问题：{question}

=== 归因数据（分析窗口：最近 {report.window_days} 天）===
当期 GMV：{report.gmv_current:,.2f}  前期 GMV：{report.gmv_previous:,.2f}
GMV 变化：{report.gmv_delta:+,.2f}  变化率：{_fmt_pct(report.gmv_change_rate)}

拆解（GMV = 订单量 × AOV）：
- 订单量效应：{report.order_effect:+,.2f}（{report.orders_current:,} vs {report.orders_previous:,} 单）
- AOV 效应：{report.aov_effect:+,.2f}（{_fmt_money(report.aov_current)} vs {_fmt_money(report.aov_previous)}）

主要{trend}维度 Top3（维度 / 对象 / 贡献额）：
{top_rows}

=== 要求 ===
- 直接回答"为什么"，聚焦 2-3 个最关键因素
- 不要重复列出上方的数字表格，用自然语言描述
- 只陈述归因数据直接支持的变化，不把相关性写成因果关系
- 禁止臆测季节性、竞品、营销、物流或供给原因；需要额外数据验证时明确说明
- 建议只能写成下一步核查方向，不能声称某项业务动作一定有效
- 简洁有洞察，100-150 字以内"""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=llm_config.get("base_url") or None,
        )
        resp = client.chat.completions.create(
            model=str(llm_config.get("model", "deepseek-chat")),
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        return build_gmv_summary(report)


def analyze_gmv_change_drivers(
    db_path: str | Path,
    window_days: int = 30,
    top_n: int = 5,
) -> GmvAttributionReport:
    days = max(7, min(120, int(window_days)))
    n = max(3, min(20, int(top_n)))

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        summary = _get_summary(conn, days)

        gmv_current = float(summary["gmv_current"] or 0.0)
        gmv_previous = float(summary["gmv_previous"] or 0.0)
        orders_current = int(summary["orders_current"] or 0)
        orders_previous = int(summary["orders_previous"] or 0)
        buyers_current = int(summary["buyers_current"] or 0)
        buyers_previous = int(summary["buyers_previous"] or 0)

        aov_current = _safe_div(gmv_current, float(orders_current))
        aov_previous = _safe_div(gmv_previous, float(orders_previous))
        gmv_delta = gmv_current - gmv_previous
        gmv_change_rate = _safe_pct_change(gmv_current, gmv_previous)

        # Exact decomposition: ΔGMV = (ΔOrders * AOV_prev) + (Orders_current * ΔAOV)
        order_effect = (orders_current - orders_previous) * (aov_previous or 0.0)
        aov_effect = orders_current * ((aov_current or 0.0) - (aov_previous or 0.0))

        state_drops = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(f.customer_state, 'UNKNOWN')",
            trend="drop",
        )
        category_drops = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(p.product_category_name, 'UNKNOWN')",
            join_sql="LEFT JOIN products p ON f.product_id = p.product_id",
            trend="drop",
        )
        seller_drops = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(CAST(f.seller_id AS VARCHAR), 'UNKNOWN')",
            trend="drop",
        )

        state_gains = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(f.customer_state, 'UNKNOWN')",
            trend="gain",
        )
        category_gains = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(p.product_category_name, 'UNKNOWN')",
            join_sql="LEFT JOIN products p ON f.product_id = p.product_id",
            trend="gain",
        )
        seller_gains = _get_dimension_driver_df(
            conn=conn,
            window_days=days,
            top_n=n,
            dimension_sql="COALESCE(CAST(f.seller_id AS VARCHAR), 'UNKNOWN')",
            trend="gain",
        )

        all_drops = _merge_dimension_tables(
            state_df=state_drops,
            category_df=category_drops,
            seller_df=seller_drops,
            trend="drop",
            top_n=n,
        )
        all_gains = _merge_dimension_tables(
            state_df=state_gains,
            category_df=category_gains,
            seller_df=seller_gains,
            trend="gain",
            top_n=n,
        )
    finally:
        conn.close()

    return GmvAttributionReport(
        window_days=days,
        anchor_date=str(summary["anchor_date"]).split(" ")[0],
        current_start=str(summary["current_start"]).split(" ")[0],
        current_end=str(summary["current_end"]).split(" ")[0],
        previous_start=str(summary["previous_start"]).split(" ")[0],
        previous_end=str(summary["previous_end"]).split(" ")[0],
        gmv_current=gmv_current,
        gmv_previous=gmv_previous,
        gmv_delta=gmv_delta,
        gmv_change_rate=gmv_change_rate,
        orders_current=orders_current,
        orders_previous=orders_previous,
        buyers_current=buyers_current,
        buyers_previous=buyers_previous,
        aov_current=aov_current,
        aov_previous=aov_previous,
        order_effect=order_effect,
        aov_effect=aov_effect,
        state_drops=state_drops,
        category_drops=category_drops,
        seller_drops=seller_drops,
        state_gains=state_gains,
        category_gains=category_gains,
        seller_gains=seller_gains,
        all_drops=all_drops,
        all_gains=all_gains,
        evidence_sql={
            "窗口汇总": _build_summary_sql(days),
            "州·下拉贡献": _build_dimension_driver_sql(days, n, "COALESCE(f.customer_state, 'UNKNOWN')", trend="drop"),
            "州·拉升贡献": _build_dimension_driver_sql(days, n, "COALESCE(f.customer_state, 'UNKNOWN')", trend="gain"),
            "品类·下拉贡献": _build_dimension_driver_sql(
                days,
                n,
                "COALESCE(p.product_category_name, 'UNKNOWN')",
                join_sql="LEFT JOIN products p ON f.product_id = p.product_id",
                trend="drop",
            ),
            "品类·拉升贡献": _build_dimension_driver_sql(
                days,
                n,
                "COALESCE(p.product_category_name, 'UNKNOWN')",
                join_sql="LEFT JOIN products p ON f.product_id = p.product_id",
                trend="gain",
            ),
            "商家·下拉贡献": _build_dimension_driver_sql(
                days,
                n,
                "COALESCE(CAST(f.seller_id AS VARCHAR), 'UNKNOWN')",
                trend="drop",
            ),
            "商家·拉升贡献": _build_dimension_driver_sql(
                days,
                n,
                "COALESCE(CAST(f.seller_id AS VARCHAR), 'UNKNOWN')",
                trend="gain",
            ),
        },
    )
