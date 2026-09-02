from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd


@dataclass
class FulfillmentDiagnosisReport:
    window_days: int
    anchor_date: str
    current_start: str
    current_end: str
    previous_start: str
    previous_end: str
    delivered_orders_current: int
    delivered_orders_previous: int
    on_time_rate_current: float
    on_time_rate_previous: float
    on_time_rate_delta: float
    avg_delivery_days_current: float | None
    avg_delivery_days_previous: float | None
    mix_effect: float
    within_effect: float
    state_contributions: pd.DataFrame
    evidence_sql: dict[str, str] = field(default_factory=dict)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _build_summary_sql(window_days: int) -> str:
    days = int(window_days)
    return f"""
WITH bounds AS (
  SELECT CAST(MAX(order_purchase_ts) AS DATE) AS anchor_dt
  FROM vw_delivered_orders
),
labeled AS (
  SELECT
    CASE
      WHEN CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {days - 1} DAY AND b.anchor_dt
        THEN 'current'
      WHEN CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt - INTERVAL {days} DAY
        THEN 'previous'
      ELSE NULL
    END AS period,
    d.order_id,
    CASE WHEN d.order_delivered_customer_ts <= d.order_estimated_delivery_ts THEN 1 ELSE 0 END AS on_time,
    date_diff('day', d.order_purchase_ts, d.order_delivered_customer_ts) AS delivery_days,
    b.anchor_dt
  FROM vw_delivered_orders d
  CROSS JOIN bounds b
  WHERE d.order_delivered_customer_ts IS NOT NULL
    AND d.order_estimated_delivery_ts IS NOT NULL
    AND CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt
)
SELECT
  CAST(MAX(anchor_dt) AS VARCHAR) AS anchor_date,
  CAST(MAX(anchor_dt - INTERVAL {days - 1} DAY) AS VARCHAR) AS current_start,
  CAST(MAX(anchor_dt) AS VARCHAR) AS current_end,
  CAST(MAX(anchor_dt - INTERVAL {2 * days - 1} DAY) AS VARCHAR) AS previous_start,
  CAST(MAX(anchor_dt - INTERVAL {days} DAY) AS VARCHAR) AS previous_end,
  COUNT(DISTINCT CASE WHEN period = 'current' THEN order_id END) AS delivered_orders_current,
  COUNT(DISTINCT CASE WHEN period = 'previous' THEN order_id END) AS delivered_orders_previous,
  AVG(CASE WHEN period = 'current' THEN on_time END) AS on_time_rate_current,
  AVG(CASE WHEN period = 'previous' THEN on_time END) AS on_time_rate_previous,
  AVG(CASE WHEN period = 'current' THEN delivery_days END) AS avg_delivery_days_current,
  AVG(CASE WHEN period = 'previous' THEN delivery_days END) AS avg_delivery_days_previous
FROM labeled
WHERE period IS NOT NULL
"""


def _build_state_sql(window_days: int) -> str:
    days = int(window_days)
    return f"""
WITH bounds AS (
  SELECT CAST(MAX(order_purchase_ts) AS DATE) AS anchor_dt
  FROM vw_delivered_orders
),
labeled AS (
  SELECT
    CASE
      WHEN CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {days - 1} DAY AND b.anchor_dt
        THEN 'current'
      WHEN CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt - INTERVAL {days} DAY
        THEN 'previous'
      ELSE NULL
    END AS period,
    COALESCE(c.customer_state, 'UNKNOWN') AS customer_state,
    d.order_id,
    CASE WHEN d.order_delivered_customer_ts <= d.order_estimated_delivery_ts THEN 1 ELSE 0 END AS on_time,
    date_diff('day', d.order_purchase_ts, d.order_delivered_customer_ts) AS delivery_days
  FROM vw_delivered_orders d
  JOIN customers c ON d.customer_id = c.customer_id
  CROSS JOIN bounds b
  WHERE d.order_delivered_customer_ts IS NOT NULL
    AND d.order_estimated_delivery_ts IS NOT NULL
    AND CAST(d.order_purchase_ts AS DATE) BETWEEN b.anchor_dt - INTERVAL {2 * days - 1} DAY AND b.anchor_dt
)
SELECT
  period,
  customer_state,
  COUNT(DISTINCT order_id) AS delivered_orders,
  SUM(on_time) AS on_time_orders,
  AVG(delivery_days) AS avg_delivery_days
FROM labeled
WHERE period IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
"""


def _decompose_state_mix(state_periods: pd.DataFrame) -> pd.DataFrame:
    current = state_periods[state_periods["period"] == "current"].copy()
    previous = state_periods[state_periods["period"] == "previous"].copy()
    current = current.rename(
        columns={
            "delivered_orders": "orders_current",
            "on_time_orders": "on_time_current",
            "avg_delivery_days": "avg_days_current",
        }
    ).drop(columns=["period"])
    previous = previous.rename(
        columns={
            "delivered_orders": "orders_previous",
            "on_time_orders": "on_time_previous",
            "avg_delivery_days": "avg_days_previous",
        }
    ).drop(columns=["period"])
    merged = current.merge(previous, on="customer_state", how="outer")
    for column in ("orders_current", "on_time_current", "orders_previous", "on_time_previous"):
        merged[column] = merged[column].fillna(0).astype(float)

    total_current = float(merged["orders_current"].sum())
    total_previous = float(merged["orders_previous"].sum())
    merged["weight_current"] = merged["orders_current"] / total_current if total_current else 0.0
    merged["weight_previous"] = merged["orders_previous"] / total_previous if total_previous else 0.0
    merged["rate_current"] = merged.apply(
        lambda row: _safe_div(row["on_time_current"], row["orders_current"]), axis=1
    )
    merged["rate_previous"] = merged.apply(
        lambda row: _safe_div(row["on_time_previous"], row["orders_previous"]), axis=1
    )

    # For a state absent in one period, use its observed-period rate as the neutral
    # counterfactual. This assigns entry/exit entirely to mix and preserves additivity.
    neutral_rate = merged["rate_current"].combine_first(merged["rate_previous"]).fillna(0.0)
    merged["rate_current"] = merged["rate_current"].fillna(neutral_rate)
    merged["rate_previous"] = merged["rate_previous"].fillna(neutral_rate)
    merged["mix_effect"] = (
        (merged["weight_current"] - merged["weight_previous"])
        * (merged["rate_current"] + merged["rate_previous"])
        / 2
    )
    merged["within_effect"] = (
        (merged["rate_current"] - merged["rate_previous"])
        * (merged["weight_current"] + merged["weight_previous"])
        / 2
    )
    merged["contribution"] = merged["mix_effect"] + merged["within_effect"]
    return merged[
        [
            "customer_state",
            "orders_current",
            "orders_previous",
            "rate_current",
            "rate_previous",
            "weight_current",
            "weight_previous",
            "mix_effect",
            "within_effect",
            "contribution",
        ]
    ].sort_values("contribution", ascending=True).reset_index(drop=True)


def build_fulfillment_summary(report: FulfillmentDiagnosisReport) -> str:
    delta_pp = report.on_time_rate_delta * 100
    direction = "提升" if delta_pp > 0 else "下降" if delta_pp < 0 else "持平"
    summary = (
        f"本期准时送达率为 {report.on_time_rate_current * 100:.2f}%，"
        f"较前期{direction} {abs(delta_pp):.2f} 个百分点。"
    )
    dominant_label, dominant_value = max(
        (("订单地域结构变化", report.mix_effect), ("各州组内履约变化", report.within_effect)),
        key=lambda item: abs(item[1]),
    )
    summary += f"较大的分解项是{dominant_label}（{dominant_value * 100:+.2f} 个百分点）。"
    candidates = report.state_contributions
    if not candidates.empty and report.on_time_rate_delta != 0:
        row = candidates.iloc[0] if report.on_time_rate_delta < 0 else candidates.iloc[-1]
        summary += (
            f"客户州 {row['customer_state']} 的总贡献为 "
            f"{float(row['contribution']) * 100:+.2f} 个百分点。"
        )
    summary += "以上是可加和的变化贡献，不代表已经证明因果关系。"
    return summary


def analyze_fulfillment_change_drivers(
    db_path: str | Path,
    window_days: int = 30,
) -> FulfillmentDiagnosisReport:
    days = max(7, min(120, int(window_days)))
    summary_sql = _build_summary_sql(days)
    state_sql = _build_state_sql(days)
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        summary_frame = connection.execute(summary_sql).fetchdf()
        state_periods = connection.execute(state_sql).fetchdf()
    finally:
        connection.close()
    if summary_frame.empty:
        raise ValueError("无法计算履约诊断：未读取到窗口汇总结果。")
    row = summary_frame.iloc[0]
    state_contributions = _decompose_state_mix(state_periods)
    rate_current = float(row["on_time_rate_current"] or 0.0)
    rate_previous = float(row["on_time_rate_previous"] or 0.0)
    return FulfillmentDiagnosisReport(
        window_days=days,
        anchor_date=str(row["anchor_date"]).split(" ")[0],
        current_start=str(row["current_start"]).split(" ")[0],
        current_end=str(row["current_end"]).split(" ")[0],
        previous_start=str(row["previous_start"]).split(" ")[0],
        previous_end=str(row["previous_end"]).split(" ")[0],
        delivered_orders_current=int(row["delivered_orders_current"] or 0),
        delivered_orders_previous=int(row["delivered_orders_previous"] or 0),
        on_time_rate_current=rate_current,
        on_time_rate_previous=rate_previous,
        on_time_rate_delta=rate_current - rate_previous,
        avg_delivery_days_current=(
            float(row["avg_delivery_days_current"])
            if pd.notna(row["avg_delivery_days_current"])
            else None
        ),
        avg_delivery_days_previous=(
            float(row["avg_delivery_days_previous"])
            if pd.notna(row["avg_delivery_days_previous"])
            else None
        ),
        mix_effect=float(state_contributions["mix_effect"].sum()),
        within_effect=float(state_contributions["within_effect"].sum()),
        state_contributions=state_contributions,
        evidence_sql={"履约窗口汇总": summary_sql, "客户州结构与组内分解": state_sql},
    )
