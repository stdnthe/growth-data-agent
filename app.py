from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.attribution import GmvAttributionReport, analyze_gmv_change_drivers, build_gmv_summary
from agent.insight import generate_insight
from agent.llm_sql import generate_sql
from agent.metrics_store import MetricsStore
from agent.sql_guard import SQLGuard, SQLGuardError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("OLIST_DB_PATH", str(PROJECT_ROOT / "olist.duckdb")))
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SAMPLE_QUESTIONS = [
    "近30天GMV走势（按天）",
    "上个月订单数、GMV、客单价分别是多少？",
    "按州(state)看GMV Top 10（近90天）",
    "准时送达率按月趋势（最近6个月）",
    "延迟送达订单平均评分 vs 准时订单平均评分（最近6个月）",
    "新客占比按月趋势（今年以来）",
]


@st.cache_data(show_spinner=False)
def load_metrics_context(metrics_file: str) -> str:
    store = MetricsStore(metrics_file)
    return store.compressed_context(max_items=30)


def run_duckdb_query(sql: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB 文件不存在：{DB_PATH}。请先运行 python warehouse/load_olist_to_duckdb.py"
        )

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


def _pick_x_column(df: pd.DataFrame) -> str | None:
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    if datetime_cols:
        return datetime_cols[0]

    preferred_names = ("dt", "date", "day", "week", "month", "year")
    for col in df.columns:
        lower = str(col).lower()
        if any(name in lower for name in preferred_names):
            return str(col)
    return None


def _prepare_chart_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    chart_df = df.copy()

    # Try parsing potential date columns for better time-series chart rendering.
    for col in chart_df.columns:
        if chart_df[col].dtype == "object":
            lower = str(col).lower()
            if any(token in lower for token in ("dt", "date", "day", "week", "month", "year")):
                parsed = pd.to_datetime(chart_df[col], errors="coerce")
                if parsed.notna().any():
                    chart_df[col] = parsed

    x_col = _pick_x_column(chart_df)
    numeric_cols = list(chart_df.select_dtypes(include="number").columns)
    y_cols = [c for c in numeric_cols if c != x_col]
    return chart_df, x_col, y_cols


def render_chart(df: pd.DataFrame, chart_type: str) -> None:
    if df.empty:
        st.info("结果为空，暂无可视化内容。")
        return

    chart_df, x_col, y_cols = _prepare_chart_data(df)
    if not y_cols:
        st.info("结果中没有可绘制的数值列，已跳过图表。")
        return

    st.subheader("图表可视化")
    if chart_type == "Auto":
        if x_col:
            chart_type = "Line"
        elif len(y_cols) >= 1:
            chart_type = "Bar"
        else:
            chart_type = "Scatter"

    if chart_type == "Line":
        if x_col:
            st.line_chart(chart_df, x=x_col, y=y_cols, use_container_width=True)
        else:
            st.line_chart(chart_df[y_cols], use_container_width=True)
        return

    if chart_type == "Area":
        if x_col:
            st.area_chart(chart_df, x=x_col, y=y_cols, use_container_width=True)
        else:
            st.area_chart(chart_df[y_cols], use_container_width=True)
        return

    if chart_type == "Bar":
        if x_col:
            st.bar_chart(chart_df, x=x_col, y=y_cols, use_container_width=True)
        else:
            st.bar_chart(chart_df[y_cols], use_container_width=True)
        return

    # Scatter chart supports one x and one y, so we pick the first two numeric columns.
    x_scatter = y_cols[0]
    y_scatter = y_cols[1] if len(y_cols) > 1 else y_cols[0]
    st.scatter_chart(chart_df, x=x_scatter, y=y_scatter, use_container_width=True)


def _fmt_currency(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _render_driver_table(df: pd.DataFrame, title: str, value_name: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.caption("无明显贡献项。")
        return

    shown = df.rename(columns={"dimension_value": value_name}).copy()
    for col in ("gmv_current", "gmv_previous", "contribution"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: round(float(x), 2))
    st.dataframe(shown, use_container_width=True)


def _render_cross_dimension_table(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.caption("无明显贡献项。")
        return

    shown = df.rename(columns={"dimension": "维度", "dimension_value": "对象"}).copy()
    for col in ("gmv_current", "gmv_previous", "contribution"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: round(float(x), 2))
    st.dataframe(shown, use_container_width=True)


def render_gmv_attribution(report: GmvAttributionReport) -> None:
    st.subheader("GMV 异动归因")
    trend = "下降" if report.gmv_delta < 0 else "增长"
    st.success(build_gmv_summary(report))
    st.caption(
        f"对比窗口：最近 {report.window_days} 天（{report.current_start} ~ {report.current_end}）"
        f" vs 前 {report.window_days} 天（{report.previous_start} ~ {report.previous_end}），"
        f"基准日期={report.anchor_date}"
    )
    st.write(
        f"结论：GMV {trend} {_fmt_currency(abs(report.gmv_delta))}，变化率 {_fmt_percent(report.gmv_change_rate)}。"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"GMV（最近{report.window_days}天）",
        _fmt_currency(report.gmv_current),
        f"{_fmt_currency(report.gmv_delta)} ({_fmt_percent(report.gmv_change_rate)})",
    )
    m2.metric(
        f"订单数（最近{report.window_days}天）",
        f"{report.orders_current:,}",
        f"{report.orders_current - report.orders_previous:,}",
    )
    m3.metric(
        "AOV（最近窗口）",
        _fmt_currency(report.aov_current),
        _fmt_currency((report.aov_current or 0) - (report.aov_previous or 0)),
    )

    st.markdown("**拆解（GMV = 订单量 × AOV）**")
    decomp_df = pd.DataFrame(
        [
            {"factor": "订单量效应", "contribution": report.order_effect},
            {"factor": "AOV效应", "contribution": report.aov_effect},
        ]
    )
    st.dataframe(decomp_df, use_container_width=True)

    if report.gmv_delta < 0:
        _render_cross_dimension_table(report.all_drops, "全维度 TopN 下拉贡献（州/品类/商家）")
        st.markdown("**主要下拉因素**")
        _render_driver_table(report.state_drops, "按州（State）", "state")
        _render_driver_table(report.category_drops, "按品类（Category）", "category")
        _render_driver_table(report.seller_drops, "按商家（Seller）", "seller_id")
    else:
        _render_cross_dimension_table(report.all_gains, "全维度 TopN 拉升贡献（州/品类/商家）")
        st.markdown("**主要拉升因素**")
        _render_driver_table(report.state_gains, "按州（State）", "state")
        _render_driver_table(report.category_gains, "按品类（Category）", "category")
        _render_driver_table(report.seller_gains, "按商家（Seller）", "seller_id")

    with st.expander("查看反向对冲因素", expanded=False):
        if report.gmv_delta < 0:
            _render_driver_table(report.state_gains, "对冲下滑：州（State）", "state")
            _render_driver_table(report.category_gains, "对冲下滑：品类（Category）", "category")
            _render_driver_table(report.seller_gains, "对冲下滑：商家（Seller）", "seller_id")
        else:
            _render_driver_table(report.state_drops, "拖累增长：州（State）", "state")
            _render_driver_table(report.category_drops, "拖累增长：品类（Category）", "category")
            _render_driver_table(report.seller_drops, "拖累增长：商家（Seller）", "seller_id")


def main() -> None:
    st.set_page_config(page_title="Olist Growth Copilot", layout="wide")
    st.title("Olist Growth Copilot")
    st.caption("电商/增长分析对话式数据分析助手（DuckDB + Streamlit + OpenAI SQL）")

    with st.sidebar:
        st.header("配置")
        use_llm = st.toggle("Use LLM to generate SQL", value=True)
        show_sql = st.toggle("Show SQL", value=True)
        show_chart = st.toggle("Show Chart", value=True)
        chart_type = st.selectbox("Chart Type", options=["Auto", "Line", "Bar", "Area", "Scatter"], index=0)
        enable_gmv_attribution = st.toggle("Enable GMV Attribution", value=True)
        attribution_window_days = st.selectbox("Attribution Window (days)", options=[7, 14, 30, 60, 90], index=2)
        attribution_top_n = st.slider("Attribution Top N", min_value=3, max_value=15, value=5, step=1)
        default_limit = st.number_input("Default LIMIT", min_value=100, max_value=10000, value=2000, step=100)

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if use_llm and not api_key:
            st.info("未配置 OPENAI_API_KEY：将自动使用规则 fallback 生成 SQL。")

        st.divider()
        st.subheader("示例问题")
        for idx, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"sample_{idx}", use_container_width=True):
                st.session_state["question"] = q

    if "question" not in st.session_state:
        st.session_state["question"] = ""

    st.text_area(
        "输入增长问题（中文为主）",
        key="question",
        height=110,
        placeholder="例如：近30天GMV走势（按天）",
    )

    run = st.button("Run", type="primary")
    if not run:
        return

    question = st.session_state.get("question", "").strip()
    if not question:
        st.warning("请先输入问题。")
        return

    try:
        metrics_context = load_metrics_context(str(METRICS_PATH))
    except Exception as e:  # noqa: BLE001
        st.error(f"读取指标字典失败：{e}")
        return

    try:
        sql, source = generate_sql(
            question=question,
            metrics_context=metrics_context,
            use_llm=use_llm,
            model=MODEL,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"SQL 生成失败：{e}")
        return

    guard = SQLGuard(default_limit=int(default_limit))
    try:
        safe_sql = guard.validate_and_rewrite(sql)
    except SQLGuardError as e:
        st.error(f"SQL Guard 拒绝执行：{e}")
        if show_sql:
            st.code(sql, language="sql")
        return

    if show_sql:
        st.code(safe_sql, language="sql")

    try:
        result_df = run_duckdb_query(safe_sql)
    except Exception as e:  # noqa: BLE001
        st.error(f"SQL 执行失败：{e}")
        return

    st.success(f"查询成功（SQL 来源：{source}，模型：{MODEL}）")
    st.dataframe(result_df, use_container_width=True)
    if show_chart:
        render_chart(result_df, chart_type)
    st.info(generate_insight(result_df))

    if enable_gmv_attribution:
        try:
            report = analyze_gmv_change_drivers(
                db_path=DB_PATH,
                window_days=int(attribution_window_days),
                top_n=int(attribution_top_n),
            )
            render_gmv_attribution(report)
        except Exception as e:  # noqa: BLE001
            st.warning(f"GMV 归因计算失败：{e}")


if __name__ == "__main__":
    main()
