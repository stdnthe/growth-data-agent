from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.attribution import GmvAttributionReport, build_gmv_summary
from agent.feedback import FeedbackRecord, FeedbackStore
from agent.llm_sql import resolve_llm_config
from agent.models import AnalysisRun
from agent.pipeline import PipelineConfig, execute_analysis

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("OLIST_DB_PATH", str(PROJECT_ROOT / "olist.duckdb")))
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
FEEDBACK_PATH = PROJECT_ROOT / ".runtime" / "feedback.jsonl"
LLM_RUNTIME = resolve_llm_config()
LLM_PROVIDER = str(LLM_RUNTIME["provider"])
MODEL = str(LLM_RUNTIME["model"])
ACTIVE_API_KEY = str(LLM_RUNTIME["api_key"] or "")
ACTIVE_API_KEY_NAME = str(LLM_RUNTIME["api_key_name"])
ACTIVE_API_KEY_SOURCE = str(LLM_RUNTIME["api_key_source"])

SAMPLE_QUESTIONS = [
    "近30天GMV走势（按天）",
    "为什么最近GMV下降？",
    "最近销售表现怎么样？",
    "为什么准时送达率下降？",
]


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
            st.line_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.line_chart(chart_df[y_cols], width="stretch")
        return

    if chart_type == "Area":
        if x_col:
            st.area_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.area_chart(chart_df[y_cols], width="stretch")
        return

    if chart_type == "Bar":
        if x_col:
            st.bar_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.bar_chart(chart_df[y_cols], width="stretch")
        return

    # Scatter chart supports one x and one y, so we pick the first two numeric columns.
    x_scatter = y_cols[0]
    y_scatter = y_cols[1] if len(y_cols) > 1 else y_cols[0]
    st.scatter_chart(chart_df, x=x_scatter, y=y_scatter, width="stretch")


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
    st.dataframe(shown, width="stretch")


def _render_cross_dimension_table(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.caption("无明显贡献项。")
        return

    shown = df.rename(columns={"dimension": "维度", "dimension_value": "对象"}).copy()
    for col in ("gmv_current", "gmv_previous", "contribution"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: round(float(x), 2))
    st.dataframe(shown, width="stretch")


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
    st.dataframe(decomp_df, width="stretch")

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


def _render_analysis_plan(run: AnalysisRun) -> None:
    intent = run.intent
    with st.expander("分析计划与口径", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("任务类型", "指标诊断" if intent.task_type == "metric_diagnosis" else "指标查询")
        c2.metric("指标", intent.metric or "待确认")
        c3.metric("时间范围", intent.time_range or "待确认")
        c4.metric("Workflow", run.workflow)
        st.caption(
            f"粒度：{intent.grain or '自动'} ｜ 比较：{intent.comparison or '无'} ｜ "
            f"维度：{', '.join(intent.dimensions) or '无'} ｜ 置信度：{intent.confidence:.0%}"
        )
        if run.metric_context:
            st.markdown("**指标口径**")
            for item in run.metric_context:
                st.write(f"- {item}")
        for assumption in intent.assumptions:
            st.warning(f"假设：{assumption}")


def _render_validations(run: AnalysisRun) -> None:
    if not run.validations:
        return
    with st.expander("确定性校验", expanded=True):
        rows = [
            {
                "状态": "通过" if item.passed else "未通过",
                "校验项": item.name,
                "级别": item.severity,
                "说明": item.message,
            }
            for item in run.validations
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_run_metadata(run: AnalysisRun) -> None:
    with st.expander("运行元数据", expanded=False):
        st.json(
            {
                "run_id": run.run_id,
                "status": run.status,
                "provider": run.provider,
                "model": run.model,
                "prompt_version": run.prompt_version,
                "metric_version": run.metric_version,
                "latency_ms": run.latency_ms,
                "sql_source": run.sql_source,
                "attempts": [attempt.to_dict() for attempt in run.attempts],
                "langsmith_trace_active": run.trace_enabled,
            }
        )


def _render_feedback(run: AnalysisRun) -> None:
    if run.status not in {"success", "success_with_warnings"}:
        return
    with st.expander("这次分析有帮助吗？", expanded=False):
        with st.form(f"feedback_{run.run_id}"):
            helpful_label = st.radio("总体评价", ["有帮助", "需要改进"], horizontal=True)
            feedback_type = st.selectbox(
                "原因",
                ["结果正确", "数字不对", "指标口径不对", "没有回答问题", "分析方法不对", "建议不可执行"],
            )
            comment = st.text_input("补充说明（可选）")
            submitted = st.form_submit_button("提交反馈")
        if submitted:
            FeedbackStore(FEEDBACK_PATH).append(
                FeedbackRecord(
                    run_id=run.run_id,
                    helpful=helpful_label == "有帮助",
                    feedback_type=feedback_type,
                    comment=comment,
                    question=run.question,
                    workflow=run.workflow,
                    failure_stage=run.failure_stage,
                )
            )
            st.success("反馈已记录，将用于补充 Golden Dataset。")


def render_analysis_run(
    run: AnalysisRun,
    *,
    show_sql: bool,
    show_chart: bool,
    chart_type: str,
) -> None:
    _render_analysis_plan(run)

    if run.status == "needs_clarification":
        st.warning(run.insight or "需要补充信息后才能继续分析。")
        _render_run_metadata(run)
        return

    if run.status == "failed":
        st.error(f"分析失败（阶段：{run.failure_stage or 'unknown'}）：{run.failure_detail or '未知错误'}")
        if show_sql and run.generated_sql:
            st.code(run.generated_sql, language="sql")
        _render_validations(run)
        _render_run_metadata(run)
        return

    if run.insight:
        st.info(run.insight)
    for caveat in run.caveats:
        st.warning(f"限制：{caveat}")

    if run.attribution_report is not None:
        render_gmv_attribution(run.attribution_report)
    elif run.result is not None:
        if show_sql and run.generated_sql:
            st.subheader("查询 SQL")
            st.code(run.generated_sql, language="sql")
        st.subheader("查询结果")
        st.dataframe(run.result, width="stretch")
        if show_chart:
            render_chart(run.result, chart_type)

    _render_validations(run)
    _render_run_metadata(run)
    _render_feedback(run)


def main() -> None:
    st.set_page_config(page_title="Olist Growth Copilot", layout="wide")
    st.title("Olist Growth Copilot")
    provider_label = "OpenAI" if LLM_PROVIDER == "openai" else "DeepSeek"
    st.caption(
        f"可信电商增长 Data Agent：自然语言问数、确定性 GMV 归因与可审计分析运行"
        f"（DuckDB + Streamlit + {provider_label}）"
    )

    proof_1, proof_2, proof_3, proof_4 = st.columns(4)
    proof_1.metric("指标语义层", "28 个指标")
    proof_2.metric("可评估 Workflow", "2 条")
    proof_3.metric("Golden Cases", "48 道")
    proof_4.metric("SQL 安全", "只读白名单")

    with st.expander("产品边界与可信性原则"):
        st.markdown(
            "高频、路径明确的问题走可评估 Workflow；模糊问题先澄清。"
            "当前确定性自动归因仅支持 GMV。Olist 不含曝光、点击、广告渠道与成本数据，"
            "因此不对 CTR、渠道转化率、CAC 或 ROAS 给出伪精确结论。"
        )

    with st.sidebar:
        st.header("配置")
        use_llm = st.toggle("Use LLM to generate SQL", value=True)
        show_sql = st.toggle("Show SQL", value=True)
        show_chart = st.toggle("Show Chart", value=True)
        chart_type = st.selectbox("Chart Type", options=["Auto", "Line", "Bar", "Area", "Scatter"], index=0)
        attribution_top_n = st.slider("Attribution Top N", min_value=3, max_value=15, value=5, step=1)
        default_limit = st.number_input("Default LIMIT", min_value=100, max_value=10000, value=2000, step=100)
        max_retries = st.selectbox("Deterministic Retry", options=[0, 1], index=1)
        enable_tracing = st.toggle(
            "LangSmith Trace",
            value=os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"},
            help="需要安装可选 langsmith 依赖并配置 LANGSMITH_API_KEY。Trace 失败不会阻塞分析。",
        )

        if use_llm and not ACTIVE_API_KEY:
            st.info(f"未检测到 {ACTIVE_API_KEY_NAME}（环境变量或 Keychain）：将自动使用规则 fallback 生成 SQL。")
        elif use_llm and ACTIVE_API_KEY_SOURCE == "keychain":
            st.caption(f"已从 macOS Keychain 读取 {ACTIVE_API_KEY_NAME}。")

        st.divider()
        st.subheader("示例问题")
        for idx, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"sample_{idx}", width="stretch"):
                st.session_state["question"] = q

    if "question" not in st.session_state:
        st.session_state["question"] = ""

    st.text_area(
        "输入增长问题（中文为主）",
        key="question",
        height=110,
        placeholder="例如：近30天GMV走势（按天）",
    )

    if st.button("运行可信分析", type="primary"):
        question = st.session_state.get("question", "").strip()
        if not question:
            st.warning("请先输入问题。")
        else:
            with st.spinner("正在理解问题、执行分析并进行确定性校验..."):
                st.session_state["last_analysis_run"] = execute_analysis(
                    question,
                    PipelineConfig(
                        db_path=DB_PATH,
                        metrics_path=METRICS_PATH,
                        use_llm=use_llm,
                        model=MODEL,
                        default_limit=int(default_limit),
                        attribution_top_n=int(attribution_top_n),
                        max_retries=int(max_retries),
                        tracing_enabled=enable_tracing,
                    ),
                )

    last_run = st.session_state.get("last_analysis_run")
    if isinstance(last_run, AnalysisRun):
        render_analysis_run(
            last_run,
            show_sql=show_sql,
            show_chart=show_chart,
            chart_type=chart_type,
        )


if __name__ == "__main__":
    main()
