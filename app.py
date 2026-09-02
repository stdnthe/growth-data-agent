from __future__ import annotations

import html
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.attribution import GmvAttributionReport
from agent.clarification import clarification_suggestions, resolve_clarification
from agent.feedback import FeedbackRecord, FeedbackStore
from agent.fulfillment import FulfillmentDiagnosisReport
from agent.intent_router import IntentRoutingError
from agent.llm_sql import resolve_llm_config, validate_compatible_base_url
from agent.models import AnalysisRun
from agent.pipeline import PipelineConfig, execute_analysis

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("OLIST_DB_PATH", str(PROJECT_ROOT / "olist.duckdb")))
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"
FEEDBACK_PATH = PROJECT_ROOT / ".runtime" / "feedback.jsonl"
LLM_RUNTIME = resolve_llm_config()
LLM_PROVIDER_OPTIONS = ("deepseek", "openai", "openai_compatible")
LLM_PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "openai_compatible": "OpenAI-compatible",
}

SAMPLE_QUESTIONS = [
    "近30天GMV趋势（按天）",
    "近30天GMV下降的原因",
    "最近销售表现怎么样？",
    "近30天GMV下降主要受哪些商品品类影响？",
]

COLUMN_LABELS = {
    "period": "周期",
    "start": "开始日期",
    "end": "结束日期",
    "dt": "日期",
    "gmv": "GMV（R$）",
    "orders": "订单量",
    "paid_orders": "支付订单数",
    "aov": "AOV（R$）",
    "dimension": "维度",
    "dimension_value": "对象",
    "gmv_current": "本期 GMV（R$）",
    "gmv_previous": "前期 GMV（R$）",
    "contribution": "变动贡献（R$）",
    "contribution_share": "贡献占比",
    "share_of_total_change": "占总变动",
    "customer_state": "客户州",
    "orders_current": "本期妥投订单",
    "orders_previous": "前期妥投订单",
    "rate_current": "本期准时率",
    "rate_previous": "前期准时率",
    "weight_current": "本期订单占比",
    "weight_previous": "前期订单占比",
    "mix_effect": "结构效应",
    "within_effect": "组内效应",
    "mix_effect_pp": "结构效应（百分点）",
    "within_effect_pp": "组内效应（百分点）",
    "contribution_pp": "总贡献（百分点）",
}


def _inject_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --paper: #fffdf8;
  --paper-soft: #f7f3eb;
  --ink: #1c1917;
  --muted: #746a62;
  --line: #ded5c8;
  --blue: #1b5e82;
  --blue-soft: #e4f0f6;
  --gold: #b58a46;
  --green: #2f7d59;
  --action: #a84f2f;
  --action-hover: #8f3f25;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--paper); }
.block-container { max-width: 920px; padding-top: 2.25rem; padding-bottom: 4rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
.gc-hero { padding: 1.1rem 0 1.35rem; border-bottom: 1px solid var(--line); margin-bottom: 1.35rem; }
.gc-kicker { color: var(--blue); font-size: .78rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
.gc-title { color: var(--ink); font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.08; font-weight: 760; margin: .35rem 0 .5rem; }
.gc-subtitle { color: var(--muted); font-size: 1rem; max-width: 760px; line-height: 1.7; }
.gc-example-label { color: var(--ink); font-size: .92rem; font-weight: 700; margin: 0 0 .45rem; }
.gc-question { background: var(--blue); color: white; border-radius: 14px 14px 3px 14px; padding: .85rem 1.1rem; margin: .35rem 0 1.1rem auto; max-width: 78%; width: fit-content; line-height: 1.55; }
.gc-answer { background: #fbf6ed; border-left: 4px solid var(--gold); border-radius: 0 10px 10px 0; padding: 1rem 1.15rem; margin-bottom: 1rem; color: var(--ink); font-size: 1rem; line-height: 1.75; }
.gc-meta { display: flex; flex-wrap: wrap; gap: .5rem; margin: .7rem 0 1.1rem; }
.gc-meta span { background: var(--blue-soft); color: var(--blue); border-radius: 999px; padding: .27rem .62rem; font-size: .76rem; font-weight: 650; }
.gc-trace { position: relative; padding-left: 1.15rem; margin: .35rem 0; }
.gc-trace:before { content: ""; position: absolute; left: .25rem; top: .55rem; bottom: -.75rem; width: 1px; background: var(--line); }
.gc-dot { position: absolute; left: 0; top: .38rem; width: .52rem; height: .52rem; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 4px var(--paper); }
.gc-dot.warning { background: var(--gold); }
.gc-dot.failed { background: #b94a48; }
.gc-trace-title { color: var(--ink); font-weight: 680; }
.gc-trace-body { color: var(--muted); margin-top: .15rem; line-height: 1.55; }
.gc-tool { color: var(--blue); font-size: .75rem; font-weight: 650; }
div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; box-shadow: 0 5px 18px rgba(45, 38, 31, .035); }
div[data-testid="stMetric"] label { color: var(--muted); }
div[data-testid="stMetricValue"] { color: var(--ink); }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); padding-left: 1rem; padding-right: 1rem; }
.stTabs [aria-selected="true"] { color: var(--blue) !important; font-weight: 700; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
  background: var(--action) !important;
  border-color: var(--action) !important;
  border-radius: 9px;
  color: white;
  box-shadow: 0 5px 14px rgba(168, 79, 47, .18);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
  background: var(--action-hover) !important;
  border-color: var(--action-hover) !important;
  color: white;
}
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primaryFormSubmit"] p { color: white !important; }
.stButton > button { border-color: var(--line); border-radius: 9px; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { background: white; border-color: var(--line); color: var(--ink); border-radius: 10px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
[data-testid="stCode"] { border: 1px solid var(--line); border-radius: 10px; }
@media (max-width: 700px) {
  .block-container { padding: 1.2rem .85rem 3rem; }
  .gc-question { max-width: 95%; }
  .gc-title { font-size: 2rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
<section class="gc-hero">
  <div class="gc-kicker">Trusted growth analytics agent</div>
  <div class="gc-title">Olist Growth Copilot</div>
  <div class="gc-subtitle">用自然语言查询、诊断并验证巴西电商增长数据。每个结论都连接到数据、SQL 与确定性校验。</div>
</section>
        """,
        unsafe_allow_html=True,
    )


def _pick_x_column(df: pd.DataFrame) -> str | None:
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    if datetime_cols:
        return str(datetime_cols[0])
    for col in df.columns:
        lower = str(col).lower()
        if any(name in lower for name in ("dt", "date", "day", "week", "month", "year")):
            return str(col)
    return None


def _prepare_chart_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    chart_df = df.copy()
    for col in chart_df.columns:
        if chart_df[col].dtype == "object" and any(
            token in str(col).lower() for token in ("dt", "date", "day", "week", "month", "year")
        ):
            parsed = pd.to_datetime(chart_df[col], errors="coerce")
            if parsed.notna().any():
                chart_df[col] = parsed
    x_col = _pick_x_column(chart_df)
    numeric_cols = [str(col) for col in chart_df.select_dtypes(include="number").columns]
    y_cols = [col for col in numeric_cols if col != x_col and not str(col).lower().endswith("_id")]
    return chart_df, x_col, y_cols


def _render_chart(df: pd.DataFrame, chart_type: str) -> None:
    if df.empty:
        st.info("结果为空，暂无可视化内容。")
        return
    chart_df, x_col, y_cols = _prepare_chart_data(df)
    if not y_cols:
        st.info("当前结果中没有适合绘图的业务数值列。")
        return
    actual_type = chart_type
    if actual_type == "自动":
        actual_type = "折线图" if x_col else "柱状图"
    if actual_type == "折线图":
        if x_col:
            st.line_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.line_chart(chart_df[y_cols], width="stretch")
    elif actual_type == "面积图":
        if x_col:
            st.area_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.area_chart(chart_df[y_cols], width="stretch")
    elif actual_type == "柱状图":
        if x_col:
            st.bar_chart(chart_df, x=x_col, y=y_cols, width="stretch")
        else:
            st.bar_chart(chart_df[y_cols], width="stretch")
    else:
        x_scatter = y_cols[0]
        y_scatter = y_cols[1] if len(y_cols) > 1 else y_cols[0]
        st.scatter_chart(chart_df, x=x_scatter, y=y_scatter, width="stretch")


def _fmt_currency(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}R$ {value:,.2f}"


def _insight_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped.replace("\n", "<br>")


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    shown = df.rename(columns=COLUMN_LABELS).copy()
    if "对象" in shown.columns:
        shown["对象"] = shown["对象"].map(lambda value: str(value).replace("_", " "))
    for col in ("贡献占比", "占总变动"):
        if col in shown.columns:
            shown[col] = shown[col].map(
                lambda value: "—" if pd.isna(value) else f"{float(value) * 100:.1f}%"
            )
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda value: round(float(value), 2) if pd.notna(value) else value)
    return shown


def _left_aligned_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Render compact attribution tables with one consistent text alignment."""
    shown = _display_df(df)

    def as_text(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    return shown.map(as_text)


def _render_question(question: str) -> None:
    st.markdown(
        f'<div class="gc-question">{html.escape(question)}</div>',
        unsafe_allow_html=True,
    )


def _render_intent_context(run: AnalysisRun) -> None:
    intent = run.intent
    runtime_label = {
        "gmv_attribution": "Agent Loop · GMV Recipe",
        "fulfillment_diagnosis": "Agent Loop · Fulfillment Recipe",
    }.get(run.workflow, "Agent Loop · Query Tool")
    st.markdown(
        "".join(
            [
                '<div class="gc-meta">',
                f"<span>{html.escape(intent.time_range or '时间待确认')}</span>",
                f"<span>{html.escape(intent.metric or '指标待确认')}</span>",
                f"<span>{html.escape(runtime_label)}</span>",
                f"<span>{'校验通过' if run.validation_passed else '校验未通过'}</span>",
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_retrieval_insight(run: AnalysisRun) -> None:
    st.markdown(
        f'<div class="gc-answer">{_insight_html(run.insight or "查询已完成。")}</div>',
        unsafe_allow_html=True,
    )
    for assumption in run.intent.assumptions:
        st.warning(f"口径假设：{assumption}")
    for caveat in run.caveats:
        st.warning(f"分析限制：{caveat}")


def _attribution_summary_df(report: GmvAttributionReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "本期",
                "start": report.current_start,
                "end": report.current_end,
                "gmv": report.gmv_current,
                "orders": report.orders_current,
                "aov": report.aov_current,
            },
            {
                "period": "前期",
                "start": report.previous_start,
                "end": report.previous_end,
                "gmv": report.gmv_previous,
                "orders": report.orders_previous,
                "aov": report.aov_previous,
            },
        ]
    )


def _render_attribution_insight(run: AnalysisRun, report: GmvAttributionReport) -> None:
    st.markdown(
        f'<div class="gc-answer">{_insight_html(run.insight or "归因分析已完成。")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"数据锚点：{report.anchor_date} ｜ 本期 {report.current_start}—{report.current_end} ｜ "
        f"对比期 {report.previous_start}—{report.previous_end}"
    )
    for caveat in run.caveats:
        st.warning(f"分析限制：{caveat}")


def _render_attribution_data(report: GmvAttributionReport) -> None:
    def contribution_share(value: float) -> float | None:
        return value / report.gmv_delta if report.gmv_delta != 0 else None

    st.markdown("#### 窗口汇总")
    st.dataframe(
        _left_aligned_display_df(_attribution_summary_df(report)),
        width="stretch",
        hide_index=True,
    )
    st.markdown("#### GMV = 订单量 × AOV")
    decomposition = pd.DataFrame(
        [
            {
                "因素": "订单量效应",
                "变动贡献（R$）": report.order_effect,
                "contribution_share": contribution_share(report.order_effect),
            },
            {
                "因素": "AOV 效应",
                "变动贡献（R$）": report.aov_effect,
                "contribution_share": contribution_share(report.aov_effect),
            },
        ]
    )
    st.dataframe(_left_aligned_display_df(decomposition), width="stretch", hide_index=True)
    trend_df = (report.all_drops if report.gmv_delta < 0 else report.all_gains).copy()
    trend_df["share_of_total_change"] = trend_df["contribution"].map(contribution_share)
    st.markdown("#### 主要维度贡献")
    st.dataframe(_left_aligned_display_df(trend_df), width="stretch", hide_index=True)
    with st.expander("查看州、品类与商家的完整 Top N 拆解"):
        for title, frame in (
            ("州", report.state_drops if report.gmv_delta < 0 else report.state_gains),
            ("品类", report.category_drops if report.gmv_delta < 0 else report.category_gains),
            ("商家", report.seller_drops if report.gmv_delta < 0 else report.seller_gains),
        ):
            st.markdown(f"**{title}**")
            shown_frame = frame.copy()
            shown_frame["share_of_total_change"] = shown_frame["contribution"].map(
                contribution_share
            )
            st.dataframe(
                _left_aligned_display_df(shown_frame),
                width="stretch",
                hide_index=True,
            )


def _render_attribution_chart(report: GmvAttributionReport) -> None:
    st.markdown("#### 结构性变动贡献")
    effects = pd.DataFrame(
        {
            "因素": ["订单量效应", "AOV 效应"],
            "变动贡献（R$）": [report.order_effect, report.aov_effect],
        }
    ).set_index("因素")
    st.bar_chart(effects, width="stretch")
    drivers = report.all_drops if report.gmv_delta < 0 else report.all_gains
    if not drivers.empty:
        st.markdown("#### Top 维度贡献")
        chart_df = drivers.copy()
        chart_df["对象"] = chart_df["dimension"].astype(str) + " · " + chart_df["dimension_value"].astype(str)
        st.bar_chart(chart_df.set_index("对象")[["contribution"]], width="stretch")


def _render_fulfillment_insight(run: AnalysisRun, report: FulfillmentDiagnosisReport) -> None:
    st.markdown(
        f'<div class="gc-answer">{_insight_html(run.insight or "履约诊断已完成。")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"数据锚点：{report.anchor_date} ｜ 本期 {report.current_start}—{report.current_end} ｜ "
        f"对比期 {report.previous_start}—{report.previous_end}"
    )
    for caveat in run.caveats:
        st.warning(f"分析限制：{caveat}")


def _render_fulfillment_data(report: FulfillmentDiagnosisReport) -> None:
    st.markdown("#### 准时送达率变化分解")
    decomposition = pd.DataFrame(
        [
            {"因素": "客户州结构效应", "贡献（百分点）": report.mix_effect * 100},
            {"因素": "各州组内履约效应", "贡献（百分点）": report.within_effect * 100},
        ]
    )
    st.dataframe(decomposition, width="stretch", hide_index=True)
    st.markdown("#### 客户州贡献")
    shown = report.state_contributions.copy()
    for column in (
        "rate_current",
        "rate_previous",
        "weight_current",
        "weight_previous",
        "mix_effect",
        "within_effect",
        "contribution",
    ):
        shown[column] = shown[column] * 100
    shown = shown.rename(
        columns={
            "mix_effect": "mix_effect_pp",
            "within_effect": "within_effect_pp",
            "contribution": "contribution_pp",
        }
    )
    st.dataframe(_display_df(shown), width="stretch", hide_index=True)


def _render_fulfillment_chart(report: FulfillmentDiagnosisReport) -> None:
    st.markdown("#### 结构与组内效应")
    effects = pd.DataFrame(
        {
            "因素": ["客户州结构效应", "各州组内履约效应"],
            "贡献（百分点）": [report.mix_effect * 100, report.within_effect * 100],
        }
    ).set_index("因素")
    st.bar_chart(effects, width="stretch")
    contributions = report.state_contributions.copy()
    if not contributions.empty:
        st.markdown("#### 客户州 Top 贡献")
        direction = contributions.head(8) if report.on_time_rate_delta < 0 else contributions.tail(8)
        direction = direction.assign(contribution_pp=direction["contribution"] * 100)
        st.bar_chart(direction.set_index("customer_state")[["contribution_pp"]], width="stretch")


def _render_sql(run: AnalysisRun) -> None:
    if run.attribution_report is not None or run.fulfillment_report is not None:
        report = run.attribution_report or run.fulfillment_report
        labels = list(report.evidence_sql)
        selected = st.selectbox("选择证据查询", labels, key=f"sql_evidence_{run.run_id}")
        st.caption("以下 SQL 均来自本次确定性分析 Recipe 实际使用的查询模板，并以当前窗口参数化执行。")
        st.code(report.evidence_sql[selected].strip(), language="sql")
    elif run.generated_sql:
        source_label = "模型生成" if run.sql_source == "llm" else "规则降级"
        st.caption(f"SQL 来源：{source_label} ｜ 已通过只读白名单与 LIMIT 检查")
        st.code(run.generated_sql, language="sql")
    else:
        st.info("本次运行没有生成 SQL。")
    if run.metric_context:
        with st.expander("查看本次使用的指标定义"):
            for item in run.metric_context:
                st.write(f"- {item}")
    if run.context_bundle:
        with st.expander("查看本次 ContextBundle"):
            st.caption("仅展示指标、Schema、Join 规则和输出契约，不展示模型隐藏推理。")
            st.json(run.context_bundle)


def _render_trace(run: AnalysisRun) -> None:
    st.caption(
        f"LangGraph 有限循环 · 最多 {run.max_steps} 个工具步骤 · "
        "展示结构化动作与验证记录，不展示模型隐藏思维链"
    )
    for step in run.steps:
        tool = f'<div class="gc-tool">{html.escape(step.tool)}</div>' if step.tool else ""
        st.markdown(
            f"""
<div class="gc-trace">
  <span class="gc-dot {html.escape(step.status)}"></span>
  <div class="gc-trace-title">{html.escape(step.label)}</div>
  {tool}
  <div class="gc-trace-body">{html.escape(step.summary)}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if step.details:
            with st.expander(f"查看“{step.label}”结构化记录"):
                st.json(step.details)

    if run.validations:
        st.markdown("#### 确定性校验")
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

    with st.expander("运行元数据"):
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
                "original_question": run.original_question,
                "clarification_history": run.clarification_history,
                "agent_runtime": run.agent_runtime,
                "max_steps": run.max_steps,
                "active_capabilities": run.active_capabilities,
                "decisions": [decision.to_dict() for decision in run.decisions],
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
                    question=run.original_question or run.question,
                    workflow=run.workflow,
                    failure_stage=run.failure_stage,
                )
            )
            st.success("反馈已记录，将作为 Golden Dataset 候选样本。")


def _render_success(run: AnalysisRun, display_question: str, chart_type: str) -> None:
    _render_question(display_question)
    _render_intent_context(run)
    insight_tab, data_tab, chart_tab, sql_tab, trace_tab = st.tabs(
        ["数据洞察", "数据表格", "可视化图表", "查询 SQL", "分析轨迹"]
    )
    with insight_tab:
        if run.attribution_report is not None:
            _render_attribution_insight(run, run.attribution_report)
        elif run.fulfillment_report is not None:
            _render_fulfillment_insight(run, run.fulfillment_report)
        else:
            _render_retrieval_insight(run)
    with data_tab:
        if run.attribution_report is not None:
            _render_attribution_data(run.attribution_report)
        elif run.fulfillment_report is not None:
            _render_fulfillment_data(run.fulfillment_report)
        elif run.result is not None:
            st.caption(f"共 {len(run.result):,} 行；表格、图表与洞察均使用本次实际查询结果。")
            st.dataframe(_display_df(run.result), width="stretch", hide_index=True)
            st.download_button(
                "下载 CSV",
                data=run.result.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"olist-analysis-{run.run_id[:8]}.csv",
                mime="text/csv",
            )
    with chart_tab:
        if run.attribution_report is not None:
            _render_attribution_chart(run.attribution_report)
        elif run.fulfillment_report is not None:
            _render_fulfillment_chart(run.fulfillment_report)
        elif run.result is not None:
            _render_chart(run.result, chart_type)
    with sql_tab:
        _render_sql(run)
    with trace_tab:
        _render_trace(run)
    _render_feedback(run)


def _pipeline_config(
    settings: dict[str, object],
    *,
    llm_runtime: dict[str, str | None],
) -> PipelineConfig:
    return PipelineConfig(
        db_path=DB_PATH,
        metrics_path=METRICS_PATH,
        use_llm=bool(settings["use_llm"]),
        provider=str(llm_runtime["provider"]),
        model=str(llm_runtime["model"] or ""),
        api_key=str(llm_runtime["api_key"] or "") or None,
        base_url=str(llm_runtime["base_url"] or "") or None,
        default_limit=int(settings["default_limit"]),
        attribution_top_n=int(settings["attribution_top_n"]),
        max_retries=int(settings["max_retries"]),
        use_agent_planner=bool(settings["use_agent_planner"]),
        tracing_enabled=bool(settings["enable_tracing"]),
    )


def _start_analysis(question: str, config: PipelineConfig) -> None:
    try:
        run = execute_analysis(question, config)
    except IntentRoutingError as exc:
        st.error(str(exc))
        st.session_state.pop("pending_clarification", None)
        st.session_state.pop("last_analysis_run", None)
        return
    st.session_state["last_display_question"] = question
    if run.status == "needs_clarification":
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": run.insight or "请补充分析条件。"},
        ]
        st.session_state["pending_clarification"] = {
            "base_question": question,
            "current_question": question,
            "run": run,
            "turns": 1,
            "messages": messages,
        }
        st.session_state.pop("last_analysis_run", None)
    else:
        run.original_question = question
        st.session_state["last_analysis_run"] = run
        st.session_state.pop("pending_clarification", None)


def _render_clarification(config: PipelineConfig) -> None:
    pending = st.session_state.get("pending_clarification")
    if not isinstance(pending, dict):
        return
    run = pending["run"]
    for message in pending["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if pending.get("closed"):
        st.info("本次澄清已达到两轮上限。请重新提交一个包含指标、时间范围和分析目标的问题。")
        if st.button("结束并重新提问"):
            st.session_state.pop("pending_clarification", None)
            st.rerun()
        return

    suggestions = clarification_suggestions(run.intent)
    with st.form(f"clarification_{pending['turns']}"):
        selected = st.radio("请选择一个方向", suggestions, horizontal=True)
        custom_answer = st.text_input("或补充更具体的条件（可选）", placeholder="例如：看 GMV，最近 60 天，按周")
        submitted = st.form_submit_button("继续分析", type="primary")
    if not submitted:
        return

    answer = custom_answer.strip() or selected
    pending["messages"].append({"role": "user", "content": answer})
    resolved_question = resolve_clarification(pending["current_question"], run.intent, answer)
    if resolved_question is None:
        st.session_state.pop("pending_clarification", None)
        st.session_state.pop("last_analysis_run", None)
        st.info("已结束本次分析，你可以重新输入问题。")
        return

    with st.spinner("正在根据补充条件继续分析..."):
        try:
            next_run = execute_analysis(resolved_question, config)
        except IntentRoutingError as exc:
            st.error(str(exc))
            return
    if next_run.status == "needs_clarification":
        if int(pending["turns"]) >= 2:
            pending["messages"].append(
                {"role": "assistant", "content": "仍缺少可执行条件，请重新整理指标、时间范围和分析目标后再试。"}
            )
            pending["closed"] = True
            st.session_state["pending_clarification"] = pending
            st.rerun()
        pending["turns"] = int(pending["turns"]) + 1
        pending["current_question"] = resolved_question
        pending["run"] = next_run
        pending["messages"].append(
            {"role": "assistant", "content": next_run.insight or "还需要确认一个条件。"}
        )
        st.session_state["pending_clarification"] = pending
        st.rerun()

    next_run.original_question = str(pending["base_question"])
    next_run.clarification_history = list(pending["messages"])
    st.session_state["last_analysis_run"] = next_run
    st.session_state["last_display_question"] = str(pending["base_question"])
    st.session_state.pop("pending_clarification", None)
    st.rerun()


def _default_settings() -> dict[str, object]:
    return {
        "use_llm": True,
        "chart_type": "自动",
        "attribution_top_n": 5,
        "default_limit": 2000,
        "max_retries": 1,
        "use_agent_planner": os.getenv("AGENT_PLANNER_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"},
        "enable_tracing": os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"},
    }


def _render_sample_questions() -> None:
    st.markdown('<div class="gc-example-label">示例问题</div>', unsafe_allow_html=True)
    columns = st.columns(2)
    for idx, question in enumerate(SAMPLE_QUESTIONS):
        with columns[idx % 2]:
            if st.button(question, key=f"sample_{idx}", width="stretch"):
                st.session_state["question"] = question


def _clear_session_api_key(state_key: str) -> None:
    st.session_state[state_key] = ""


def _render_llm_settings() -> tuple[dict[str, str | None], bool]:
    with st.sidebar:
        st.markdown("### 连接模型")
        default_provider = str(LLM_RUNTIME["provider"])
        default_index = (
            LLM_PROVIDER_OPTIONS.index(default_provider)
            if default_provider in LLM_PROVIDER_OPTIONS
            else 0
        )
        provider = st.selectbox(
            "模型服务商",
            LLM_PROVIDER_OPTIONS,
            index=default_index,
            format_func=lambda value: LLM_PROVIDER_LABELS[value],
            key="llm_provider",
        )
        provider_label = LLM_PROVIDER_LABELS[provider]
        provider_runtime = resolve_llm_config(provider=provider, allow_stored_api_key=False)
        model = st.text_input(
            "模型名称",
            value=str(provider_runtime["model"] or ""),
            key=f"{provider}_model",
            placeholder="输入模型 ID",
        ).strip()
        base_url: str | None = None
        if provider == "openai_compatible":
            base_url = st.text_input(
                "Base URL",
                value=str(provider_runtime["base_url"] or ""),
                key="openai_compatible_base_url",
                placeholder="https://example.com/v1",
            ).strip() or None

        api_key_state_key = f"{provider}_api_key"
        session_api_key = st.text_input(
            f"{provider_label} API Key",
            type="password",
            key=api_key_state_key,
            placeholder="sk-...",
            help="关闭会话或点击清除后，本应用不再保留该 Key。",
        ).strip()
        runtime = resolve_llm_config(
            provider=provider,
            model=model,
            api_key_override=session_api_key or None,
            base_url_override=base_url,
            allow_stored_api_key=False,
        )
        compatible_url_valid = True
        compatible_url_error = None
        if provider == "openai_compatible":
            compatible_url_valid, compatible_url_error = validate_compatible_base_url(
                runtime["base_url"]
            )
        is_ready = bool(
            runtime["api_key"]
            and runtime["model"]
            and compatible_url_valid
        )

        if session_api_key and is_ready:
            st.success(f"已连接：本次分析将使用你的 {provider_label} 会话 Key。")
            st.button(
                "清除 Key",
                key=f"clear_{api_key_state_key}",
                on_click=_clear_session_api_key,
                args=(api_key_state_key,),
                width="stretch",
            )
        elif provider == "openai_compatible" and not compatible_url_valid:
            st.info(compatible_url_error or "请填写兼容服务的 Base URL。")
        elif not model:
            st.info("请填写模型名称。")
        else:
            st.info(f"请输入 {provider_label} API Key 后开始体验。")
        return runtime, is_ready


def main() -> None:
    st.set_page_config(
        page_title="Olist Growth Copilot",
        page_icon="↗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    settings = _default_settings()
    _render_hero()
    llm_runtime, llm_ready = _render_llm_settings()
    config = _pipeline_config(settings, llm_runtime=llm_runtime)

    if "question" not in st.session_state:
        st.session_state["question"] = ""

    _render_sample_questions()

    with st.form("analysis_form"):
        st.text_area(
            "你想了解什么增长问题？",
            key="question",
            height=96,
            placeholder="例如：近30天 GMV 趋势如何？请按天展示。",
        )
        submitted = st.form_submit_button(
            "开始可信分析",
            type="primary",
            disabled=not llm_ready,
        )

    if submitted:
        question = st.session_state.get("question", "").strip()
        if not question:
            st.warning("请先输入一个增长问题。")
        else:
            with st.spinner("正在理解问题、查询 Olist 数据并执行确定性校验..."):
                _start_analysis(question, config)
            st.rerun()

    _render_clarification(config)

    run = st.session_state.get("last_analysis_run")
    if isinstance(run, AnalysisRun):
        if run.status == "failed":
            _render_question(str(st.session_state.get("last_display_question", run.question)))
            st.error(f"分析失败（{run.failure_stage or 'unknown'}）：{run.failure_detail or '未知错误'}")
            with st.expander("查看失败轨迹", expanded=True):
                _render_trace(run)
        else:
            _render_success(
                run,
                str(st.session_state.get("last_display_question", run.original_question or run.question)),
                str(settings["chart_type"]),
            )


if __name__ == "__main__":
    main()
