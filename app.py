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

COLUMN_LABELS = {
    "dt": "日期",
    "gmv": "GMV（R$）",
    "paid_orders": "支付订单数",
    "aov": "客单价（R$）",
    "dimension": "维度",
    "dimension_value": "对象",
    "gmv_current": "本期 GMV（R$）",
    "gmv_previous": "前期 GMV（R$）",
    "contribution": "变动贡献（R$）",
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
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--paper); }
[data-testid="stSidebar"] { background: #f5f1e9; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 4rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
.gc-hero { padding: 1.1rem 0 1.35rem; border-bottom: 1px solid var(--line); margin-bottom: 1.35rem; }
.gc-kicker { color: var(--blue); font-size: .78rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
.gc-title { color: var(--ink); font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.08; font-weight: 760; margin: .35rem 0 .5rem; }
.gc-subtitle { color: var(--muted); font-size: 1rem; max-width: 760px; line-height: 1.7; }
.gc-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
.gc-badge { background: var(--paper-soft); border: 1px solid var(--line); color: #4a423c; border-radius: 999px; padding: .32rem .68rem; font-size: .78rem; }
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
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); border-radius: 9px; }
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
  <div class="gc-badges">
    <span class="gc-badge">28 个指标定义</span>
    <span class="gc-badge">2 条可评估 Workflow</span>
    <span class="gc-badge">48 道 Golden Cases</span>
    <span class="gc-badge">只读 SQL Guard</span>
  </div>
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
        st.line_chart(chart_df, x=x_col, y=y_cols, width="stretch") if x_col else st.line_chart(
            chart_df[y_cols], width="stretch"
        )
    elif actual_type == "面积图":
        st.area_chart(chart_df, x=x_col, y=y_cols, width="stretch") if x_col else st.area_chart(
            chart_df[y_cols], width="stretch"
        )
    elif actual_type == "柱状图":
        st.bar_chart(chart_df, x=x_col, y=y_cols, width="stretch") if x_col else st.bar_chart(
            chart_df[y_cols], width="stretch"
        )
    else:
        x_scatter = y_cols[0]
        y_scatter = y_cols[1] if len(y_cols) > 1 else y_cols[0]
        st.scatter_chart(chart_df, x=x_scatter, y=y_scatter, width="stretch")


def _fmt_currency(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}R$ {value:,.2f}"


def _fmt_currency_compact(value: float | None) -> str:
    if value is None:
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"R$ {value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"R$ {value / 1_000:,.1f}K"
    return f"R$ {value:,.2f}"


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}%"


def _insight_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped.replace("\n", "<br>")


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    shown = df.rename(columns=COLUMN_LABELS).copy()
    if "对象" in shown.columns:
        shown["对象"] = shown["对象"].map(lambda value: str(value).replace("_", " "))
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda value: round(float(value), 2) if pd.notna(value) else value)
    return shown


def _render_question(question: str) -> None:
    st.markdown(
        f'<div class="gc-question">{html.escape(question)}</div>',
        unsafe_allow_html=True,
    )


def _render_intent_context(run: AnalysisRun) -> None:
    intent = run.intent
    st.markdown(
        "".join(
            [
                '<div class="gc-meta">',
                f"<span>{html.escape(intent.time_range or '时间待确认')}</span>",
                f"<span>{html.escape(intent.metric or '指标待确认')}</span>",
                f"<span>{html.escape(run.workflow)}</span>",
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
    df = run.result
    if df is not None and not df.empty:
        numeric_cols = list(df.select_dtypes(include="number").columns)
        cards = st.columns(min(3, max(1, len(numeric_cols) + 1)))
        cards[0].metric("返回记录", f"{len(df):,} 行")
        for idx, col in enumerate(numeric_cols[: len(cards) - 1], start=1):
            total = float(df[col].fillna(0).sum())
            label = COLUMN_LABELS.get(str(col), str(col))
            cards[idx].metric(label, _fmt_currency(total) if "gmv" in str(col).lower() else f"{total:,.2f}")
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
                "buyers": report.buyers_current,
                "aov": report.aov_current,
            },
            {
                "period": "前期",
                "start": report.previous_start,
                "end": report.previous_end,
                "gmv": report.gmv_previous,
                "orders": report.orders_previous,
                "buyers": report.buyers_previous,
                "aov": report.aov_previous,
            },
        ]
    )


def _render_attribution_insight(run: AnalysisRun, report: GmvAttributionReport) -> None:
    st.markdown(
        f'<div class="gc-answer">{_insight_html(run.insight or "归因分析已完成。")}</div>',
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m1.metric("本期 GMV", _fmt_currency_compact(report.gmv_current), _fmt_percent(report.gmv_change_rate))
    m2.metric("GMV 变动", _fmt_currency_compact(report.gmv_delta))
    m3, m4 = st.columns(2)
    m3.metric("支付订单数", f"{report.orders_current:,}", f"{report.orders_current - report.orders_previous:+,}")
    aov_delta = (report.aov_current or 0) - (report.aov_previous or 0)
    m4.metric("客单价（R$）", f"{report.aov_current:,.2f}" if report.aov_current is not None else "—", f"{aov_delta:+,.2f}")
    st.caption(
        f"数据锚点：{report.anchor_date} ｜ 本期 {report.current_start}—{report.current_end} ｜ "
        f"对比期 {report.previous_start}—{report.previous_end}"
    )
    for caveat in run.caveats:
        st.warning(f"分析限制：{caveat}")


def _render_attribution_data(report: GmvAttributionReport) -> None:
    st.markdown("#### 窗口汇总")
    st.dataframe(_display_df(_attribution_summary_df(report)), width="stretch", hide_index=True)
    st.markdown("#### GMV = 订单量 × AOV")
    decomposition = pd.DataFrame(
        [
            {"因素": "订单量效应", "变动贡献（R$）": report.order_effect},
            {"因素": "AOV 效应", "变动贡献（R$）": report.aov_effect},
        ]
    )
    st.dataframe(_display_df(decomposition), width="stretch", hide_index=True)
    trend_df = report.all_drops if report.gmv_delta < 0 else report.all_gains
    st.markdown("#### 主要维度贡献")
    st.dataframe(_display_df(trend_df), width="stretch", hide_index=True)
    with st.expander("查看州、品类与商家的完整 Top N 拆解"):
        for title, frame in (
            ("州", report.state_drops if report.gmv_delta < 0 else report.state_gains),
            ("品类", report.category_drops if report.gmv_delta < 0 else report.category_gains),
            ("商家", report.seller_drops if report.gmv_delta < 0 else report.seller_gains),
        ):
            st.markdown(f"**{title}**")
            st.dataframe(_display_df(frame), width="stretch", hide_index=True)


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


def _render_sql(run: AnalysisRun) -> None:
    if run.attribution_report is not None:
        report: GmvAttributionReport = run.attribution_report
        labels = list(report.evidence_sql)
        selected = st.selectbox("选择证据查询", labels, key=f"sql_evidence_{run.run_id}")
        st.caption("以下 SQL 均来自本次确定性归因实际使用的查询模板，并以当前分析窗口参数化执行。")
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


def _render_trace(run: AnalysisRun) -> None:
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
        else:
            _render_retrieval_insight(run)
    with data_tab:
        if run.attribution_report is not None:
            _render_attribution_data(run.attribution_report)
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
        elif run.result is not None:
            _render_chart(run.result, chart_type)
    with sql_tab:
        _render_sql(run)
    with trace_tab:
        _render_trace(run)
    _render_feedback(run)


def _pipeline_config(settings: dict[str, object]) -> PipelineConfig:
    return PipelineConfig(
        db_path=DB_PATH,
        metrics_path=METRICS_PATH,
        use_llm=bool(settings["use_llm"]),
        model=MODEL,
        default_limit=int(settings["default_limit"]),
        attribution_top_n=int(settings["attribution_top_n"]),
        max_retries=int(settings["max_retries"]),
        tracing_enabled=bool(settings["enable_tracing"]),
    )


def _start_analysis(question: str, config: PipelineConfig) -> None:
    run = execute_analysis(question, config)
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
        next_run = execute_analysis(resolved_question, config)
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


def _render_sidebar() -> dict[str, object]:
    with st.sidebar:
        st.markdown("### 示例问题")
        st.caption("覆盖可信问数、GMV 归因与主动澄清")
        for idx, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample_{idx}", width="stretch"):
                st.session_state["question"] = question

        st.divider()
        with st.expander("产品边界", expanded=False):
            st.write(
                "当前确定性自动归因聚焦 GMV。Olist 不含曝光、点击、广告渠道与成本数据，"
                "因此不会对 CTR、CAC 或 ROAS 给出伪精确结论。"
            )

        with st.expander("高级设置", expanded=False):
            use_llm = st.toggle("使用模型生成 SQL", value=True)
            chart_type = st.selectbox("默认图表", ["自动", "折线图", "柱状图", "面积图", "散点图"])
            attribution_top_n = st.slider("归因 Top N", min_value=3, max_value=15, value=5)
            default_limit = st.number_input("默认最大行数", min_value=100, max_value=10000, value=2000, step=100)
            max_retries = st.selectbox("确定性纠错次数", options=[0, 1], index=1)
            enable_tracing = st.toggle(
                "LangSmith Trace",
                value=os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"},
            )
            if use_llm and not ACTIVE_API_KEY:
                st.info(f"未检测到 {ACTIVE_API_KEY_NAME}，将使用有限规则降级路径。")
            elif use_llm and ACTIVE_API_KEY_SOURCE == "keychain":
                st.caption(f"已从 macOS Keychain 读取 {ACTIVE_API_KEY_NAME}。")

    return {
        "use_llm": use_llm,
        "chart_type": chart_type,
        "attribution_top_n": attribution_top_n,
        "default_limit": default_limit,
        "max_retries": max_retries,
        "enable_tracing": enable_tracing,
    }


def main() -> None:
    st.set_page_config(page_title="Olist Growth Copilot", page_icon="↗", layout="wide")
    _inject_theme()
    settings = _render_sidebar()
    config = _pipeline_config(settings)
    _render_hero()

    if "question" not in st.session_state:
        st.session_state["question"] = ""

    with st.form("analysis_form"):
        st.text_area(
            "你想了解什么增长问题？",
            key="question",
            height=96,
            placeholder="例如：近30天 GMV 走势如何？请按天展示。",
        )
        submitted = st.form_submit_button("开始可信分析", type="primary")

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
