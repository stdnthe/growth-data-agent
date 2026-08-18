from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .attribution import analyze_gmv_change_drivers, build_gmv_summary
from .insight import generate_insight
from .intent_router import route
from .llm_sql import generate_sql, resolve_llm_config
from .metrics_store import MetricsStore
from .models import AnalysisRun, AnalysisStep, AttemptRecord, ValidationResult
from .observability import TraceRecorder
from .sql_guard import SQLGuard, SQLGuardError
from .validators import validate_gmv_attribution, validate_retrieval_result


@dataclass
class PipelineConfig:
    db_path: str | Path
    metrics_path: str | Path
    use_llm: bool = True
    model: str | None = None
    default_limit: int = 2000
    attribution_top_n: int = 5
    max_retries: int = 1
    tracing_enabled: bool | None = None
    trace_project: str = "growth-analysis-agent"
    prompt_version: str = "growth-sql-v2"


def _run_query(db_path: Path, sql: str):
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB 文件不存在：{db_path}")
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        return connection.execute(sql).fetchdf()
    finally:
        connection.close()


def _validation_feedback(validations: list[ValidationResult]) -> str:
    return "; ".join(item.message for item in validations if not item.passed and item.severity == "error")


def _finalize(run: AnalysisRun, started: float) -> AnalysisRun:
    run.latency_ms = int((time.perf_counter() - started) * 1000)
    if run.status not in {"failed", "needs_clarification"}:
        if not run.validation_passed:
            run.status = "failed"
            run.failure_stage = run.failure_stage or "validation"
            run.failure_detail = run.failure_detail or _validation_feedback(run.validations)
        elif run.warning_count:
            run.status = "success_with_warnings"
        else:
            run.status = "success"
    return run


def _add_step(
    run: AnalysisRun,
    *,
    key: str,
    label: str,
    category: str,
    summary: str,
    status: str = "completed",
    tool: str | None = None,
    **details: object,
) -> None:
    run.steps.append(
        AnalysisStep(
            key=key,
            label=label,
            category=category,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            summary=summary,
            tool=tool,
            details=details,
        )
    )


def execute_analysis(question: str, config: PipelineConfig) -> AnalysisRun:
    started = time.perf_counter()
    intent = route(question)
    llm_runtime = resolve_llm_config(model=config.model)
    metrics_store = MetricsStore(config.metrics_path)
    metrics_store.load()
    selected_metrics = [
        metric
        for metric_id in (intent.metrics or ([intent.metric] if intent.metric else []))
        if (metric := metrics_store.get(metric_id)) is not None
    ]
    tracer = TraceRecorder(enabled=config.tracing_enabled, project=config.trace_project)

    run = AnalysisRun(
        question=question,
        intent=intent,
        workflow=intent.workflow,
        provider=str(llm_runtime["provider"]),
        model=str(llm_runtime["model"]),
        prompt_version=config.prompt_version,
        metric_version=metrics_store.version,
        trace_enabled=tracer.available,
        metric_context=[
            f"{metric.id}: {metric.definition}; formula={metric.formula}"
            for metric in selected_metrics
        ],
    )
    _add_step(
        run,
        key="parse_intent",
        label="理解业务问题",
        category="decision",
        summary=(
            f"识别指标 {intent.metric or '待确认'}，任务类型 {intent.task_type}，"
            f"时间范围 {intent.time_range or '待确认'}。"
        ),
        action=intent.action,
        ambiguities=intent.ambiguities,
        dimensions=intent.dimensions,
        grain=intent.grain,
    )
    _add_step(
        run,
        key="select_workflow",
        label="选择指标语义与 Workflow",
        category="context",
        summary=f"路由到 {intent.workflow}，加载 {len(run.metric_context)} 条指标口径。",
        tool="MetricsStore",
        metric_context=run.metric_context,
        metric_version=run.metric_version,
    )

    root_metadata = {
        "run_id": run.run_id,
        "workflow": run.workflow,
        "provider": run.provider,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "metric_version": run.metric_version,
    }
    with tracer.span(
        "growth_analysis_run",
        inputs={"question": question},
        metadata=root_metadata,
        tags=[run.workflow, intent.metric or "unknown_metric"],
    ) as root_span:
        with tracer.span("parse_intent", inputs={"question": question}) as span:
            span.set_outputs(intent=intent.to_dict())

        if intent.action in {"clarify", "confirm"}:
            run.status = "needs_clarification"
            run.insight = intent.clarification_question
            _add_step(
                run,
                key="human_confirmation",
                label="等待用户确认",
                category="decision",
                status="warning",
                summary=intent.clarification_question or "需要补充信息后继续。",
                ambiguities=intent.ambiguities,
            )
            _finalize(run, started)
            root_span.set_outputs(**run.to_trace_dict())
            return run

        run.status = "running"

        try:
            if intent.workflow == "gmv_attribution":
                with tracer.span(
                    "calculate_gmv_attribution",
                    run_type="tool",
                    inputs={"window_days": intent.window_days, "top_n": config.attribution_top_n},
                ) as span:
                    report = analyze_gmv_change_drivers(
                        db_path=config.db_path,
                        window_days=intent.window_days,
                        top_n=config.attribution_top_n,
                    )
                    run.attribution_report = report
                    _add_step(
                        run,
                        key="calculate_gmv_attribution",
                        label="执行 GMV 归因",
                        category="tool",
                        summary=(
                            f"比较 {report.current_start} 至 {report.current_end} 与前一等长窗口，"
                            "完成 GMV、订单量、AOV 及维度贡献拆解。"
                        ),
                        tool="DuckDB · deterministic attribution",
                        window_days=report.window_days,
                        queries=len(getattr(report, "evidence_sql", {})),
                    )
                    span.set_outputs(
                        gmv_current=report.gmv_current,
                        gmv_previous=report.gmv_previous,
                        gmv_delta=report.gmv_delta,
                    )

                with tracer.span("validate_attribution", run_type="tool") as span:
                    run.validations.extend(validate_gmv_attribution(report))
                    _add_step(
                        run,
                        key="validate_attribution",
                        label="验证归因结果",
                        category="validation",
                        status="completed" if run.validation_passed else "failed",
                        summary=(
                            f"完成 {len(run.validations)} 项确定性校验；"
                            f"{'全部通过' if run.validation_passed else '存在未通过项'}。"
                        ),
                        tool="Attribution validators",
                        validations=[item.to_dict() for item in run.validations],
                    )
                    span.set_outputs(validations=[item.to_dict() for item in run.validations])

                if run.validation_passed:
                    with tracer.span("generate_attribution_insight", run_type="llm") as span:
                        run.insight = build_gmv_summary(report)
                        _add_step(
                            run,
                            key="generate_insight",
                            label="生成证据化洞察",
                            category="output",
                            summary="仅基于通过校验的 GMV 分解与维度贡献生成结论。",
                            tool="deterministic summary",
                        )
                        span.set_outputs(insight=run.insight)
                else:
                    run.caveats.append("归因确定性校验未通过，已停止生成解释性结论。")
            else:
                metrics_context = metrics_store.compressed_context(max_items=30)
                retry_context: str | None = None
                final_result = None

                for attempt_number in range(1, max(0, config.max_retries) + 2):
                    with tracer.span(
                        "generate_sql",
                        run_type="llm",
                        inputs={"question": question, "retry_context": retry_context or ""},
                        metadata={"attempt": attempt_number},
                    ) as span:
                        sql, source = generate_sql(
                            question=question,
                            metrics_context=metrics_context,
                            use_llm=config.use_llm,
                            model=config.model,
                            retry_context=retry_context,
                        )
                        run.generated_sql = sql
                        run.sql_source = source
                        attempt = AttemptRecord(attempt=attempt_number, sql_source=source, sql=sql)
                        run.attempts.append(attempt)
                        _add_step(
                            run,
                            key=f"generate_sql_{attempt_number}",
                            label=f"生成查询 SQL（第 {attempt_number} 次）",
                            category="tool",
                            summary=f"SQL 来源：{source}。",
                            tool="Text-to-SQL",
                            sql_source=source,
                        )
                        span.set_outputs(sql=sql, source=source)

                    guard = SQLGuard(default_limit=config.default_limit)
                    try:
                        with tracer.span("validate_sql", run_type="tool", inputs={"sql": sql}) as span:
                            safe_sql = guard.validate_and_rewrite(sql)
                            run.generated_sql = safe_sql
                            attempt.stage = "guard"
                            _add_step(
                                run,
                                key=f"validate_sql_{attempt_number}",
                                label="执行 SQL 安全检查",
                                category="validation",
                                summary="只读、单语句、表白名单与 LIMIT 检查通过。",
                                tool="SQLGuard",
                            )
                            span.set_outputs(passed=True, safe_sql=safe_sql)
                    except SQLGuardError as exc:
                        attempt.stage = "guard"
                        attempt.error = str(exc)
                        _add_step(
                            run,
                            key=f"validate_sql_{attempt_number}",
                            label="执行 SQL 安全检查",
                            category="validation",
                            status="warning" if source == "llm" and attempt_number <= config.max_retries else "failed",
                            summary=f"SQL 被安全规则拒绝：{exc}",
                            tool="SQLGuard",
                        )
                        retry_context = f"SQL Guard rejected the query: {exc}"
                        can_retry = source == "llm" and attempt_number <= config.max_retries
                        if can_retry:
                            continue
                        raise

                    try:
                        with tracer.span("execute_duckdb", run_type="tool", inputs={"sql": safe_sql}) as span:
                            final_result = _run_query(Path(config.db_path), safe_sql)
                            attempt.stage = "execution"
                            _add_step(
                                run,
                                key=f"execute_duckdb_{attempt_number}",
                                label="查询 Olist 数据",
                                category="tool",
                                summary=f"DuckDB 返回 {len(final_result)} 行、{len(final_result.columns)} 列。",
                                tool="DuckDB",
                                row_count=len(final_result),
                                columns=[str(col) for col in final_result.columns],
                            )
                            span.set_outputs(row_count=len(final_result), columns=list(final_result.columns))
                    except Exception as exc:  # noqa: BLE001
                        attempt.stage = "execution"
                        attempt.error = str(exc)
                        _add_step(
                            run,
                            key=f"execute_duckdb_{attempt_number}",
                            label="查询 Olist 数据",
                            category="tool",
                            status="warning" if source == "llm" and attempt_number <= config.max_retries else "failed",
                            summary=f"查询执行失败：{type(exc).__name__}: {exc}",
                            tool="DuckDB",
                        )
                        retry_context = f"DuckDB execution failed: {type(exc).__name__}: {exc}"
                        can_retry = source == "llm" and attempt_number <= config.max_retries
                        if can_retry:
                            continue
                        raise

                    with tracer.span("validate_result", run_type="tool") as span:
                        attempt_validations = validate_retrieval_result(final_result, intent)
                        validation_passed = not any(
                            not item.passed and item.severity == "error" for item in attempt_validations
                        )
                        _add_step(
                            run,
                            key=f"validate_result_{attempt_number}",
                            label="验证查询结果",
                            category="validation",
                            status="completed" if validation_passed else (
                                "warning" if source == "llm" and attempt_number <= config.max_retries else "failed"
                            ),
                            summary=(
                                f"完成 {len(attempt_validations)} 项校验；"
                                f"{'通过' if validation_passed else '需要纠错'}。"
                            ),
                            tool="Result validators",
                            validations=[item.to_dict() for item in attempt_validations],
                        )
                        span.set_outputs(validations=[item.to_dict() for item in attempt_validations])

                    validation_errors = _validation_feedback(attempt_validations)
                    can_retry = bool(validation_errors) and source == "llm" and attempt_number <= config.max_retries
                    if can_retry:
                        retry_context = validation_errors
                        attempt.stage = "validation"
                        attempt.error = validation_errors
                        continue

                    run.validations.extend(attempt_validations)
                    break

                if final_result is None:
                    raise RuntimeError("查询未产生可用结果。")
                run.result = final_result
                if run.validation_passed:
                    with tracer.span("generate_insight", run_type="chain") as span:
                        run.insight = generate_insight(final_result)
                        _add_step(
                            run,
                            key="generate_insight",
                            label="生成证据化洞察",
                            category="output",
                            summary="从最终查询结果生成摘要，不引入结果之外的业务事实。",
                            tool="deterministic insight generator",
                        )
                        span.set_outputs(insight=run.insight)

        except SQLGuardError as exc:
            run.status = "failed"
            run.failure_stage = "guard"
            run.failure_detail = str(exc)
        except FileNotFoundError as exc:
            run.status = "failed"
            run.failure_stage = "data"
            run.failure_detail = str(exc)
        except Exception as exc:  # noqa: BLE001
            run.status = "failed"
            run.failure_stage = "execution"
            run.failure_detail = f"{type(exc).__name__}: {exc}"

        _finalize(run, started)
        root_span.set_outputs(**run.to_trace_dict())
        return run
