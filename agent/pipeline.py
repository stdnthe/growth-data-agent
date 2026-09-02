from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .agent_loop import BoundedLoopController, LoopObservation
from .attribution import analyze_gmv_change_drivers, build_gmv_summary
from .context_builder import ContextBuilder
from .fulfillment import analyze_fulfillment_change_drivers, build_fulfillment_summary
from .insight import generate_insight
from .intent_router import route
from .langgraph_runtime import run_bounded_agent_graph
from .llm_sql import generate_sql, resolve_llm_config
from .metrics_store import MetricsStore
from .models import AgentDecision, AnalysisRun, AnalysisStep, AttemptRecord, ValidationResult
from .observability import TraceRecorder
from .planner import AgentController
from .sql_guard import SQLGuard, SQLGuardError
from .validators import (
    validate_fulfillment_diagnosis,
    validate_gmv_attribution,
    validate_retrieval_result,
)


@dataclass
class PipelineConfig:
    db_path: str | Path
    metrics_path: str | Path
    use_llm: bool = True
    provider: str | None = None
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    default_limit: int = 2000
    attribution_top_n: int = 5
    max_retries: int = 1
    max_steps: int = 3
    use_agent_planner: bool = True
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


def _record_decision(run: AnalysisRun, decision: AgentDecision) -> None:
    run.decisions.append(decision)
    status = "warning" if decision.action in {"clarify", "unsupported", "stop"} else "completed"
    _add_step(
        run,
        key=f"decide_next_action_{decision.sequence}",
        label="决定下一步动作",
        category="decision",
        status=status,
        summary=f"{decision.action}: {decision.reason_summary}",
        tool=decision.tool,
        action=decision.action,
        completion_reason=decision.completion_reason,
        decision_source=decision.decision_source,
        policy_status=decision.policy_status,
        parameters=decision.parameters,
    )


def execute_analysis(question: str, config: PipelineConfig) -> AnalysisRun:
    started = time.perf_counter()
    metrics_store = MetricsStore(config.metrics_path)
    metrics_store.load()
    intent = route(
        question,
        metrics_store,
        model=config.model,
        api_key=config.api_key,
        provider=config.provider,
        base_url=config.base_url,
    )
    llm_runtime = resolve_llm_config(
        model=config.model,
        provider=config.provider,
        api_key_override=config.api_key,
        base_url_override=config.base_url,
    )
    selected_metrics = [
        metric
        for metric_id in (intent.metrics or ([intent.metric] if intent.metric else []))
        if (metric := metrics_store.get(metric_id)) is not None
    ]
    context_bundle = ContextBuilder(metrics_store).build(intent)
    loop = BoundedLoopController(max_steps=config.max_steps)
    agent_controller = AgentController(
        use_llm=config.use_llm and config.use_agent_planner,
        model=config.model,
        api_key=config.api_key,
        provider=config.provider,
        base_url=config.base_url,
    )
    tracer = TraceRecorder(enabled=config.tracing_enabled, project=config.trace_project)
    active_capabilities = agent_controller.registry.capabilities.select(intent)

    run = AnalysisRun(
        question=question,
        intent=intent,
        workflow=intent.workflow,
        provider=str(llm_runtime["provider"]),
        model=str(llm_runtime["model"]),
        prompt_version=config.prompt_version,
        metric_version=metrics_store.version,
        trace_enabled=tracer.available,
        agent_runtime="langgraph",
        max_steps=config.max_steps,
        active_capabilities=[capability.id for capability in active_capabilities],
        metric_context=[
            f"{metric.id}: {metric.definition}; formula={metric.formula}"
            for metric in selected_metrics
        ],
        context_bundle=context_bundle.to_dict(),
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
        key="validate_intent",
        label="验证结构化意图",
        category="validation",
        summary=(
            f"IntentValidator 已校验指标、时间、粒度与维度，并给出 "
            f"{intent.workflow} 能力提示；具体工具由单一 Agent Controller 在 Policy Gate 内选择。"
        ),
        tool="IntentValidator",
        metric=intent.metric,
        workflow=intent.workflow,
        action=intent.action,
    )
    _add_step(
        run,
        key="assemble_context",
        label="组装分析上下文",
        category="context",
        summary=(
            f"加载 {len(run.metric_context)} 条指标口径、"
            f"{len(context_bundle.schemas)} 个相关 Schema 与输出契约。"
        ),
        tool="ContextBuilder",
        metric_context=run.metric_context,
        metric_version=run.metric_version,
        allowed_tables=context_bundle.allowed_tables,
        output_contract=context_bundle.output_contract.to_dict(),
        active_capabilities=[capability.to_dict() for capability in active_capabilities],
    )

    root_metadata = {
        "run_id": run.run_id,
        "workflow": run.workflow,
        "provider": run.provider,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "metric_version": run.metric_version,
        "active_capabilities": run.active_capabilities,
    }
    with tracer.span(
        "growth_analysis_run",
        inputs={"question": question},
        metadata=root_metadata,
        tags=[run.workflow, intent.metric or "unknown_metric"],
    ) as root_span:
        with tracer.span("parse_intent", inputs={"question": question}) as span:
            span.set_outputs(intent=intent.to_dict())

        def decide_next(observation: LoopObservation | None) -> AgentDecision:
            observation_payload = (
                None
                if observation is None
                else {
                    "stage": observation.stage,
                    "passed": observation.passed,
                    "evidence_sufficient": observation.evidence_sufficient,
                    "retryable": observation.retryable,
                    "schema_inspected": observation.schema_inspected,
                    "error_summary": observation.error_summary,
                }
            )
            with tracer.span(
                "select_next_action",
                run_type="chain",
                inputs={
                    "intent": intent.to_dict(),
                    "active_capabilities": run.active_capabilities,
                    "observation": observation_payload,
                    "remaining_steps": loop.remaining_steps,
                },
            ) as span:
                next_decision = agent_controller.decide(
                    question=question,
                    intent=intent,
                    observation=observation,
                    controller=loop,
                )
                span.set_outputs(decision=next_decision.to_dict())
                return next_decision

        decision = decide_next(None)
        _record_decision(run, decision)

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
            metrics_context = context_bundle.to_prompt()
            retry_context: str | None = None
            schema_inspected = False
            attempt_number = 0

            def execute_loop_action(action_decision: AgentDecision) -> LoopObservation:
                nonlocal retry_context, schema_inspected, attempt_number

                if action_decision.action == "inspect_schema":
                    schema_inspected = True
                    schema_context = context_bundle.schema_summary()
                    retry_context = "\n".join(
                        part
                        for part in (
                            retry_context,
                            "Relevant schema retrieved from ContextBundle:\n" + schema_context,
                        )
                        if part
                    )
                    _add_step(
                        run,
                        key=f"inspect_schema_{loop.action_count}",
                        label="补充相关 Schema 上下文",
                        category="context",
                        summary=f"读取 {len(context_bundle.schemas)} 个相关表或视图的字段契约。",
                        tool="ContextBundle Schema",
                        schemas={table: list(columns) for table, columns in context_bundle.schemas.items()},
                    )
                    return LoopObservation(stage="schema", passed=True, schema_inspected=True)

                if action_decision.action == "run_gmv_recipe":
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
                            label="执行 GMV 归因 Recipe",
                            category="tool",
                            summary=(
                                f"比较 {report.current_start} 至 {report.current_end} 与前一等长窗口，"
                                "完成 GMV、订单量、AOV 及维度贡献拆解。"
                            ),
                            tool="DuckDB · GMV Attribution Recipe",
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
                        passed = run.validation_passed
                        _add_step(
                            run,
                            key="validate_attribution",
                            label="验证归因结果",
                            category="validation",
                            status="completed" if passed else "failed",
                            summary=(
                                f"完成 {len(run.validations)} 项确定性校验；"
                                f"{'全部通过' if passed else '存在未通过项'}。"
                            ),
                            tool="Attribution validators",
                            validations=[item.to_dict() for item in run.validations],
                        )
                        span.set_outputs(validations=[item.to_dict() for item in run.validations])
                    if not passed:
                        run.caveats.append("归因确定性校验未通过，已停止生成解释性结论。")
                    return LoopObservation(
                        stage="validation",
                        passed=passed,
                        retryable=False,
                        error_summary=None if passed else _validation_feedback(run.validations),
                    )

                if action_decision.action == "run_fulfillment_recipe":
                    with tracer.span(
                        "calculate_fulfillment_diagnosis",
                        run_type="tool",
                        inputs={"window_days": intent.window_days},
                    ) as span:
                        report = analyze_fulfillment_change_drivers(
                            db_path=config.db_path,
                            window_days=intent.window_days,
                        )
                        run.fulfillment_report = report
                        _add_step(
                            run,
                            key="calculate_fulfillment_diagnosis",
                            label="执行履约时效诊断 Recipe",
                            category="tool",
                            summary=(
                                f"比较 {report.current_start} 至 {report.current_end} 与前一等长窗口，"
                                "将准时送达率变化拆解为客户州结构效应与组内履约效应。"
                            ),
                            tool="DuckDB · Fulfillment Diagnosis Recipe",
                            window_days=report.window_days,
                            states=len(report.state_contributions),
                        )
                        span.set_outputs(
                            on_time_rate_current=report.on_time_rate_current,
                            on_time_rate_previous=report.on_time_rate_previous,
                            on_time_rate_delta=report.on_time_rate_delta,
                        )
                    with tracer.span("validate_fulfillment_diagnosis", run_type="tool") as span:
                        run.validations.extend(validate_fulfillment_diagnosis(report))
                        passed = run.validation_passed
                        _add_step(
                            run,
                            key="validate_fulfillment_diagnosis",
                            label="验证履约诊断结果",
                            category="validation",
                            status="completed" if passed else "failed",
                            summary=(
                                f"完成 {len(run.validations)} 项确定性校验；"
                                f"{'全部通过' if passed else '存在未通过项'}。"
                            ),
                            tool="Fulfillment validators",
                            validations=[item.to_dict() for item in run.validations],
                        )
                        span.set_outputs(validations=[item.to_dict() for item in run.validations])
                    run.caveats.append("结构与组内贡献用于解释指标变化，不代表已证明业务因果关系。")
                    if not passed:
                        run.caveats.append("履约诊断校验未通过，已停止生成解释性结论。")
                    return LoopObservation(
                        stage="validation",
                        passed=passed,
                        retryable=False,
                        error_summary=None if passed else _validation_feedback(run.validations),
                    )

                if action_decision.action != "query_metric":
                    return LoopObservation(
                        stage="routing",
                        passed=False,
                        retryable=False,
                        error_summary=f"Unsupported graph action: {action_decision.action}",
                    )

                attempt_number += 1
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
                        api_key=config.api_key,
                        provider=config.provider,
                        base_url=config.base_url,
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
                    retry_context = f"SQL Guard rejected the query: {exc}"
                    retryable = source == "llm" and attempt_number <= config.max_retries
                    _add_step(
                        run,
                        key=f"validate_sql_{attempt_number}",
                        label="执行 SQL 安全检查",
                        category="validation",
                        status="warning" if retryable else "failed",
                        summary=f"SQL 被安全规则拒绝：{exc}",
                        tool="SQLGuard",
                    )
                    return LoopObservation(
                        stage="guard",
                        passed=False,
                        retryable=retryable,
                        error_summary=retry_context,
                        schema_inspected=schema_inspected,
                    )

                try:
                    with tracer.span("execute_duckdb", run_type="tool", inputs={"sql": safe_sql}) as span:
                        result = _run_query(Path(config.db_path), safe_sql)
                        attempt.stage = "execution"
                        _add_step(
                            run,
                            key=f"execute_duckdb_{attempt_number}",
                            label="查询 Olist 数据",
                            category="tool",
                            summary=f"DuckDB 返回 {len(result)} 行、{len(result.columns)} 列。",
                            tool="DuckDB",
                            row_count=len(result),
                            columns=[str(col) for col in result.columns],
                        )
                        span.set_outputs(row_count=len(result), columns=list(result.columns))
                except Exception as exc:  # noqa: BLE001
                    attempt.stage = "execution"
                    attempt.error = str(exc)
                    retry_context = f"DuckDB execution failed: {type(exc).__name__}: {exc}"
                    retryable = source == "llm" and attempt_number <= config.max_retries
                    _add_step(
                        run,
                        key=f"execute_duckdb_{attempt_number}",
                        label="查询 Olist 数据",
                        category="tool",
                        status="warning" if retryable else "failed",
                        summary=f"查询执行失败：{type(exc).__name__}: {exc}",
                        tool="DuckDB",
                    )
                    return LoopObservation(
                        stage="execution",
                        passed=False,
                        retryable=retryable,
                        error_summary=retry_context,
                        schema_inspected=schema_inspected,
                    )

                with tracer.span("validate_result", run_type="tool") as span:
                    attempt_validations = validate_retrieval_result(
                        result,
                        intent,
                        context_bundle=context_bundle,
                    )
                    passed = not any(
                        not item.passed and item.severity == "error" for item in attempt_validations
                    )
                    validation_errors = _validation_feedback(attempt_validations)
                    retryable = bool(validation_errors) and source == "llm" and attempt_number <= config.max_retries
                    _add_step(
                        run,
                        key=f"validate_result_{attempt_number}",
                        label="验证查询结果",
                        category="validation",
                        status="completed" if passed else ("warning" if retryable else "failed"),
                        summary=f"完成 {len(attempt_validations)} 项校验；{'通过' if passed else '需要纠错'}。",
                        tool="Result validators",
                        validations=[item.to_dict() for item in attempt_validations],
                    )
                    span.set_outputs(validations=[item.to_dict() for item in attempt_validations])
                if retryable:
                    retry_context = validation_errors
                    attempt.stage = "validation"
                    attempt.error = validation_errors
                else:
                    run.validations.extend(attempt_validations)
                if passed:
                    run.result = result
                return LoopObservation(
                    stage="validation",
                    passed=passed,
                    evidence_sufficient=intent.task_type == "metric_query",
                    retryable=retryable,
                    error_summary=validation_errors or None,
                    schema_inspected=schema_inspected,
                )

            graph_state = run_bounded_agent_graph(
                initial_decision=decision,
                controller=loop,
                execute_action=execute_loop_action,
                record_decision=lambda item: _record_decision(run, item),
                decide_next=decide_next,
            )
            decision = graph_state["decision"]
            final_observation = graph_state.get("observation")
            if decision.action == "stop":
                run.status = "failed"
                run.failure_stage = final_observation.stage if final_observation else "loop"
                run.failure_detail = (
                    final_observation.error_summary
                    if final_observation and final_observation.error_summary
                    else decision.reason_summary
                )
            elif decision.action == "finish" and run.validation_passed:
                if run.attribution_report is not None:
                    with tracer.span("generate_attribution_insight", run_type="chain") as span:
                        run.insight = build_gmv_summary(run.attribution_report)
                        _add_step(
                            run,
                            key="generate_insight",
                            label="生成证据化洞察",
                            category="output",
                            summary="仅基于通过校验的 GMV Recipe 结果生成结论。",
                            tool="deterministic summary",
                        )
                        span.set_outputs(insight=run.insight)
                elif run.fulfillment_report is not None:
                    with tracer.span("generate_fulfillment_insight", run_type="chain") as span:
                        run.insight = build_fulfillment_summary(run.fulfillment_report)
                        _add_step(
                            run,
                            key="generate_insight",
                            label="生成证据化履约洞察",
                            category="output",
                            summary="仅基于通过校验的履约 Recipe 结果生成变化贡献结论。",
                            tool="deterministic fulfillment summary",
                        )
                        span.set_outputs(insight=run.insight)
                elif run.result is not None:
                    with tracer.span("generate_insight", run_type="chain") as span:
                        run.insight = generate_insight(run.result)
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
