from __future__ import annotations

from dataclasses import dataclass

from .models import AgentDecision, AnalysisIntent


@dataclass(frozen=True)
class LoopObservation:
    stage: str
    passed: bool
    evidence_sufficient: bool = True
    retryable: bool = False
    error_summary: str | None = None
    schema_inspected: bool = False


class BoundedLoopController:
    """Deterministic control plane around model-generated analysis actions."""

    def __init__(self, *, max_steps: int = 3) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.max_steps = max_steps
        self.action_count = 0
        self.decision_count = 0

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.action_count)

    def initial(self, intent: AnalysisIntent) -> AgentDecision:
        if intent.action in {"clarify", "confirm"}:
            return self._decision(
                action="clarify",
                reason="关键信息不足或分析边界需要用户确认。",
                completion_reason="needs_clarification",
            )
        if intent.workflow == "gmv_attribution":
            return self._decision(
                action="run_gmv_recipe",
                tool="GMV Attribution Recipe",
                reason="问题要求解释 GMV 变化，优先使用可验算的确定性归因 Recipe。",
            )
        if intent.workflow == "fulfillment_diagnosis":
            return self._decision(
                action="run_fulfillment_recipe",
                tool="Fulfillment Diagnosis Recipe",
                reason="问题要求解释履约指标变化，优先使用可加和的结构/组内分解 Recipe。",
            )
        return self._decision(
            action="query_metric",
            tool="Text-to-SQL + DuckDB",
            reason="指标、时间与粒度已明确，先执行受指标语义约束的查询。",
        )

    def mark_action_executed(self) -> None:
        self.action_count += 1

    def after_observation(self, observation: LoopObservation) -> AgentDecision:
        if observation.passed:
            return self._decision(
                action="finish",
                reason="确定性执行与结果校验已通过，现有证据足以回答问题。",
                completion_reason="validated",
            )
        if not observation.retryable:
            return self._decision(
                action="stop",
                reason=observation.error_summary or "当前错误不可安全重试。",
                completion_reason=f"{observation.stage}_failed",
            )
        if self.remaining_steps <= 0:
            return self._decision(
                action="stop",
                reason="已达到本次分析允许的最大工具步数。",
                completion_reason="step_budget_exhausted",
            )
        if observation.stage == "execution" and not observation.schema_inspected and self.remaining_steps >= 2:
            return self._decision(
                action="inspect_schema",
                tool="ContextBundle Schema",
                reason="查询执行失败，先补充相关表字段上下文，再生成修复后的 SQL。",
            )
        return self._decision(
            action="query_metric",
            tool="Text-to-SQL + DuckDB",
            reason=observation.error_summary or "根据确定性校验反馈修复查询。",
        )

    def after_schema_inspection(self) -> AgentDecision:
        if self.remaining_steps <= 0:
            return self._decision(
                action="stop",
                reason="Schema 上下文已获取，但没有剩余工具步数执行修复。",
                completion_reason="step_budget_exhausted",
            )
        return self._decision(
            action="query_metric",
            tool="Text-to-SQL + DuckDB",
            reason="已获取相关字段与 Join 规则，重新生成查询。",
        )

    def _decision(
        self,
        *,
        action: str,
        reason: str,
        tool: str | None = None,
        completion_reason: str | None = None,
        decision_source: str = "policy",
        policy_status: str = "approved",
        parameters: dict[str, object] | None = None,
    ) -> AgentDecision:
        self.decision_count += 1
        return AgentDecision(
            sequence=self.decision_count,
            action=action,  # type: ignore[arg-type]
            reason_summary=reason,
            tool=tool,
            completion_reason=completion_reason,
            decision_source=decision_source,  # type: ignore[arg-type]
            policy_status=policy_status,  # type: ignore[arg-type]
            parameters=parameters or {},
        )

    def decision(
        self,
        *,
        action: str,
        reason: str,
        tool: str | None = None,
        completion_reason: str | None = None,
        decision_source: str = "policy",
        policy_status: str = "approved",
        parameters: dict[str, object] | None = None,
    ) -> AgentDecision:
        return self._decision(
            action=action,
            reason=reason,
            tool=tool,
            completion_reason=completion_reason,
            decision_source=decision_source,
            policy_status=policy_status,
            parameters=parameters,
        )
