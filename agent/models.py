from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import pandas as pd


IntentAction = Literal["execute", "clarify", "confirm"]
RunStatus = Literal[
    "created",
    "needs_clarification",
    "running",
    "success",
    "success_with_warnings",
    "failed",
]


@dataclass
class AnalysisIntent:
    task_type: Literal["metric_query", "metric_diagnosis"]
    workflow: Literal["retrieval", "gmv_attribution"]
    metric: str | None = None
    metrics: list[str] = field(default_factory=list)
    time_range: str | None = None
    window_days: int = 30
    comparison: str | None = None
    grain: str | None = None
    dimensions: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    business_goal: str | None = None
    ambiguities: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    action: IntentAction = "execute"
    clarification_question: str | None = None
    matched_keywords: list[str] = field(default_factory=list)

    @property
    def intent(self) -> str:
        """Backward-compatible label used by the original Streamlit app."""
        return "attribution" if self.workflow == "gmv_attribution" else "retrieval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "workflow": self.workflow,
            "metric": self.metric,
            "metrics": self.metrics,
            "time_range": self.time_range,
            "window_days": self.window_days,
            "comparison": self.comparison,
            "grain": self.grain,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "business_goal": self.business_goal,
            "ambiguities": self.ambiguities,
            "assumptions": self.assumptions,
            "confidence": self.confidence,
            "action": self.action,
            "clarification_question": self.clarification_question,
        }


@dataclass
class ValidationResult:
    name: str
    passed: bool
    severity: Literal["error", "warning", "info"]
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AttemptRecord:
    attempt: int
    sql_source: str
    sql: str = ""
    stage: str = "generation"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "sql_source": self.sql_source,
            "sql": self.sql,
            "stage": self.stage,
            "error": self.error,
        }


@dataclass
class AnalysisStep:
    key: str
    label: str
    category: Literal["decision", "context", "tool", "validation", "output"]
    status: Literal["completed", "warning", "failed", "skipped"] = "completed"
    summary: str = ""
    tool: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "status": self.status,
            "summary": self.summary,
            "tool": self.tool,
            "details": self.details,
        }


@dataclass
class AnalysisRun:
    question: str
    intent: AnalysisIntent
    run_id: str = field(default_factory=lambda: str(uuid4()))
    status: RunStatus = "created"
    workflow: str = ""
    metric_context: list[str] = field(default_factory=list)
    generated_sql: str | None = None
    sql_source: str | None = None
    result: pd.DataFrame | None = None
    attribution_report: Any | None = None
    validations: list[ValidationResult] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    insight: str | None = None
    caveats: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_version: str = "growth-sql-v1"
    metric_version: str = ""
    latency_ms: int = 0
    failure_stage: str | None = None
    failure_detail: str | None = None
    trace_enabled: bool = False
    original_question: str | None = None
    clarification_history: list[dict[str, str]] = field(default_factory=list)

    @property
    def validation_passed(self) -> bool:
        return not any(not item.passed and item.severity == "error" for item in self.validations)

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.validations if not item.passed and item.severity == "warning")

    def result_summary(self) -> dict[str, Any]:
        if self.result is None:
            return {"row_count": 0, "columns": []}
        return {
            "row_count": len(self.result),
            "columns": [str(col) for col in self.result.columns],
        }

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "workflow": self.workflow,
            "intent": self.intent.to_dict(),
            "sql_source": self.sql_source,
            "result": self.result_summary(),
            "validations": [item.to_dict() for item in self.validations],
            "steps": [item.to_dict() for item in self.steps],
            "caveats": self.caveats,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "metric_version": self.metric_version,
            "latency_ms": self.latency_ms,
            "failure_stage": self.failure_stage,
            "failure_detail": self.failure_detail,
        }
