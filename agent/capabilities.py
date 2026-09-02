from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import AgentAction, AnalysisIntent


CapabilityId = Literal[
    "metric_query",
    "gmv_diagnosis",
    "fulfillment_diagnosis",
]


@dataclass(frozen=True)
class CapabilitySpec:
    """A versionable contract for one user-facing analytical capability."""

    id: CapabilityId
    name: str
    purpose: str
    task_types: tuple[str, ...]
    workflows: tuple[str, ...]
    metrics: tuple[str, ...]
    actions: tuple[AgentAction, ...]
    preferred_action: AgentAction
    required_context: tuple[str, ...]
    output_artifact: str
    validators: tuple[str, ...]

    def supports(self, intent: AnalysisIntent) -> bool:
        return (
            intent.task_type in self.task_types
            and intent.workflow in self.workflows
            and (not self.metrics or intent.metric in self.metrics)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "metrics": list(self.metrics),
            "actions": list(self.actions),
            "required_context": list(self.required_context),
            "output_artifact": self.output_artifact,
            "validators": list(self.validators),
        }


class CapabilityRegistry:
    """Select a small, non-overlapping capability set for the current intent."""

    def __init__(self, specs: tuple[CapabilitySpec, ...] | None = None) -> None:
        self._specs = self._default_specs() if specs is None else specs
        ids = [spec.id for spec in self._specs]
        if len(ids) != len(set(ids)):
            raise ValueError("capability ids must be unique")

    @staticmethod
    def _default_specs() -> tuple[CapabilitySpec, ...]:
        return (
            CapabilitySpec(
                id="metric_query",
                name="Metric Query",
                purpose="Answer a scoped metric question with guarded SQL and a validated result.",
                task_types=("metric_query",),
                workflows=("retrieval",),
                metrics=(),
                actions=("query_metric",),
                preferred_action="query_metric",
                required_context=("metric_semantics", "schema", "join_rules", "output_contract"),
                output_artifact="ValidatedQueryResult",
                validators=("SQLGuard", "validate_retrieval_result"),
            ),
            CapabilitySpec(
                id="gmv_diagnosis",
                name="GMV Diagnosis",
                purpose="Explain GMV movement with an auditable Orders x AOV decomposition.",
                task_types=("metric_diagnosis",),
                workflows=("gmv_attribution",),
                metrics=("gmv",),
                actions=("run_gmv_recipe", "query_metric"),
                preferred_action="run_gmv_recipe",
                required_context=("gmv_definition", "comparison_window", "supported_dimensions"),
                output_artifact="GmvAttributionReport",
                validators=("validate_gmv_attribution",),
            ),
            CapabilitySpec(
                id="fulfillment_diagnosis",
                name="Fulfillment Diagnosis",
                purpose="Explain delivery-rate movement with additive mix and within-state effects.",
                task_types=("metric_diagnosis",),
                workflows=("fulfillment_diagnosis",),
                metrics=(
                    "on_time_delivery_rate",
                    "late_delivery_rate",
                    "avg_delivery_days",
                    "p90_delivery_days",
                ),
                actions=("run_fulfillment_recipe", "query_metric"),
                preferred_action="run_fulfillment_recipe",
                required_context=("delivery_definition", "comparison_window", "geography_join_rules"),
                output_artifact="FulfillmentDiagnosisReport",
                validators=("validate_fulfillment_diagnosis",),
            ),
        )

    def select(self, intent: AnalysisIntent) -> tuple[CapabilitySpec, ...]:
        return tuple(spec for spec in self._specs if spec.supports(intent))

    def actions_for(self, intent: AnalysisIntent) -> tuple[AgentAction, ...]:
        actions: list[AgentAction] = []
        for spec in self.select(intent):
            for action in spec.actions:
                if action not in actions:
                    actions.append(action)
        return tuple(actions)

    def preferred_action(self, intent: AnalysisIntent) -> AgentAction | None:
        selected = self.select(intent)
        return selected[0].preferred_action if selected else None

    def describe(self, intent: AnalysisIntent) -> list[dict[str, object]]:
        return [spec.to_dict() for spec in self.select(intent)]
