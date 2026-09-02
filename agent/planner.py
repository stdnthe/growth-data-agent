from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from .agent_loop import BoundedLoopController, LoopObservation
from .capabilities import CapabilityRegistry
from .llm_sql import resolve_llm_config
from .models import AgentAction, AgentDecision, AnalysisIntent


CompletionCreate = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    action: AgentAction
    name: str
    description: str


class ToolRegistry:
    """Expose only tools valid for the current analytical state."""

    def __init__(self, capabilities: CapabilityRegistry | None = None) -> None:
        self.capabilities = capabilities or CapabilityRegistry()
        self._tools = {
            "query_metric": ToolSpec(
                "query_metric",
                "Metric Query",
                "Generate guarded DuckDB SQL and validate the requested result contract.",
            ),
            "inspect_schema": ToolSpec(
                "inspect_schema",
                "Schema Context",
                "Retrieve scoped fields and join rules after a repairable execution failure.",
            ),
            "run_gmv_recipe": ToolSpec(
                "run_gmv_recipe",
                "GMV Attribution Recipe",
                "Run the deterministic GMV = Orders × AOV decomposition and contribution drill-down.",
            ),
            "run_fulfillment_recipe": ToolSpec(
                "run_fulfillment_recipe",
                "Fulfillment Diagnosis Recipe",
                "Decompose on-time delivery-rate movement into mix and within-state effects.",
            ),
            "clarify": ToolSpec(
                "clarify",
                "Ask Clarification",
                "Ask for a missing metric, scope or capability confirmation.",
            ),
            "finish": ToolSpec(
                "finish",
                "Finish",
                "Finish only after deterministic validation confirms sufficient evidence.",
            ),
            "stop": ToolSpec(
                "stop",
                "Safe Stop",
                "Stop when the task cannot be completed safely within policy or budget.",
            ),
        }

    def allowed_actions(
        self,
        intent: AnalysisIntent,
        observation: LoopObservation | None,
    ) -> tuple[AgentAction, ...]:
        if intent.action in {"clarify", "confirm"}:
            return ("clarify",)
        if observation is not None:
            if observation.stage == "schema":
                return ("query_metric", "stop")
            if observation.passed:
                if not observation.evidence_sufficient:
                    preferred = self.capabilities.preferred_action(intent)
                    if preferred not in {None, "query_metric"}:
                        return (preferred, "stop")
                return ("finish",)
            if not observation.retryable:
                return ("stop",)
            if observation.stage == "execution" and not observation.schema_inspected:
                return ("inspect_schema", "query_metric", "stop")
            return ("query_metric", "stop")
        actions = self.capabilities.actions_for(intent)
        return (*actions, "stop") if actions else ("stop",)

    def describe(self, actions: tuple[AgentAction, ...]) -> str:
        return "\n".join(f"- {action}: {self._tools[action].description}" for action in actions)

    def tool_name(self, action: AgentAction) -> str | None:
        spec = self._tools.get(action)
        return spec.name if spec else None

    def describe_capabilities(self, intent: AnalysisIntent) -> list[dict[str, object]]:
        return self.capabilities.describe(intent)


class AgentPolicyGate:
    """Approve a model proposal only when it satisfies deterministic runtime policy."""

    def validate(
        self,
        action: str,
        *,
        allowed_actions: tuple[AgentAction, ...],
        remaining_steps: int,
    ) -> tuple[AgentAction | None, str | None]:
        if action not in allowed_actions:
            return None, f"action {action!r} is outside the current tool policy"
        if action not in {"finish", "clarify", "stop"} and remaining_steps <= 0:
            return None, "tool-step budget is exhausted"
        return action, None  # type: ignore[return-value]


class AgentController:
    """Single LLM controller with capability selection and deterministic policy gates."""

    def __init__(
        self,
        *,
        use_llm: bool,
        model: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        completion_create: CompletionCreate | None = None,
        registry: ToolRegistry | None = None,
        policy_gate: AgentPolicyGate | None = None,
    ) -> None:
        self.use_llm = use_llm
        self.runtime = resolve_llm_config(
            model=model,
            provider=provider,
            api_key_override=api_key,
            base_url_override=base_url,
        )
        self.completion_create = completion_create
        self.registry = registry or ToolRegistry()
        self.policy_gate = policy_gate or AgentPolicyGate()

    def decide(
        self,
        *,
        question: str,
        intent: AnalysisIntent,
        observation: LoopObservation | None,
        controller: BoundedLoopController,
    ) -> AgentDecision:
        allowed = self.registry.allowed_actions(intent, observation)
        if not self.use_llm or len(allowed) == 1:
            fallback = self._fallback(intent, observation, controller)
            fallback.decision_source = "policy"
            return fallback

        try:
            payload = self._call_model(
                question=question,
                intent=intent,
                observation=observation,
                allowed_actions=allowed,
                remaining_steps=controller.remaining_steps,
            )
            raw_action = str(payload.get("action", ""))
            action, rejection = self.policy_gate.validate(
                raw_action,
                allowed_actions=allowed,
                remaining_steps=controller.remaining_steps,
            )
            if action is None:
                fallback = self._fallback(intent, observation, controller)
                fallback.decision_source = "fallback_policy"
                fallback.policy_status = "corrected"
                fallback.reason_summary = (
                    f"模型动作被 Policy Gate 拒绝（{rejection}）；{fallback.reason_summary}"
                )
                return fallback
            return controller.decision(
                action=action,
                tool=self.registry.tool_name(action),
                reason=str(payload.get("reason_summary") or "根据当前意图与执行观察选择下一步动作。"),
                completion_reason=(
                    "validated"
                    if action == "finish"
                    else "needs_clarification"
                    if action == "clarify"
                    else None
                ),
                decision_source="llm",
                policy_status="approved",
                parameters=(
                    payload.get("parameters")
                    if isinstance(payload.get("parameters"), dict)
                    else {}
                ),
            )
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback(intent, observation, controller)
            fallback.decision_source = "fallback_policy"
            fallback.policy_status = "corrected"
            fallback.reason_summary = f"Controller 不可用（{type(exc).__name__}）；{fallback.reason_summary}"
            return fallback

    def _fallback(
        self,
        intent: AnalysisIntent,
        observation: LoopObservation | None,
        controller: BoundedLoopController,
    ) -> AgentDecision:
        if observation is None:
            return controller.initial(intent)
        if observation.stage == "schema":
            return controller.after_schema_inspection()
        if observation.passed and not observation.evidence_sufficient:
            preferred = self.registry.capabilities.preferred_action(intent)
            if preferred == "run_gmv_recipe":
                return controller.decision(
                    action="run_gmv_recipe",
                    tool="GMV Attribution Recipe",
                    reason="趋势证据已通过校验，但不足以回答驱动因素，继续执行 GMV 归因 Recipe。",
                )
            if preferred == "run_fulfillment_recipe":
                return controller.decision(
                    action="run_fulfillment_recipe",
                    tool="Fulfillment Diagnosis Recipe",
                    reason="趋势证据已通过校验，但不足以回答履约驱动因素，继续执行履约诊断 Recipe。",
                )
        return controller.after_observation(observation)

    def _call_model(
        self,
        *,
        question: str,
        intent: AnalysisIntent,
        observation: LoopObservation | None,
        allowed_actions: tuple[AgentAction, ...],
        remaining_steps: int,
    ) -> dict[str, Any]:
        api_key = str(self.runtime.get("api_key") or "")
        if not api_key and self.completion_create is None:
            raise RuntimeError("missing planner API key")
        create = self.completion_create
        if create is None:
            client = OpenAI(
                api_key=api_key,
                base_url=str(self.runtime.get("base_url") or "") or None,
            )
            create = client.chat.completions.create

        observation_payload = None
        if observation is not None:
            observation_payload = {
                "stage": observation.stage,
                "passed": observation.passed,
                "evidence_sufficient": observation.evidence_sufficient,
                "retryable": observation.retryable,
                "error_summary": observation.error_summary,
                "schema_inspected": observation.schema_inspected,
            }
        response = create(
            model=str(self.runtime.get("model") or ""),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the single bounded controller for a trusted analytics agent. "
                        "Choose exactly one allowed action. Never invent a tool. Return JSON with "
                        "action, reason_summary, parameters. Do not reveal hidden chain of thought; "
                        "reason_summary must be a short operational explanation.\n\nAllowed tools:\n"
                        + self.registry.describe(allowed_actions)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "intent": intent.to_dict(),
                            "active_capabilities": self.registry.describe_capabilities(intent),
                            "observation": observation_payload,
                            "remaining_tool_steps": remaining_steps,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("planner response must be a JSON object")
        return payload


# Backward-compatible import for existing callers and saved examples.
AgentPlanner = AgentController
