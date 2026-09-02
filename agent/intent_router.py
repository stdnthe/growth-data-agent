from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from .intent_validator import IntentValidationError, IntentValidator
from .llm_sql import resolve_llm_config, validate_compatible_base_url
from .metrics_store import MetricsStore
from .models import AnalysisIntent


class IntentRoutingError(RuntimeError):
    """Raised when structured intent parsing cannot produce a validated intent."""


class MissingIntentAPIKeyError(IntentRoutingError):
    """Raised when intent parsing has no configured model credential."""


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_type": {"type": "string", "enum": ["metric_query", "metric_diagnosis"]},
        "metric": {"type": ["string", "null"]},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "time_range": {"type": ["string", "null"]},
        "window_days": {"type": ["integer", "null"]},
        "comparison": {
            "type": ["string", "null"],
            "enum": ["previous_period", "month_over_month", "year_over_year", None],
        },
        "grain": {
            "type": ["string", "null"],
            "enum": ["day", "week", "month", "cohort_month", None],
        },
        "dimensions": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["state", "city", "category", "seller", "product", "payment_type"],
            },
        },
        "filters": {"type": "object", "additionalProperties": {"type": "string"}},
        "business_goal": {"type": ["string", "null"]},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "task_type",
        "metric",
        "metrics",
        "time_range",
        "window_days",
        "comparison",
        "grain",
        "dimensions",
        "filters",
        "business_goal",
        "ambiguities",
        "assumptions",
        "confidence",
        "clarification_question",
    ],
}


def _build_system_prompt(metrics_store: MetricsStore) -> str:
    return (
        "You parse ecommerce analytics questions into one JSON intent object. "
        "Return JSON only and include every schema field.\n"
        "Use metric ids only from the semantic layer below. Never invent a metric id. "
        "Set metric=null and metrics=[] when the requested concept is unsupported or unclear.\n"
        "task_type=metric_diagnosis only when the user asks why a metric changed or asks for drivers; "
        "otherwise use metric_query. Do not choose a workflow or execution action; local policy does that.\n"
        "Relative dates are anchored to the dataset maximum date. Convert stated durations to window_days "
        "using 7 days per week and 30 days per month. If absent, use time_range=null and window_days=null.\n"
        "Allowed dimensions: state, city, category, seller, product, payment_type. "
        "Allowed grains: day, week, month, cohort_month.\n"
        "Use comparison=previous_period for generic change diagnosis, month_over_month for 环比, "
        "year_over_year for 同比, otherwise null.\n"
        "If clarification is needed, explain the ambiguity and provide one concise Chinese question.\n"
        "Example JSON output:\n"
        '{"task_type":"metric_query","metric":"gmv","metrics":["gmv"],'
        '"time_range":"最近30天","window_days":30,"comparison":null,"grain":"day",'
        '"dimensions":[],"filters":{},"business_goal":"retrieve_metric","ambiguities":[],'
        '"assumptions":[],"confidence":0.98,"clarification_question":null}\n'
        "JSON schema (enforced again by local code):\n"
        + json.dumps(INTENT_SCHEMA, ensure_ascii=False)
        + "\n"
        "Metric semantic layer:\n"
        + metrics_store.intent_context()
    )


def _response_format() -> dict[str, str]:
    """JSON Object mode; IntentValidator enforces the full schema locally."""
    return {"type": "json_object"}


def route(
    question: str,
    metrics_store: MetricsStore,
    *,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    completion_create: Callable[..., Any] | None = None,
) -> AnalysisIntent:
    """Parse an intent with structured LLM output, then validate it locally."""
    validator = IntentValidator(metrics_store.metrics or metrics_store.load())
    if not question.strip():
        return validator.validate(
            {
                "task_type": "metric_query",
                "metric": None,
                "metrics": [],
                "time_range": None,
                "window_days": None,
                "comparison": None,
                "grain": None,
                "dimensions": [],
                "filters": {},
                "business_goal": None,
                "ambiguities": ["问题为空"],
                "assumptions": [],
                "confidence": 0.0,
                "clarification_question": "请告诉我你想分析的指标和时间范围。",
            }
        )

    config = resolve_llm_config(
        model=model,
        provider=provider,
        api_key_override=api_key,
        base_url_override=base_url,
    )
    api_key = str(config["api_key"] or "")
    if not api_key:
        raise MissingIntentAPIKeyError(
            f"未配置 {config['api_key_name']}，LLM structured intent 无法运行；"
            "系统不会回退到关键词意图路由。"
        )
    if config["provider"] == "openai_compatible":
        base_url_valid, base_url_error = validate_compatible_base_url(config["base_url"])
        if not base_url_valid:
            raise IntentRoutingError(base_url_error or "OpenAI-compatible Base URL 无效。")
    if not config["model"]:
        raise IntentRoutingError("当前模型服务需要配置模型名称。")

    create = completion_create
    if create is None:
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
        create = client.chat.completions.create

    try:
        response = create(
            model=str(config["model"]),
            temperature=0,
            max_tokens=1200,
            response_format=_response_format(),
            messages=[
                {"role": "system", "content": _build_system_prompt(metrics_store)},
                {"role": "user", "content": question},
            ],
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise IntentRoutingError("Intent model returned empty content.")
        payload = json.loads(content)
        return validator.validate(payload)
    except IntentValidationError as exc:
        raise IntentRoutingError(f"Intent validation failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IntentRoutingError("Intent model returned invalid JSON.") from exc
    except IntentRoutingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IntentRoutingError(f"Intent model call failed: {type(exc).__name__}: {exc}") from exc
