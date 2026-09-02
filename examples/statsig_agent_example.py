from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentRequest:
    user_id: str
    question: str
    email: str | None = None
    plan: str = "free"
    country: str = "US"


@dataclass
class AgentResult:
    answer: str
    model: str
    prompt_name: str
    prompt_version: str
    latency_ms: int
    used_new_agent: bool


def _require_optional_deps() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from openai import OpenAI
        from statsig_ai import StatsigAI, StatsigCreateConfig
        from statsig_python_core import StatsigOptions, StatsigUser
    except ImportError as exc:  # pragma: no cover - example-only dependency guard
        raise SystemExit(
            "Missing optional dependencies. Install with:\n"
            "  pip install statsig-python-core statsig-ai openai\n"
        ) from exc

    return OpenAI, StatsigAI, StatsigCreateConfig, StatsigOptions, StatsigUser


def _build_statsig_user(statsig_user_cls: Any, request: AgentRequest) -> Any:
    return statsig_user_cls(
        user_id=request.user_id,
        email=request.email,
        country=request.country,
        custom={
            "plan": request.plan,
            "surface": "growth-analysis-agent",
        },
    )


def _maybe_wait(result: Any) -> None:
    wait_fn = getattr(result, "wait", None)
    if callable(wait_fn):
        wait_fn()


def run_agent(request: AgentRequest) -> AgentResult:
    OpenAI, StatsigAI, StatsigCreateConfig, StatsigOptions, StatsigUser = _require_optional_deps()

    statsig_key = os.environ["STATSIG_SERVER_SECRET_KEY"]
    deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]

    statsig_options = StatsigOptions()
    statsig_options.environment = os.getenv("STATSIG_ENV", "development")

    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(
            server_secret_key=statsig_key,
            statsig_options=statsig_options,
        )
    )
    statsig_ai.initialize()

    client = OpenAI(
        api_key=deepseek_api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    user = _build_statsig_user(StatsigUser, request)

    started = time.perf_counter()
    prompt_name = "growth_sql_analyst"
    prompt = statsig_ai.get_prompt(user, prompt_name)
    live_prompt = prompt.get_live()

    statsig = statsig_ai.get_statsig()
    new_agent_enabled = statsig.check_gate(user, "agent_v2_enabled")

    routing = statsig.get_dynamic_config(user, "agent_routing")
    fallback_model = routing.get_string("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    tool_enabled = routing.get_bool("tool_enabled", True)
    max_tool_steps = routing.get_integer("max_tool_steps", 4)

    fallback_system_prompt = (
        "You are a careful data copilot. Answer clearly, stay grounded in evidence, "
        "and explain assumptions when data is incomplete."
    )
    get_system_prompt = getattr(live_prompt, "get_system_prompt", None)
    if callable(get_system_prompt):
        system_prompt = get_system_prompt(fallback=fallback_system_prompt)
    else:
        system_prompt = fallback_system_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question},
    ]

    response = client.chat.completions.create(
        model=live_prompt.get_model(fallback=fallback_model),
        temperature=live_prompt.get_temperature(),
        max_tokens=live_prompt.get_max_tokens(),
        messages=messages,
    )
    answer = response.choices[0].message.content or ""
    latency_ms = int((time.perf_counter() - started) * 1000)

    statsig.log_event(
        user=user,
        event_name="agent_run_completed",
        value=float(latency_ms),
        metadata={
            "prompt_name": prompt_name,
            "prompt_version": str(getattr(live_prompt, "version", "live")),
            "model": live_prompt.get_model(fallback=fallback_model),
            "used_new_agent": str(new_agent_enabled).lower(),
            "tool_enabled": str(tool_enabled).lower(),
            "max_tool_steps": str(max_tool_steps),
        },
    )

    statsig.log_event(
        user=user,
        event_name="agent_output_accepted",
        value="1",
        metadata={
            "acceptance_source": "demo",
        },
    )

    # Example online eval: very simple production score you can replace with
    # a real grader or human feedback pipeline.
    score = 1.0 if answer.strip() else 0.0
    statsig_ai.log_eval_grade(
        user,
        live_prompt,
        score,
        "non_empty_answer",
        {
            "question": request.question[:120],
            "latency_ms": latency_ms,
        },
    )

    _maybe_wait(statsig_ai.flush())
    _maybe_wait(statsig_ai.shutdown())

    return AgentResult(
        answer=answer,
        model=live_prompt.get_model(fallback=fallback_model),
        prompt_name=prompt_name,
        prompt_version=str(getattr(live_prompt, "version", "live")),
        latency_ms=latency_ms,
        used_new_agent=new_agent_enabled,
    )


if __name__ == "__main__":
    demo_request = AgentRequest(
        user_id="demo-user-123",
        email="demo@example.com",
        plan="pro",
        question="请总结最近30天 GMV 变化，并给出两个可能原因。",
    )
    result = run_agent(demo_request)
    print("model:", result.model)
    print("prompt:", result.prompt_name, result.prompt_version)
    print("latency_ms:", result.latency_ms)
    print("used_new_agent:", result.used_new_agent)
    print("answer:", result.answer)
