from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.agent_loop import BoundedLoopController, LoopObservation
from agent.llm_sql import resolve_llm_config
from agent.models import AnalysisIntent
from agent.planner import AgentPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "eval" / "trajectory_cases.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_case(case: dict[str, Any], *, use_llm: bool, model: str | None) -> dict[str, Any]:
    intent = AnalysisIntent(**case["intent"])
    controller = BoundedLoopController(max_steps=int(case.get("max_steps", 3)))
    planner = AgentPlanner(use_llm=use_llm, model=model)
    decision = planner.decide(
        question=str(case["question"]),
        intent=intent,
        observation=None,
        controller=controller,
    )
    actions = [decision.action]
    decision_sources = [decision.decision_source]
    for raw_observation in case.get("observations", []):
        if decision.action in {"finish", "clarify", "unsupported", "stop"}:
            break
        controller.mark_action_executed()
        observation = LoopObservation(**raw_observation)
        decision = planner.decide(
            question=str(case["question"]),
            intent=intent,
            observation=observation,
            controller=controller,
        )
        actions.append(decision.action)
        decision_sources.append(decision.decision_source)
    expected = list(case["expected_actions"])
    fallback_count = sum(source == "fallback_policy" for source in decision_sources)
    return {
        "id": case["id"],
        "question": case["question"],
        "passed": actions == expected and (not use_llm or fallback_count == 0),
        "expected_actions": expected,
        "actual_actions": actions,
        "decision_sources": decision_sources,
        "planner_fallback_count": fallback_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bounded Agent action trajectories.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()

    if args.use_llm and not resolve_llm_config(model=args.model).get("api_key"):
        parser.error("--use-llm requires DEEPSEEK_API_KEY; no policy fallback is scored as model quality")

    results = [
        run_case(case, use_llm=args.use_llm, model=args.model)
        for case in load_cases(args.cases)
    ]
    passed = sum(bool(item["passed"]) for item in results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "llm_planner" if args.use_llm else "deterministic_policy_contract",
        "model": args.model,
        "case_count": len(results),
        "passed": passed,
        "trajectory_success_rate": passed / len(results) if results else 0.0,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
