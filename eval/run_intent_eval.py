from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.intent_router import IntentRoutingError, route
from agent.llm_sql import resolve_llm_config
from agent.metrics_store import MetricsStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "eval" / "intent_cases.jsonl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.yml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runtime = resolve_llm_config()
    if not runtime["api_key"]:
        raise SystemExit(
            f"{runtime['api_key_name']} is required for structured intent eval; "
            "there is no keyword-routing fallback."
        )
    metrics_store = MetricsStore(METRICS_PATH)
    metrics_store.load()
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        try:
            intent = route(case["question"], metrics_store)
        except IntentRoutingError as exc:
            results.append(
                {
                    "id": case["id"],
                    "passed": False,
                    "failure_stage": "structured_intent",
                    "failure_detail": str(exc),
                    "checks": {},
                }
            )
            continue
        checks = {
            "metric": intent.metric == case.get("expected_metric"),
            "workflow": intent.workflow == case["expected_workflow"],
            "action": intent.action == case["expected_action"],
            "window_days": intent.window_days == case["expected_window_days"],
        }
        if "expected_dimensions" in case:
            checks["dimensions"] = intent.dimensions == case["expected_dimensions"]
        if "expected_grain" in case:
            checks["grain"] = intent.grain == case["expected_grain"]
        results.append({"id": case["id"], "passed": all(checks.values()), "checks": checks})

    passed = sum(result["passed"] for result in results)
    report = {
        "evaluation_type": "llm_structured_intent_with_deterministic_validation",
        "evaluation_metadata": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": runtime["provider"],
            "model": runtime["model"],
            "intent_contract": "LLM JSON Output + IntentValidator",
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "intent_accuracy_pct": round(passed / len(results) * 100, 2) if results else 0.0,
        },
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
