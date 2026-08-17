from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.intent_router import route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "eval" / "intent_cases.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        intent = route(case["question"])
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
        "evaluation_type": "deterministic_intent_routing",
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
