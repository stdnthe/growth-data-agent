# Evaluation Snapshots

These artifacts are dated, reproducible quality baselines. They are not production-adoption or business-impact claims.

## Local architecture verification — 2026-08-20

| Check | Result | Interpretation |
| --- | ---: | --- |
| Offline unit/integration tests | 54 / 54 passed | Intent, ContextBundle, Planner policy, evidence sufficiency, LangGraph recovery, SQL Guard, both Recipes, validators and evaluators |
| Deterministic policy trajectory Cases | 12 / 12 passed | Tool whitelist, expected action order, schema recovery, step budget, clarification and safe stop |
| Fulfillment Recipe validation | 5 / 5 checks passed | Window, rate range, order count, mix/within additivity and state-contribution additivity on the local Olist database |

Raw artifact:

- [`trajectory-policy-2026-08-20.json`](trajectory-policy-2026-08-20.json)

**Evidence boundary:** this snapshot verifies local code and deterministic policy contracts. It does not measure DeepSeek intent accuracy, Planner tool-selection accuracy, Text-to-SQL result quality, production adoption or business impact. A configured `DEEPSEEK_API_KEY` is required to produce the new real-model intent, trajectory and SQL snapshots.

## Portfolio verification — 2026-08-17

> Historical baseline: the intent result below was produced by the removed keyword router. It does not measure the current DeepSeek structured-intent implementation and must not be presented as current intent accuracy.

| Check | Result | Interpretation |
| --- | ---: | --- |
| Offline unit/integration tests | 21 / 21 passed | Pipeline, routing, validators, SQL Guard, feedback, and evaluator behavior |
| Intent/clarification Golden Cases | 8 / 8 passed | Historical keyword-routing contract; superseded by structured intent + `IntentValidator` |
| LLM SQL generation | 40 / 40 generated | All cases used the configured remote model, not fallback |
| SQL Guard pass | 97.5% | One generated query violated the allowlist |
| DuckDB execution success | 77.5% | Execution failures remain actionable Bad Cases |
| Metric compliance | 95.0% | Correct-table and required-pattern checks |
| Strict complete-result equivalence | 10.0% | Exact output columns and all values must match Golden SQL |
| Strict end-to-end task success | 10.0% (4 / 40) | Unified denominator; no failed stage is excluded |

Raw artifacts:

- [`intent-routing-2026-08-17.json`](intent-routing-2026-08-17.json)
- [`deepseek-v3-2026-08-17.json`](deepseek-v3-2026-08-17.json)

## Validation assessment

**Status: diagnostic baseline; not ready to use as a headline model-quality claim.**

The strict comparator is intentionally demanding: aliases, output shape, time representation, row set, and every value must match the Golden SQL. Many semantic failures are output-contract mismatches such as `day` versus `dt`, string month versus timestamp month, or omitted audit columns. Those cases show that the current Agent does not provide a canonical output schema to SQL generation; they should not be silently counted as successes.

Execution failures are genuine product issues, including invalid CTE references, ambiguous columns, unsupported series generation, and incorrect view-schema assumptions. The next model-quality iteration should improve the structured analysis/output contract and schema context before changing the comparator.

For a resume, it is safe to claim that the project implements a 40-case strict evaluation pipeline and Bad Case classification. Do not quote 10% as a positive quality result or replace it with the older conditional metric.

## Reproduce

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python eval/run_trajectory_eval.py --output eval/results/trajectory-policy-YYYY-MM-DD.json
PYTHONPATH=. python eval/run_intent_eval.py --output eval/results/intent-routing-YYYY-MM-DD.json
PYTHONPATH=. python eval/run_trajectory_eval.py --use-llm --output eval/results/trajectory-model-YYYY-MM-DD.json
PYTHONPATH=. python eval/run_eval.py --use-llm --output eval/results/model-YYYY-MM-DD.json
```

The remote-model run uses the configured provider and may incur API cost. Each new snapshot should preserve its date, model, denominator, and raw failure list.
The current `run_intent_eval.py` requires `DEEPSEEK_API_KEY`; it has no keyword or no-Key fallback.
