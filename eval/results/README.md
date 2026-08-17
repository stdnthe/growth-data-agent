# Evaluation Snapshots

These artifacts are dated, reproducible quality baselines. They are not production-adoption or business-impact claims.

## Portfolio verification — 2026-08-17

| Check | Result | Interpretation |
| --- | ---: | --- |
| Offline unit/integration tests | 21 / 21 passed | Pipeline, routing, validators, SQL Guard, feedback, and evaluator behavior |
| Intent/clarification Golden Cases | 8 / 8 passed | Deterministic routing contract on the checked cases |
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
PYTHONPATH=. python eval/run_intent_eval.py --output eval/results/intent-routing-YYYY-MM-DD.json
PYTHONPATH=. python eval/run_eval.py --use-llm --output eval/results/model-YYYY-MM-DD.json
```

The remote-model run uses the configured provider and may incur API cost. Each new snapshot should preserve its date, model, denominator, and raw failure list.
