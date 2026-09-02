# Portfolio Demo Guide

This walkthrough is designed to prove product judgment, analytical methodology and Agent quality controls in about two minutes.

## Before the interview

1. Start the app with `streamlit run app.py`.
2. Use the same model and configuration recorded in the latest dated evaluation snapshot.
3. Confirm the three golden questions below and one error-recovery trajectory before sharing the screen.
4. Keep the “分析轨迹” and evidence SQL tabs available for follow-up questions.

## Two-minute walkthrough

### 1. Positioning — 10 seconds

> Growth Copilot is a trusted ecommerce analytics Agent. The model can choose a bounded next action, while business methods, data access and completion criteria remain auditable and testable.

Point out four design choices: metric semantics, Tool Registry + Policy Gate, deterministic Recipes and layered Evals.

### 2. Metric query — 25 seconds

Ask: `近30天GMV走势（按天）`

Show:

- structured intent and ContextBundle;
- the Planner-selected `query_metric` action and its decision source;
- generated SQL, SQL Guard and output-contract validation;
- result, chart and `AnalysisRun` metadata.

Key message: executable SQL is not enough; the output must also match the requested metric, grain and schema contract.

### 3. GMV attribution Recipe — 30 seconds

Ask: `为什么最近GMV下降？`

Show:

- Planner selection of `run_gmv_recipe` from the allowed tools;
- exact `GMV = Orders × AOV` decomposition;
- state, category and seller contribution drill-down;
- additivity and non-overlapping-window validation.

Key message: a mature analytical method is packaged as a reusable, testable tool rather than recreated through free-form prompting.

### 4. Fulfillment diagnosis Recipe — 35 seconds

Ask: `为什么最近准时送达率下降？`

Show:

- Planner selection of `run_fulfillment_recipe`;
- equal-window comparison of on-time delivery rate and delivery duration;
- exact split between customer-state mix effect and within-state performance effect;
- state contributions adding back to the overall rate movement;
- the caveat that contribution analysis does not establish causality.

Key message: the architecture is extensible because a second business methodology uses the same Tool → Validate → Observe contract.

### 5. Human-in-the-loop and safe stop — 15 seconds

Ask: `最近销售表现怎么样？`

Show that missing metric scope is clarified before the Agent spends a tool step. If asked about unsupported data such as CAC or ROAS, show that the product states the Olist data boundary instead of inventing an answer.

### 6. Close — 5 seconds

> The Agent is autonomous inside a measurable boundary: the LLM proposes an action, Policy Gate controls it, deterministic validators return observations, and failures become trajectory or result Golden Cases.

## Optional failure-recovery deep dive

Use the tested trajectory:

```text
query_metric
→ DuckDB reports a missing column
→ inspect_schema
→ query_metric with repaired context
→ deterministic validation
→ finish
```

Explain that only structured action summaries are shown; hidden chain-of-thought is neither required nor exposed.

## Claims that are safe to make

- Implemented a runnable Olist ecommerce Agent with structured intent, ContextBundle, LLM-proposed tool choice, deterministic Policy Gate and LangGraph bounded execution.
- Productized two checkable analytical Recipes: GMV decomposition and on-time-delivery mix/within decomposition.
- Built 28 metric definitions, 40 SQL/result Cases, 8 intent Cases and 12 trajectory Cases.
- Records actions, decision source, policy status, SQL, validations, versions, latency, retries and failure stage in a unified `AnalysisRun`.
- Verified the current offline code and policy contracts with 54 unit/integration tests and a dated 12-case policy-trajectory snapshot.

Do not claim current production adoption, business impact, causal effects or real-model Planner accuracy until a dated remote-model run exists. The 12/12 trajectory snapshot verifies deterministic policy contracts, not LLM planning quality.
