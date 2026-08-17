# Portfolio Demo Guide

This guide keeps the portfolio demonstration focused on product judgment rather than feature breadth. The complete walkthrough takes about 90 seconds.

## Before the interview

1. Start the app with `streamlit run app.py`.
2. Keep SQL and charts visible.
3. Use the same model and configuration recorded in the latest evaluation snapshot.
4. Confirm the three questions below still produce the expected route before sharing the screen.

## 90-second walkthrough

### 1. Positioning — 10 seconds

> Growth Copilot is a trusted ecommerce Data Agent. It turns natural-language questions into auditable analysis runs instead of stopping at executable SQL.

Point out the four proof cards: metric semantics, explicit Workflows, Golden Cases, and read-only SQL safety.

### 2. Metric retrieval — 25 seconds

Ask: `近30天GMV走势（按天）`

Show:

- structured intent and time grain;
- the selected metric definition;
- generated SQL and SQL Guard result;
- query result, chart, deterministic validations, and run metadata.

Key message: an executable query is not sufficient; the result must also match the requested metric and time shape.

### 3. Deterministic attribution — 35 seconds

Ask: `为什么最近GMV下降？`

Show:

- routing from a diagnosis request to the GMV Attribution Workflow;
- the exact `GMV = Orders × AOV` decomposition;
- state, category, and seller contribution drill-down;
- additivity and period-window validations before narrative generation.

Key message: deterministic business logic is used where the conclusion can be checked; the model only explains evidence that has passed validation.

### 4. Human-in-the-loop — 15 seconds

Ask: `最近销售表现怎么样？`

Show that the Agent asks which metric should be analyzed instead of silently choosing one.

Optional boundary question: `为什么准时送达率下降？` The product should explain that automatic attribution currently supports GMV and ask for confirmation before taking a narrower path.

### 5. Close — 5 seconds

> The project deliberately favors a narrow, measurable Workflow over an open-ended Agent Loop. Feedback and failures become candidates for the Golden Dataset after human review.

## Claims that are safe to make

- Implemented a runnable Olist ecommerce Data Agent with structured intent, metric semantics, safe SQL execution, deterministic validation, and GMV attribution.
- Built 28 metric definitions, 40 SQL/result Golden Cases, and 8 intent/clarification cases.
- Records Workflow, SQL, validation, versions, latency, retries, and failure stage in a unified `AnalysisRun`.

Do not claim production adoption, business impact, causal effects, or support for advertising and browsing metrics. Model-quality percentages must cite a dated result under `eval/results/` and its exact denominator.
