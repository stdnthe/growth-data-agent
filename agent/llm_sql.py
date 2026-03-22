from __future__ import annotations

import os
import re
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

from .sql_guard import ALLOWED_TABLES

load_dotenv()


DATA_MAX_DATE_EXPR = "(SELECT CAST(MAX(order_purchase_ts) AS DATE) FROM orders)"


def _read_key_from_keychain(service_name: str, account_name: str | None = None) -> str:
    if sys.platform != "darwin":
        return ""
    if not service_name.strip():
        return ""

    cmd = ["security", "find-generic-password", "-s", service_name, "-w"]
    if account_name and account_name.strip():
        cmd.extend(["-a", account_name.strip()])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def resolve_llm_config(model: str | None = None) -> dict[str, str | None]:
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()

    if provider == "openai":
        resolved_provider = "openai"
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        api_key_name = "OPENAI_API_KEY"
        keychain_service = os.getenv("OPENAI_KEYCHAIN_SERVICE", "growth-analysis-agent/OPENAI_API_KEY").strip()
        keychain_account = os.getenv("OPENAI_KEYCHAIN_ACCOUNT", os.getenv("USER", "")).strip()
    else:
        resolved_provider = "deepseek"
        resolved_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip() or None
        api_key_name = "DEEPSEEK_API_KEY"
        keychain_service = os.getenv("DEEPSEEK_KEYCHAIN_SERVICE", "growth-analysis-agent/DEEPSEEK_API_KEY").strip()
        keychain_account = os.getenv("DEEPSEEK_KEYCHAIN_ACCOUNT", os.getenv("USER", "")).strip()

    api_key_source = "env"
    if not api_key:
        api_key = _read_key_from_keychain(keychain_service, keychain_account)
        api_key_source = "keychain" if api_key else "missing"

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "api_key": api_key or None,
        "base_url": base_url,
        "api_key_name": api_key_name,
        "api_key_source": api_key_source,
        "keychain_service": keychain_service,
        "keychain_account": keychain_account or None,
    }


def _clean_llm_output(text: str) -> str:
    sql = text.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)
    return sql.strip()


def _build_system_prompt(metrics_context: str) -> str:
    allowed = ", ".join(sorted(ALLOWED_TABLES))
    return (
        "You are a senior data analyst generating DuckDB SQL for Olist growth analytics.\n"
        "Rules (must follow):\n"
        "1) Output pure SQL only. No markdown.\n"
        "2) Use only allowed tables/views: "
        + allowed
        + "\n"
        + "3) Query must be read-only SELECT/CTE only.\n"
        + "4) For GMV use vw_fact_items and SUM(price + freight_value) or SUM(line_gmv).\n"
        + "5) For delivered/fulfillment metrics prioritize vw_delivered_orders.\n"
        + "6) For relative dates (today/last 30 days/last month), use dataset max date as anchor, not system date.\n"
        + "   Use max(order_purchase_ts) from the same main table/view used by the query.\n"
        + "7) DuckDB date arithmetic: use interval syntax only, e.g. date - INTERVAL 30 DAY, date - INTERVAL 1 MONTH.\n"
        + "   Never use DATE_SUB(), DATE_ADD(), or DATEADD() — they are not supported in DuckDB.\n"
        + "8) Never use generate_series() — it is not supported in DuckDB. Use GROUP BY on existing data instead.\n"
        + "9) Never add SQL comments (-- or /* */). Output clean SQL only.\n"
        + "10) For date formatting use strftime('%Y-%m', col), never DATE_FORMAT().\n"
        + "11) order_reviews table has no order_purchase_ts column. To filter reviews by date use review_creation_date.\n"
        + "    To anchor dates for review queries use: (SELECT CAST(MAX(review_creation_date) AS DATE) FROM order_reviews).\n"
        + "12) For period-over-period comparisons (current vs previous), always use UNION ALL to produce one row per period.\n"
        + "    Do not output a wide table with two columns side by side.\n"
        + "13) Only SELECT the metrics explicitly requested by the user. Do not add extra columns.\n"
        + "14) Keep metric definitions consistent with the metric dictionary below.\n"
        + metrics_context
    )


def _pick_anchor_expr(sql: str) -> str:
    lowered = sql.lower()
    if "vw_fact_items" in lowered:
        return "(SELECT CAST(MAX(order_purchase_ts) AS DATE) FROM vw_fact_items)"
    if "vw_delivered_orders" in lowered:
        return "(SELECT CAST(MAX(order_purchase_ts) AS DATE) FROM vw_delivered_orders)"
    if "vw_eligible_orders" in lowered:
        return "(SELECT CAST(MAX(order_purchase_ts) AS DATE) FROM vw_eligible_orders)"
    return DATA_MAX_DATE_EXPR


def _anchor_relative_date(sql: str) -> str:
    # Normalize relative date semantics: treat \"today\" as max date in dataset.
    anchor = _pick_anchor_expr(sql)
    return re.sub(r"(?is)\bcurrent_date\b", anchor, sql)


def _fallback_sql(question: str) -> str:
    q = re.sub(r"\s+", "", question.lower())

    if "近30天" in question and "gmv" in q and "走势" in question:
        return """
SELECT
  date_trunc('day', order_purchase_ts) AS dt,
  SUM(line_gmv) AS gmv,
  COUNT(DISTINCT order_id) AS paid_orders
FROM vw_fact_items
WHERE order_purchase_ts >= current_date - INTERVAL 30 DAY
GROUP BY 1
ORDER BY 1
""".strip()

    if "上个月" in question and ("订单数" in question or "客单价" in question or "gmv" in q):
        return """
WITH last_month AS (
  SELECT
    date_trunc('month', current_date) - INTERVAL 1 MONTH AS start_dt,
    date_trunc('month', current_date) AS end_dt
),
base AS (
  SELECT f.*
  FROM vw_fact_items f, last_month m
  WHERE f.order_purchase_ts >= m.start_dt
    AND f.order_purchase_ts < m.end_dt
)
SELECT
  COUNT(DISTINCT order_id) AS paid_orders,
  SUM(line_gmv) AS gmv,
  CASE
    WHEN COUNT(DISTINCT order_id) = 0 THEN NULL
    ELSE SUM(line_gmv) / COUNT(DISTINCT order_id)
  END AS aov
FROM base
""".strip()

    if ("按州" in question or "state" in q) and "top" in q:
        return """
SELECT
  customer_state,
  SUM(line_gmv) AS gmv,
  COUNT(DISTINCT order_id) AS paid_orders
FROM vw_fact_items
WHERE order_purchase_ts >= current_date - INTERVAL 90 DAY
GROUP BY 1
ORDER BY gmv DESC
LIMIT 10
""".strip()

    if "准时送达率" in question and ("按月" in question or "趋势" in question):
        return """
SELECT
  date_trunc('month', order_purchase_ts) AS month,
  AVG(CASE WHEN order_delivered_customer_ts <= order_estimated_delivery_ts THEN 1 ELSE 0 END) AS on_time_delivery_rate,
  COUNT(*) AS delivered_orders
FROM vw_delivered_orders
WHERE order_purchase_ts >= date_trunc('month', current_date) - INTERVAL 5 MONTH
GROUP BY 1
ORDER BY 1
""".strip()

    if "延迟送达" in question and "平均评分" in question:
        return """
WITH base AS (
  SELECT
    d.order_id,
    CASE
      WHEN d.order_delivered_customer_ts <= d.order_estimated_delivery_ts THEN 'on_time'
      ELSE 'delayed'
    END AS delivery_type,
    r.review_score
  FROM vw_delivered_orders d
  LEFT JOIN order_reviews r ON d.order_id = r.order_id
  WHERE d.order_purchase_ts >= date_trunc('month', current_date) - INTERVAL 5 MONTH
    AND r.review_score IS NOT NULL
)
SELECT
  delivery_type,
  AVG(review_score) AS avg_review_score,
  COUNT(*) AS review_cnt
FROM base
GROUP BY 1
ORDER BY 1
""".strip()

    if "新客占比" in question and ("按月" in question or "趋势" in question):
        return """
WITH first_order AS (
  SELECT
    customer_unique_id,
    MIN(order_purchase_ts) AS first_purchase_ts
  FROM vw_fact_items
  GROUP BY 1
),
base AS (
  SELECT
    date_trunc('month', f.order_purchase_ts) AS month,
    f.order_id,
    f.customer_unique_id,
    fo.first_purchase_ts
  FROM vw_fact_items f
  LEFT JOIN first_order fo ON f.customer_unique_id = fo.customer_unique_id
  WHERE f.order_purchase_ts >= date_trunc('year', current_date)
),
monthly AS (
  SELECT
    month,
    COUNT(DISTINCT order_id) AS paid_orders,
    COUNT(DISTINCT CASE WHEN date_trunc('month', first_purchase_ts) = month THEN order_id END) AS new_customer_orders
  FROM base
  GROUP BY 1
)
SELECT
  month,
  paid_orders,
  new_customer_orders,
  CASE WHEN paid_orders = 0 THEN NULL ELSE CAST(new_customer_orders AS DOUBLE) / paid_orders END AS new_customer_order_ratio
FROM monthly
ORDER BY 1
""".strip()

    return """
SELECT
  date_trunc('day', order_purchase_ts) AS dt,
  COUNT(DISTINCT order_id) AS paid_orders
FROM vw_eligible_orders
WHERE order_purchase_ts >= current_date - INTERVAL 30 DAY
GROUP BY 1
ORDER BY 1
""".strip()


def generate_sql(
    question: str,
    metrics_context: str,
    use_llm: bool = True,
    model: str | None = None,
) -> tuple[str, str]:
    config = resolve_llm_config(model=model)
    provider = str(config["provider"])
    resolved_model = str(config["model"])
    api_key = str(config["api_key"] or "")
    base_url = config["base_url"]

    if not question.strip():
        raise ValueError("Question is empty.")

    if use_llm and api_key:
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": _build_system_prompt(metrics_context)},
                    {"role": "user", "content": question},
                ],
            )
            text = response.choices[0].message.content or ""
            sql = _clean_llm_output(text)
            if not sql:
                raise ValueError("LLM returned empty SQL")
            return _anchor_relative_date(sql), "llm"
        except Exception as e:  # noqa: BLE001
            return _anchor_relative_date(_fallback_sql(question)), f"fallback_llm_error: {type(e).__name__}"

    reason = f"fallback_no_key_{provider}" if use_llm and not api_key else "fallback_disabled"
    return _anchor_relative_date(_fallback_sql(question)), reason
