from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .metrics_store import MetricDefinition, MetricsStore
from .models import AnalysisIntent


TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "orders": (
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_ts",
        "order_approved_ts",
        "order_delivered_customer_ts",
        "order_estimated_delivery_ts",
    ),
    "vw_eligible_orders": (
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_ts",
        "order_delivered_customer_ts",
        "order_estimated_delivery_ts",
    ),
    "vw_fact_items": (
        "order_id",
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
        "order_purchase_ts",
        "order_delivered_customer_ts",
        "order_estimated_delivery_ts",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
        "line_gmv",
    ),
    "vw_delivered_orders": (
        "order_id",
        "customer_id",
        "order_purchase_ts",
        "order_delivered_customer_ts",
        "order_estimated_delivery_ts",
    ),
    "customers": ("customer_id", "customer_unique_id", "customer_city", "customer_state"),
    "order_reviews": ("order_id", "review_score", "review_creation_date", "review_answer_timestamp"),
    "order_payments": ("order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"),
    "products": ("product_id", "product_category_name"),
    "sellers": ("seller_id", "seller_city", "seller_state"),
}

DIMENSION_COLUMNS: dict[str, tuple[str, ...]] = {
    "state": ("customer_state", "seller_state"),
    "city": ("customer_city", "seller_city"),
    "category": ("product_category_name", "category"),
    "seller": ("seller_id",),
    "product": ("product_id",),
    "payment_type": ("payment_type",),
}

METRIC_OUTPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "on_time_delivery_rate": ("on_time_delivery_rate", "on_time_rate"),
    "late_delivery_rate": ("late_delivery_rate", "late_rate"),
    "average_review_score": ("average_review_score", "avg_review_score", "avg_score"),
    "new_customer_order_share": ("new_customer_order_share", "new_customer_order_ratio", "new_customer_ratio"),
    "new_customers": ("new_customers", "new_buyers"),
    "unique_buyers": ("unique_buyers", "buyers"),
    "installment_order_share": ("installment_order_share", "installment_ratio"),
}

GRAIN_COLUMNS: dict[str, tuple[str, ...]] = {
    "day": ("dt", "date", "day"),
    "week": ("week", "week_start", "dt"),
    "month": ("month", "month_start", "dt"),
    "cohort_month": ("cohort_month", "month"),
}

METRIC_TABLE_HINTS: dict[str, tuple[str, ...]] = {
    "cancellation_rate": ("orders",),
    "approval_rate": ("orders",),
    "delivered_orders": ("vw_delivered_orders",),
    "on_time_delivery_rate": ("vw_delivered_orders", "customers"),
    "late_delivery_rate": ("vw_delivered_orders", "customers"),
    "avg_delivery_days": ("vw_delivered_orders",),
    "p90_delivery_days": ("vw_delivered_orders",),
    "average_review_score": ("order_reviews",),
    "low_rating_rate": ("order_reviews",),
    "high_rating_rate": ("order_reviews",),
    "installment_order_share": ("order_payments", "vw_eligible_orders"),
    "avg_payment_value": ("order_payments", "vw_eligible_orders"),
}


@dataclass(frozen=True)
class OutputContract:
    required_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, list[str]]:
        return {label: list(aliases) for label, aliases in self.required_columns.items()}

    def to_prompt(self) -> str:
        if not self.required_columns:
            return "No fixed output columns were inferred."
        return "\n".join(
            f"- {label}: output one of [{', '.join(aliases)}]"
            for label, aliases in self.required_columns.items()
        )


@dataclass(frozen=True)
class ContextBundle:
    metric_ids: tuple[str, ...]
    metric_version: str
    metric_semantics: tuple[str, ...]
    allowed_tables: tuple[str, ...]
    schemas: dict[str, tuple[str, ...]]
    join_rules: tuple[str, ...]
    mandatory_rules: tuple[str, ...]
    output_contract: OutputContract

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_contract"] = self.output_contract.to_dict()
        payload["schemas"] = {table: list(columns) for table, columns in self.schemas.items()}
        return payload

    def schema_summary(self) -> str:
        return "\n".join(
            f"- {table}: {', '.join(columns)}" for table, columns in self.schemas.items()
        )

    def to_prompt(self) -> str:
        sections = [
            "Selected metric semantics:",
            *(f"- {item}" for item in self.metric_semantics),
            "Relevant table schemas:",
            self.schema_summary(),
            "Join and grain rules:",
            *(f"- {rule}" for rule in self.join_rules),
            "Mandatory data rules:",
            *(f"- {rule}" for rule in self.mandatory_rules),
            "Output contract:",
            self.output_contract.to_prompt(),
        ]
        return "\n".join(sections)


class ContextBuilder:
    def __init__(self, metrics_store: MetricsStore) -> None:
        self.metrics_store = metrics_store

    def build(self, intent: AnalysisIntent) -> ContextBundle:
        metric_ids = tuple(intent.metrics or ([intent.metric] if intent.metric else []))
        metrics = [
            metric
            for metric_id in metric_ids
            if (metric := self.metrics_store.get(metric_id)) is not None
        ]
        allowed_tables = self._select_tables(metrics, intent)
        output_contract = self._build_output_contract(metric_ids, intent)
        return ContextBundle(
            metric_ids=metric_ids,
            metric_version=self.metrics_store.version,
            metric_semantics=tuple(self._metric_semantics(metric) for metric in metrics),
            allowed_tables=allowed_tables,
            schemas={table: TABLE_SCHEMAS[table] for table in allowed_tables if table in TABLE_SCHEMAS},
            join_rules=self._join_rules(intent, allowed_tables),
            mandatory_rules=(
                "Anchor relative dates to the maximum date in the query's primary dataset, never system time.",
                "Use customer_unique_id for buyer-level counting and retention analysis.",
                "Aggregate order_payments to order_id before joining to item-level data.",
                "Do not infer exposure, clicks, sessions, advertising cost, CAC or ROAS from Olist transactions.",
            ),
            output_contract=output_contract,
        )

    @staticmethod
    def _metric_semantics(metric: MetricDefinition) -> str:
        return (
            f"{metric.id} ({metric.name_zh}): {metric.definition}; formula={metric.formula}; "
            f"grain={metric.grain}; sql_hint={metric.sql_hint}"
        )

    @staticmethod
    def _select_tables(
        metrics: list[MetricDefinition],
        intent: AnalysisIntent,
    ) -> tuple[str, ...]:
        selected: list[str] = []
        for metric in metrics:
            selected.extend(METRIC_TABLE_HINTS.get(metric.id, ("vw_fact_items",)))
        if "category" in intent.dimensions:
            selected.extend(("vw_fact_items", "products"))
        if "payment_type" in intent.dimensions:
            selected.extend(("order_payments", "vw_eligible_orders"))
        if any(dimension in intent.dimensions for dimension in ("state", "city")) and any(
            table == "vw_delivered_orders" for table in selected
        ):
            selected.append("customers")
        return tuple(dict.fromkeys(selected or ["vw_fact_items"]))

    @staticmethod
    def _join_rules(intent: AnalysisIntent, tables: tuple[str, ...]) -> tuple[str, ...]:
        rules = ["Use order_id for order-level joins; never join item and payment rows without pre-aggregation."]
        if "products" in tables:
            rules.append("Join vw_fact_items to products on product_id for category analysis.")
        if "customers" in tables and "vw_delivered_orders" in tables:
            rules.append("Join vw_delivered_orders to customers on customer_id for buyer geography.")
        if "order_payments" in tables:
            rules.append("Join order_payments to vw_eligible_orders on order_id after selecting the payment grain.")
        if intent.grain:
            rules.append(f"Return a {intent.grain}-grain time column and group by the same grain.")
        return tuple(rules)

    @staticmethod
    def _build_output_contract(metric_ids: tuple[str, ...], intent: AnalysisIntent) -> OutputContract:
        required: dict[str, tuple[str, ...]] = {}
        if intent.grain:
            required[f"time_grain:{intent.grain}"] = GRAIN_COLUMNS.get(intent.grain, (intent.grain,))
        for dimension in intent.dimensions:
            required[f"dimension:{dimension}"] = DIMENSION_COLUMNS.get(dimension, (dimension,))
        for metric_id in metric_ids:
            required[f"metric:{metric_id}"] = METRIC_OUTPUT_ALIASES.get(metric_id, (metric_id,))
        return OutputContract(required_columns=required)
