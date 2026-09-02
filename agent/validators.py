from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .attribution import GmvAttributionReport
from .context_builder import ContextBundle
from .fulfillment import FulfillmentDiagnosisReport
from .models import AnalysisIntent, ValidationResult


_EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "gmv": ("gmv", "line_gmv"),
    "paid_orders": ("paid_orders", "orders", "order_count"),
    "aov": ("aov", "avg_order_value"),
    "on_time_delivery_rate": ("on_time_delivery_rate", "on_time_rate"),
    "late_delivery_rate": ("late_delivery_rate", "late_rate"),
    "average_review_score": ("average_review_score", "avg_review_score", "avg_score"),
    "new_customer_order_share": ("new_customer_order_share", "new_customer_order_ratio"),
    "repeat_purchase_rate": ("repeat_purchase_rate",),
    "new_customers": ("new_customers", "new_buyers"),
    "unique_buyers": ("unique_buyers", "buyers"),
    "gmv_per_buyer": ("gmv_per_buyer",),
    "orders_per_buyer": ("orders_per_buyer",),
    "returning_customers": ("returning_customers",),
    "d7_retention": ("d7_retention", "retained_d7"),
    "d30_retention": ("d30_retention", "retained_d30"),
    "cancellation_rate": ("cancellation_rate",),
    "approval_rate": ("approval_rate",),
    "delivered_orders": ("delivered_orders",),
    "avg_delivery_days": ("avg_delivery_days",),
    "p90_delivery_days": ("p90_delivery_days",),
    "low_rating_rate": ("low_rating_rate",),
    "high_rating_rate": ("high_rating_rate",),
    "active_sku": ("active_sku", "active_skus"),
    "active_sellers": ("active_sellers",),
    "gmv_top10_state_share": ("gmv_top10_state_share", "top10_state_share"),
    "freight_ratio": ("freight_ratio",),
    "installment_order_share": ("installment_order_share", "installment_ratio"),
    "avg_payment_value": ("avg_payment_value",),
}


def _result(name: str, passed: bool, message: str, severity: str = "error", **details: object) -> ValidationResult:
    return ValidationResult(
        name=name,
        passed=passed,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        details=details,
    )


def validate_retrieval_result(
    df: pd.DataFrame,
    intent: AnalysisIntent,
    context_bundle: ContextBundle | None = None,
) -> list[ValidationResult]:
    validations = [
        _result(
            "result_non_empty",
            not df.empty,
            "查询返回了数据。" if not df.empty else "查询结果为空，请检查时间范围或筛选条件。",
        )
    ]
    if df.empty:
        return validations

    lower_columns = {str(col).lower() for col in df.columns}
    requested_metrics = intent.metrics or ([intent.metric] if intent.metric else [])
    expected_by_metric = {
        metric: _EXPECTED_COLUMNS.get(metric, ())
        for metric in requested_metrics
        if _EXPECTED_COLUMNS.get(metric, ())
    }
    if expected_by_metric:
        missing = [
            metric
            for metric, aliases in expected_by_metric.items()
            if not any(any(alias in col for alias in aliases) for col in lower_columns)
        ]
        validations.append(
            _result(
                "requested_metrics_present",
                not missing,
                "结果包含全部请求指标。" if not missing else f"结果缺少请求指标：{', '.join(missing)}。",
                expected={metric: list(aliases) for metric, aliases in expected_by_metric.items()},
                actual=sorted(lower_columns),
            )
        )

    numeric = df.select_dtypes(include="number")
    finite = True
    if not numeric.empty:
        finite = all(
            math.isfinite(float(value))
            for value in numeric.to_numpy().flatten()
            if not pd.isna(value)
        )
    validations.append(
        _result(
            "numeric_values_finite",
            finite,
            "数值结果均为有限值。" if finite else "结果包含无穷大或非法数值。",
        )
    )

    ratio_columns = [
        col for col in numeric.columns if any(token in str(col).lower() for token in ("rate", "ratio", "share"))
    ]
    ratios_valid = all(
        numeric[col].dropna().between(0, 1).all()
        for col in ratio_columns
    )
    if ratio_columns:
        validations.append(
            _result(
                "ratio_range",
                ratios_valid,
                "比例指标位于 0 到 1 之间。" if ratios_valid else "比例指标超出 0 到 1 的合理范围。",
                columns=[str(col) for col in ratio_columns],
            )
        )

    if intent.grain:
        time_tokens = ("dt", "date", "day", "week", "month", "year")
        has_time_column = any(any(token in col for token in time_tokens) for col in lower_columns)
        validations.append(
            _result(
                "time_grain_present",
                has_time_column,
                "结果包含时间粒度字段。" if has_time_column else "用户要求趋势粒度，但结果缺少时间字段。",
                grain=intent.grain,
            )
        )

    if context_bundle and context_bundle.output_contract.required_columns:
        actual_columns = {str(col).lower() for col in df.columns}
        missing_contract_fields = [
            label
            for label, aliases in context_bundle.output_contract.required_columns.items()
            if not any(alias.lower() in actual_columns for alias in aliases)
        ]
        validations.append(
            _result(
                "output_contract",
                not missing_contract_fields,
                "结果满足 ContextBundle 输出契约。"
                if not missing_contract_fields
                else f"结果缺少输出契约字段：{', '.join(missing_contract_fields)}。",
                expected=context_bundle.output_contract.to_dict(),
                actual=sorted(actual_columns),
            )
        )

    return validations


def validate_gmv_attribution(report: GmvAttributionReport) -> list[ValidationResult]:
    tolerance = max(0.01, abs(report.gmv_delta) * 1e-8)
    decomposition_diff = abs((report.order_effect + report.aov_effect) - report.gmv_delta)
    windows_valid = bool(
        report.current_start
        and report.current_end
        and report.previous_start
        and report.previous_end
        and report.current_start <= report.current_end
        and report.previous_start <= report.previous_end
        and report.previous_end < report.current_start
    )
    values_non_negative = all(
        value >= 0
        for value in (
            report.gmv_current,
            report.gmv_previous,
            report.orders_current,
            report.orders_previous,
        )
    )
    drop_directions_valid = report.all_drops.empty or bool((report.all_drops["contribution"] < 0).all())
    gain_directions_valid = report.all_gains.empty or bool((report.all_gains["contribution"] > 0).all())

    return [
        _result(
            "comparison_windows_disjoint",
            windows_valid,
            "当前期与对比期窗口有效且互不重叠。" if windows_valid else "当前期与对比期窗口存在重叠或顺序错误。",
        ),
        _result(
            "gmv_values_non_negative",
            values_non_negative,
            "GMV 与订单数均为非负值。" if values_non_negative else "GMV 或订单数出现负值。",
        ),
        _result(
            "gmv_decomposition_additive",
            decomposition_diff <= tolerance,
            "订单量效应与 AOV 效应可精确加和至 GMV 变化。"
            if decomposition_diff <= tolerance
            else "GMV 分解无法加和，请停止输出归因结论。",
            difference=decomposition_diff,
            tolerance=tolerance,
        ),
        _result(
            "dimension_contribution_direction",
            drop_directions_valid and gain_directions_valid,
            "下拉与拉升贡献方向一致。"
            if drop_directions_valid and gain_directions_valid
            else "维度贡献方向与分类不一致。",
        ),
    ]


def validate_fulfillment_diagnosis(
    report: FulfillmentDiagnosisReport,
) -> list[ValidationResult]:
    tolerance = 1e-10
    effects_diff = abs((report.mix_effect + report.within_effect) - report.on_time_rate_delta)
    contribution_diff = abs(
        float(report.state_contributions["contribution"].sum()) - report.on_time_rate_delta
    )
    windows_valid = bool(
        report.current_start
        and report.current_end
        and report.previous_start
        and report.previous_end
        and report.current_start <= report.current_end
        and report.previous_start <= report.previous_end
        and report.previous_end < report.current_start
    )
    rates_valid = all(
        0 <= value <= 1
        for value in (report.on_time_rate_current, report.on_time_rate_previous)
    )
    orders_valid = report.delivered_orders_current >= 0 and report.delivered_orders_previous >= 0
    return [
        _result(
            "fulfillment_windows_disjoint",
            windows_valid,
            "当前期与对比期履约窗口有效且互不重叠。"
            if windows_valid
            else "履约诊断窗口存在重叠或顺序错误。",
        ),
        _result(
            "on_time_rate_range",
            rates_valid,
            "准时送达率位于 0 到 1 之间。" if rates_valid else "准时送达率超出合理范围。",
        ),
        _result(
            "delivered_orders_non_negative",
            orders_valid,
            "履约订单数为非负值。" if orders_valid else "履约订单数出现负值。",
        ),
        _result(
            "fulfillment_mix_within_additive",
            effects_diff <= tolerance,
            "结构效应与组内效应可精确加和至准时送达率变化。"
            if effects_diff <= tolerance
            else "结构/组内分解无法加和，请停止输出诊断结论。",
            difference=effects_diff,
            tolerance=tolerance,
        ),
        _result(
            "fulfillment_state_contributions_additive",
            contribution_diff <= tolerance,
            "客户州贡献可精确加和至整体准时送达率变化。"
            if contribution_diff <= tolerance
            else "客户州贡献无法加和，请停止输出诊断结论。",
            difference=contribution_diff,
            tolerance=tolerance,
        ),
    ]


def summarize_validation_status(validations: Iterable[ValidationResult]) -> tuple[bool, int]:
    items = list(validations)
    has_error = any(not item.passed and item.severity == "error" for item in items)
    warnings = sum(1 for item in items if not item.passed and item.severity == "warning")
    return not has_error, warnings
