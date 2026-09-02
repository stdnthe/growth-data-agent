from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .metrics_store import MetricDefinition
from .models import AnalysisIntent


class IntentValidationError(ValueError):
    """Raised when a model-produced intent violates the local contract."""


class IntentValidator:
    """Validate model output and make routing decisions deterministically."""

    ALLOWED_DIMENSIONS = {
        "state",
        "city",
        "category",
        "seller",
        "product",
        "payment_type",
    }
    ALLOWED_GRAINS = {"day", "week", "month", "cohort_month"}
    ALLOWED_COMPARISONS = {"previous_period", "month_over_month", "year_over_year"}
    REQUIRED_FIELDS = {
        "task_type",
        "metric",
        "metrics",
        "time_range",
        "window_days",
        "comparison",
        "grain",
        "dimensions",
        "filters",
        "business_goal",
        "ambiguities",
        "assumptions",
        "confidence",
        "clarification_question",
    }
    FULFILLMENT_DIAGNOSIS_METRICS = {
        "on_time_delivery_rate",
        "late_delivery_rate",
        "avg_delivery_days",
        "p90_delivery_days",
    }

    def __init__(self, metrics: list[MetricDefinition]) -> None:
        self._metrics = {metric.id: metric for metric in metrics}

    @property
    def allowed_metric_ids(self) -> set[str]:
        return set(self._metrics)

    def validate(self, payload: Mapping[str, Any]) -> AnalysisIntent:
        if not isinstance(payload, Mapping):
            raise IntentValidationError("Intent output must be a JSON object.")

        missing_fields = self.REQUIRED_FIELDS - set(payload)
        extra_fields = set(payload) - self.REQUIRED_FIELDS
        if missing_fields:
            raise IntentValidationError(
                f"Intent output is missing fields: {', '.join(sorted(missing_fields))}."
            )
        if extra_fields:
            raise IntentValidationError(
                f"Intent output contains unsupported fields: {', '.join(sorted(extra_fields))}."
            )

        task_type = self._required_choice(
            payload["task_type"], "task_type", {"metric_query", "metric_diagnosis"}
        )
        metrics = self._string_list(payload["metrics"], "metrics")
        primary = self._optional_string(payload["metric"], "metric")
        if primary:
            metrics.insert(0, primary)
        metrics = list(dict.fromkeys(metrics))

        unknown_metrics = [metric for metric in metrics if metric not in self._metrics]
        if unknown_metrics:
            raise IntentValidationError(
                f"Intent references metrics outside the semantic layer: {', '.join(unknown_metrics)}."
            )
        primary = metrics[0] if metrics else None

        dimensions = self._string_list(payload["dimensions"], "dimensions")
        invalid_dimensions = [item for item in dimensions if item not in self.ALLOWED_DIMENSIONS]
        if invalid_dimensions:
            raise IntentValidationError(
                f"Intent contains unsupported dimensions: {', '.join(invalid_dimensions)}."
            )
        dimensions = list(dict.fromkeys(dimensions))

        grain = self._optional_string(payload["grain"], "grain")
        if grain and grain not in self.ALLOWED_GRAINS:
            raise IntentValidationError(f"Unsupported grain: {grain}.")

        comparison = self._optional_string(payload["comparison"], "comparison")
        if comparison and comparison not in self.ALLOWED_COMPARISONS:
            raise IntentValidationError(f"Unsupported comparison: {comparison}.")

        raw_window = payload["window_days"]
        if raw_window is not None and (isinstance(raw_window, bool) or not isinstance(raw_window, int)):
            raise IntentValidationError("window_days must be an integer or null.")
        if isinstance(raw_window, int) and raw_window < 1:
            raise IntentValidationError("window_days must be positive.")
        window_days = min(raw_window or 30, 365)

        confidence_value = payload["confidence"]
        if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
            raise IntentValidationError("confidence must be a number between 0 and 1.")
        confidence = float(confidence_value)
        if not 0 <= confidence <= 1:
            raise IntentValidationError("confidence must be a number between 0 and 1.")

        filters = payload["filters"]
        if not isinstance(filters, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in filters.items()
        ):
            raise IntentValidationError("filters must be a JSON object with string values.")

        ambiguities = self._string_list(payload["ambiguities"], "ambiguities")
        assumptions = self._string_list(payload["assumptions"], "assumptions")
        time_range = self._optional_string(payload["time_range"], "time_range")
        if not time_range:
            time_range = "最近30天"
            assumptions.append("未指定时间范围，默认使用数据集最近30天")
        if isinstance(raw_window, int) and raw_window > 365:
            assumptions.append("查询窗口最长为365天，已将请求范围收敛为最近365天")

        business_goal = self._optional_string(payload["business_goal"], "business_goal")
        clarification_question = self._optional_string(
            payload["clarification_question"], "clarification_question"
        )

        workflow = "retrieval"
        action = "execute"
        if not metrics:
            action = "clarify"
            ambiguities.append("未识别明确的业务指标")
            clarification_question = clarification_question or (
                "你希望分析哪个指标？例如 GMV、订单数、客单价、复购率或准时送达率。"
            )
        elif confidence < 0.55:
            action = "clarify"
            ambiguities.append("意图识别置信度不足")
            clarification_question = clarification_question or "请补充要分析的指标、时间范围和分析目标。"
        elif task_type == "metric_diagnosis" and "gmv" in metrics:
            workflow = "gmv_attribution"
            primary = "gmv"
            comparison = "previous_period"
            business_goal = business_goal or "identify_change_drivers"
            if window_days > 120:
                assumptions.append("GMV 归因窗口最长为120天，已将请求范围收敛为最近120天")
                window_days = 120
        elif task_type == "metric_diagnosis" and primary in self.FULFILLMENT_DIAGNOSIS_METRICS:
            workflow = "fulfillment_diagnosis"
            comparison = "previous_period"
            business_goal = business_goal or "identify_fulfillment_change_drivers"
            if window_days > 120:
                assumptions.append("履约诊断窗口最长为120天，已将请求范围收敛为最近120天")
                window_days = 120
        elif task_type == "metric_diagnosis":
            action = "confirm"
            ambiguities.append("当前确定性诊断 Recipe 仅支持 GMV 与履约时效指标")
            clarification_question = clarification_question or (
                "当前可以先查询该指标的趋势，但自动诊断只支持 GMV 与履约时效指标。是否先执行趋势分析？"
            )

        if primary and grain:
            supported_grains = set(self._metrics[primary].grain.split("/"))
            if grain not in supported_grains:
                action = "clarify"
                ambiguities.append(f"指标 {primary} 不支持 {grain} 粒度")
                clarification_question = (
                    f"{self._metrics[primary].name_zh}支持的粒度为"
                    f"{self._metrics[primary].grain}，请选择其中一种。"
                )

        return AnalysisIntent(
            task_type=task_type,  # type: ignore[arg-type]
            workflow=workflow,  # type: ignore[arg-type]
            metric=primary,
            metrics=metrics,
            time_range=time_range,
            window_days=window_days,
            comparison=comparison,
            grain=grain,
            dimensions=dimensions,
            filters=dict(filters),
            business_goal=business_goal,
            ambiguities=list(dict.fromkeys(ambiguities)),
            assumptions=list(dict.fromkeys(assumptions)),
            confidence=confidence,
            action=action,  # type: ignore[arg-type]
            clarification_question=clarification_question,
        )

    @staticmethod
    def _required_choice(value: Any, field: str, allowed: set[str]) -> str:
        if not isinstance(value, str) or value not in allowed:
            raise IntentValidationError(f"{field} must be one of: {', '.join(sorted(allowed))}.")
        return value

    @staticmethod
    def _optional_string(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise IntentValidationError(f"{field} must be a string or null.")
        return value.strip() or None

    @staticmethod
    def _string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise IntentValidationError(f"{field} must be an array of strings.")
        return [item.strip() for item in value if item.strip()]
