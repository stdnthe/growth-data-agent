from __future__ import annotations

import re

from .models import AnalysisIntent


_ATTRIBUTION_KEYWORDS = [
    "为什么",
    "原因",
    "归因",
    "驱动",
    "导致",
    "异动",
    "波动",
    "下滑",
    "下降",
    "增长原因",
    "变化原因",
    "怎么了",
    "如何解释",
]

_METRIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("gmv_top10_state_share", ("top10州gmv占比", "州gmv集中度")),
    ("new_customer_order_share", ("新客订单占比", "新客占比")),
    ("installment_order_share", ("分期订单占比", "分期占比")),
    ("on_time_delivery_rate", ("准时送达率", "准时率")),
    ("late_delivery_rate", ("延迟送达率", "延迟率")),
    ("avg_delivery_days", ("平均送达天数", "平均配送天数")),
    ("p90_delivery_days", ("p90送达天数", "p90配送天数")),
    ("average_review_score", ("平均评分", "评价均分", "评分")),
    ("low_rating_rate", ("低评分率",)),
    ("high_rating_rate", ("高评分率",)),
    ("d7_retention", ("d7留存率", "7日留存率")),
    ("d30_retention", ("d30留存率", "30日留存率")),
    ("repeat_purchase_rate", ("复购率",)),
    ("cancellation_rate", ("取消率",)),
    ("approval_rate", ("审批通过率", "通过率")),
    ("freight_ratio", ("运费占比",)),
    ("gmv_per_buyer", ("人均gmv", "人均成交额")),
    ("orders_per_buyer", ("人均订单数",)),
    ("avg_payment_value", ("平均支付金额",)),
    ("unique_buyers", ("下单买家数", "买家数", "用户数")),
    ("new_customers", ("新客数", "新增买家", "新增客户")),
    ("returning_customers", ("老客数", "老客户数")),
    ("delivered_orders", ("完成妥投订单数", "妥投订单数")),
    ("active_sku", ("活跃sku数", "活跃商品数")),
    ("active_sellers", ("活跃商家数", "活跃卖家数")),
    ("paid_orders", ("支付订单数", "支付订单", "订单数", "订单量")),
    ("aov", ("aov", "客单价")),
    ("gmv", ("gmv", "成交总额", "销售额", "收入")),
]

_DIMENSION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("state", ("按州", "各州", "state")),
    ("city", ("按城市", "各城市", "city")),
    ("category", ("按品类", "各品类", "category")),
    ("seller", ("按卖家", "按商家", "seller")),
    ("product", ("按产品", "sku", "product")),
    ("payment_type", ("支付方式", "payment_type")),
]

_WINDOW_PATTERNS: list[tuple[str, int | None, str]] = [
    (r"(?:近|最近|过去)\s*(\d+)\s*天", None, "最近{n}天"),
    (r"(?:近|最近|过去)\s*(\d+)\s*周", None, "最近{n}周"),
    (r"(?:近|最近|过去)\s*(\d+)\s*个?月", None, "最近{n}个月"),
    (r"上个?月|上月", 30, "上个月"),
    (r"近一周|最近一周", 7, "最近7天"),
    (r"近两周|最近两周", 14, "最近14天"),
    (r"近一个?月|最近一个?月", 30, "最近30天"),
    (r"近三个?月|最近三个?月", 90, "最近90天"),
    (r"今年以来", 365, "今年以来"),
    (r"过去一年|近一年|最近一年", 365, "最近一年"),
    (r"今天", 1, "数据集最新一天"),
]


def _extract_metrics(question: str) -> list[str]:
    lowered = question.lower()
    candidates: list[tuple[int, int, str, int]] = []
    for metric, aliases in _METRIC_KEYWORDS:
        for alias in aliases:
            alias_lower = alias.lower()
            start = lowered.find(alias_lower)
            while start >= 0:
                candidates.append((start, start + len(alias_lower), metric, len(alias_lower)))
                start = lowered.find(alias_lower, start + 1)

    selected: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    seen_metrics: set[str] = set()
    for start, end, metric, _ in sorted(candidates, key=lambda item: (-item[3], item[0])):
        if metric in seen_metrics or any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        seen_metrics.add(metric)
        selected.append((start, metric))
    return [metric for _, metric in sorted(selected)]


def _extract_dimensions(question: str) -> list[str]:
    lowered = question.lower()
    return [dimension for dimension, aliases in _DIMENSION_KEYWORDS if any(alias.lower() in lowered for alias in aliases)]


def _extract_grain(question: str) -> str | None:
    if re.search(r"按\s*天|每天|每日", question):
        return "day"
    if re.search(r"按\s*周|每周", question):
        return "week"
    if re.search(r"按\s*月|每月|月度", question):
        return "month"
    return None


def _extract_time_range(question: str) -> tuple[str | None, int | None]:
    for pattern, fixed_days, label in _WINDOW_PATTERNS:
        match = re.search(pattern, question)
        if not match:
            continue
        if fixed_days is not None:
            return label, fixed_days
        value = int(match.group(1))
        if "周" in match.group(0):
            return label.format(n=value), value * 7
        if "月" in match.group(0):
            return label.format(n=value), value * 30
        return label.format(n=value), value
    return None, None


def route(question: str) -> AnalysisIntent:
    normalized = question.strip()
    if not normalized:
        return AnalysisIntent(
            task_type="metric_query",
            workflow="retrieval",
            action="clarify",
            confidence=0.0,
            ambiguities=["问题为空"],
            clarification_question="请告诉我你想分析的指标和时间范围。",
        )

    matched = [keyword for keyword in _ATTRIBUTION_KEYWORDS if keyword in normalized]
    metrics = _extract_metrics(normalized)
    metric = metrics[0] if metrics else None
    time_range, extracted_days = _extract_time_range(normalized)
    dimensions = _extract_dimensions(normalized)
    grain = _extract_grain(normalized)
    is_diagnosis = bool(matched)
    if is_diagnosis and "gmv" in metrics:
        metric = "gmv"
    ambiguities: list[str] = []
    assumptions: list[str] = []
    action = "execute"
    clarification_question: str | None = None

    if metric is None:
        ambiguities.append("未识别明确的业务指标")
        action = "clarify"
        clarification_question = "你希望分析哪个指标？例如 GMV、订单数、客单价、复购率或准时送达率。"

    window_days = extracted_days or 30
    if time_range is None:
        time_range = "最近30天"
        assumptions.append("未指定时间范围，默认使用数据集最近30天")

    workflow = "gmv_attribution" if is_diagnosis and "gmv" in metrics else "retrieval"
    task_type = "metric_diagnosis" if is_diagnosis else "metric_query"

    if workflow == "gmv_attribution" and window_days > 120:
        assumptions.append("GMV 归因窗口最长为120天，已将请求范围收敛为最近120天")
        window_days = 120

    if is_diagnosis and metric not in {None, "gmv"}:
        ambiguities.append("当前确定性归因 Workflow 仅支持 GMV")
        action = "confirm"
        clarification_question = (
            "当前可以先查询该指标的趋势，但自动归因只支持 GMV。是否先执行趋势分析？"
        )

    comparison = "previous_period" if is_diagnosis or any(token in normalized for token in ("对比", "环比", "同比")) else None
    confidence = 0.95 if metric and extracted_days else 0.8 if metric else 0.35

    return AnalysisIntent(
        task_type=task_type,
        workflow=workflow,
        metric=metric,
        metrics=metrics,
        time_range=time_range,
        window_days=max(1, min(365, window_days)),
        comparison=comparison,
        grain=grain,
        dimensions=dimensions,
        business_goal="identify_change_drivers" if is_diagnosis else "retrieve_metric",
        ambiguities=ambiguities,
        assumptions=assumptions,
        confidence=confidence,
        action=action,
        clarification_question=clarification_question,
        matched_keywords=matched,
    )
