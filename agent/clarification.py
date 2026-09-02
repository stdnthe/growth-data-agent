from __future__ import annotations

from .models import AnalysisIntent


_METRIC_QUERY_LABELS = {
    "gmv": "GMV",
    "paid_orders": "订单量",
    "aov": "客单价",
    "repeat_purchase_rate": "复购率",
    "on_time_delivery_rate": "准时送达率",
    "late_delivery_rate": "延迟送达率",
    "average_review_score": "平均评分",
}


def clarification_suggestions(intent: AnalysisIntent) -> list[str]:
    if intent.action == "confirm":
        return ["先查看该指标趋势", "改为分析 GMV 变化", "暂不分析"]
    if intent.metric is None:
        return ["GMV", "订单量", "客单价", "复购率"]
    return ["按默认口径继续", "补充其他条件"]


def resolve_clarification(
    original_question: str,
    intent: AnalysisIntent,
    answer: str,
) -> str | None:
    normalized = answer.strip()
    if not normalized or normalized in {"暂不分析", "取消", "不用了"}:
        return None

    if intent.action == "confirm":
        if "GMV" in normalized.upper() or "成交" in normalized or "销售额" in normalized:
            return f"为什么{intent.time_range or '最近30天'}GMV发生变化？"

        metric_label = _METRIC_QUERY_LABELS.get(intent.metric or "", intent.metric or "该指标")
        return f"{intent.time_range or '最近30天'}{metric_label}趋势"

    metric_answer = normalized.lower().replace(" ", "")
    if metric_answer in {"gmv", "成交总额", "销售额"}:
        return "近30天GMV趋势（按天）"
    if metric_answer in {"订单量", "订单数", "支付订单数"}:
        return "近30天订单量走势（按天）"
    if metric_answer in {"aov", "客单价"}:
        return "近30天客单价走势（按天）"
    if metric_answer == "复购率":
        return "近30天复购率"

    return f"{original_question}\n用户补充：{normalized}"
