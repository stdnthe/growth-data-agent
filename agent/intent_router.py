from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_ATTRIBUTION_KEYWORDS = [
    "为什么", "原因", "归因", "驱动", "导致",
    "异动", "波动", "下滑原因", "增长原因", "变化原因",
    "怎么了", "分析原因", "什么原因", "如何解释",
]

_WINDOW_PATTERNS: list[tuple[str, int]] = [
    (r"近\s*(\d+)\s*天", 0),        # 近N天 → N（动态）
    (r"最近\s*(\d+)\s*天", 0),
    (r"过去\s*(\d+)\s*天", 0),
    (r"近\s*(\d+)\s*周", -1),        # 近N周 → N*7（动态）
    (r"上个?月|上月", 30),
    (r"近一个?月|最近一个?月", 30),
    (r"近三个?月|最近三个?月", 90),
    (r"近\s*7\s*天|近一周|最近一周", 7),
    (r"近\s*14\s*天|近两周", 14),
    (r"近\s*90\s*天", 90),
    (r"近\s*60\s*天", 60),
]


@dataclass
class RouteResult:
    intent: Literal["attribution", "retrieval"]
    window_days: int = 30
    matched_keywords: list[str] = field(default_factory=list)


def _extract_window_days(question: str) -> int:
    for pattern, fixed_days in _WINDOW_PATTERNS:
        m = re.search(pattern, question)
        if m:
            if fixed_days == 0:
                return int(m.group(1))
            if fixed_days == -1:
                return int(m.group(1)) * 7
            return fixed_days
    return 30


def route(question: str) -> RouteResult:
    matched = [kw for kw in _ATTRIBUTION_KEYWORDS if kw in question]
    if matched:
        return RouteResult(
            intent="attribution",
            window_days=_extract_window_days(question),
            matched_keywords=matched,
        )
    return RouteResult(intent="retrieval")
