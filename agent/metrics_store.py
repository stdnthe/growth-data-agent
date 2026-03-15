from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MetricDefinition:
    id: str
    name_zh: str
    definition: str
    formula: str
    grain: str
    sql_hint: str


class MetricsStore:
    def __init__(self, metrics_path: str | Path):
        self.metrics_path = Path(metrics_path)
        self.metrics: list[MetricDefinition] = []

    def load(self) -> list[MetricDefinition]:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")

        raw: dict[str, Any] = yaml.safe_load(self.metrics_path.read_text(encoding="utf-8")) or {}
        metric_items = raw.get("metrics", [])

        if not isinstance(metric_items, list):
            raise ValueError("metrics.yml format error: 'metrics' must be a list")

        parsed: list[MetricDefinition] = []
        for item in metric_items:
            if not isinstance(item, dict):
                continue
            parsed.append(
                MetricDefinition(
                    id=str(item.get("id", "")).strip(),
                    name_zh=str(item.get("name_zh", "")).strip(),
                    definition=str(item.get("definition", "")).strip(),
                    formula=str(item.get("formula", "")).strip(),
                    grain=str(item.get("grain", "")).strip(),
                    sql_hint=str(item.get("sql_hint", "")).strip(),
                )
            )

        self.metrics = [m for m in parsed if m.id]
        return self.metrics

    def compressed_context(self, max_items: int = 30) -> str:
        if not self.metrics:
            self.load()

        lines: list[str] = [
            "Follow these metric semantics strictly. If ambiguous, use the metric formula and sql_hint.",
            "Do not include canceled/unavailable orders in paid metrics unless explicitly requested.",
        ]

        for metric in self.metrics[:max_items]:
            lines.append(
                f"- {metric.id} ({metric.name_zh}): {metric.definition}; formula={metric.formula}; "
                f"grain={metric.grain}; hint={metric.sql_hint}"
            )

        return "\n".join(lines)
