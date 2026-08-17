from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _enabled_from_env() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() in {"1", "true", "yes", "on"}


def _safe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)[:500]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item, depth + 1) for item in value]
    return str(value)[:2000]


@dataclass
class SpanHandle:
    outputs: dict[str, Any] = field(default_factory=dict)

    def set_outputs(self, **outputs: Any) -> None:
        self.outputs.update(outputs)


class TraceRecorder:
    """Optional LangSmith adapter. Observability must never block analysis."""

    def __init__(self, enabled: bool | None = None, project: str | None = None):
        self.enabled = _enabled_from_env() if enabled is None else bool(enabled)
        self.project = project or os.getenv("LANGSMITH_PROJECT", "growth-analysis-agent")
        self.available = False
        self._trace_factory: Any | None = None
        if self.enabled:
            try:
                from langsmith import trace

                self._trace_factory = trace
                self.available = True
            except ImportError:
                self.available = False

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[SpanHandle]:
        handle = SpanHandle()
        if not self.available or self._trace_factory is None:
            yield handle
            return

        try:
            manager = self._trace_factory(
                name=name,
                run_type=run_type,
                inputs=_safe(inputs or {}),
                metadata=_safe(metadata or {}),
                tags=tags or [],
                project_name=self.project,
            )
            run_tree = manager.__enter__()
        except Exception:  # noqa: BLE001 - tracing cannot take down the product
            yield handle
            return

        try:
            yield handle
        except BaseException as exc:
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:  # noqa: BLE001
                pass
            raise
        else:
            try:
                if handle.outputs and hasattr(run_tree, "end"):
                    run_tree.end(outputs=_safe(handle.outputs))
                manager.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
