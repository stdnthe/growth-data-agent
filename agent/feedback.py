from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FeedbackRecord:
    run_id: str
    feedback_type: str
    helpful: bool
    comment: str = ""
    question: str = ""
    workflow: str = ""
    failure_stage: str | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: FeedbackRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
