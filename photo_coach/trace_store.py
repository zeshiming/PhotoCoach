"""PhotoCoach 本地运行 Trace 存储。"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TraceRecord(BaseModel):
    """一次 Agent 运行的可审计摘要。"""

    trace_id: str
    session_id: str
    model: str
    vision_model: str | None = None
    status: str
    duration_ms: int
    retry_count: int = 0
    failure_type: str | None = None
    guard_checks: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class JsonlTraceStore:
    """以追加写入方式保存 Trace，适合本地开发和调试。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("TRACE_FILE", "data/traces.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: TraceRecord) -> None:
        line = json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
