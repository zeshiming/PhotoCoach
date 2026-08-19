"""会话级 Scope Memory。

SQLiteSession 保存对话消息；本模块只保存护栏需要的范围状态，二者职责分离。
不保存模型思维链或图片像素。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .scope_models import ScopeResult


@dataclass
class ScopeMemoryState:
    """一个会话当前的最小范围状态。"""

    session_id: str
    goal: str | None = None
    risk_level: str = "low"
    last_decision: str | None = None
    denied_topics: list[str] = field(default_factory=list)
    last_image_id: str | None = None
    updated_at: str | None = None

    def summary(self) -> str:
        """生成可发送给 Scope Guard 的短摘要。"""

        denied = ", ".join(self.denied_topics) or "无"
        return (
            f"会话目标：{self.goal or '尚未确定'}；"
            f"最近决策：{self.last_decision or '无'}；"
            f"风险等级：{self.risk_level}；"
            f"已拒绝主题：{denied}；"
            f"最近图片：{self.last_image_id or '无'}"
        )


class ScopeMemoryStore:
    """使用 SQLite 持久化跨轮 Scope 状态。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_scope_state (
                    session_id TEXT PRIMARY KEY,
                    goal TEXT,
                    risk_level TEXT NOT NULL,
                    last_decision TEXT,
                    denied_topics_json TEXT NOT NULL,
                    last_image_id TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ScopeMemoryState:
        try:
            denied_topics = json.loads(row["denied_topics_json"] or "[]")
        except json.JSONDecodeError:
            denied_topics = []
        if not isinstance(denied_topics, list):
            denied_topics = []
        return ScopeMemoryState(
            session_id=row["session_id"],
            goal=row["goal"],
            risk_level=row["risk_level"],
            last_decision=row["last_decision"],
            denied_topics=[str(item) for item in denied_topics],
            last_image_id=row["last_image_id"],
            updated_at=row["updated_at"],
        )

    def get(self, session_id: str) -> ScopeMemoryState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_scope_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def summary(self, session_id: str) -> str:
        state = self.get(session_id)
        return state.summary() if state else "当前会话还没有 Scope Memory。"

    def upsert(self, state: ScopeMemoryState) -> None:
        state.updated_at = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_scope_state
                    (session_id, goal, risk_level, last_decision,
                     denied_topics_json, last_image_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    goal = excluded.goal,
                    risk_level = excluded.risk_level,
                    last_decision = excluded.last_decision,
                    denied_topics_json = excluded.denied_topics_json,
                    last_image_id = excluded.last_image_id,
                    updated_at = excluded.updated_at
                """,
                (
                    state.session_id,
                    state.goal,
                    state.risk_level,
                    state.last_decision,
                    json.dumps(state.denied_topics, ensure_ascii=False),
                    state.last_image_id,
                    state.updated_at,
                ),
            )

    def record(
        self,
        session_id: str,
        result: ScopeResult,
        *,
        goal: str | None = None,
        last_image_id: str | None = None,
    ) -> ScopeMemoryState:
        """将一次 Guard 决策合并进会话状态。"""

        state = self.get(session_id) or ScopeMemoryState(session_id=session_id)
        if goal and not state.goal:
            state.goal = goal
        state.risk_level = result.risk_level
        state.last_decision = result.decision.value
        if last_image_id:
            state.last_image_id = last_image_id
        if result.decision.value == "refuse" and result.reason_code not in state.denied_topics:
            state.denied_topics.append(result.reason_code)
        self.upsert(state)
        return state

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM session_scope_state WHERE session_id = ?",
                (session_id,),
            )
