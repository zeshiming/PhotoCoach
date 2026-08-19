"""PhotoCoach 的用户长期记忆和会话摘要存储。"""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class UserMemory:
    user_id: str
    memory_key: str
    memory_value: str
    source_session_id: str | None = None
    confidence: float = 1.0
    updated_at: str | None = None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    summary: str
    open_questions: list[str]
    updated_at: str | None = None


class LongTermMemoryStore:
    """SQLite 长期记忆；V0 使用关键词检索，后续可替换为混合检索。"""

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
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    source_session_id TEXT,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, memory_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert(self, memory: UserMemory) -> None:
        updated_at = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memories
                    (user_id, memory_key, memory_value, source_session_id, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) DO UPDATE SET
                    memory_value = excluded.memory_value,
                    source_session_id = excluded.source_session_id,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    memory.user_id,
                    memory.memory_key,
                    memory.memory_value,
                    memory.source_session_id,
                    memory.confidence,
                    updated_at,
                ),
            )

    def list(self, user_id: str, limit: int = 50) -> list[UserMemory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM user_memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, max(1, limit)),
            ).fetchall()
        return [
            UserMemory(
                user_id=row["user_id"],
                memory_key=row["memory_key"],
                memory_value=row["memory_value"],
                source_session_id=row["source_session_id"],
                confidence=float(row["confidence"]),
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def search(self, user_id: str, query: str, limit: int = 5) -> list[UserMemory]:
        """按词重叠召回相关记忆；没有匹配时返回最近的少量记忆。"""

        memories = self.list(user_id, limit=200)
        terms = {term for term in re.findall(r"[\w一-龥]+", query.lower()) if len(term) > 1}
        scored = []
        for memory in memories:
            haystack = f"{memory.memory_key} {memory.memory_value}".lower()
            score = sum(1 for term in terms if term in haystack)
            scored.append((score, memory.updated_at or "", memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        matched = [memory for score, _, memory in scored if score > 0]
        return (matched or [memory for _, _, memory in scored])[: max(1, limit)]

    def context_text(self, user_id: str, query: str, limit: int = 5) -> str:
        memories = self.search(user_id, query, limit)
        if not memories:
            return "暂无用户长期记忆。"
        return "；".join(f"{item.memory_key}：{item.memory_value}" for item in memories)

    def delete_user(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))

    def remember_explicit(
        self,
        user_id: str,
        text: str,
        source_session_id: str | None = None,
    ) -> UserMemory | None:
        """只保存用户明确要求“记住”的非敏感信息。"""

        if "记住" not in text:
            return None
        value = text.split("记住", 1)[1].lstrip("：:，, ").strip()
        if not value or any(item in value for item in ("身份证", "年龄", "身份", "住址", "电话")):
            return None
        if any(item in value for item in ("喜欢", "偏好", "风格")):
            key = "preference"
        elif any(item in value for item in ("相机", "镜头", "设备")):
            key = "equipment"
        else:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
            key = f"explicit_note_{digest}"
        memory = UserMemory(
            user_id=user_id,
            memory_key=key,
            memory_value=value,
            source_session_id=source_session_id,
        )
        self.upsert(memory)
        return memory

    def upsert_summary(
        self,
        session_id: str,
        summary: str,
        open_questions: list[str] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_summaries
                    (session_id, summary, open_questions_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    open_questions_json = excluded.open_questions_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    summary,
                    json.dumps(open_questions or [], ensure_ascii=False),
                    self._now(),
                ),
            )

    def get_summary(self, session_id: str) -> SessionSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            questions = json.loads(row["open_questions_json"] or "[]")
        except json.JSONDecodeError:
            questions = []
        return SessionSummary(
            session_id=row["session_id"],
            summary=row["summary"],
            open_questions=questions if isinstance(questions, list) else [],
            updated_at=row["updated_at"],
        )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
