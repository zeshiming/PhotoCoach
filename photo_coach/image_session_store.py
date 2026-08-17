"""会话级图片引用存储。

对话消息由 Agents SDK Session 保存；图片只保存引用和元数据，
不把像素内容塞进 LLM 对话历史。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    ref: str
    turn_no: int
    added_at: datetime
    image_hash: str | None = None
    expires_at: datetime | None = None


class ImageReferenceResolver:
    """把自然语言图片引用解析为稳定的 image_id。"""

    _EXPLICIT = re.compile(r"\bimage#(\d+)\b", re.IGNORECASE)
    _VAGUE_MARKERS = (
        "这张",
        "这张照片",
        "这张图",
        "那张",
        "那张照片",
        "那张图",
        "那光线",
        "那构图",
        "那曝光",
    )

    @classmethod
    def resolve(cls, records: list[ImageRecord], text: str) -> str | None:
        if not records:
            return None
        normalized = text.strip().lower()
        explicit = cls._EXPLICIT.search(normalized)
        if explicit:
            image_id = f"image#{explicit.group(1)}"
            return image_id if any(item.image_id == image_id for item in records) else None

        ordered = sorted(records, key=lambda item: (item.turn_no, item.added_at))
        if "上一张" in normalized or "上张" in normalized:
            return ordered[-2].image_id if len(ordered) > 1 else ordered[-1].image_id

        if any(marker in normalized for marker in cls._VAGUE_MARKERS):
            return ordered[-1].image_id if len(ordered) == 1 else None
        if "那" in normalized and len(ordered) == 1:
            return ordered[-1].image_id
        return None


class ImageSessionStore:
    """使用 SQLite 保存会话级图片引用和生命周期元数据。"""

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
                CREATE TABLE IF NOT EXISTS session_images (
                    session_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    turn_no INTEGER NOT NULL,
                    image_hash TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    PRIMARY KEY (session_id, image_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_images_turn "
                "ON session_images(session_id, turn_no)"
            )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _hash_ref(ref: str) -> str | None:
        path = Path(ref).expanduser()
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _record(cls, row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            image_id=row["image_id"],
            ref=row["ref"],
            turn_no=row["turn_no"],
            added_at=cls._parse_datetime(row["created_at"]) or cls._now(),
            image_hash=row["image_hash"],
            expires_at=cls._parse_datetime(row["expires_at"]),
        )

    def next_turn_no(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_no), 0) + 1 FROM session_images WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row else 1

    def register(
        self,
        session_id: str,
        ref: str,
        turn_no: int,
        *,
        expires_at: datetime | None = None,
    ) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_id FROM session_images WHERE session_id = ? "
                "ORDER BY turn_no DESC, created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            previous = 0
            if row and str(row["image_id"]).startswith("image#"):
                try:
                    previous = int(str(row["image_id"])[6:])
                except ValueError:
                    pass
            image_id = f"image#{previous + 1}"
            conn.execute(
                "INSERT INTO session_images "
                "(session_id, image_id, ref, turn_no, image_hash, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    image_id,
                    ref,
                    turn_no,
                    self._hash_ref(ref),
                    self._now().isoformat(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )
            return image_id

    def get(self, session_id: str, image_id: str) -> ImageRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_images WHERE session_id = ? AND image_id = ?",
                (session_id, image_id),
            ).fetchone()
        return self._record(row) if row else None

    def list(self, session_id: str) -> list[ImageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_images WHERE session_id = ? ORDER BY turn_no ASC, created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def latest(self, session_id: str) -> ImageRecord | None:
        records = self.list(session_id)
        return records[-1] if records else None

    def resolve_reference(self, session_id: str, text: str) -> str | None:
        return ImageReferenceResolver.resolve(self.list(session_id), text)

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM session_images WHERE session_id = ?", (session_id,))
