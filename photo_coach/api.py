"""PhotoCoach FastAPI HTTP 接口和静态前端入口。"""

from __future__ import annotations

import os
import re
import json
import sqlite3
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents_sdk_app import run_photo_agent
from .models import AgentRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
WEB_ROOT = PROJECT_ROOT / "web"
SESSION_DB = PROJECT_ROOT / "data" / "agent_sessions.db"
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    answer: str


class HealthResponse(BaseModel):
    status: str
    service: str


class SessionSummary(BaseModel):
    session_id: str
    created_at: str | None = None
    updated_at: str | None = None
    title: str = "新建会话"
    message_count: int = 0


class SessionMessages(BaseModel):
    session_id: str
    messages: list[dict[str, str]]


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


app = FastAPI(
    title="PhotoCoach API",
    version="0.3.0",
    description="文字和图片对话的单 Agent 摄影助手",
)

origins = [
    item.strip()
    for item in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="photocoach")


def _message_text(payload: str) -> tuple[str | None, str | None]:
    """从 Agents SDK 保存的输入项中提取 UI 可展示的角色和文本。"""

    try:
        item = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    role = item.get("role")
    if role == "user" and isinstance(item.get("content"), str):
        return "user", item["content"]
    if role == "assistant" and isinstance(item.get("content"), list):
        texts = [
            str(block.get("text", ""))
            for block in item["content"]
            if isinstance(block, dict) and block.get("type") in {"output_text", "text"}
        ]
        text = "\n".join(part for part in texts if part).strip()
        return ("assistant", text) if text else (None, None)
    return None, None


def _session_rows(limit: int) -> list[SessionSummary]:
    if not SESSION_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(SESSION_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT session_id, created_at, updated_at FROM agent_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        summaries: list[SessionSummary] = []
        for row in rows:
            messages = conn.execute(
                "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id ASC",
                (row["session_id"],),
            ).fetchall()
            title = "新建会话"
            for message in messages:
                role, text = _message_text(message["message_data"])
                if role == "user" and text:
                    title = text[:28] + ("…" if len(text) > 28 else "")
                    break
            summaries.append(
                SessionSummary(
                    session_id=row["session_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    title=title,
                    message_count=len(messages),
                )
            )
        conn.close()
        return summaries
    except sqlite3.OperationalError:
        return []


@app.get("/api/v1/sessions", response_model=list[SessionSummary])
async def list_sessions(limit: int = 30) -> list[SessionSummary]:
    return _session_rows(max(1, min(limit, 100)))


@app.get("/api/v1/sessions/{session_id}/messages", response_model=SessionMessages)
async def session_messages(session_id: str) -> SessionMessages:
    safe_session_id = _safe_session_id(session_id)
    if not SESSION_DB.exists():
        return SessionMessages(session_id=safe_session_id, messages=[])
    try:
        conn = sqlite3.connect(str(SESSION_DB))
        rows = conn.execute(
            "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id ASC",
            (safe_session_id,),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []
    messages: list[dict[str, str]] = []
    for row in rows:
        role, text = _message_text(row[0])
        if role and text:
            messages.append({"role": role, "content": text})
    return SessionMessages(session_id=safe_session_id, messages=messages)


@app.delete("/api/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    safe_session_id = _safe_session_id(session_id)
    if not SESSION_DB.exists():
        return DeleteSessionResponse(session_id=safe_session_id, deleted=False)
    try:
        conn = sqlite3.connect(str(SESSION_DB))
        exists = conn.execute(
            "SELECT 1 FROM agent_sessions WHERE session_id = ? "
            "UNION SELECT 1 FROM session_images WHERE session_id = ? LIMIT 1",
            (safe_session_id, safe_session_id),
        ).fetchone()
        if exists is None:
            conn.close()
            return DeleteSessionResponse(session_id=safe_session_id, deleted=False)
        conn.execute("DELETE FROM agent_messages WHERE session_id = ?", (safe_session_id,))
        conn.execute("DELETE FROM agent_sessions WHERE session_id = ?", (safe_session_id,))
        conn.execute("DELETE FROM session_images WHERE session_id = ?", (safe_session_id,))
        conn.commit()
        conn.close()
        return DeleteSessionResponse(session_id=safe_session_id, deleted=True)
    except sqlite3.OperationalError:
        return DeleteSessionResponse(session_id=safe_session_id, deleted=False)


def _safe_session_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]", "_", value.strip())
    return normalized[:100] or "default"


async def _save_image(session_id: str, image: UploadFile) -> str:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件。")

    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 10MB。")

    suffix = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".jpg"
    target_dir = UPLOAD_ROOT / session_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(data)
    return str(target)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(default=""),
    session_id: str = Form(default="default"),
    image: UploadFile | None = File(default=None),
) -> ChatResponse:
    safe_session_id = _safe_session_id(session_id)
    text = message.strip()
    image_refs: list[str] = []
    if image is not None:
        image_refs.append(await _save_image(safe_session_id, image))
    if not text and not image_refs:
        raise HTTPException(status_code=400, detail="请输入问题或上传图片。")

    try:
        result = await run_photo_agent(
            AgentRequest(
                text=text,
                images=image_refs,
                session_id=safe_session_id,
            )
        )
    except Exception as exc:
        # 对外不返回 Key、Base URL 或完整 SDK 堆栈；详细信息由进程日志记录。
        print(f"PhotoCoach request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="模型服务暂时不可用，请稍后重试。",
        ) from exc

    return ChatResponse(
        session_id=result.session_id,
        trace_id=result.trace_id,
        answer=result.final_output,
    )


if WEB_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
