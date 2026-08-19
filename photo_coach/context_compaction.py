"""非破坏式上下文压缩管理器。

完整 Session 历史仍由 SQLiteSession 保存；超过预算时，只向模型返回摘要和最近消息。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactionConfig:
    max_tokens: int = 12000
    keep_recent_items: int = 12
    chars_per_token: int = 4

    @classmethod
    def from_env(cls) -> "CompactionConfig":
        return cls(
            max_tokens=max(1000, int(os.getenv("PHOTOCOACH_CONTEXT_MAX_TOKENS", "12000"))),
            keep_recent_items=max(2, int(os.getenv("PHOTOCOACH_KEEP_RECENT_ITEMS", "12"))),
        )


@dataclass(frozen=True)
class CompactionResult:
    items: list[dict[str, Any]]
    compacted: bool
    estimated_tokens: int
    removed_items: int


class ContextCompactor:
    def __init__(self, config: CompactionConfig | None = None):
        self.config = config or CompactionConfig.from_env()

    def estimate_tokens(self, items: list[Any]) -> int:
        payload = json.dumps(items, ensure_ascii=False, default=str)
        return max(0, len(payload) // self.config.chars_per_token)

    def should_compact(self, items: list[Any]) -> bool:
        return self.estimate_tokens(items) > self.config.max_tokens

    def summarize_items(self, items: list[Any]) -> str:
        """生成可审计的确定性摘要；后续可替换成专用摘要模型。"""

        user_texts: list[str] = []
        tool_names: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                user_texts.append(item["content"][:240])
            if item.get("type") in {"function_call", "tool_call"}:
                name = item.get("name") or item.get("tool_name")
                if name:
                    tool_names.append(str(name))
        recent_user = "；".join(user_texts[-5:]) or "无用户文本"
        tools = ", ".join(tool_names[-10:]) or "无"
        return f"历史消息数：{len(items)}；最近用户主题：{recent_user}；已调用工具：{tools}。"

    def compact(self, items: list[Any], summary: str | None = None) -> CompactionResult:
        estimated = self.estimate_tokens(items)
        if estimated <= self.config.max_tokens:
            return CompactionResult(list(items), False, estimated, 0)
        recent = list(items[-self.config.keep_recent_items :])
        summary_text = summary or self.summarize_items(items)
        summary_item = {
            "role": "system",
            "content": f"[PhotoCoach 会话摘要]\n{summary_text}",
        }
        return CompactionResult(
            [summary_item, *recent],
            True,
            estimated,
            max(0, len(items) - len(recent)),
        )


class CompactingSession:
    """包装 Agents SDK Session 的非破坏式压缩适配器。"""

    def __init__(self, base_session: Any, compactor: ContextCompactor, summary_provider):
        self.base_session = base_session
        self.compactor = compactor
        self.summary_provider = summary_provider

    async def get_items(self, limit: int | None = None):
        items = await self.base_session.get_items(limit)
        if limit is not None:
            return items
        summary = self.summary_provider()
        return self.compactor.compact(items, summary).items

    async def add_items(self, items):
        return await self.base_session.add_items(items)

    async def pop_item(self):
        return await self.base_session.pop_item()

    async def clear_session(self):
        return await self.base_session.clear_session()

    async def close(self):
        close = getattr(self.base_session, "close", None)
        if close is not None:
            return close()
