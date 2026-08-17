"""PhotoCoach 文本模型配置。

模型调用和 Agent Loop 已由 OpenAI Agents SDK 负责；本文件只保留统一的
OpenAI-compatible 配置读取，供 DeepSeek 等 Chat Completions Provider 使用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class LLMProviderError(RuntimeError):
    """Provider 配置或调用失败时使用的异常类型。"""


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible 文本模型的运行时配置。"""

    api_key: str
    base_url: str | None
    model: str
    timeout: float = 30.0
    max_retries: int = 2
    max_tokens: int = 1024
    disable_thinking: bool = True

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "LLMConfig":
        """从环境变量或本地 .env 读取配置，不输出 API Key。"""

        requested_env_file = env_file or os.getenv("PHOTOCOACH_ENV_FILE") or os.getenv(
            "FRIDGEMATE_ENV_FILE"
        )
        local_env_file = Path.cwd() / ".env"
        try:
            from dotenv import load_dotenv

            if requested_env_file:
                load_dotenv(requested_env_file, override=False)
            elif local_env_file.exists():
                load_dotenv(local_env_file, override=False)
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMProviderError(
                "python-dotenv is required when loading an env file"
            ) from exc

        api_key = (
            os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
        )
        if not api_key:
            auth_file = os.getenv("OPENAI_AUTH_FILE", "").strip()
            if auth_file:
                try:
                    payload = json.loads(
                        Path(auth_file).expanduser().read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise LLMProviderError(
                        f"Unable to read OPENAI_AUTH_FILE: {auth_file}"
                    ) from exc
                api_key = str(payload.get("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise LLMProviderError(
                "Missing OPENAI_API_KEY, LLM_API_KEY, or OPENAI_AUTH_FILE key"
            )

        return cls(
            api_key=api_key,
            base_url=(
                os.getenv("OPENAI_BASE_URL", "").strip()
                or os.getenv("LLM_BASE_URL", "").strip()
                or "https://api.deepseek.com"
            ),
            model=(
                os.getenv("MODEL_NAME", "").strip()
                or os.getenv("LLM_MODEL", "").strip()
                or "deepseek-v4-flash"
            ),
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
            max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", "2"))),
            max_tokens=max(128, int(os.getenv("LLM_MAX_TOKENS", "1024"))),
            disable_thinking=os.getenv(
                "LLM_DISABLE_THINKING", "true"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        )
