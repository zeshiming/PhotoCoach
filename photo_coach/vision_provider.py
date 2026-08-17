"""智谱 GLM-4V 图片理解 Provider。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_provider import LLMProviderError


@dataclass(frozen=True)
class GLMVisionConfig:
    """GLM 视觉接口的运行时配置。"""

    api_key: str
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    model: str = "glm-4v-flash"
    timeout: float = 60.0
    max_retries: int = 2
    max_tokens: int = 1024

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "GLMVisionConfig":
        """从本地 .env 或环境变量读取 GLM 配置，不输出 Key。"""

        requested_env = env_file or os.getenv("GLM_ENV_FILE")
        local_env = Path.cwd() / ".env"
        try:
            from dotenv import load_dotenv

            if requested_env:
                load_dotenv(requested_env, override=False)
            elif local_env.exists():
                load_dotenv(local_env, override=False)
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMProviderError(
                "python-dotenv is required when loading GLM configuration"
            ) from exc

        api_key = (
            os.getenv("GLM_API_KEY", "").strip()
            or os.getenv("ZAI_API_KEY", "").strip()
        )
        if not api_key:
            raise LLMProviderError("Missing GLM_API_KEY or ZAI_API_KEY")

        return cls(
            api_key=api_key,
            base_url=(
                os.getenv("GLM_BASE_URL", "").strip()
                or cls.base_url
            ),
            model=(
                os.getenv("GLM_VISION_MODEL", "").strip()
                or "glm-4v-flash"
            ),
            timeout=float(os.getenv("GLM_TIMEOUT", "60")),
            max_retries=max(0, int(os.getenv("GLM_MAX_RETRIES", "2"))),
            max_tokens=max(128, int(os.getenv("GLM_MAX_TOKENS", "1024"))),
        )


class GLMVisionProvider:
    """使用智谱 OpenAI-compatible Chat Completions 接口理解图片。"""

    def __init__(self, config: GLMVisionConfig, *, client: Any | None = None):
        self.config = config
        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMProviderError(
                "openai is required; run `pip install openai`"
            ) from exc

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        client: Any | None = None,
    ) -> "GLMVisionProvider":
        return cls(GLMVisionConfig.from_env(env_file), client=client)

    @staticmethod
    def _image_value(image_ref: str) -> str:
        """把远程 URL 或本地图片转换为 GLM 可接收的值。"""

        if image_ref.startswith(("http://", "https://", "data:")):
            return image_ref

        path = Path(image_ref).expanduser()
        if not path.is_file():
            raise LLMProviderError(f"图片不存在：{path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        # 智谱文档支持 image_url 中传入 URL 或 Base64 图片数据。
        # 使用 data URL 形式也兼容 OpenAI-compatible 图片接口。
        return f"data:{mime_type};base64,{encoded}"

    def analyze(self, image_ref: str, prompt: str) -> str:
        """向视觉模型发送一张图片和分析问题，返回文字观察结果。"""

        image_value = self._image_value(image_ref)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_value},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:  # SDK 暴露厂商相关的异常类型
            raise LLMProviderError(f"GLM 视觉请求失败：{exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMProviderError("GLM 响应中没有 message.content") from exc
        if not content or not content.strip():
            raise LLMProviderError("GLM 返回了空的图片分析结果")
        return content.strip()
