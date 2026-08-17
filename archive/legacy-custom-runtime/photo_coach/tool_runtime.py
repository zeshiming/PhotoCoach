"""PhotoCoach 的最小工具运行时。

这里把 LLM 输出的 capability 连接到可执行工具。当前先提供一个确定性的
MockImageAnalyzer，后续可以替换成真实视觉模型，而不改 AgentLoop 接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import AgentRequest, TaskUnderstanding
from .vision_provider import GLMVisionProvider


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的标准结果。"""

    tool_name: str
    success: bool
    observation: str
    error: str | None = None


class Tool(Protocol):
    """AgentLoop 可以调用的最小工具协议。"""

    name: str
    capabilities: list[str]

    def run(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
    ) -> ToolResult:
        """执行工具并返回观察结果。"""


class MockImageAnalyzer:
    """用于打通闭环的确定性图片分析工具。"""

    name = "mock_image_analyzer"
    capabilities = [
        "image_understanding",
        "exposure_analysis",
    ]

    def run(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
    ) -> ToolResult:
        if not request.has_image:
            return ToolResult(
                tool_name=self.name,
                success=False,
                observation="",
                error="没有提供图片",
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            observation=(
                "图片中人物脸部可能存在欠曝、逆光或光比过大的问题；"
                "这是 Mock 工具结果，尚未进行真实像素分析。"
            ),
        )


class GLMImageAnalyzer:
    """使用 GLM-4V-Flash 执行真实的单图摄影分析。"""

    name = "glm_image_analyzer"
    capabilities = [
        "image_understanding",
        "exposure_analysis",
        "composition_analysis",
    ]

    def __init__(self, provider: GLMVisionProvider):
        self.provider = provider

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "GLMImageAnalyzer":
        return cls(GLMVisionProvider.from_env(env_file))

    def run(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
    ) -> ToolResult:
        if not request.has_image:
            return ToolResult(
                tool_name=self.name,
                success=False,
                observation="",
                error="没有提供图片",
            )
        if len(request.images) > 1:
            return ToolResult(
                tool_name=self.name,
                success=False,
                observation="",
                error="GLM-4V-Flash 当前工具只处理一张图片",
            )

        prompt = (
            f"用户目标：{task.goal}\n"
            f"需要能力：{', '.join(task.capabilities)}\n"
            "请只描述图片中可以观察到的摄影事实，并分析曝光、光线和构图问题。"
            "不要猜测无法从图片确认的相机参数。"
        )
        try:
            observation = self.provider.analyze(request.images[0], prompt)
        except LLMProviderError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                observation="",
                error=str(exc),
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            observation=observation,
        )


class ToolExecutor:
    """根据 TaskUnderstanding 的 capabilities 选择并执行工具。"""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """注册一个可执行工具，重名时直接报错。"""

        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def execute(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
    ) -> ToolResult:
        """选择与任务能力最匹配的工具并执行一次。"""

        required = set(task.capabilities)
        candidates = [
            tool
            for tool in self._tools.values()
            if required.intersection(tool.capabilities)
        ]
        if not candidates:
            return ToolResult(
                tool_name="tool_executor",
                success=False,
                observation="",
                error=f"没有工具提供这些能力：{sorted(required)}",
            )

        # 当前采用简单确定性选择：覆盖能力最多的工具优先；后续再加入
        # 成本、延迟、质量和可用性等排序因素。
        tool = max(
            candidates,
            key=lambda item: len(required.intersection(item.capabilities)),
        )
        return tool.run(request, task)
