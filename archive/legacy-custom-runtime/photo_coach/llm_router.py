"""带确定性规则护栏的 LLM 任务理解路由器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from .capability_registry import CapabilityRegistry, build_default_capability_registry
from .guard_layer import GuardLayer
from .models import AgentRequest, TaskUnderstanding


class StructuredLLMProvider(Protocol):
    """任务路由器所需的最小 Provider 接口。"""

    def complete_json(self, *, system_prompt: str, payload: dict) -> Mapping:
        """返回一个符合任务 Schema 的 JSON 对象。"""


class MockLLMProvider:
    """用于单元测试和本地开发的确定性 Provider。"""

    def __init__(self, responder: Callable[[str, dict], Mapping]):
        self.responder = responder
        self.calls: list[dict] = []

    def complete_json(self, *, system_prompt: str, payload: dict) -> Mapping:
        self.calls.append({"system_prompt": system_prompt, "payload": payload})
        return self.responder(system_prompt, payload)


class LLMTaskRouter:
    """先执行护栏，再让 LLM 进行基于能力的任务理解。

    当前类只负责任务理解阶段。
    工具选择、执行、结果观察和重新规划属于后续 Agent Loop。
    """

    SYSTEM_PROMPT = """你是 PhotoCoach 的任务理解器。
只理解摄影相关任务，不直接回答用户。
请输出 JSON：goal、capabilities、needs_image、needs_external_search、
should_clarify、clarification_question、risk_level、decision、is_oos。
capabilities 必须来自提供的能力注册表；不要虚构工具名称。
risk_level 必须且只能是 "low"、"medium" 或 "high"，不要输出中文等级。
decision 必须且只能是以下值之一：
"answer"（直接回答）、"clarify"（需要澄清）、"plan"（需要执行能力或工具）、
"refuse"（安全拒绝）、"unsupported"（当前能力未实现）。
不要使用 proceed、continue、execute 等其他值。
"""

    def __init__(
        self,
        provider: StructuredLLMProvider,
        *,
        capability_registry: CapabilityRegistry | None = None,
        guard: GuardLayer | None = None,
    ):
        self.provider = provider
        self.capability_registry = capability_registry or build_default_capability_registry()
        self.guard = guard or GuardLayer()

    def route(self, request: AgentRequest) -> TaskUnderstanding:
        # 1. 在任何付费或远程模型调用之前，先执行确定性检查。
        guard_result = self.guard.check(request)
        if not guard_result.allowed:
            return self.guard.blocked_task(request, guard_result)

        # 2. 注册表同时承担两种职责：
        #    - 作为上下文，把可用能力告诉 LLM；
        #    - 作为白名单，校验 LLM 返回的能力。
        payload = {
            "text": request.text,
            "images": request.images,
            "has_image": request.has_image,
            "history": request.history[-6:],
            "capabilities": [
                {
                    "name": capability.name,
                    "description": capability.description,
                    "requires_image": capability.requires_image,
                    "retrieval_text": capability.retrieval_text,
                }
                for capability in self.capability_registry.list()
            ],
        }
        raw = self.provider.complete_json(
            system_prompt=self.SYSTEM_PROMPT,
            payload=payload,
        )

        # 3. 把不可信的 JSON 转成领域模型，
        #    并在返回运行时之前校验能力和图片契约。
        understanding = TaskUnderstanding.from_dict(dict(raw))
        understanding.validate(self.capability_registry.names())
        if understanding.needs_image and not request.has_image:
            raise ValueError("task requires an image but request has no image")
        return understanding
