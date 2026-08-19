"""Scope Guard 使用的结构化决策模型。

这个模块只描述 Guard 的判断结果，不负责真正执行规则或调用模型。
这样输入护栏、工具护栏和未来的语义 LLM Guard 可以共用同一种结果格式。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScopeDecision(str, Enum):
    """Guard 对当前请求采取的动作。"""

    ALLOW = "allow"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class ScopeResult(BaseModel):
    """一次 Scope 检查的机器可读结果。"""

    decision: ScopeDecision
    scope: str
    risk_level: str
    reason_code: str
    message: str
    capabilities: list[str] = Field(default_factory=list)

    @property
    def should_block(self) -> bool:
        """是否应该阻止 Agent 或工具继续执行。"""

        return self.decision is not ScopeDecision.ALLOW
