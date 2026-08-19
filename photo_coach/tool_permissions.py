"""工具权限策略。

权限层与工具实现分离：只读工具默认放行，产生外部副作用的工具要求确认，
高风险工具默认拒绝。OpenAI Agents SDK 会把 ask 映射为可恢复的 approval interruption。
"""

from __future__ import annotations

from enum import Enum


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ToolPermissionPolicy:
    """按工具名给出最小权限决策。"""

    def __init__(self):
        self._rules = {
            "analyze_current_image": PermissionDecision.ALLOW,
            "read_image_metadata": PermissionDecision.ALLOW,
            "search_xhs": PermissionDecision.ALLOW,
            "edit_image": PermissionDecision.ASK,
            "generate_image": PermissionDecision.ASK,
            "upload_external": PermissionDecision.ASK,
            "delete_image": PermissionDecision.ASK,
            "execute_shell": PermissionDecision.DENY,
        }

    def decision(self, tool_name: str) -> PermissionDecision:
        return self._rules.get(tool_name, PermissionDecision.ASK)

    def set_rule(self, tool_name: str, decision: PermissionDecision) -> None:
        self._rules[tool_name] = decision

    def reason(self, tool_name: str) -> str:
        decision = self.decision(tool_name)
        if decision is PermissionDecision.ASK:
            return f"工具 {tool_name} 可能产生外部副作用，需要用户确认。"
        if decision is PermissionDecision.DENY:
            return f"工具 {tool_name} 被当前 PhotoCoach 权限策略禁止。"
        return "工具为只读或分析操作，可以直接执行。"


DEFAULT_TOOL_PERMISSION_POLICY = ToolPermissionPolicy()
