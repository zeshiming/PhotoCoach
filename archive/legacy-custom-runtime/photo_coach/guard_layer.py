"""在 LLM 任务理解之前执行的低成本确定性检查。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AgentRequest, Decision, RiskLevel, TaskUnderstanding


@dataclass(frozen=True)
class GuardDecision:
    """确定性输入检查和安全检查的结果。"""

    allowed: bool
    decision: Decision | None = None
    reason: str = ""
    matched_rules: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


class GuardLayer:
    """结构和安全护栏，不是语义意图分类器。

    这一层故意保持低成本和确定性。
    它处理不能依赖 LLM 的情况，正常摄影问题继续进入语义任务理解阶段。
    """

    SENSITIVE_PATTERNS = (
        "真实身份",
        "身份证",
        "几岁",
        "年龄",
        "是不是未成年",
        "人脸识别",
    )
    OUT_OF_SCOPE_PATTERNS = (
        "写简历",
        "写代码",
        "股票",
        "法律咨询",
        "算命",
    )

    def check(self, request: AgentRequest) -> GuardDecision:
        text = request.text.strip()

        # 检查顺序是：输入结构问题 → 图片引用问题 → 安全问题 → 越界问题。
        # 被护栏拦截的请求不会消耗一次 LLM 调用。
        if not text and request.has_image:
            return GuardDecision(
                allowed=False,
                decision=Decision.CLARIFY,
                reason="图片已收到，但用户没有说明希望完成的任务",
                matched_rules=["image_only_requires_task"],
            )
        if not text:
            return GuardDecision(
                allowed=False,
                decision=Decision.CLARIFY,
                reason="输入为空",
                matched_rules=["empty_input"],
            )
        if self._references_missing_image(text, request):
            return GuardDecision(
                allowed=False,
                decision=Decision.CLARIFY,
                reason="用户引用了图片，但当前请求没有图片",
                matched_rules=["missing_referenced_image"],
            )
        if any(pattern in text for pattern in self.SENSITIVE_PATTERNS) and request.has_image:
            return GuardDecision(
                allowed=False,
                decision=Decision.REFUSE,
                reason="不能根据图片推断身份、年龄等敏感个人信息",
                matched_rules=["sensitive_inference_guard"],
                risk_level=RiskLevel.HIGH,
            )
        if any(pattern in text for pattern in self.OUT_OF_SCOPE_PATTERNS):
            return GuardDecision(
                allowed=False,
                decision=Decision.CLARIFY,
                reason="当前 Agent 只处理摄影相关任务",
                matched_rules=["out_of_scope_guard"],
            )
        return GuardDecision(allowed=True)

    @staticmethod
    def _references_missing_image(text: str, request: AgentRequest) -> bool:
        references = ("这张照片", "这张图", "图片里", "照片中", "这张片子")
        return not request.has_image and any(item in text for item in references)

    def blocked_task(self, request: AgentRequest, result: GuardDecision) -> TaskUnderstanding:
        """把被护栏拦截的请求转换成和 LLM 路径相同的数据结构。"""

        # 这样调用方不需要区分结果来自规则还是 LLM，后续层只处理一种结构。
        if result.decision is None:
            raise ValueError("blocked guard decision must include a decision")
        clarification = result.decision == Decision.CLARIFY
        return TaskUnderstanding(
            goal=request.text.strip() or "未提供任务",
            should_clarify=clarification,
            clarification_question=(
                "你希望我具体分析、回答还是制定拍摄方案？" if clarification else None
            ),
            risk_level=result.risk_level,
            decision=result.decision,
            is_oos="out_of_scope_guard" in result.matched_rules,
        )
