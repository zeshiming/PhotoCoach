"""PhotoCoach Scope 策略配置。

策略与 Guard 执行逻辑分离：调整边界时修改这里，不需要重写 Guard 流程。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopePolicy:
    """一套可注入的摄影 Agent 范围策略。"""

    allowed_scope: str
    out_of_scope_topics: tuple[str, ...]
    sensitive_topics: tuple[str, ...]
    image_reference_markers: tuple[str, ...]

    def prompt_text(self) -> str:
        """生成给语义 Guard 的策略片段。"""

        return (
            f"允许范围：{self.allowed_scope}\n"
            f"拒绝范围：{'、'.join(self.out_of_scope_topics)} 等非摄影任务。\n"
            f"高风险范围：{'、'.join(self.sensitive_topics)} 等敏感个人信息推断。"
        )


DEFAULT_SCOPE_POLICY = ScopePolicy(
    allowed_scope="摄影知识、构图、光线、曝光、后期、拍摄计划、图片分析和摄影推荐",
    out_of_scope_topics=("写简历", "写代码", "股票", "法律咨询", "算命"),
    sensitive_topics=("真实身份", "身份证", "几岁", "年龄", "是不是未成年", "人脸识别"),
    image_reference_markers=("这张照片", "这张图", "图片里", "这张片子"),
)
