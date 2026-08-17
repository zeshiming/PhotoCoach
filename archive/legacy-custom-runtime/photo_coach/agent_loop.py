"""PhotoCoach 的最小 Agent Loop。

当前版本先验证控制流，不直接执行真实工具：

    请求 → 任务理解 → 根据 Decision 分支

后续再把 ``pending_tool`` 分支连接到 ToolRegistry、ToolExecutor 和结果
观察/重新规划逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from .answer_generator import LLMAnswerGenerator, MockAnswerGenerator
from .llm_router import LLMTaskRouter
from .models import AgentRequest, Decision, TaskUnderstanding
from .tool_runtime import ToolExecutor


@dataclass(frozen=True)
class AgentResponse:
    """Agent Loop 对外返回的统一响应。"""

    status: str
    message: str
    task: TaskUnderstanding


class AgentLoop:
    """协调任务理解和一次工具执行。"""

    def __init__(
        self,
        router: LLMTaskRouter,
        executor: ToolExecutor | None = None,
        answer_generator: LLMAnswerGenerator | MockAnswerGenerator | None = None,
    ):
        self.router = router
        self.executor = executor
        self.answer_generator = answer_generator

    def run(self, request: AgentRequest) -> AgentResponse:
        # Router 负责护栏和 LLM 任务理解；Loop 只负责根据决策推进流程。
        task = self.router.route(request)

        if task.decision == Decision.CLARIFY:
            return AgentResponse(
                status="clarify",
                message=task.clarification_question or "请补充更多信息。",
                task=task,
            )

        if task.decision == Decision.REFUSE:
            return AgentResponse(
                status="refuse",
                message="这个请求我无法处理。",
                task=task,
            )

        if task.decision == Decision.UNSUPPORTED:
            return AgentResponse(
                status="unsupported",
                message="当前版本暂不支持这个能力。",
                task=task,
            )

        if task.decision == Decision.ANSWER:
            # 纯文字问题没有工具 observation，直接进入回答生成层。
            if self.answer_generator is not None:
                return AgentResponse(
                    status="answer",
                    message=self.answer_generator.generate(request, task),
                    task=task,
                )

            # 没有注入回答生成器时保留占位行为，便于只测试路由分支。
            return AgentResponse(
                status="answer",
                message="当前进入直接回答流程。",
                task=task,
            )

        if task.decision == Decision.PLAN:
            # 没有执行器时保留 pending_tool 状态，方便先测试路由阶段。
            if self.executor is None:
                return AgentResponse(
                    status="pending_tool",
                    message=f"需要执行能力：{', '.join(task.capabilities)}",
                    task=task,
                )

            # 有执行器时，真正完成一次“能力 → 工具 → observation”调用。
            tool_result = self.executor.execute(request, task)
            if tool_result.success:
                if self.answer_generator is not None:
                    return AgentResponse(
                        status="answer",
                        message=self.answer_generator.generate(
                            request,
                            task,
                            tool_result.observation,
                        ),
                        task=task,
                    )

                return AgentResponse(
                    status="tool_result",
                    message=tool_result.observation,
                    task=task,
                )

            if tool_result.error and tool_result.error.startswith("没有工具提供"):
                return AgentResponse(
                    status="unsupported",
                    message=tool_result.error,
                    task=task,
                )

            return AgentResponse(
                status="tool_error",
                message=tool_result.error or "工具执行失败。",
                task=task,
            )

        raise ValueError(f"未知决策：{task.decision}")
