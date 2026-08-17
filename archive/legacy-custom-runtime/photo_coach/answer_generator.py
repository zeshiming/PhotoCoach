"""根据任务理解和工具 observation 生成最终用户回答。"""

from __future__ import annotations

from typing import Protocol

from .models import AgentRequest, TaskUnderstanding


class TextLLMProvider(Protocol):
    """回答生成器所需的文本 LLM 接口。"""

    def complete_text(self, *, system_prompt: str, payload: dict) -> str:
        """根据输入返回自然语言回答。"""


class LLMAnswerGenerator:
    """使用真实 LLM 将 observation 转换成面向用户的回答。"""

    SYSTEM_PROMPT = """你是 PhotoCoach 的回答生成器。
请根据用户问题、任务理解结果和工具观察结果，用中文回答用户。
要求：
1. 只使用工具观察结果中明确提供的图片事实，不要编造 EXIF 或不可见细节；
2. 区分“观察到的事实”和“摄影建议”；
3. 建议具体、可执行，优先给出不超过 3 条；
4. 如果工具结果是 Mock 或信息不足，要明确说明不确定性；
5. 只返回最终回答，不要输出内部 JSON、路由过程或思考过程。
"""

    def __init__(self, provider: TextLLMProvider):
        self.provider = provider

    def generate(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
        observation: str = "",
    ) -> str:
        """根据当前请求、任务理解和 observation 生成最终文本。"""

        payload = {
            "user_text": request.text,
            "history": request.history[-6:],
            "goal": task.goal,
            "capabilities": task.capabilities,
            "constraints": task.constraints,
            "observation": observation,
        }
        answer = self.provider.complete_text(
            system_prompt=self.SYSTEM_PROMPT,
            payload=payload,
        )
        if not answer.strip():
            raise ValueError("LLM 返回了空回答")
        return answer.strip()


class MockAnswerGenerator:
    """测试和本地开发使用的确定性回答生成器。"""

    def generate(
        self,
        request: AgentRequest,
        task: TaskUnderstanding,
        observation: str = "",
    ) -> str:
        return f"根据当前分析：{observation or '这是一个摄影知识问题。'}"
