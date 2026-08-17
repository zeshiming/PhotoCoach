import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.agent_loop import AgentLoop
from photo_coach.answer_generator import MockAnswerGenerator
from photo_coach.llm_router import LLMTaskRouter, MockLLMProvider
from photo_coach.models import AgentRequest, TaskUnderstanding
from photo_coach.tool_runtime import MockImageAnalyzer, ToolExecutor


class AgentLoopTests(unittest.TestCase):
    def _loop_for(self, response: dict) -> AgentLoop:
        provider = MockLLMProvider(lambda _system, _payload: response)
        return AgentLoop(LLMTaskRouter(provider))

    def test_answer_branch(self):
        loop = self._loop_for(
            {
                "goal": "回答摄影知识问题",
                "capabilities": ["photo_knowledge_qa"],
                "decision": "answer",
            }
        )

        result = loop.run(AgentRequest(text="怎么拍出电影感？"))

        self.assertEqual(result.status, "answer")

    def test_clarify_branch_from_image_only_guard(self):
        loop = self._loop_for({})

        result = loop.run(AgentRequest(images=["portrait.jpg"]))

        self.assertEqual(result.status, "clarify")
        self.assertIn("具体分析", result.message)

    def test_refuse_branch_from_sensitive_guard(self):
        loop = self._loop_for({})

        result = loop.run(
            AgentRequest(
                text="告诉我照片里这个人的真实身份",
                images=["person.jpg"],
            )
        )

        self.assertEqual(result.status, "refuse")

    def test_pending_tool_branch(self):
        loop = self._loop_for(
            {
                "goal": "分析人物脸部曝光",
                "capabilities": ["image_understanding", "exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )

        result = loop.run(
            AgentRequest(
                text="这张照片为什么脸很暗？",
                images=["portrait.jpg"],
            )
        )

        self.assertEqual(result.status, "pending_tool")
        self.assertIn("exposure_analysis", result.message)

    def test_tool_result_branch(self):
        loop = self._loop_for(
            {
                "goal": "分析人物脸部曝光",
                "capabilities": ["image_understanding", "exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )
        loop.executor = ToolExecutor([MockImageAnalyzer()])

        result = loop.run(
            AgentRequest(
                text="这张照片为什么脸很暗？",
                images=["portrait.jpg"],
            )
        )

        self.assertEqual(result.status, "tool_result")
        self.assertIn("Mock 工具结果", result.message)

    def test_tool_result_is_synthesized_into_final_answer(self):
        loop = self._loop_for(
            {
                "goal": "分析人物脸部曝光",
                "capabilities": ["image_understanding", "exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )
        loop.executor = ToolExecutor([MockImageAnalyzer()])
        loop.answer_generator = MockAnswerGenerator()

        result = loop.run(
            AgentRequest(
                text="这张照片为什么脸很暗？",
                images=["portrait.jpg"],
            )
        )

        self.assertEqual(result.status, "answer")
        self.assertIn("根据当前分析", result.message)

    def test_unknown_capability_returns_unsupported(self):
        loop = self._loop_for(
            {
                "goal": "分析构图",
                "capabilities": ["composition_analysis"],
                "decision": "plan",
            }
        )
        loop.executor = ToolExecutor([MockImageAnalyzer()])

        result = loop.run(AgentRequest(text="分析构图"))

        self.assertEqual(result.status, "unsupported")

    def test_tool_failure_returns_tool_error(self):
        executor = ToolExecutor([MockImageAnalyzer()])
        task = TaskUnderstanding.from_dict(
            {
                "goal": "分析人物脸部曝光",
                "capabilities": ["exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )

        result = executor.execute(AgentRequest(text="分析照片曝光"), task)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "没有提供图片")

    def test_unsupported_branch(self):
        loop = self._loop_for(
            {
                "goal": "生成图片",
                "capabilities": ["image_generation"],
                "decision": "unsupported",
            }
        )

        result = loop.run(AgentRequest(text="生成一张电影感人像"))

        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
