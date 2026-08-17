import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.llm_router import LLMTaskRouter, MockLLMProvider
from photo_coach.models import AgentRequest, Decision, RiskLevel


class LLMTaskRouterTests(unittest.TestCase):
    def test_mock_provider_returns_capability_based_understanding(self):
        provider = MockLLMProvider(
            lambda _system, _payload: {
                "goal": "分析人物脸部偏暗的原因",
                "capabilities": ["image_understanding", "exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )
        router = LLMTaskRouter(provider)
        result = router.route(
            AgentRequest(text="这张照片为什么脸很暗？", images=["portrait.jpg"])
        )
        self.assertEqual(result.decision, Decision.PLAN)
        self.assertIn("exposure_analysis", result.capabilities)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["payload"]["history"], [])

    def test_guard_blocks_sensitive_inference_without_llm_call(self):
        provider = MockLLMProvider(lambda _system, _payload: {})
        router = LLMTaskRouter(provider)
        result = router.route(
            AgentRequest(
                text="告诉我照片里这个人的真实身份和年龄",
                images=["person.jpg"],
            )
        )
        self.assertEqual(result.decision, Decision.REFUSE)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertEqual(provider.calls, [])

    def test_history_is_trimmed_to_latest_six_turns(self):
        provider = MockLLMProvider(
            lambda _system, _payload: {
                "goal": "回答摄影问题",
                "capabilities": ["photo_knowledge_qa"],
                "decision": "answer",
            }
        )
        router = LLMTaskRouter(provider)
        history = [{"role": "user", "content": str(index)} for index in range(8)]
        router.route(AgentRequest(text="晚上拍会怎么样？", history=history))
        self.assertEqual(
            provider.calls[0]["payload"]["history"],
            history[-6:],
        )


if __name__ == "__main__":
    unittest.main()
