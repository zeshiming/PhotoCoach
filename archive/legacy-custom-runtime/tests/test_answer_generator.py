import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.answer_generator import LLMAnswerGenerator
from photo_coach.models import AgentRequest, TaskUnderstanding


class FakeTextProvider:
    def __init__(self):
        self.calls = []

    def complete_text(self, *, system_prompt: str, payload: dict) -> str:
        self.calls.append({"system_prompt": system_prompt, "payload": payload})
        return "建议先补光，再降低背景亮度。"


class AnswerGeneratorTests(unittest.TestCase):
    def test_passes_task_and_observation_to_provider(self):
        provider = FakeTextProvider()
        generator = LLMAnswerGenerator(provider)
        task = TaskUnderstanding.from_dict(
            {
                "goal": "分析曝光",
                "capabilities": ["exposure_analysis"],
                "decision": "plan",
            }
        )

        answer = generator.generate(
            AgentRequest(text="为什么脸很暗？"),
            task,
            "人物脸部可能欠曝。",
        )

        self.assertEqual(answer, "建议先补光，再降低背景亮度。")
        self.assertEqual(provider.calls[0]["payload"]["observation"], "人物脸部可能欠曝。")

    def test_empty_provider_answer_is_rejected(self):
        class EmptyProvider(FakeTextProvider):
            def complete_text(self, *, system_prompt: str, payload: dict) -> str:
                return "  "

        generator = LLMAnswerGenerator(EmptyProvider())
        task = TaskUnderstanding.from_dict({"goal": "回答摄影问题"})

        with self.assertRaises(ValueError):
            generator.generate(AgentRequest(text="怎么拍照？"), task)


if __name__ == "__main__":
    unittest.main()
