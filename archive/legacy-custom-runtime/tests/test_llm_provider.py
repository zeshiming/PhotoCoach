import json
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.llm_provider import LLMConfig, OpenAICompatibleJSONProvider


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content: str):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class OpenAICompatibleJSONProviderTests(unittest.TestCase):
    def test_calls_json_mode_and_parses_object(self):
        client = FakeClient('{"goal": "分析曝光"}')
        provider = OpenAICompatibleJSONProvider(
            LLMConfig(api_key="test", base_url="https://proxy.test", model="test-model"),
            client=client,
        )

        result = provider.complete_json(
            system_prompt="输出 JSON",
            payload={"text": "脸很暗"},
        )

        self.assertEqual(result, {"goal": "分析曝光"})
        self.assertEqual(client.completions.kwargs["model"], "test-model")
        self.assertEqual(
            client.completions.kwargs["response_format"], {"type": "json_object"}
        )
        self.assertEqual(
            json.loads(client.completions.kwargs["messages"][1]["content"]),
            {"text": "脸很暗"},
        )

    def test_generates_text_without_json_mode(self):
        client = FakeClient("这是最终回答")
        provider = OpenAICompatibleJSONProvider(
            LLMConfig(api_key="test", base_url="https://proxy.test", model="test-model"),
            client=client,
        )

        result = provider.complete_text(
            system_prompt="生成回答",
            payload={"observation": "人物脸部欠曝"},
        )

        self.assertEqual(result, "这是最终回答")
        self.assertNotIn("response_format", client.completions.kwargs)

    def test_disables_thinking_for_deepseek_structured_requests(self):
        client = FakeClient('{"goal": "分析曝光"}')
        provider = OpenAICompatibleJSONProvider(
            LLMConfig(
                api_key="test",
                base_url="https://api.deepseek.com",
                model="deepseek-v4-flash",
            ),
            client=client,
        )

        provider.complete_json(
            system_prompt="输出 JSON",
            payload={"text": "脸很暗"},
        )

        self.assertEqual(
            client.completions.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()
