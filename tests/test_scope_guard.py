import json
import unittest
from types import SimpleNamespace

from photo_coach.llm_provider import LLMConfig
from photo_coach.scope_guard import SemanticScopeGuard


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload, ensure_ascii=False)
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self, payload):
        self.chat = SimpleNamespace(completions=FakeCompletions(payload))


class SemanticScopeGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_structured_allow_result(self):
        client = FakeClient(
            {
                "decision": "allow",
                "scope": "in_scope",
                "risk_level": "low",
                "reason_code": "photography_question",
                "message": "属于摄影问题",
            }
        )
        guard = SemanticScopeGuard(
            LLMConfig(api_key="test", base_url="https://example.com/v1", model="judge"),
            client=client,
        )

        result = await guard.evaluate("怎么拍出背景虚化？")

        self.assertEqual(result.decision.value, "allow")
        self.assertFalse(result.should_block)
        self.assertEqual(client.chat.completions.calls[0]["temperature"], 0)
        prompt = client.chat.completions.calls[0]["messages"][0]["content"]
        self.assertIn("analyze_image", prompt)

    async def test_passes_tool_context_to_judge(self):
        client = FakeClient(
            {
                "decision": "refuse",
                "scope": "restricted",
                "risk_level": "high",
                "reason_code": "sensitive_inference",
                "message": "不能推断年龄",
            }
        )
        guard = SemanticScopeGuard(
            LLMConfig(api_key="test", base_url="https://example.com/v1", model="judge"),
            client=client,
        )

        result = await guard.evaluate(
            "分析一下这个人",
            context_summary="上一轮用户询问人物年龄",
            tool_name="analyze_current_image",
            tool_arguments={"focus": "人物年龄"},
        )

        self.assertTrue(result.should_block)
        sent = json.loads(client.chat.completions.calls[0]["messages"][1]["content"])
        self.assertEqual(sent["proposed_tool"], "analyze_current_image")
        self.assertEqual(sent["tool_arguments"]["focus"], "人物年龄")


if __name__ == "__main__":
    unittest.main()
