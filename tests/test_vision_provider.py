import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.llm_provider import LLMProviderError
from photo_coach.vision_provider import GLMVisionConfig, GLMVisionProvider


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="人物脸部可能存在欠曝。")
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class GLMVisionProviderTests(unittest.TestCase):
    def test_sends_image_url_and_text(self):
        client = FakeClient()
        provider = GLMVisionProvider(
            GLMVisionConfig(api_key="test", model="glm-4v-flash"),
            client=client,
        )

        result = provider.analyze(
            "https://example.com/photo.jpg",
            "分析曝光",
        )

        self.assertIn("欠曝", result)
        content = client.completions.kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["url"], "https://example.com/photo.jpg")
        self.assertEqual(content[1], {"type": "text", "text": "分析曝光"})

    def test_missing_local_image_is_rejected(self):
        provider = GLMVisionProvider(
            GLMVisionConfig(api_key="test"),
            client=FakeClient(),
        )

        with self.assertRaises(LLMProviderError):
            provider.analyze("/tmp/photo-coach-not-found.jpg", "分析图片")


if __name__ == "__main__":
    unittest.main()
