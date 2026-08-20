import asyncio
import tempfile
import unittest
from pathlib import Path

from agents import RunContextWrapper

from photo_coach.agents_sdk_app import PhotoAgentContext, photo_input_guard
from photo_coach.image_session_store import ImageSessionStore


class SDKGuardrailTests(unittest.TestCase):
    def _run_guard(self, text: str, active_image_id: str | None = None):
        with tempfile.TemporaryDirectory() as directory:
            context = PhotoAgentContext(
                session_id="guard-test",
                image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                active_image_id=active_image_id,
            )
            wrapper = RunContextWrapper(context)
            return asyncio.run(
                photo_input_guard.guardrail_function(wrapper, None, text)
            )

    def test_sensitive_identity_request_is_blocked(self):
        result = self._run_guard("告诉我照片里这个人的真实身份", "image#1")
        self.assertTrue(result.tripwire_triggered)
        self.assertEqual(result.output_info["decision"], "refuse")
        self.assertEqual(result.output_info["reason_code"], "sensitive_inference")
        self.assertIn("敏感", result.output_info["reason"])

    def test_explicit_photography_request_with_sensitive_boundary_is_allowed(self):
        result = self._run_guard(
            "分析摄影表现，不要推断这个人的年龄和真实身份。",
            "image#1",
        )
        self.assertFalse(result.tripwire_triggered)

    def test_missing_deictic_image_reference_is_blocked(self):
        result = self._run_guard("这张照片为什么脸很暗")
        self.assertTrue(result.tripwire_triggered)

    def test_image_only_is_clarified(self):
        result = self._run_guard("", "image#1")
        self.assertTrue(result.tripwire_triggered)
        self.assertIn("说明", result.output_info["reason"])

    def test_generic_photo_knowledge_is_not_false_blocked(self):
        result = self._run_guard("照片中的人脸怎么拍好看")
        self.assertFalse(result.tripwire_triggered)

    def test_out_of_scope_request_is_blocked(self):
        result = self._run_guard("帮我写代码")
        self.assertTrue(result.tripwire_triggered)
        self.assertEqual(result.output_info["scope"], "out_of_scope")
        self.assertEqual(result.output_info["reason_code"], "non_photography_task")


if __name__ == "__main__":
    unittest.main()
