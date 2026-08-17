import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.agents_sdk_app import (
    AgentRunResult,
    ImageAnalysisArgs,
    PhotoAgentContext,
    analyze_current_image,
    build_photo_agent,
    photo_input_guard,
    vision_tool_enabled,
)
from photo_coach.image_session_store import ImageSessionStore
from photo_coach.llm_provider import LLMConfig


class AgentsSdkAppTests(unittest.TestCase):
    def test_pydantic_tool_args_are_validated(self):
        args = ImageAnalysisArgs(focus="曝光")
        self.assertEqual(args.focus, "曝光")
        self.assertIn("focus", ImageAnalysisArgs.model_json_schema()["properties"])
        self.assertIn("image_id", ImageAnalysisArgs.model_json_schema()["properties"])

    def test_context_can_hold_vision_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            context = PhotoAgentContext(
                session_id="s1",
                image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                active_image_id=None,
            )
            self.assertEqual(context.session_id, "s1")

    def test_vision_tool_is_disabled_after_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            context = PhotoAgentContext(
                session_id="s1",
                image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                active_image_id="image#1",
                analysis_done=False,
            )
            wrapper = type("Wrapper", (), {"context": context})()
            self.assertTrue(vision_tool_enabled(wrapper, None))
            context.analysis_done = True
            self.assertFalse(vision_tool_enabled(wrapper, None))

    def test_run_result_contains_session_id(self):
        result = AgentRunResult(session_id="s1", final_output="ok", trace_id="t1")
        self.assertEqual(result.session_id, "s1")
        self.assertEqual(result.trace_id, "t1")

    def test_agent_has_sdk_tool(self):
        agent = build_photo_agent(
            LLMConfig(
                api_key="test",
                base_url="https://example.com/v1",
                model="test-model",
            )
        )
        self.assertEqual(agent.name, "PhotoCoach")
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(agent.tools[0].name, analyze_current_image.name)
        self.assertEqual(len(agent.input_guardrails), 1)
        self.assertEqual(
            agent.input_guardrails[0].get_name(),
            photo_input_guard.get_name(),
        )


if __name__ == "__main__":
    unittest.main()
