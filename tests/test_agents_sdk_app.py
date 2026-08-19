import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from agents.tool_context import ToolContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.agents_sdk_app import (
    AgentRunResult,
    ImageAnalysisArgs,
    ImageMetadataArgs,
    ImageMetadataResult,
    PhotoAgentContext,
    analyze_current_image,
    build_photo_agent,
    generic_output_guard,
    metadata_tool_enabled,
    metadata_output_guard,
    photo_tool_scope_guard,
    photo_input_guard,
    read_image_metadata,
    make_tool_needs_approval,
    vision_output_guard,
    vision_tool_enabled,
)
from photo_coach.image_session_store import ImageSessionStore
from photo_coach.llm_provider import LLMConfig
from photo_coach.scope_guard import SemanticScopeGuard
from photo_coach.scope_models import ScopeDecision, ScopeResult


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

    def test_dynamic_tool_exposure_uses_selected_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            context = PhotoAgentContext(
                session_id="s1",
                image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                active_image_id="image#1",
                selected_capabilities=["read_metadata"],
            )
            wrapper = type("Wrapper", (), {"context": context})()
            self.assertFalse(vision_tool_enabled(wrapper, None))
            self.assertTrue(metadata_tool_enabled(wrapper, None))

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
        self.assertEqual(len(agent.tools), 2)
        self.assertEqual(agent.tools[0].name, analyze_current_image.name)
        self.assertEqual(agent.tools[1].name, read_image_metadata.name)
        self.assertEqual(len(agent.input_guardrails), 1)
        self.assertEqual(
            agent.input_guardrails[0].get_name(),
            photo_input_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[0].tool_input_guardrails[0].get_name(),
            photo_tool_scope_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[1].tool_input_guardrails[0].get_name(),
            photo_tool_scope_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[1].tool_output_guardrails[0].get_name(),
            generic_output_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[1].tool_output_guardrails[1].get_name(),
            metadata_output_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[0].tool_output_guardrails[0].get_name(),
            generic_output_guard.get_name(),
        )
        self.assertEqual(
            agent.tools[0].tool_output_guardrails[1].get_name(),
            vision_output_guard.get_name(),
        )

    def test_metadata_tool_args_and_enabled_state(self):
        args = ImageMetadataArgs(image_id="image#1")
        self.assertEqual(args.image_id, "image#1")
        self.assertIn("image_id", ImageMetadataArgs.model_json_schema()["properties"])
        with tempfile.TemporaryDirectory() as directory:
            context = PhotoAgentContext(
                session_id="s1",
                image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                active_image_id="image#1",
            )
            wrapper = type("Wrapper", (), {"context": context})()
            self.assertTrue(metadata_tool_enabled(wrapper, None))
            context.metadata_done = True
            self.assertFalse(metadata_tool_enabled(wrapper, None))

    def test_read_only_tools_do_not_need_approval(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                context = PhotoAgentContext(
                    session_id="s1",
                    image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                )
                from agents import RunContextWrapper
                return await make_tool_needs_approval("read_image_metadata")(
                    RunContextWrapper(context),
                    {},
                    "call-1",
                )

        self.assertFalse(asyncio.run(run()))

    def test_metadata_tool_reads_dimensions_and_format(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / "photo.png"
                Image.new("RGB", (7, 5), color=(20, 40, 60)).save(image_path)
                store = ImageSessionStore(Path(directory) / "sessions.db")
                store.register("s1", str(image_path), 1)
                context = PhotoAgentContext(
                    session_id="s1",
                    image_store=store,
                    active_image_id="image#1",
                )
                tool_context = ToolContext(
                    context,
                    tool_name="read_image_metadata",
                    tool_call_id="call-1",
                    tool_arguments=json.dumps({"image_id": "image#1"}),
                )
                implementation = read_image_metadata.on_invoke_tool._get_wrapped_callable()
                return await implementation(
                    tool_context,
                    ImageMetadataArgs(image_id="image#1"),
                )

        result = asyncio.run(run())
        self.assertIsInstance(result, ImageMetadataResult)
        self.assertEqual(result.width, 7)
        self.assertEqual(result.height, 5)
        self.assertEqual(result.format, "PNG")

    def test_metadata_output_guard_rejects_invalid_result(self):
        data = type(
            "Data",
            (),
            {"output": ImageMetadataResult(ok=True, image_id="image#1", width=0, height=5)},
        )()
        result = metadata_output_guard.guardrail_function(data)
        self.assertEqual(result.behavior["type"], "reject_content")

    def test_generic_output_guard_rejects_empty_text(self):
        data = type("Data", (), {"output": ""})()
        result = generic_output_guard.guardrail_function(data)
        self.assertEqual(result.behavior["type"], "reject_content")

    def test_vision_output_guard_rejects_sensitive_claim(self):
        data = type("Data", (), {"output": "这个人的年龄是 25 岁"})()
        result = vision_output_guard.guardrail_function(data)
        self.assertEqual(result.behavior["type"], "reject_content")

    def test_vision_output_guard_allows_safe_refusal(self):
        data = type("Data", (), {"output": "无法判断照片中人物的真实年龄"})()
        result = vision_output_guard.guardrail_function(data)
        self.assertEqual(result.behavior["type"], "allow")

    def _run_tool_guard(self, *, request_text: str, tool_args: dict, scope_guard=None):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                context = PhotoAgentContext(
                    session_id="s1",
                    image_store=ImageSessionStore(Path(directory) / "sessions.db"),
                    request_text=request_text,
                    scope_guard=scope_guard,
                )
                tool_context = type(
                    "ToolContext",
                    (),
                    {
                        "context": context,
                        "tool_name": "analyze_current_image",
                        "tool_arguments": json.dumps(tool_args),
                    },
                )()
                data = type("Data", (), {"context": tool_context})()
                return await photo_tool_scope_guard.guardrail_function(data)

        return asyncio.run(run())

    def test_tool_guard_allows_photography_focus(self):
        result = self._run_tool_guard(
            request_text="请分析这张照片的构图",
            tool_args={"focus": "构图"},
        )
        self.assertEqual(result.behavior["type"], "allow")

    def test_tool_guard_rejects_sensitive_focus(self):
        class FakeScopeGuard(SemanticScopeGuard):
            def __init__(self):
                pass

            async def evaluate(self, text, **kwargs):
                return ScopeResult(
                    decision=ScopeDecision.REFUSE,
                    scope="restricted",
                    risk_level="high",
                    reason_code="sensitive_inference",
                    message="不能根据照片推断人物年龄。",
                )

        result = self._run_tool_guard(
            request_text="请分析这个人",
            tool_args={"focus": "人物年龄"},
            scope_guard=FakeScopeGuard(),
        )
        self.assertEqual(result.behavior["type"], "raise_exception")


if __name__ == "__main__":
    unittest.main()
