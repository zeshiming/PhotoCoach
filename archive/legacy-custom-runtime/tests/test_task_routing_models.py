import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.capability_registry import (
    ToolRegistry,
    build_default_capability_registry,
)
from photo_coach.models import (
    Decision,
    ExecutionPlan,
    PlanAction,
    PlanStep,
    RiskLevel,
    TaskUnderstanding,
    ToolSpec,
)


class TaskUnderstandingTests(unittest.TestCase):
    def test_from_dict_validates_structured_llm_output(self):
        understanding = TaskUnderstanding.from_dict(
            {
                "goal": "分析人物脸部偏暗的原因",
                "capabilities": ["image_understanding", "exposure_analysis"],
                "needs_image": True,
                "decision": "plan",
            }
        )
        understanding.validate(build_default_capability_registry().names())
        self.assertEqual(understanding.decision, Decision.PLAN)
        self.assertTrue(understanding.needs_image)

    def test_clarification_requires_question(self):
        with self.assertRaises(ValueError):
            TaskUnderstanding.from_dict(
                {
                    "goal": "分析照片",
                    "should_clarify": True,
                }
            )

    def test_oos_must_not_be_answered_as_normal_task(self):
        with self.assertRaises(ValueError):
            TaskUnderstanding.from_dict(
                {
                    "goal": "写一份简历",
                    "oos_flag": True,
                    "decision": "answer",
                }
            )

    def test_common_llm_decision_alias_is_normalized(self):
        understanding = TaskUnderstanding.from_dict(
            {
                "goal": "分析照片曝光",
                "capabilities": ["exposure_analysis"],
                "needs_image": True,
                "decision": "proceed",
            }
        )
        self.assertEqual(understanding.decision, Decision.PLAN)

    def test_chinese_risk_level_alias_is_normalized(self):
        understanding = TaskUnderstanding.from_dict(
            {
                "goal": "分析照片",
                "risk_level": "低",
            }
        )
        self.assertEqual(understanding.risk_level, RiskLevel.LOW)


class PlanAndRegistryTests(unittest.TestCase):
    def test_plan_validates_dependencies_and_tool_steps(self):
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id="analyze",
                    action=PlanAction.CALL_TOOL,
                    tool_name="vision_model",
                    capability="image_understanding",
                    description="分析图片",
                ),
                PlanStep(
                    step_id="answer",
                    action=PlanAction.ANSWER,
                    description="根据分析结果回答",
                    depends_on=["analyze"],
                ),
            ]
        )
        plan.validate()

    def test_default_registry_is_capability_oriented(self):
        registry = build_default_capability_registry()
        self.assertIn("image_understanding", registry.names())
        self.assertIn("web_search", registry.names())
        self.assertNotIn("photo_analysis", registry.names())

    def test_tool_registry_filters_by_capability(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="vision_model",
                description="分析图片",
                provides=["image_understanding"],
            )
        )
        self.assertEqual(
            [tool.name for tool in registry.for_capability("image_understanding")],
            ["vision_model"],
        )


if __name__ == "__main__":
    unittest.main()
