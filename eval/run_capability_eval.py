"""评测 Scope Guard 的 Capability 选择和动态 Tool 映射。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.capability_registry import DEFAULT_CAPABILITY_REGISTRY
from photo_coach.llm_provider import LLMConfig
from photo_coach.scope_guard import ScopeGuardError, SemanticScopeGuard
from photo_coach.scope_models import ScopeDecision, ScopeResult


CASES_PATH = PROJECT_ROOT / "eval" / "capability_eval.jsonl"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MockCapabilityGuard:
    """离线契约测试，不代表真实模型的能力选择质量。"""

    async def evaluate(self, text: str, *, context_summary: str = "", **kwargs: Any) -> ScopeResult:
        value = text.lower()
        if any(item in value for item in ("写 python", "写代码")):
            return ScopeResult(
                decision="refuse", scope="out_of_scope", risk_level="low",
                reason_code="non_photography_task", message="只处理摄影问题。"
            )
        if "几岁" in value or "年龄" in value:
            return ScopeResult(
                decision="refuse", scope="restricted", risk_level="high",
                reason_code="sensitive_inference", message="不能推断年龄。"
            )
        if "那构图" in value and not context_summary:
            return ScopeResult(
                decision="clarify", scope="ambiguous", risk_level="low",
                reason_code="missing_context", message="请补充上下文。"
            )
        if any(item in value for item in ("iso", "快门", "光圈")):
            caps = ["read_metadata"]
        elif any(item in value for item in ("构图", "曝光", "光线", "照片", "图片", "比较")):
            caps = ["analyze_image"]
        elif any(item in value for item in ("计划", "街拍")):
            caps = ["shoot_planning"]
        elif any(item in value for item in ("推荐", "镜头")):
            caps = ["photo_recommendation"]
        else:
            caps = ["explain_photography"]
        return ScopeResult(
            decision="allow", scope="in_scope", risk_level="low",
            reason_code="mock_capability", message="属于摄影范围。", capabilities=caps
        )


async def evaluate_case(case: dict[str, Any], guard: Any) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await guard.evaluate(
            case["text"],
            context_summary=case.get("context_summary", ""),
        )
        actual_decision = result.decision.value
        actual_capabilities = DEFAULT_CAPABILITY_REGISTRY.validate(result.capabilities)
        actual_tools = sorted(
            DEFAULT_CAPABILITY_REGISTRY.tool_names_for(actual_capabilities)
        )
        error = None
    except ScopeGuardError as exc:
        result = None
        actual_decision = "error"
        actual_capabilities = []
        actual_tools = []
        error = str(exc)

    expected_capabilities = case.get("expected_capabilities", [])
    expected_tools = sorted(case.get("expected_tools", []))
    return {
        "id": case["id"],
        "expected_decision": case["expected_decision"],
        "actual_decision": actual_decision,
        "decision_passed": actual_decision == case["expected_decision"],
        "expected_capabilities": expected_capabilities,
        "actual_capabilities": actual_capabilities,
        "capability_passed": actual_capabilities == expected_capabilities,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "tool_exposure_passed": actual_tools == expected_tools,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "error": error,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    decision_passed = sum(item["decision_passed"] for item in results)
    capability_passed = sum(item["capability_passed"] for item in results)
    tool_passed = sum(item["tool_exposure_passed"] for item in results)
    return {
        "total": total,
        "decision_accuracy": decision_passed / total if total else 0.0,
        "capability_accuracy": capability_passed / total if total else 0.0,
        "tool_exposure_accuracy": tool_passed / total if total else 0.0,
        "avg_latency_ms": sum(item["duration_ms"] for item in results) / total if total else 0.0,
        "all_checks_passed": decision_passed == total and capability_passed == total and tool_passed == total,
    }


async def run(mode: str) -> dict[str, Any]:
    guard: Any = MockCapabilityGuard()
    if mode == "real":
        guard = SemanticScopeGuard(LLMConfig.from_env())
    results = [await evaluate_case(case, guard) for case in load_cases()]
    return {"mode": mode, "summary": summarize(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run(args.mode))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["summary"]["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
