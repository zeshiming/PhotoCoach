"""Run the V0 intent-routing evaluation set.

Usage from the project root:
    python3 eval/run_eval.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.intent_router import route_message


DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "eval_cases.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per line and fail clearly on malformed data."""

    cases: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            case = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
        cases.append(case)
    return cases


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one case and return a compact, printable evaluation record."""

    result = route_message(
        text=case.get("text", ""),
        image=case.get("image"),
        history=case.get("history", []),
    )
    actual_intent = result.intent.value
    expected_intent = case["expected_intent"]
    actual_clarify = result.should_clarify
    expected_clarify = case.get("should_clarify", False)

    return {
        "id": case["id"],
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "intent_correct": actual_intent == expected_intent,
        "expected_clarify": expected_clarify,
        "actual_clarify": actual_clarify,
        "clarify_correct": actual_clarify == expected_clarify,
        "confidence": result.confidence,
        "reason": result.reason,
        "matched_rules": result.matched_rules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PhotoCoach intent routing")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="JSONL evaluation case file",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        print("没有找到评测用例")
        return 1

    results = [evaluate_case(case) for case in cases]

    for item in results:
        status = "PASS" if item["intent_correct"] else "FAIL"
        print(
            f'{item["id"]} {status} '
            f'expected={item["expected_intent"]} '
            f'actual={item["actual_intent"]} '
            f'rule={",".join(item["matched_rules"])}'
        )
        if not item["intent_correct"]:
            print(f'  reason: {item["reason"]}')

    total = len(results)
    intent_correct = sum(item["intent_correct"] for item in results)
    clarify_correct = sum(item["clarify_correct"] for item in results)

    print("\nSummary")
    print(f"total={total}")
    print(f"intent_correct={intent_correct}")
    print(f"intent_accuracy={intent_correct / total:.3f}")
    print(f"clarification_accuracy={clarify_correct / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
