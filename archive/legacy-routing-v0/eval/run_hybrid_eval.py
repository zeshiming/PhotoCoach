"""Evaluate the hybrid rule-first plus real-embedding router."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.embedding_router import EmbeddingRouter, SentenceTransformerProvider
from photo_coach.hybrid_router import HybridRouter
from photo_coach.intent_router import IntentRouter
from photo_coach.models import RouteRequest
from run_eval import load_cases


def main() -> int:
    provider = SentenceTransformerProvider("BAAI/bge-small-zh-v1.5")
    router = HybridRouter(
        rule_router=IntentRouter(),
        embedding_router=EmbeddingRouter(
            provider=provider,
            examples_path=PROJECT_ROOT / "data" / "intent_examples.jsonl",
        ),
    )

    cases = load_cases(PROJECT_ROOT / "eval" / "eval_cases.jsonl")
    correct = 0
    clarify_correct = 0

    for case in cases:
        result = router.route(
            RouteRequest(
                text=case.get("text", ""),
                image=case.get("image"),
                history=case.get("history", []),
            )
        )
        intent_ok = result.intent.value == case["expected_intent"]
        clarify_ok = result.should_clarify == case.get("should_clarify", False)
        correct += int(intent_ok)
        clarify_correct += int(clarify_ok)
        status = "PASS" if intent_ok else "FAIL"
        print(
            f'{case["id"]} {status} expected={case["expected_intent"]} '
            f'actual={result.intent.value} source={result.source}'
        )

    total = len(cases)
    print("\nSummary")
    print(f"total={total}")
    print(f"intent_accuracy={correct / total:.3f}")
    print(f"clarification_accuracy={clarify_correct / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
