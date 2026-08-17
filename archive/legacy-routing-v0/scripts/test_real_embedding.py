"""Smoke test for the real local embedding model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from photo_coach.embedding_router import (
    EmbeddingRouter,
    SentenceTransformerProvider,
)


def main() -> None:
    provider = SentenceTransformerProvider("BAAI/bge-small-zh-v1.5")
    router = EmbeddingRouter(
        provider=provider,
        examples_path=PROJECT_ROOT / "data" / "intent_examples.jsonl",
    )

    queries = [
        ("这张片子的光比不太舒服", True),
        ("周末适合去哪里拍日落", False),
        ("帮我生成一张电影感人像", False),
    ]

    for text, has_image in queries:
        print(f"\nquery: {text}")
        candidates = router.route(text=text, has_image=has_image, top_k=3)
        for candidate in candidates:
            print(
                f"  {candidate.intent.value:<18} "
                f"score={candidate.score:.4f} "
                f"example={candidate.matched_example_id}"
            )


if __name__ == "__main__":
    main()
