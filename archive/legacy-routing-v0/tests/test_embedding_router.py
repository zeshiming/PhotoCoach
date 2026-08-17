import unittest
from pathlib import Path

from photo_coach.embedding_router import (
    EmbeddingRouter,
    HashEmbeddingProvider,
    cosine_similarity,
    load_intent_examples,
)
from photo_coach.models import Intent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = PROJECT_ROOT / "data" / "intent_examples.jsonl"


class EmbeddingRouterTests(unittest.TestCase):
    def test_loads_five_examples_per_intent(self):
        examples = load_intent_examples(EXAMPLES_PATH)
        self.assertEqual(len(examples), 30)
        self.assertEqual(
            {example.intent for example in examples},
            set(Intent),
        )

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_router_returns_sorted_top_k_candidates(self):
        router = EmbeddingRouter(
            provider=HashEmbeddingProvider(),
            examples_path=EXAMPLES_PATH,
        )
        candidates = router.route("这张照片的构图和光线怎么样？", has_image=True)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            candidates,
            sorted(candidates, key=lambda item: item.score, reverse=True),
        )
        self.assertTrue(all(candidate.matched_example_id for candidate in candidates))

    def test_empty_text_returns_no_candidates(self):
        router = EmbeddingRouter(
            provider=HashEmbeddingProvider(),
            examples_path=EXAMPLES_PATH,
        )
        self.assertEqual(router.route("", has_image=True), [])


if __name__ == "__main__":
    unittest.main()
