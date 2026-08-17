import unittest

from photo_coach.hybrid_router import HybridRouter
from photo_coach.intent_router import IntentRouter
from photo_coach.embedding_router import EmbeddingCandidate
from photo_coach.models import Intent, RouteRequest


class StubEmbeddingRouter:
    def __init__(self, candidates):
        self.candidates = candidates
        self.called = False

    def route(self, text, has_image, top_k):
        self.called = True
        return self.candidates[:top_k]


class HybridRouterTests(unittest.TestCase):
    def test_explicit_rule_does_not_call_embedding(self):
        embedding = StubEmbeddingRouter([])
        router = HybridRouter(IntentRouter(), embedding)

        result = router.route(RouteRequest(text="帮我生成一张人像"))

        self.assertEqual(result.intent, Intent.IMAGE_GENERATION)
        self.assertFalse(embedding.called)

    def test_embedding_handles_rule_fallback(self):
        embedding = StubEmbeddingRouter(
            [
                EmbeddingCandidate(Intent.PHOTO_ANALYSIS, 0.82, "analysis-1"),
                EmbeddingCandidate(Intent.PHOTO_QUESTION, 0.60, "question-1"),
            ]
        )
        router = HybridRouter(IntentRouter(), embedding)

        result = router.route(RouteRequest(text="这张片子的光比不太舒服", image="x.jpg"))

        self.assertEqual(result.intent, Intent.PHOTO_ANALYSIS)
        self.assertEqual(result.source, "embedding")
        self.assertTrue(embedding.called)

    def test_low_margin_requires_clarification(self):
        embedding = StubEmbeddingRouter(
            [
                EmbeddingCandidate(Intent.PHOTO_QUESTION, 0.61, "question-1"),
                EmbeddingCandidate(Intent.SHOOT_PLANNING, 0.59, "planning-1"),
            ]
        )
        router = HybridRouter(IntentRouter(), embedding, min_margin=0.05)

        result = router.route(RouteRequest(text="帮我看看怎么拍", image="x.jpg"))

        self.assertEqual(result.intent, Intent.CLARIFICATION)
        self.assertTrue(result.should_clarify)

    def test_image_required_candidate_without_image_requires_clarification(self):
        embedding = StubEmbeddingRouter(
            [EmbeddingCandidate(Intent.PHOTO_ANALYSIS, 0.80, "analysis-1")]
        )
        router = HybridRouter(IntentRouter(), embedding)

        result = router.route(RouteRequest(text="这张片子的光比不太舒服"))

        self.assertEqual(result.intent, Intent.CLARIFICATION)
        self.assertTrue(result.should_clarify)


if __name__ == "__main__":
    unittest.main()
