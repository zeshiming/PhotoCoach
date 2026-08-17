"""Hybrid rule-first plus embedding intent routing."""

from __future__ import annotations

from .embedding_router import EmbeddingCandidate, EmbeddingRouter
from .intent_router import IntentRouter
from .models import Intent, RouteRequest, RouteResult


class HybridRouter:
    """Use deterministic rules first and embeddings for the fallback path."""

    IMAGE_REQUIRED_INTENTS = {Intent.PHOTO_ANALYSIS, Intent.IMAGE_EDITING}

    def __init__(
        self,
        rule_router: IntentRouter,
        embedding_router: EmbeddingRouter,
        min_score: float = 0.50,
        min_margin: float = 0.05,
    ) -> None:
        self.rule_router = rule_router
        self.embedding_router = embedding_router
        self.min_score = min_score
        self.min_margin = min_margin

    def route(self, request: RouteRequest) -> RouteResult:
        rule_result = self.rule_router.route(request)

        # Explicit rules include safety, missing-input, generation, editing,
        # planning and image-analysis decisions. Do not override them.
        if rule_result.matched_rules != ["default_photo_question"]:
            return rule_result

        candidates = self.embedding_router.route(
            text=request.text,
            has_image=bool(request.image),
            top_k=3,
        )
        candidate_payload = [self._candidate_payload(candidate) for candidate in candidates]

        if not candidates:
            return rule_result

        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - second_score

        if top.intent in self.IMAGE_REQUIRED_INTENTS and not request.image:
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=max(0.0, top.score),
                should_clarify=True,
                reason="语义候选需要图片，但当前请求没有提供图片",
                matched_rules=["embedding_requires_image"],
                source="embedding",
                candidates=candidate_payload,
            )

        if top.score < self.min_score or margin < self.min_margin:
            # The current product scope intentionally keeps pure-text
            # photography questions under one stable entry point. Do not
            # over-clarify when no image/tool input is required.
            if not request.image and top.intent not in self.IMAGE_REQUIRED_INTENTS:
                return RouteResult(
                    intent=Intent.PHOTO_QUESTION,
                    confidence=max(0.0, min(1.0, top.score)),
                    should_clarify=False,
                    reason="纯文字摄影问题候选不够明确，回退到统一摄影问答入口",
                    matched_rules=["embedding_fallback_photo_question"],
                    source="embedding",
                    candidates=candidate_payload,
                )
            return RouteResult(
                intent=Intent.CLARIFICATION,
                confidence=max(0.0, top.score),
                should_clarify=True,
                reason="Embedding 候选置信度或候选间隔不足，需要澄清",
                matched_rules=["embedding_low_confidence"],
                source="embedding",
                candidates=candidate_payload,
            )

        return RouteResult(
            intent=top.intent,
            confidence=max(0.0, min(1.0, top.score)),
            should_clarify=top.intent == Intent.CLARIFICATION,
            reason="规则未命中特定意图，由 Embedding 召回候选",
            matched_rules=["embedding_candidate"],
            source="embedding",
            candidates=candidate_payload,
        )

    @staticmethod
    def _candidate_payload(candidate: EmbeddingCandidate) -> dict[str, str | float]:
        return {
            "intent": candidate.intent.value,
            "score": candidate.score,
            "matched_example_id": candidate.matched_example_id,
        }
