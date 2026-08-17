"""Embedding-based intent candidate retrieval.

The router only recalls semantic candidates. It does not call tools and does
not replace the deterministic safety/input rules in ``IntentRouter``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .models import Intent


class EmbeddingProvider(Protocol):
    """Minimal interface that any embedding backend must implement."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each input text."""


@dataclass(frozen=True)
class IntentExample:
    id: str
    intent: Intent
    text: str
    has_image: bool


@dataclass(frozen=True)
class EmbeddingCandidate:
    intent: Intent
    score: float
    matched_example_id: str


def load_intent_examples(path: str | Path) -> list[IntentExample]:
    """Load and validate one intent example per JSONL line."""

    examples: list[IntentExample] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
            examples.append(
                IntentExample(
                    id=str(data["id"]),
                    intent=Intent(data["intent"]),
                    text=str(data["text"]),
                    has_image=bool(data["has_image"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"意图样本第 {line_number} 行格式错误: {exc}"
            ) from exc
    if not examples:
        raise ValueError("意图样本文件为空")
    return examples


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity for two vectors."""

    if len(left) != len(right):
        raise ValueError("两个向量的维度必须一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


class HashEmbeddingProvider:
    """Offline lexical vectorizer for smoke tests only.

    This is not a production semantic model. It lets the project run without
    downloading a model and validates the provider/router contract.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须大于 0")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", "", text.lower())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector

        # Character n-grams work for Chinese text without tokenization.
        grams = [normalized]
        grams.extend(normalized[index : index + 2] for index in range(len(normalized) - 1))
        grams.extend(normalized[index : index + 3] for index in range(len(normalized) - 2))
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerProvider:
    """Adapter for a real local sentence-transformers model.

    Install the optional dependency separately, then pass a multilingual or
    Chinese embedding model name when constructing this provider.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，请先安装后再使用真实 Embedding 模型"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, vector)) for vector in vectors]


class EmbeddingRouter:
    """Retrieve top-k intent candidates from example embeddings."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        examples_path: str | Path,
        image_mismatch_penalty: float = 0.05,
    ) -> None:
        self.provider = provider
        self.examples = load_intent_examples(examples_path)
        self.image_mismatch_penalty = image_mismatch_penalty
        vectors = provider.embed([example.text for example in self.examples])
        if len(vectors) != len(self.examples):
            raise ValueError("Embedding 返回的向量数量与样本数量不一致")
        self._example_vectors = vectors

    def route(
        self,
        text: str,
        has_image: bool = False,
        top_k: int = 3,
    ) -> list[EmbeddingCandidate]:
        """Return the best intent candidate for each intent, sorted by score."""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not text.strip():
            return []

        query_vector = self.provider.embed([text])[0]
        best_by_intent: dict[Intent, EmbeddingCandidate] = {}

        for example, example_vector in zip(self.examples, self._example_vectors):
            score = cosine_similarity(query_vector, example_vector)
            if example.has_image != has_image:
                score -= self.image_mismatch_penalty
            candidate = EmbeddingCandidate(
                intent=example.intent,
                score=score,
                matched_example_id=example.id,
            )
            previous = best_by_intent.get(example.intent)
            if previous is None or candidate.score > previous.score:
                best_by_intent[example.intent] = candidate

        return sorted(
            best_by_intent.values(),
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:top_k]
