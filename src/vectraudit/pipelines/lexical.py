"""Inspectable BM25/vector hybrid baseline with reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from vectraudit.pipelines.models import Hit, PipelineRequest
from vectraudit.pipelines.runtime import PythonPipeline, Stage


def hybrid_pipeline(documents: list[dict[str, Any]]) -> PythonPipeline:
    """Build a local baseline. Optional vectors must share the query embedding space.

    ``tenant`` in query context restricts matching document metadata. Customers should
    use their own adapter when auditing their actual authorization implementation.
    """
    ids = [str(d["doc_id"]) for d in documents]
    if not documents or len(set(ids)) != len(ids):
        raise ValueError("corpus must have documents with unique doc_id values")
    tokens = [Counter(re.findall(r"\w+", d["text"].casefold())) for d in documents]
    lengths = [sum(t.values()) for t in tokens]
    average = sum(lengths) / len(lengths) or 1
    df = Counter(word for words in tokens for word in words)
    by_id = dict(zip(ids, documents, strict=True))

    def keyword(q: PipelineRequest, parents: dict[str, list[Hit]]) -> list[Hit]:
        terms = set(re.findall(r"\w+", (q.text or "").casefold()))
        scores = []
        for doc, words, length in zip(ids, tokens, lengths, strict=True):
            score = sum(
                math.log(1 + (len(ids) - df[t] + 0.5) / (df[t] + 0.5))
                * words[t]
                * 2.2
                / (words[t] + 1.2 * (0.25 + 0.75 * length / average))
                for t in terms
                if words[t]
            )
            if score > 0:
                scores.append(Hit(id=doc, score=score))
        return sorted(scores, key=lambda h: (-(h.score or 0), h.id))[: _depth(q)]

    def vector(q: PipelineRequest, parents: dict[str, list[Hit]]) -> list[Hit]:
        weight = _weight(q)
        if weight == 0:
            return []
        if q.vector is None or any("vector" not in d for d in documents):
            raise ValueError("hybrid vector search requires corpus and query vectors")
        matrix = np.asarray([d["vector"] for d in documents], dtype=float)
        query = np.asarray(q.vector, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(query) or not np.isfinite(matrix).all():
            raise ValueError("corpus vectors must be finite and match query dimensions")
        scores = (
            matrix
            @ query
            / np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(query), 1e-12)
        )
        order = np.argsort(-scores, kind="stable")[: _depth(q)]
        return [Hit(id=ids[i], score=float(scores[i])) for i in order]

    def fusion(q: PipelineRequest, parents: dict[str, list[Hit]]) -> list[Hit]:
        scores: dict[str, float] = {}
        weight = _weight(q)
        for name, w in (("keyword", 1 - weight), ("vector", weight)):
            if not w:
                continue
            for rank, hit in enumerate(parents[name], 1):
                scores[hit.id] = scores.get(hit.id, 0) + w / (60 + rank)
        return [
            Hit(id=doc, score=score)
            for doc, score in sorted(scores.items(), key=lambda t: (-t[1], t[0]))
        ]

    def filtering(q: PipelineRequest, parents: dict[str, list[Hit]]) -> list[Hit]:
        required = dict(q.filters)
        if "tenant" in q.context:
            required["tenant"] = q.context["tenant"]
        return [
            h
            for h in parents["fusion"]
            if all(
                by_id[h.id].get("metadata", {}).get(key) == value for key, value in required.items()
            )
        ]

    def rerank(q: PipelineRequest, parents: dict[str, list[Hit]]) -> list[Hit]:
        hits = parents["filter"]
        mode = q.params.get("reranker", "none")
        if mode == "none":
            return hits
        if mode != "phrase":
            raise ValueError("baseline reranker must be none or phrase")
        phrase = (q.text or "").strip().casefold()
        return sorted(
            hits, key=lambda h: -(int(bool(phrase) and phrase in by_id[h.id]["text"].casefold()))
        )

    return PythonPipeline(
        [
            Stage("keyword", "keyword", keyword),
            Stage("vector", "vector", vector),
            Stage("fusion", "fusion", fusion, ("keyword", "vector")),
            Stage("filter", "filter", filtering, ("fusion",)),
            Stage("rerank", "rerank", rerank, ("filter",)),
            Stage("final", "truncate", lambda q, p: p["rerank"][: q.k], ("rerank",)),
        ]
    )


def _depth(q: PipelineRequest) -> int:
    value = q.params.get("depth", max(q.k, 50))
    if type(value) is not int or not q.k <= value <= 10000:
        raise ValueError("depth must be an integer between k and 10000")
    return value


def _weight(q: PipelineRequest) -> float:
    value = q.params.get("vector_weight", 0)
    if type(value) not in {int, float} or not 0 <= value <= 1:
        raise ValueError("vector_weight must be in [0, 1]")
    return float(value)
