"""Experiment embedders and the embedding cache."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vectraudit.experiments.embedders import (
    EmbedderSpec,
    EmbeddingCache,
    HashingEmbedder,
    make_embedder,
)


def cosine(a, b) -> float:
    return float(a @ b)


def test_hashing_embedder_is_lexical_and_deterministic() -> None:
    e = HashingEmbedder(256)
    docs = e.embed_documents(
        [
            "refund policy for annual plans",
            "password reset email",
            "Refund POLICY for annual plans",
            "",
        ]
    )
    assert docs.shape == (4, 256)
    assert docs.dtype == np.float32
    assert np.allclose(docs[0], docs[2])  # case-insensitive
    assert np.isclose(np.linalg.norm(docs[0]), 1.0)
    assert np.linalg.norm(docs[3]) == 0  # empty text: zero vector, no NaN
    q = e.embed_queries(["how do refunds work for annual plans"])[0]
    assert cosine(q, docs[0]) > cosine(q, docs[1])


def test_specs() -> None:
    assert EmbedderSpec(type="hashing", dim=64).label == "hashing-64"
    st = EmbedderSpec(model="intfloat/e5-small-v2", query_prefix="query: ")
    assert st.label == "intfloat/e5-small-v2"
    assert st.key() != EmbedderSpec(model="intfloat/e5-small-v2").key()
    assert (
        st.key()
        == EmbedderSpec(model="intfloat/e5-small-v2", query_prefix="query: ", batch_size=8).key()
    )  # batch size doesn't change vectors
    for kwargs, message in [
        ({"type": "openai"}, "type"),
        ({"model": None}, "model name"),
        ({"type": "hashing", "dim": 2}, "dim"),
    ]:
        with pytest.raises(ValueError, match=message):
            EmbedderSpec(**kwargs)


def test_custom_registry() -> None:
    mine = HashingEmbedder(32)
    assert make_embedder(EmbedderSpec(type="custom", model="mine"), {"mine": mine}) is mine
    with pytest.raises(ValueError, match="no custom embedder"):
        make_embedder(EmbedderSpec(type="custom", model="other"), {"mine": mine})


def test_sentence_transformer_prefixes(monkeypatch) -> None:
    calls = []

    class FakeModel:
        def __init__(self, name):
            calls.append(("load", name))

        def encode(self, texts, **kwargs):
            calls.append(("encode", list(texts), kwargs["normalize_embeddings"]))
            return np.ones((len(texts), 4))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    spec = EmbedderSpec(
        model="m", query_prefix="query: ", document_prefix="passage: ", normalize=False
    )
    e = make_embedder(spec)
    assert e.embed_documents(["a"]).dtype == np.float32
    e.embed_queries(["b"])
    assert calls == [
        ("load", "m"),
        ("encode", ["passage: a"], False),
        ("encode", ["query: b"], False),
    ]


def test_embedding_cache(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path)
    spec = EmbedderSpec(type="hashing", dim=16)
    computed = []

    def compute():
        computed.append(1)
        return np.ones((3, 16))

    first, hit = cache.get_or_compute(spec, "corpus", "words-100", 3, compute)
    assert not hit
    again, hit = cache.get_or_compute(spec, "corpus", "words-100", 3, compute)
    assert hit
    assert np.array_equal(first, again)
    assert len(computed) == 1
    cache.get_or_compute(spec, "corpus", "words-200", 3, compute)  # other chunking
    cache.get_or_compute(
        EmbedderSpec(type="hashing", dim=32), "corpus", "words-100", 3, lambda: np.ones((3, 32))
    )
    assert len(computed) == 2
    custom = EmbedderSpec(type="custom", model="x")
    cache.get_or_compute(custom, "corpus", "words-100", 3, compute)
    cache.get_or_compute(custom, "corpus", "words-100", 3, compute)
    assert len(computed) == 4  # custom embedders are never cached
    assert EmbeddingCache(None).get_or_compute(spec, "c", "w", 3, compute)[1] is False
