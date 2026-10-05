"""Embedders for experiments, and a disk cache for document embeddings.

- ``hashing``: dependency-free lexical baseline (feature-hashed word counts). Useful as a
  floor every semantic model should beat, and deterministic for tests.
- ``sentence-transformers``: any model from that library, with the query and document
  prefixes many retrieval models expect (e.g. ``"query: "`` / ``"passage: "`` for E5).
- ``custom``: an embedder registered from Python (``run_experiment(..., embedders=...)``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
TYPES = ("hashing", "sentence-transformers", "custom")
_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> FloatArray: ...

    def embed_queries(self, texts: Sequence[str]) -> FloatArray: ...


@dataclass(frozen=True)
class EmbedderSpec:
    type: str = "sentence-transformers"
    model: str | None = None
    """sentence-transformers model name, or the registered name for ``custom``."""
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    dim: int = 1024
    """Dimensions of the ``hashing`` embedder."""
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"embedder type must be one of {', '.join(TYPES)}")
        if self.type != "hashing" and not self.model:
            raise ValueError(f"a {self.type} embedder needs a model name")
        if self.dim < 8 or self.batch_size < 1:
            raise ValueError("dim must be >= 8 and batch_size >= 1")

    @property
    def label(self) -> str:
        return f"hashing-{self.dim}" if self.type == "hashing" else str(self.model)

    def key(self) -> str:
        """Identifies the vectors this spec produces (for caching)."""
        relevant = {k: v for k, v in asdict(self).items() if k != "batch_size"}
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


class HashingEmbedder:
    """Signed feature hashing of lowercased word counts (log-scaled), L2-normalised."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, texts: Sequence[str]) -> FloatArray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in _TOKEN.findall(text.lower()):
                counts[token] = counts.get(token, 0) + 1
            for token, n in counts.items():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                sign = 1.0 if h & 1 else -1.0
                out[row, (h >> 1) % self.dim] += sign * (1.0 + math.log(n))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalized: FloatArray = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        return normalized

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> FloatArray:
        return self._embed(texts)


class SentenceTransformerEmbedder:
    def __init__(self, spec: EmbedderSpec) -> None:
        from vectraudit.core.embed import EmbedderUnavailableError

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailableError() from e
        self.spec = spec
        self._model: Any = SentenceTransformer(spec.model)

    def _encode(self, texts: Sequence[str], prefix: str) -> FloatArray:
        vectors = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        return self._encode(texts, self.spec.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> FloatArray:
        return self._encode(texts, self.spec.query_prefix)


def make_embedder(spec: EmbedderSpec, registry: Mapping[str, Embedder] | None = None) -> Embedder:
    if spec.type == "hashing":
        return HashingEmbedder(spec.dim)
    if spec.type == "custom":
        found = (registry or {}).get(str(spec.model))
        if found is None:
            raise ValueError(f"no custom embedder registered as {spec.model!r}")
        return found
    return SentenceTransformerEmbedder(spec)


class EmbeddingCache:
    """Document embeddings on disk, keyed by embedder, corpus and chunking."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _path(self, spec: EmbedderSpec, corpus_sha: str, chunking: str) -> Path | None:
        if self.root is None or spec.type == "custom":
            return None  # custom embedders can change without their name changing
        key = hashlib.sha256(f"{spec.key()}|{corpus_sha}|{chunking}".encode()).hexdigest()[:24]
        return self.root / f"{key}.npy"

    def get_or_compute(
        self,
        spec: EmbedderSpec,
        corpus_sha: str,
        chunking: str,
        n: int,
        compute: Callable[[], FloatArray],
    ) -> tuple[FloatArray, bool]:
        """Vectors and whether they came from the cache."""
        path = self._path(spec, corpus_sha, chunking)
        if path is not None and path.exists():
            vectors = np.load(path)
            if vectors.shape[0] == n:
                return np.ascontiguousarray(vectors, dtype=np.float32), True
        vectors = np.ascontiguousarray(compute(), dtype=np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, vectors)
            tmp.replace(path)
        return vectors, False
