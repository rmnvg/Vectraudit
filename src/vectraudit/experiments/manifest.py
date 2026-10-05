"""Experiment manifests (JSON): corpus, queries, k and the candidates to compare.

::

    {"name": "embedding-bakeoff",
     "corpus": {"path": "corpus.jsonl", "id_field": "doc_id", "text_field": "text"},
     "queries": "queries.jsonl",
     "k": 10,
     "candidates": [
       {"name": "lexical", "embedder": {"type": "hashing"},
        "chunking": {"method": "words", "size": 200}},
       {"name": "minilm-200", "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "chunking": {"method": "words", "size": 200, "overlap": 20},
        "index": {"factory": "HNSW32", "params": {"ef_search": 64}}}
     ]}

Paths are relative to the manifest. ``index.factory`` is a FAISS factory string (default
``Flat``: exact search); ``index.metric`` is ``ip`` (default, for normalised embeddings) or
``l2``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.experiments.chunking import ChunkingSpec
from vectraudit.experiments.embedders import EmbedderSpec

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INDEX_PARAMS = ("nprobe", "ef_search")


class ExperimentError(ValueError):
    """The manifest or its inputs are invalid."""


@dataclass(frozen=True)
class IndexSpec:
    factory: str = "Flat"
    metric: str = "ip"
    params: dict[str, int] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return self.factory == "Flat"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    embedder: EmbedderSpec
    chunking: ChunkingSpec
    index: IndexSpec = IndexSpec()

    def describe(self) -> dict[str, Any]:
        e, c, i = self.embedder, self.chunking, self.index
        return {
            "name": self.name,
            "embedder": {
                "type": e.type,
                "model": e.model,
                "label": e.label,
                "normalize": e.normalize,
                "query_prefix": e.query_prefix,
                "document_prefix": e.document_prefix,
            },
            "chunking": {
                "method": c.method,
                "size": c.size,
                "overlap": c.overlap,
                "label": c.label,
            },
            "index": {"factory": i.factory, "metric": i.metric, "params": dict(i.params)},
        }


@dataclass(frozen=True)
class Manifest:
    name: str
    corpus_path: Path
    queries_path: Path
    k: int
    candidates: tuple[CandidateSpec, ...]
    id_field: str = "doc_id"
    text_field: str = "text"


def _obj(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ExperimentError(f"{where} must be an object.")
    return raw


def _known(raw: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = set(raw) - keys
    if unknown:
        raise ExperimentError(
            f"{where}: unknown field(s) {sorted(unknown)}; allowed {sorted(keys)}."
        )


def _candidate(raw: Any, n: int) -> CandidateSpec:
    where = f"candidates[{n}]"
    raw = _obj(raw, where)
    _known(raw, {"name", "embedder", "chunking", "index"}, where)
    name = raw.get("name")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ExperimentError(f"{where}.name must be 1-64 letters, digits, '.', '_' or '-'.")
    try:
        embedder = EmbedderSpec(**_obj(raw.get("embedder", {}), f"{where}.embedder"))
        chunking = ChunkingSpec(**_obj(raw.get("chunking", {}), f"{where}.chunking"))
        index_raw = _obj(raw.get("index", {}), f"{where}.index")
        _known(index_raw, {"factory", "metric", "params"}, f"{where}.index")
        params = _obj(index_raw.get("params", {}), f"{where}.index.params")
        bad = [k for k, v in params.items() if k not in INDEX_PARAMS or type(v) is not int or v < 1]
        if bad:
            raise ExperimentError(
                f"{where}.index.params: {bad} must be positive integers among {INDEX_PARAMS}."
            )
        metric = index_raw.get("metric", "ip")
        if metric not in ("ip", "l2"):
            raise ExperimentError(f"{where}.index.metric must be ip or l2.")
        factory = index_raw.get("factory", "Flat")
        if not isinstance(factory, str) or not factory:
            raise ExperimentError(f"{where}.index.factory must be a FAISS factory string.")
        index = IndexSpec(factory, metric, dict(params))
    except TypeError as e:  # unexpected keyword in embedder/chunking
        raise ExperimentError(f"{where}: {e}") from e
    except ValueError as e:
        if isinstance(e, ExperimentError):
            raise
        raise ExperimentError(f"{where}: {e}") from e
    return CandidateSpec(name, embedder, chunking, index)


def parse_manifest(raw: Any, base_dir: Path = Path()) -> Manifest:
    raw = _obj(raw, "The manifest")
    _known(raw, {"name", "corpus", "queries", "k", "candidates"}, "manifest")
    name = raw.get("name", "experiment")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ExperimentError("name must be 1-64 letters, digits, '.', '_' or '-'.")
    corpus = raw.get("corpus")
    if isinstance(corpus, str):
        corpus = {"path": corpus}
    corpus = _obj(corpus, "corpus")
    _known(corpus, {"path", "id_field", "text_field"}, "corpus")
    queries = raw.get("queries")
    if not isinstance(corpus.get("path"), str) or not isinstance(queries, str):
        raise ExperimentError("corpus.path and queries must be file paths.")
    k = raw.get("k", 10)
    if type(k) is not int or k < 1:
        raise ExperimentError("k must be a positive integer.")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ExperimentError("candidates must be a non-empty list.")
    specs = tuple(_candidate(c, n) for n, c in enumerate(candidates))
    names = [c.name for c in specs]
    if len(set(names)) != len(names):
        raise ExperimentError("candidate names must be unique.")

    def resolve(p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else base_dir / path

    return Manifest(
        name=name,
        corpus_path=resolve(corpus["path"]),
        queries_path=resolve(queries),
        k=k,
        candidates=specs,
        id_field=str(corpus.get("id_field", "doc_id")),
        text_field=str(corpus.get("text_field", "text")),
    )


def load_manifest(path: str | Path) -> Manifest:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ExperimentError(f"Manifest not found: {p}") from e
    except ValueError as e:
        raise ExperimentError(f"{p} is not valid JSON: {e}") from e
    return parse_manifest(raw, p.parent)
