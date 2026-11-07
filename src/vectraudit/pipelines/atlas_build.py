"""Build an atlas bundle from the files a workbench already has.

Reads a corpus (JSONL: ``doc_id`` or ``id``, ``text``, ``metadata``, ``vector``), the benchmark
dataset, and optionally the runs, a hubness report, a corpus scan and a poison suite, then
writes ``atlas-<name>.json`` (the sidecar) and ``atlas-<name>.bin`` (the points) beside the
other workbench artifacts. Nothing leaves the machine; the sidecar holds document titles, so it
is marked ``local_only`` and is never part of an upload.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit.core.atlas import (
    DEFAULT_MAX_POINTS,
    Atlas,
    AtlasError,
    AtlasQuestion,
    build_atlas,
)
from vectraudit.core.metadata import TITLE_COLUMNS
from vectraudit.evaluation.dataset import Dataset

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
Embed = Callable[[str], Any]


def read_corpus(
    path: Path, embed: Embed | None = None
) -> tuple[list[str], np.ndarray[Any, Any], list[str | None], str]:
    """``(ids, vectors, titles, sha256)`` of a corpus JSONL; embeds only if ``embed`` is given."""
    ids: list[str] = []
    vectors: list[Any] = []
    titles: list[str | None] = []
    raw = path.read_bytes()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise AtlasError(f"{path.name} line {n} is not JSON.") from e
        doc = row.get("doc_id", row.get("id")) if isinstance(row, dict) else None
        if doc is None:
            raise AtlasError(f"{path.name} line {n} has no doc_id.")
        vector = row.get("vector")
        if vector is None:
            if embed is None or not row.get("text"):
                raise AtlasError(
                    f"Document {doc!r} has no vector. Add vectors to the corpus, or pass "
                    "--embedder with the benchmark's embedding model (needs vectraudit[text])."
                )
            vector = embed(str(row["text"]))
        ids.append(str(doc))
        vectors.append(vector)
        meta = row.get("metadata") or {}
        title = next((str(meta[c]) for c in TITLE_COLUMNS if meta.get(c)), None)
        first = (str(row.get("text") or "").strip().splitlines() or [""])[0]
        titles.append(title or first or None)
    if not ids:
        raise AtlasError(f"{path.name} holds no documents.")
    return ids, np.asarray(vectors, dtype=np.float32), titles, hashlib.sha256(raw).hexdigest()


def questions_from(dataset: Dataset, embed: Embed | None = None) -> list[AtlasQuestion]:
    """The benchmark's questions with their vectors (embedded from text when asked to)."""
    out = []
    for q in dataset.queries:
        vector = q.vector
        if vector is None and embed is not None and q.text:
            vector = embed(q.text)
        relevant = frozenset(d for d, grade in q.relevant.items() if grade > 0) | frozenset(
            s.doc_id for s in q.relevant_spans
        )
        out.append(AtlasQuestion(q.query_id, vector, relevant, q.no_answer, q.split == "heldout"))
    return out


def flagged_documents(
    scan: Mapping[str, Any] | None, suite: Mapping[str, Any] | None
) -> dict[str, str]:
    """Documents the corpus scan or a poison suite names, with why (never the findings' text)."""
    out: dict[str, str] = {}
    for d in (scan or {}).get("documents") or []:
        detectors = sorted({f.get("detector", "?") for f in d.get("findings") or []})
        out[str(d["doc_id"])] = "corpus scan: " + ", ".join(detectors)
    for d in (suite or {}).get("documents") or []:
        out[str(d["doc_id"])] = f"poison suite: {d.get('attack', 'attack')}"
    return out


def hub_documents(report: Mapping[str, Any] | None) -> set[str]:
    return {str(h["doc_id"]) for h in (report or {}).get("hubs") or []}


def write_bundle(atlas: Atlas, directory: Path, name: str) -> tuple[Path, Path]:
    """Write ``atlas-<name>.json`` and ``atlas-<name>.bin``; returns both paths."""
    if not NAME.match(name):
        raise AtlasError("The atlas name may use letters, digits, '.', '_' and '-' (64 at most).")
    directory.mkdir(parents=True, exist_ok=True)
    bin_path = directory / f"atlas-{name}.bin"
    json_path = directory / f"atlas-{name}.json"
    bin_path.write_bytes(atlas.payload())
    atlas.sidecar["bin"]["file"] = bin_path.name
    json_path.write_text(json.dumps(atlas.sidecar, allow_nan=False) + "\n", encoding="utf-8")
    return json_path, bin_path


def build_from_files(
    corpus: Path,
    dataset: Dataset,
    *,
    selected: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
    hubness: Mapping[str, Any] | None = None,
    scan: Mapping[str, Any] | None = None,
    suite: Mapping[str, Any] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    show_heldout: bool = False,
    name: str = "atlas",
    embed: Embed | None = None,
) -> Atlas:
    """Read the corpus, assemble the inputs and build the atlas."""
    ids, vectors, titles, corpus_sha = read_corpus(corpus, embed)
    flagged = flagged_documents(scan, suite)
    known = set(ids)
    atlas = build_atlas(
        ids,
        vectors,
        questions_from(dataset, embed),
        doc_titles=titles,
        selected_run=selected,
        baseline_run=baseline,
        hubs=hub_documents(hubness) & known,
        flagged={d: why for d, why in flagged.items() if d in known},
        max_points=max_points,
        seed=seed,
        show_heldout=show_heldout,
        name=name,
        inputs={
            "corpus_sha256": corpus_sha,
            "dataset_sha256": dataset.sha256,
            "hubness": hubness is not None,
            "corpus_scan": scan is not None,
            "poison_suite": suite is not None,
            "flagged_not_in_corpus": sum(1 for d in flagged if d not in known),
        },
    )
    return atlas
