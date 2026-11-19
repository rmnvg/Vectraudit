"""Source-to-context lineage: where a required passage was first lost.

A benchmark labels required passages as character spans of source documents
(``relevant_spans``). A user-exported **lineage manifest** says what their ingestion
did with each source: its current revision, how much text extraction produced, and the
character range, index state and indexed revision of every chunk. One JSON object per line::

    {"source_id": "refund-policy", "revision": "r8", "extracted_chars": 1840,
     "chunks": [{"chunk_id": "refund-policy#0", "start": 0, "end": 900,
                 "indexed_revision": "r8"},
                {"chunk_id": "refund-policy#1", "start": 900, "end": 1840,
                 "indexed_revision": "r8", "text_sha256": "…"}]}

Each passage is followed through the stages in order, and the first one that loses it is
named: the source is missing from the export; extraction produced no text or stopped before
the passage; the passage is in no chunk, or only across a chunk boundary; its chunk was not
indexed, or was indexed from an older revision; and, given a run, the chunk was not
retrieved, was dropped before delivery, or did not reach the model. Losses before retrieval
are ingestion problems: changing the embedding model will not recover them.

The manifest describes the user's pipeline; Vectraudit does not re-run ingestion or read
document text. Findings are only as complete as the export: a stage the export does not
describe is reported as unknown, never as passed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vectraudit.evaluation.dataset import Dataset

MAX_SOURCES = 200_000
STAGES = (
    "source_missing",
    "empty_extraction",
    "beyond_extraction",
    "not_chunked",
    "split_across_chunks",
    "not_indexed",
    "stale_index",
    "not_retrieved",
    "lost_downstream",
    "lost_in_context",
)
"""First-loss classes in path order. Everything up to ``stale_index`` is ingestion."""
INGESTION = frozenset(STAGES[:7])
EXPLAIN = {
    "source_missing": "The source is not in the lineage export: it was never ingested, or the "
    "export is incomplete.",
    "empty_extraction": "Text extraction produced nothing for this source (a scanned PDF, an "
    "unsupported format, or a parser failure).",
    "beyond_extraction": "Extraction stopped before the passage: it is past the end of the "
    "extracted text (a dropped table, a truncated page or a size limit).",
    "not_chunked": "Part of the passage is in no chunk: the chunker skipped that text.",
    "split_across_chunks": "No single chunk holds the whole passage; it is split across a "
    "chunk boundary, so no retrieved chunk can carry all of it.",
    "not_indexed": "The chunk holding the passage is not in the index.",
    "stale_index": "The chunk was indexed from an older revision of the source, so the "
    "current text is not what retrieval searches.",
    "not_retrieved": "The chunk holding the passage was indexed but appears in no recorded "
    "retrieval stage.",
    "lost_downstream": "The chunk was retrieved but dropped before delivery.",
    "lost_in_context": "The chunk was delivered but did not reach the model intact.",
}


class LineageError(ValueError):
    """The lineage manifest cannot be used; the message says why."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChunkRecord(_Model):
    chunk_id: str = Field(min_length=1, max_length=512)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    indexed: bool = True
    indexed_revision: str | None = Field(default=None, min_length=1, max_length=256)
    text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def ordered(self) -> ChunkRecord:
        if self.start >= self.end:
            raise ValueError(f"chunk {self.chunk_id}: start must be before end")
        return self


class SourceRecord(_Model):
    source_id: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, min_length=1, max_length=256)
    extracted_chars: int | None = Field(default=None, ge=0)
    """Characters of extracted text; spans and chunk offsets refer to that text."""
    chunks: list[ChunkRecord] = Field(default_factory=list, max_length=100_000)

    @model_validator(mode="after")
    def within(self) -> SourceRecord:
        ids = [c.chunk_id for c in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"source {self.source_id}: chunk IDs repeat")
        if self.extracted_chars is not None and any(
            c.end > self.extracted_chars for c in self.chunks
        ):
            raise ValueError(f"source {self.source_id}: a chunk ends past the extracted text")
        return self


def read_manifest(path: str | Path) -> dict[str, SourceRecord]:
    """Sources by ID from a JSONL lineage manifest; chunk IDs must be unique overall."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LineageError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    sources: dict[str, SourceRecord] = {}
    owners: dict[str, str] = {}
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = SourceRecord.model_validate(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise LineageError(f"line {line}: not JSON ({exc.msg})") from exc
        except ValidationError as exc:
            problem = exc.errors()[0]
            where = ".".join(str(p) for p in problem["loc"])
            raise LineageError(f"line {line}: {where}: {problem['msg']}") from None
        if record.source_id in sources:
            raise LineageError(f"line {line}: source {record.source_id} appears twice")
        for chunk in record.chunks:
            if chunk.chunk_id in owners:
                raise LineageError(
                    f"line {line}: chunk {chunk.chunk_id} also belongs to {owners[chunk.chunk_id]}"
                )
            owners[chunk.chunk_id] = record.source_id
        sources[record.source_id] = record
        if len(sources) > MAX_SOURCES:
            raise LineageError(f"more than {MAX_SOURCES} sources; split the manifest")
    if not sources:
        raise LineageError(f"{path} lists no sources")
    return sources


def _holders(source: SourceRecord, start: int, end: int) -> tuple[list[ChunkRecord], bool]:
    """Chunks that hold the whole span, and whether chunks together cover every character."""
    whole = [c for c in source.chunks if c.start <= start and end <= c.end]
    covered = start
    for chunk in sorted(source.chunks, key=lambda c: c.start):
        if chunk.start <= covered < chunk.end:
            covered = chunk.end
        if covered >= end:
            break
    return whole, covered >= end


def ingestion_stage(source: SourceRecord | None, start: int, end: int) -> dict[str, Any]:
    """The first ingestion loss for one span, or the chunks that carry it intact."""
    if source is None:
        return {"loss": "source_missing"}
    if source.extracted_chars is not None:
        if source.extracted_chars == 0:
            return {"loss": "empty_extraction"}
        if end > source.extracted_chars:
            return {"loss": "beyond_extraction", "extracted_chars": source.extracted_chars}
    whole, covered = _holders(source, start, end)
    if not whole:
        if not covered:
            return {"loss": "not_chunked"}
        boundary = sorted({c.chunk_id for c in source.chunks if c.start < end and start < c.end})
        return {"loss": "split_across_chunks", "chunks": boundary}
    indexed = [c for c in whole if c.indexed]
    if not indexed:
        return {"loss": "not_indexed", "chunks": [c.chunk_id for c in whole]}
    current = [
        c
        for c in indexed
        if source.revision is None
        or c.indexed_revision is None
        or c.indexed_revision == source.revision
    ]
    if not current:
        return {
            "loss": "stale_index",
            "chunks": [c.chunk_id for c in indexed],
            "indexed_revisions": sorted({c.indexed_revision or "" for c in indexed}),
            "revision": source.revision,
        }
    unknown = source.revision is None or any(c.indexed_revision is None for c in current)
    return {
        "loss": None,
        "chunks": [c.chunk_id for c in current],
        **({"revision_unverified": True} if unknown else {}),
    }


def _run_stage(
    query: Mapping[str, Any], chunks: Sequence[str], source_id: str, ids: str
) -> dict[str, Any]:
    """Where the run lost a span's chunks: retrieval, delivery or context."""
    if query.get("error"):
        return {"loss": None, "unknown": "the question failed in this run"}
    wanted = set(chunks) if ids == "chunk" else {source_id}
    evidence = query.get("evidence") if isinstance(query.get("evidence"), Mapping) else None
    delivered = wanted & set(query.get("ids") or [])
    if delivered:
        # The raw context record, not the per-question analysis: that one follows only
        # document-level labels, and lineage follows span-labelled chunks.
        context = query.get("model_context")
        if not isinstance(context, Mapping):
            return {"loss": None, "delivered": sorted(delivered), "unknown": "model context"}
        items = {item["id"]: item for item in context.get("items") or []}
        reached = [d for d in delivered if d in items and items[d].get("truncated") is False]
        if reached:
            return {"loss": None, "delivered": sorted(delivered), "in_context": sorted(reached)}
        trimmed = [d for d in delivered if d in items and items[d].get("truncated") is True]
        if trimmed:
            return {"loss": "lost_in_context", "chunks": sorted(trimmed), "detail": "truncated"}
        if not any(d in items for d in delivered) and context.get("coverage") == "observed":
            return {"loss": "lost_in_context", "chunks": sorted(delivered), "detail": "omitted"}
        return {"loss": None, "delivered": sorted(delivered), "unknown": "model context"}
    stages = (evidence or {}).get("stages") or []
    seen = [s["name"] for s in stages if wanted & {h["id"] for h in s.get("hits") or []}]
    if seen:
        return {"loss": "lost_downstream", "last_stage": seen[-1]}
    if not stages:
        return {"loss": "not_retrieved", "unknown": "no stage trace; only the final ranking"}
    return {"loss": "not_retrieved"}


def check_lineage(
    dataset: Dataset,
    sources: Mapping[str, SourceRecord],
    run: Mapping[str, Any] | None = None,
    *,
    ids: str = "chunk",
) -> dict[str, Any]:
    """Follow every labelled passage from source to model context; name the first loss.

    ``ids`` says what the run's result IDs are: ``chunk`` IDs from the manifest, or
    ``source`` document IDs. Questions without ``relevant_spans`` are not followed.
    """
    if ids not in ("chunk", "source"):
        raise LineageError("ids must be chunk or source")
    queries = {q["query_id"]: q for q in (run or {}).get("queries") or []}
    rows = []
    for q in dataset.queries:
        for span in q.relevant_spans:
            if span.grade <= 0:
                continue
            source = sources.get(span.doc_id)
            stage = ingestion_stage(source, span.start, span.end)
            row: dict[str, Any] = {
                "query_id": q.query_id,
                "source_id": span.doc_id,
                "start": span.start,
                "end": span.end,
                **stage,
            }
            if stage["loss"] is None and run is not None:
                measured = queries.get(q.query_id)
                if measured is None:
                    row["unknown"] = "the question is not in this run"
                else:
                    after = _run_stage(measured, stage["chunks"], span.doc_id, ids)
                    row["loss"] = after.pop("loss")
                    row.update(after)
            rows.append(row)
    return {
        "format": "vectraudit.lineage",
        "format_version": 1,
        "dataset_sha256": dataset.sha256,
        "run_id": (run or {}).get("run_id"),
        "ids": ids,
        "passages": len(rows),
        "first_loss": dict(Counter(r["loss"] for r in rows if r["loss"])),
        "ingestion_losses": sum(r["loss"] in INGESTION for r in rows),
        "rows": rows,
        "corpus": corpus_findings(sources, run, ids=ids),
        "advice": _advice(rows),
    }


def corpus_findings(
    sources: Mapping[str, SourceRecord], run: Mapping[str, Any] | None = None, *, ids: str = "chunk"
) -> dict[str, Any]:
    """Problems visible in the export alone: empty sources, stale chunks still indexed beside
    current ones, and identical chunks that compete for the same result slots."""
    empty = sorted(s.source_id for s in sources.values() if s.extracted_chars == 0)
    stale = []
    for s in sources.values():
        if s.revision is None:
            continue
        old = [
            c.chunk_id
            for c in s.chunks
            if c.indexed and c.indexed_revision not in (None, s.revision)
        ]
        if old:
            stale.append({"source_id": s.source_id, "revision": s.revision, "chunks": old})
    by_hash: dict[str, list[str]] = defaultdict(list)
    for s in sources.values():
        for c in s.chunks:
            if c.indexed and c.text_sha256:
                by_hash[c.text_sha256].append(c.chunk_id)
    duplicates = [sorted(group) for group in by_hash.values() if len(group) > 1]
    crowded = []
    if run is not None and ids == "chunk" and duplicates:
        twin = {chunk: tuple(group) for group in duplicates for chunk in group}
        for q in run.get("queries") or []:
            delivered = [i for i in q.get("ids") or [] if i in twin]
            groups = Counter(twin[i] for i in delivered)
            wasted = sum(n - 1 for n in groups.values())
            if wasted:
                crowded.append({"query_id": q["query_id"], "slots_taken_by_duplicates": wasted})
    return {
        "empty_extraction": empty,
        "stale_chunks_indexed": stale,
        "duplicate_chunks": duplicates,
        "duplicates_in_results": crowded,
    }


def _advice(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    losses = Counter(r["loss"] for r in rows if r["loss"])
    advice = []
    ingestion = sum(n for loss, n in losses.items() if loss in INGESTION)
    if ingestion:
        advice.append(
            f"{ingestion} passage(s) were lost during ingestion. Fix extraction, chunking or "
            "indexing first: a different embedding model cannot retrieve text that is not "
            "indexed intact."
        )
    if losses.get("split_across_chunks"):
        advice.append(
            "Passages split across chunk boundaries: try overlapping chunks or splitting on "
            "section boundaries, then re-check lineage."
        )
    if losses.get("stale_index"):
        advice.append("Re-index sources whose chunks carry an older revision.")
    if losses.get("not_retrieved") and not ingestion:
        advice.append(
            "Passages were indexed intact but not retrieved: this is where retrieval changes "
            "(depth, hybrid weight, embeddings) can help."
        )
    return advice


def render(report: Mapping[str, Any]) -> str:
    """The lineage report as plain text for the terminal."""
    lines = [f"{report['passages']} required passage(s) followed from source to context."]
    if not report["first_loss"]:
        lines.append("No passage was lost at any stage the export and run describe.")
    for loss in STAGES:
        n = report["first_loss"].get(loss)
        if n:
            lines.append(f"  {n} first lost at {loss.replace('_', ' ')}: {EXPLAIN[loss]}")
    corpus = report["corpus"]
    if corpus["empty_extraction"]:
        lines.append(f"Sources with empty extraction: {', '.join(corpus['empty_extraction'][:10])}")
    if corpus["stale_chunks_indexed"]:
        lines.append(
            f"{len(corpus['stale_chunks_indexed'])} source(s) still have chunks from an older "
            "revision in the index."
        )
    if corpus["duplicate_chunks"]:
        lines.append(f"{len(corpus['duplicate_chunks'])} group(s) of identical indexed chunks.")
    if corpus["duplicates_in_results"]:
        lines.append(
            f"Duplicates took result slots in {len(corpus['duplicates_in_results'])} question(s)."
        )
    if report["advice"]:
        lines += ["", *report["advice"]]
    return "\n".join(lines)
