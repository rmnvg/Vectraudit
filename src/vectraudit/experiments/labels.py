"""Map labels on documents and text spans onto any chunking.

Document level (the fair comparison across chunkings): retrieved chunks become their
source documents, first occurrence wins, and metrics are computed on documents. Span level
(finer, when spans are labelled): a chunk is relevant when it overlaps a labelled span,
with the highest overlapping span's grade.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from vectraudit.evaluation.dataset import DatasetQuery, Span
from vectraudit.experiments.chunking import Chunk


def documents_in_order(chunk_ids: Iterable[str], doc_of: Mapping[str, str]) -> list[str]:
    """Source documents of ranked chunks, deduplicated, best first."""
    seen: set[str] = set()
    out = []
    for cid in chunk_ids:
        doc = doc_of[cid]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def document_labels(query: DatasetQuery) -> dict[str, float]:
    """Document grades: explicit document labels, plus the best span grade per document."""
    labels = dict(query.relevant)
    for span in query.relevant_spans:
        labels[span.doc_id] = max(labels.get(span.doc_id, 0.0), span.grade)
    return labels


def chunk_labels(spans: Sequence[Span], chunks: Iterable[Chunk]) -> dict[str, float]:
    """Chunk grades: the highest grade among labelled spans the chunk overlaps."""
    by_doc: dict[str, list[Span]] = {}
    for span in spans:
        by_doc.setdefault(span.doc_id, []).append(span)
    out: dict[str, float] = {}
    for chunk in chunks:
        for span in by_doc.get(chunk.doc_id, ()):
            if chunk.start < span.end and span.start < chunk.end:
                out[chunk.chunk_id] = max(out.get(chunk.chunk_id, 0.0), span.grade)
    return out
