"""Split documents into chunks that remember their source document and character span.

Chunk ids are ``<doc_id>#<n>`` and are only meaningful within one chunking: labels live on
documents (or text spans) and are mapped onto each chunking, never on chunk ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vectraudit.experiments.corpus import Corpus, Document

METHODS = ("words", "paragraphs", "document")
_WORD = re.compile(r"\S+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    """Character offset in the source document (inclusive)."""
    end: int
    """Character offset in the source document (exclusive)."""
    text: str


@dataclass(frozen=True)
class ChunkingSpec:
    method: str = "words"
    size: int = 200
    """Words per chunk (``words``), or the merge limit in words (``paragraphs``)."""
    overlap: int = 0
    """Words shared by consecutive chunks (``words`` only)."""

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"chunking method must be one of {', '.join(METHODS)}")
        if self.size < 1:
            raise ValueError("chunk size must be >= 1")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be >= 0 and smaller than size")

    @property
    def label(self) -> str:
        if self.method == "document":
            return "document"
        if self.method == "paragraphs":
            return f"paragraphs-{self.size}"
        return f"words-{self.size}" + (f"-overlap{self.overlap}" if self.overlap else "")


def _word_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    return [(m.start() + offset, m.end() + offset) for m in _WORD.finditer(text)]


def _windows(spans: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of word windows; the last window ends at the last word."""
    if not spans:
        return []
    step = size - overlap
    out = []
    i = 0
    while True:
        window = spans[i : i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            return out
        i += step


def _paragraph_ranges(doc: Document, size: int) -> list[tuple[int, int]]:
    """Merge consecutive paragraphs up to ``size`` words; split longer paragraphs by words."""
    ranges: list[tuple[int, int]] = []
    pieces: list[tuple[int, int, int]] = []  # (start, end, words)
    cursor = 0
    for brk in [*_PARAGRAPH_BREAK.finditer(doc.text), None]:
        end = brk.start() if brk else len(doc.text)
        spans = _word_spans(doc.text[cursor:end], cursor)
        if len(spans) > size:
            pieces += [(s, e, size) for s, e in _windows(spans, size, 0)]
        elif spans:
            pieces.append((spans[0][0], spans[-1][1], len(spans)))
        cursor = brk.end() if brk else end
    current: list[tuple[int, int, int]] = []
    for piece in pieces:
        if current and sum(p[2] for p in current) + piece[2] > size:
            ranges.append((current[0][0], current[-1][1]))
            current = []
        current.append(piece)
    if current:
        ranges.append((current[0][0], current[-1][1]))
    return ranges


def chunk_document(doc: Document, spec: ChunkingSpec) -> list[Chunk]:
    if spec.method == "document":
        spans = _word_spans(doc.text)
        ranges = [(spans[0][0], spans[-1][1])] if spans else []
    elif spec.method == "words":
        ranges = _windows(_word_spans(doc.text), spec.size, spec.overlap)
    else:
        ranges = _paragraph_ranges(doc, spec.size)
    return [
        Chunk(f"{doc.doc_id}#{n}", doc.doc_id, start, end, doc.text[start:end])
        for n, (start, end) in enumerate(ranges)
    ]


def chunk_corpus(corpus: Corpus, spec: ChunkingSpec) -> list[Chunk]:
    """Every document's chunks, in corpus order."""
    return [chunk for doc in corpus.documents for chunk in chunk_document(doc, spec)]
