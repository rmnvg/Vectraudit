"""A corpus of source documents (before chunking), identified by a content hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CorpusError(ValueError):
    """The corpus file is malformed; the message names the line."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Corpus:
    documents: tuple[Document, ...]
    sha256: str
    by_id: dict[str, Document] = field(default_factory=dict, compare=False, repr=False)

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, doc_id: str) -> Document | None:
        return self.by_id.get(doc_id)


def load_corpus(path: str | Path, *, id_field: str = "doc_id", text_field: str = "text") -> Corpus:
    """Read JSONL documents: ``{"doc_id": ..., "text": ..., ...metadata}``."""
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise CorpusError(f"Corpus file not found: {p}") from e
    docs: list[Document] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise CorpusError(f"Line {n}: not JSON ({e}).") from e
        if not isinstance(obj, dict):
            raise CorpusError(f"Line {n}: each line must be a JSON object.")
        doc_id, text = obj.get(id_field), obj.get(text_field)
        if not isinstance(doc_id, (str, int)) or isinstance(doc_id, bool) or str(doc_id) == "":
            raise CorpusError(f"Line {n}: {id_field!r} must be a non-empty string or integer.")
        if not isinstance(text, str) or not text.strip():
            raise CorpusError(f"Line {n}: {text_field!r} must be non-empty text.")
        doc_id = str(doc_id)
        if doc_id in seen:
            raise CorpusError(f"Line {n}: duplicate {id_field} {doc_id!r}.")
        seen.add(doc_id)
        meta = {k: v for k, v in obj.items() if k not in (id_field, text_field)}
        docs.append(Document(doc_id, text, meta))
    if not docs:
        raise CorpusError("The corpus has no documents.")
    return make_corpus(docs)


def make_corpus(documents: list[Document]) -> Corpus:
    digest = hashlib.sha256()
    for d in sorted(documents, key=lambda d: d.doc_id):
        digest.update(json.dumps([d.doc_id, d.text], ensure_ascii=False).encode())
        digest.update(b"\n")
    return Corpus(tuple(documents), digest.hexdigest(), {d.doc_id: d for d in documents})
