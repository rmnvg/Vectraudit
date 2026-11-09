"""A BEIR-layout folder as an experiment's corpus and queries.

BEIR is the layout public retrieval benchmarks (and MTEB's retrieval tasks) are distributed in::

    corpus.jsonl            {"_id": "d1", "title": "...", "text": "..."}
    queries.jsonl           {"_id": "q1", "text": "..."}
    qrels/<split>.tsv       query-id <TAB> corpus-id <TAB> score   (with a header row)

The queries and labels go through the same importer as ``vectraudit dataset import --format beir``
(:func:`vectraudit.evaluation.dataset_formats.import_beir`), so a question is labelled exactly as
it is anywhere else. The corpus becomes documents whose text is ``title`` and ``text`` on two
lines (a document with only one of them uses it), the usual way a BEIR document is embedded, with
the title kept as metadata. A document with neither is refused rather than dropped: dropping it
would quietly change the corpus a model is scored on.

Nothing is downloaded. Fetch a BEIR or MTEB dataset yourself and point at the folder; each
public dataset has its own licence.
"""

from __future__ import annotations

import json
from pathlib import Path

from vectraudit.evaluation.dataset import Dataset, DatasetError, parse_dataset
from vectraudit.evaluation.dataset_formats import import_beir
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.experiments.corpus import Corpus, CorpusError, Document, make_corpus


def load_beir_corpus(directory: str | Path) -> Corpus:
    """``corpus.jsonl`` of a BEIR folder as documents."""
    path = Path(directory) / "corpus.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise CorpusError(f"No corpus.jsonl in {directory}; a BEIR folder holds one.") from e
    docs: list[Document] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise CorpusError(f"corpus.jsonl line {n}: not JSON ({e}).") from e
        if not isinstance(obj, dict) or obj.get("_id") in (None, ""):
            raise CorpusError(f"corpus.jsonl line {n}: each line needs an _id.")
        doc_id = str(obj["_id"])
        title, text = str(obj.get("title") or "").strip(), str(obj.get("text") or "").strip()
        body = f"{title}\n{text}" if title and text else (title or text)
        if not body:
            raise CorpusError(
                f"corpus.jsonl line {n}: document {doc_id!r} has neither a title nor text. "
                "It is refused, not dropped, because dropping it changes the corpus being scored."
            )
        if doc_id in seen:
            raise CorpusError(f"corpus.jsonl line {n}: duplicate _id {doc_id!r}.")
        seen.add(doc_id)
        docs.append(Document(doc_id, body, {"title": title} if title else {}))
    if not docs:
        raise CorpusError("The corpus has no documents.")
    return make_corpus(docs)


def load_beir(directory: str | Path, split: str = "test") -> tuple[Corpus, Dataset]:
    """The corpus and the labelled queries of a BEIR folder for one qrels split."""
    d = Path(directory)
    try:
        imported = import_beir(d, split=split)
        dataset = parse_dataset(
            render_dataset(
                imported.rows, {"name": f"beir:{d.name}", "provenance": imported.provenance}
            )
        )
    except DatasetError as e:
        raise CorpusError(str(e)) from e
    return load_beir_corpus(d), dataset
