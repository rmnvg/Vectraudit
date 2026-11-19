"""Import labelled datasets that teams already keep in other tools' formats.

Users bring their own labels; these importers only translate them. Nothing here
invents a question or a relevance judgment.

- **TREC qrels** (``qid iter docno grade``) with a query file, and **BEIR** directories
  (``queries.jsonl`` plus ``qrels/<split>.tsv``): graded relevance by document id.
- **Ragas** test sets (``user_input`` with ``reference_context_ids`` or
  ``reference_contexts``) and **DeepEval** goldens (``input`` with ``context``). When a test
  set names its reference passages only by their text, each passage is located in the
  user's corpus file to find the document id the backend returns. Passages that are not
  found, or that match too many documents, are reported rather than guessed.

Every importer returns validated-ready rows plus provenance (source format, file hashes and
what was skipped and why) for the dataset header.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.evaluation.dataset import DatasetError
from vectraudit.evaluation.dataset_tools import read_records

CONTEXT_FORMATS = ("ragas", "deepeval")
MAX_MATCHES = 5
"""Default cap on corpus documents one reference passage may match before it is ambiguous."""
MIN_PASSAGE_CHARS = 20
"""Shorter reference passages are too generic to locate reliably and are reported."""
_SPACE = re.compile(r"\s+")


@dataclass
class Imported:
    """Dataset rows translated from another format, with what happened along the way."""

    rows: list[dict[str, Any]]
    provenance: dict[str, Any]
    skipped: Counter[str] = field(default_factory=Counter)
    """Source queries left out, by reason."""
    warnings: list[str] = field(default_factory=list)
    corpus: list[dict[str, Any]] | None = None
    """Documents with the IDs the rows refer to, when the source has none of its own (MTCB)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as e:
        raise DatasetError(f"File not found: {path}") from e
    except UnicodeDecodeError as e:
        raise DatasetError(f"{path} is not UTF-8 text.") from e


# --- TREC and BEIR -------------------------------------------------------------------------


def read_qrels(path: str | Path) -> tuple[dict[str, dict[str, float]], int]:
    """Read relevance judgments, keeping the order queries first appear in.

    Accepts TREC lines (``qid iter docno grade``, whitespace-separated) and BEIR TSV
    (``query-id<TAB>corpus-id<TAB>score`` with an optional header). Negative grades, which
    some collections use for spam or junk, are dropped; the second value is their count.
    """
    p = Path(path)
    qrels: dict[str, dict[str, float]] = {}
    negative = 0
    first = True
    for n, line in enumerate(_read_text(p).splitlines(), 1):
        if not line.strip():
            continue
        header_allowed, first = first, False
        parts = line.rstrip("\r\n").split("\t") if "\t" in line else line.split()
        parts = [x.strip() for x in parts]
        if len(parts) == 4:
            qid, doc, grade_text = parts[0], parts[2], parts[3]
        elif len(parts) == 3:
            qid, doc, grade_text = parts
        else:
            raise DatasetError(
                f"{p.name} line {n}: expected 'qid iter docno grade' (TREC) or "
                f"'query-id corpus-id score' (BEIR), got {len(parts)} fields."
            )
        try:
            grade = float(grade_text)
        except ValueError as e:
            if header_allowed:
                continue  # a header row such as "query-id corpus-id score"
            raise DatasetError(f"{p.name} line {n}: grade {grade_text!r} is not a number.") from e
        if not qid or not doc:
            raise DatasetError(f"{p.name} line {n}: empty query or document id.")
        if grade < 0:
            negative += 1
            continue
        judged = qrels.setdefault(qid, {})
        if doc in judged and judged[doc] != grade:
            raise DatasetError(
                f"{p.name} line {n}: {qid} / {doc} is judged twice with different grades "
                f"({judged[doc]:g} and {grade:g})."
            )
        judged[doc] = grade
    if not qrels:
        raise DatasetError(f"{p.name} has no relevance judgments.")
    return qrels, negative


def read_query_texts(path: str | Path) -> dict[str, str]:
    """Query texts by id from JSONL (``_id``/``query_id``/``id`` and ``text``) or TSV.

    TSV lines are ``qid<TAB>text``, as in MS MARCO and TREC topic conversions.
    """
    p = Path(path)
    texts: dict[str, str] = {}
    if p.suffix.lower() in (".jsonl", ".json"):
        for n, rec in enumerate(read_records(p), 1):
            if not isinstance(rec, Mapping):
                raise DatasetError(f"{p.name} record {n} is not an object.")
            qid = next((rec[k] for k in ("_id", "query_id", "id") if k in rec), None)
            text = rec.get("text")
            if qid is None or not isinstance(text, str):
                raise DatasetError(f"{p.name} record {n}: needs an id (_id) and a text.")
            _add_text(texts, str(qid), text, f"{p.name} record {n}")
        return texts
    for n, line in enumerate(_read_text(p).splitlines(), 1):
        if not line.strip():
            continue
        qid, sep, text = line.partition("\t")
        if not sep:
            raise DatasetError(f"{p.name} line {n}: expected 'qid<TAB>text'.")
        _add_text(texts, qid.strip(), text, f"{p.name} line {n}")
    return texts


def _add_text(texts: dict[str, str], qid: str, text: str, where: str) -> None:
    if qid in texts:
        raise DatasetError(f"{where}: duplicate query id {qid!r}.")
    texts[qid] = text.strip()


def import_qrels(
    queries: str | Path, qrels: str | Path, *, source_format: str = "trec"
) -> Imported:
    """Join query texts with relevance judgments.

    Only judged queries are imported. A query with no text, or whose judgments are all
    zero, is skipped and counted: it could only contribute latency, not quality.
    """
    qp, rp = Path(queries), Path(qrels)
    texts = read_query_texts(qp)
    judged, negative = read_qrels(rp)
    out = Imported(
        rows=[],
        provenance={
            "source_format": source_format,
            "queries_file": qp.name,
            "queries_sha256": _sha256(qp),
            "qrels_file": rp.name,
            "qrels_sha256": _sha256(rp),
        },
    )
    for qid, grades in judged.items():
        text = texts.get(qid)
        if not text:
            out.skipped["no query text"] += 1
        elif not any(g > 0 for g in grades.values()):
            out.skipped["no relevant document"] += 1
        else:
            out.rows.append({"query_id": qid, "text": text, "relevant": grades})
    if negative:
        out.warnings.append(f"Dropped {negative:,} negative judgments (treated as not relevant).")
        out.provenance["negative_judgments_dropped"] = negative
    unjudged = len(set(texts) - set(judged))
    if unjudged:
        out.warnings.append(f"{unjudged:,} queries in {qp.name} have no judgments; not imported.")
    _finish(out)
    return out


def import_beir(directory: str | Path, *, split: str = "test") -> Imported:
    """Import a BEIR-format directory: ``queries.jsonl`` and ``qrels/<split>.tsv``."""
    d = Path(directory)
    if not d.is_dir():
        raise DatasetError(f"{d} is not a directory; BEIR datasets are folders.")
    qrels = d / "qrels" / f"{split}.tsv"
    if not qrels.is_file():
        available = sorted(p.stem for p in (d / "qrels").glob("*.tsv"))
        hint = f" Available splits: {', '.join(available)}." if available else ""
        raise DatasetError(f"No qrels/{split}.tsv in {d}.{hint}")
    out = import_qrels(d / "queries.jsonl", qrels, source_format="beir")
    out.provenance["split"] = split
    return out


# --- Ragas and DeepEval --------------------------------------------------------------------


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _list_cell(value: Any, *, delimiter: str | None) -> list[str]:
    """A list field that may have been flattened by a CSV export."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v) != ""]
    text = str(value).strip()
    if text.startswith("["):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None and str(v) != ""]
    if delimiter:
        return [part for part in (p.strip() for p in text.split(delimiter)) if part]
    return [text]


class CorpusIndex:
    """Locate reference passages in the documents a backend retrieves, by text."""

    def __init__(self, documents: Iterable[tuple[str, str]]) -> None:
        self._exact: dict[str, list[str]] = {}
        ids: list[str] = []
        parts: list[str] = []
        self._starts: list[int] = []
        offset = 0
        for doc_id, text in documents:
            normalized = _norm(text)
            self._exact.setdefault(normalized, []).append(doc_id)
            ids.append(doc_id)
            self._starts.append(offset)
            parts.append(normalized)
            offset += len(normalized) + 1  # the separator below
        if not ids:
            raise DatasetError("The corpus has no documents.")
        self._ids = ids
        # NUL separates documents so a passage cannot match across a boundary (document
        # text practically never contains it).
        self._haystack = "\x00".join(parts)

    def __len__(self) -> int:
        return len(self._ids)

    def find(self, passage: str, *, limit: int) -> list[str]:
        """Ids of documents equal to or containing the passage, at most ``limit + 1``."""
        needle = _norm(passage)
        exact = self._exact.get(needle)
        if exact:
            return list(dict.fromkeys(exact))[: limit + 1]
        found: dict[str, None] = {}
        start = self._haystack.find(needle)
        while start != -1 and len(found) <= limit:
            doc = bisect.bisect_right(self._starts, start) - 1
            found[self._ids[doc]] = None
            # Continue after this document: one hit per document is enough.
            nxt = self._starts[doc + 1] if doc + 1 < len(self._starts) else len(self._haystack)
            start = self._haystack.find(needle, nxt)
        return list(found)


def load_corpus(
    path: str | Path, *, id_field: str | None = None, text_field: str | None = None
) -> CorpusIndex:
    """Read corpus documents (JSONL, JSON or CSV) whose ids are what the backend returns.

    Without explicit fields, the id is ``_id``, ``doc_id`` or ``id`` and the text is
    ``text``, ``content`` or ``page_content``.
    """
    p = Path(path)
    records = read_records(p)
    docs: list[tuple[str, str]] = []
    for n, rec in enumerate(records, 1):
        if not isinstance(rec, Mapping):
            raise DatasetError(f"{p.name} record {n} is not an object.")
        id_key = id_field or next((k for k in ("_id", "doc_id", "id") if k in rec), None)
        text_key = text_field or next(
            (k for k in ("text", "content", "page_content") if k in rec), None
        )
        if id_key is None or id_key not in rec:
            raise DatasetError(f"{p.name} record {n}: no document id; pass --corpus-id-field.")
        if text_key is None or not isinstance(rec.get(text_key), str):
            raise DatasetError(f"{p.name} record {n}: no document text; pass --corpus-text-field.")
        docs.append((str(rec[id_key]), rec[text_key]))
    return CorpusIndex(docs)


_FIELDS: dict[str, dict[str, Any]] = {
    # ragas.dataset_schema.SingleTurnSample; Testset.to_pandas adds synthesizer_name.
    "ragas": {
        "question": ("user_input", "question"),
        "ids": ("reference_context_ids",),
        "passages": ("reference_contexts",),
        "tag": "synthesizer_name",
        "delimiter": None,
    },
    # deepeval.dataset.Golden: ``context`` is the ideal context; ``retrieval_context`` is
    # what a system retrieved, so it is never read as a label. CSV joins lists with "|".
    "deepeval": {
        "question": ("input",),
        "ids": (),
        "passages": ("context",),
        "tag": None,
        "delimiter": "|",
    },
}


def _query_id(rec: Mapping[str, Any], question: str) -> str:
    for key in ("query_id", "id"):
        if rec.get(key) not in (None, ""):
            return str(rec[key])
    # Content-derived, so the id survives reordering the source file.
    return "q-" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def import_context_testset(
    path: str | Path,
    *,
    source_format: str,
    corpus: CorpusIndex | None = None,
    max_matches: int = MAX_MATCHES,
) -> Imported:
    """Import a Ragas test set or DeepEval goldens.

    Reference context ids are used directly. Otherwise each reference passage must be found
    in ``corpus``; a query keeps the documents its passages were found in and is skipped if
    none was. Every passage that is not found or is ambiguous is counted in the warnings.
    """
    if source_format not in _FIELDS:
        raise DatasetError(f"Format must be one of {', '.join(CONTEXT_FORMATS)}.")
    spec = _FIELDS[source_format]
    p = Path(path)
    records = read_records(p)
    out = Imported(
        rows=[],
        provenance={"source_format": source_format, "file": p.name, "sha256": _sha256(p)},
    )
    passages = Counter[str]()
    seen: set[str] = set()
    for n, rec in enumerate(records, 1):
        if not isinstance(rec, Mapping):
            raise DatasetError(f"{p.name} record {n} is not an object.")
        question = next(
            (rec[k] for k in spec["question"] if isinstance(rec.get(k), str) and rec[k].strip()),
            None,
        )
        if question is None:
            raise DatasetError(
                f"{p.name} record {n}: no question ({' or '.join(spec['question'])}). "
                f"Is this a {source_format} file?"
            )
        relevant = _relevant_for(rec, spec, corpus, max_matches, passages, n, p.name)
        if relevant is None:
            out.skipped["no reference passage found in the corpus"] += 1
            continue
        if not relevant:
            out.skipped["no reference context"] += 1
            continue
        qid = _query_id(rec, question)
        if qid in seen:
            out.skipped["duplicate question"] += 1
            continue
        seen.add(qid)
        row: dict[str, Any] = {"query_id": qid, "text": question.strip(), "relevant": relevant}
        tag = spec["tag"] and rec.get(spec["tag"])
        if isinstance(tag, str) and tag:
            row["tags"] = [tag]
        out.rows.append(row)
    if passages:
        out.provenance["passage_matching"] = dict(sorted(passages.items()))
        missed = passages["not found"] + passages["ambiguous"] + passages["too short"]
        if missed:
            out.warnings.append(
                f"{missed:,} of {sum(passages.values()):,} reference passages could not be "
                f"located ({passages['not found']:,} not found, {passages['ambiguous']:,} in "
                f"more than {max_matches} documents, {passages['too short']:,} too short). "
                "Queries keep only the documents that were located, so their labels may be "
                "incomplete. Use the corpus the test set was generated from, at the same "
                "chunking."
            )
    _finish(out)
    return out


def _relevant_for(
    rec: Mapping[str, Any],
    spec: Mapping[str, Any],
    corpus: CorpusIndex | None,
    max_matches: int,
    passages: Counter[str],
    n: int,
    name: str,
) -> dict[str, float] | None:
    """Relevant ids for one record; ``{}`` if it names no context, ``None`` if none located."""
    for key in spec["ids"]:
        ids = _list_cell(rec.get(key), delimiter=None)
        if ids:
            return dict.fromkeys(ids, 1.0)
    texts: list[str] = []
    for key in spec["passages"]:
        texts = _list_cell(rec.get(key), delimiter=spec["delimiter"])
        if texts:
            break
    if not texts:
        return {}
    if corpus is None:
        raise DatasetError(
            f"{name} record {n} names its reference contexts by text, not id. Pass --corpus "
            "with the documents your backend returns so each passage can be located."
        )
    relevant: dict[str, float] = {}
    for text in texts:
        if len(_norm(text)) < MIN_PASSAGE_CHARS:
            passages["too short"] += 1
            continue
        found = corpus.find(text, limit=max_matches)
        if not found:
            passages["not found"] += 1
        elif len(found) > max_matches:
            passages["ambiguous"] += 1
        else:
            passages["located"] += 1
            relevant.update(dict.fromkeys(found, 1.0))
    return relevant or None


def _finish(out: Imported) -> None:
    if out.skipped:
        out.provenance["skipped"] = dict(sorted(out.skipped.items()))
        out.warnings.extend(
            f"Skipped {c:,} queries: {why}." for why, c in sorted(out.skipped.items())
        )
    if not out.rows:
        reasons = "; ".join(f"{why} ({c})" for why, c in out.skipped.items()) or "empty input"
        raise DatasetError(f"No labelled queries could be imported: {reasons}.")


MTCB_PASSAGE_FIELDS = ("chunk-must-contain", "chunk_must_contain")
MTCB_MAX_OCCURRENCES = 100


def _mtcb_records(directory: Path, names: tuple[str, ...], what: str) -> tuple[list[Any], Path]:
    """The records of ``<name>.jsonl|json|csv|parquet`` or ``<name>/*.parquet`` in ``directory``."""
    for name in names:
        for suffix in (".jsonl", ".json", ".csv"):
            path = directory / f"{name}{suffix}"
            if path.is_file():
                return read_records(path), path
        parquet = sorted([*directory.glob(f"{name}.parquet"), *directory.glob(f"{name}/*.parquet")])
        if parquet:
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise DatasetError(
                    'Reading .parquet needs pyarrow: pip install "vectraudit[parquet]".'
                ) from e
            rows: list[Any] = []
            for path in parquet:
                rows.extend(pq.read_table(path).to_pylist())
            return rows, parquet[0]
    raise DatasetError(
        f"{directory} has no {what}: expected {names[0]}.jsonl (or .json, .csv, .parquet). "
        "Fetch the MTCB dataset yourself; Vectraudit never downloads datasets."
    )


def import_mtcb(directory: str | Path) -> Imported:
    """An MTCB chunking-benchmark dataset as questions with relevant spans.

    MTCB (read in its source, commit 6fd594a345aa) publishes each dataset as a ``corpus`` of
    documents (``text``; no IDs) and ``questions`` (``question`` and ``chunk-must-contain``, the
    passage a retrieved chunk must contain). Its dataset generator writes ``document`` and
    ``chunk_must_contain`` instead; both spellings are read. Documents get the IDs
    ``doc-<row>`` (or ``doc-<document_id>``), returned in ``Imported.corpus`` so the experiment
    uses the same ones. Every exact occurrence of a passage, in any document, becomes a
    relevant span, because MTCB counts a chunk that contains the passage wherever it comes
    from. A passage that occurs nowhere exactly, or more than 100 times, is skipped and
    counted, never matched loosely.
    """
    d = Path(directory)
    if not d.is_dir():
        raise DatasetError(f"{d} is not a directory holding an MTCB corpus and questions.")
    corpus_rows, corpus_path = _mtcb_records(d, ("corpus",), "corpus")
    question_rows, questions_path = _mtcb_records(d, ("questions", "qa"), "questions")
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, rec in enumerate(corpus_rows):
        text = rec.get("text", rec.get("document")) if isinstance(rec, Mapping) else None
        if not isinstance(text, str) or not text:
            raise DatasetError(f"{corpus_path.name} row {n + 1} has no text (or document) field.")
        doc_id = f"doc-{rec['document_id']}" if rec.get("document_id") is not None else f"doc-{n}"
        if doc_id in seen:
            raise DatasetError(f"{corpus_path.name} repeats the document id {doc_id!r}.")
        seen.add(doc_id)
        documents.append({"doc_id": doc_id, "text": text})
    out = Imported(
        rows=[],
        provenance={
            "format": "mtcb",
            "directory": d.name,
            "corpus_file": corpus_path.name,
            "corpus_sha256": _sha256(corpus_path),
            "questions_file": questions_path.name,
            "questions_sha256": _sha256(questions_path),
            "documents": len(documents),
            "passage_rule": "every exact occurrence of chunk-must-contain is a relevant span",
        },
        corpus=documents,
    )
    for n, rec in enumerate(question_rows):
        question = rec.get("question") if isinstance(rec, Mapping) else None
        passage = next(
            (rec[f] for f in MTCB_PASSAGE_FIELDS if isinstance(rec, Mapping) and rec.get(f)), None
        )
        if not isinstance(question, str) or not question.strip():
            out.skipped["no question text"] += 1
            continue
        if not isinstance(passage, str) or not passage:
            out.skipped["no chunk-must-contain passage"] += 1
            continue
        spans: list[dict[str, Any]] = []
        for doc in documents:
            start = doc["text"].find(passage)
            while start >= 0 and len(spans) <= MTCB_MAX_OCCURRENCES:
                spans.append({"doc_id": doc["doc_id"], "start": start, "end": start + len(passage)})
                start = doc["text"].find(passage, start + 1)
        if not spans:
            out.skipped["passage not found exactly in any document"] += 1
            continue
        if len(spans) > MTCB_MAX_OCCURRENCES:
            out.skipped[f"passage occurs more than {MTCB_MAX_OCCURRENCES} times"] += 1
            continue
        out.rows.append({"query_id": f"mtcb-{n}", "text": question, "relevant_spans": spans})
    _finish(out)
    return out


def import_format(
    source_format: str,
    source: str | Path,
    *,
    qrels: str | Path | None = None,
    split: str = "test",
    corpus: str | Path | None = None,
    corpus_id_field: str | None = None,
    corpus_text_field: str | None = None,
    max_matches: int = MAX_MATCHES,
) -> Imported:
    """Run the importer for ``source_format``: beir, trec, mtcb, ragas or deepeval."""
    if source_format == "mtcb":
        return import_mtcb(source)
    if source_format == "beir":
        return import_beir(source, split=split)
    if source_format == "trec":
        if qrels is None:
            raise DatasetError("The trec format needs --qrels alongside the query file.")
        return import_qrels(source, qrels, source_format="trec")
    if source_format in CONTEXT_FORMATS:
        index = (
            load_corpus(corpus, id_field=corpus_id_field, text_field=corpus_text_field)
            if corpus is not None
            else None
        )
        out = import_context_testset(
            source, source_format=source_format, corpus=index, max_matches=max_matches
        )
        if corpus is not None:
            cp = Path(corpus)
            out.provenance["corpus_file"] = cp.name
            out.provenance["corpus_sha256"] = _sha256(cp)
        return out
    raise DatasetError(
        f"Unknown format {source_format!r}; use beir, trec, mtcb, ragas, deepeval, or omit it for "
        "CSV/JSON with column mapping."
    )
