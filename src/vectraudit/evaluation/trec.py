"""TREC run and qrels interchange: take a run to another IR tool, or bring one in.

The TREC run format (``qid Q0 docno rank score tag``) and qrels format (``qid 0 docno grade``)
are what ``trec_eval``, pytrec_eval, ir-measures, ranx and most retrieval research use, so a
team with existing tooling can move its data in and out of Vectraudit without a converter.

**Export** writes a run's first-repeat ranking per question. A Vectraudit run records ranks,
not scores, so the score column holds descending ranks and the tag says so; use the rank
column, which trec_eval itself ignores in favour of scores (and breaks equal scores by
document id, descending, which a rank-valued score never produces). Held-out questions are not
exported unless asked for, errored questions are not exported at all (an unanswered question
must not look like an empty answer), and an identifier containing whitespace is refused rather
than altered, because TREC cannot represent it.

**Import** builds a ``vectraudit.run`` from another system's TREC file, on this dataset. It is
marked ``source: "external:trec"``: rankings only, no stage evidence and no latency (the file
says nothing about either, so they are *not measured*, never passed). Quality metrics are
computed here from the dataset's labels with the same per-query code a recorded run uses, so
they are comparable with any other run on quality. A question the file does not mention is an
error by default (a partial run, never a pass); ``missing_as_empty`` treats it as "retrieved
nothing", which is the TREC convention but should be a choice.

Credit: the formats are those of NIST's ``trec_eval``; ir-measures (Macdonald, MacAvaney) and
pytrec_eval (Van Gysel, de Rijke) are the interchange and parity references. Our own implementation;
see docs/credits.md.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    Distance,
    SearchRequest,
)
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.pass_rates import PassRates
from vectraudit.evaluation.runner import RunOptions, _run_query, _skeleton, summarize

SOURCE = "external:trec"
_TAG = re.compile(r"[^A-Za-z0-9_.-]+")
_SPACE = re.compile(r"\s")


class TrecError(ValueError):
    """A TREC file or export cannot be used; the message says why."""


def _tag(text: str | None, fallback: str) -> str:
    cleaned = _TAG.sub("-", text or "").strip("-")
    return cleaned or fallback


# --- export ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Export:
    """The lines of a TREC file and what was left out."""

    lines: list[str]
    queries: int
    heldout_left_out: int = 0
    errored_left_out: int = 0
    empty: int = 0
    """Questions that retrieved nothing: present in the run, absent from the file (as in TREC)."""


def export_run(
    run: Mapping[str, Any],
    dataset: Dataset,
    *,
    include_heldout: bool = False,
    tag: str | None = None,
) -> Export:
    """The run's first-repeat rankings as TREC run lines (``qid Q0 docno rank score tag``)."""
    if run["dataset"]["sha256"] != dataset.sha256:
        raise TrecError(
            "The run was not recorded on this dataset version; pass the dataset it was run on."
        )
    split = {q.query_id: q.split for q in dataset.queries}
    label = _tag(tag or run.get("label"), "vectraudit-" + str(run.get("run_id", ""))[:8])
    lines: list[str] = []
    stats = {"heldout": 0, "errored": 0, "empty": 0, "queries": 0}
    for entry in run.get("queries") or []:
        qid = entry["query_id"]
        if split.get(qid) == "heldout" and not include_heldout:
            stats["heldout"] += 1
            continue
        if entry.get("error") or entry.get("ids") is None:
            stats["errored"] += 1
            continue
        ranking = [str(d) for d in entry["ids"]]
        if _SPACE.search(qid) or any(_SPACE.search(d) for d in ranking):
            raise TrecError(
                f"Question {qid!r} or one of its documents has whitespace in its ID, which TREC "
                "cannot represent. Rename the IDs; Vectraudit will not alter them."
            )
        stats["queries"] += 1
        stats["empty"] += not ranking
        n = len(ranking)
        lines += [f"{qid} Q0 {d} {i + 1} {n - i}.0 {label}" for i, d in enumerate(ranking)]
    return Export(lines, stats["queries"], stats["heldout"], stats["errored"], stats["empty"])


def export_qrels(dataset: Dataset, *, include_heldout: bool = False) -> tuple[list[str], int]:
    """TREC qrels lines (``qid 0 docno grade``) and the number of held-out questions left out.

    Relevant documents with their grades (a judged non-relevant document keeps its 0); documents
    that only appear as ``relevant_spans`` get grade 1. Grades must be integers, as trec_eval
    requires.
    """
    lines: list[str] = []
    held = 0
    for q in dataset.queries:
        if q.split == "heldout" and not include_heldout:
            held += 1
            continue
        grades = {d: g for d, g in q.relevant.items()}
        for span in q.relevant_spans:
            grades.setdefault(span.doc_id, 1.0)
        if not grades:
            continue
        if _SPACE.search(q.query_id) or any(_SPACE.search(d) for d in grades):
            raise TrecError(
                f"Question {q.query_id!r} or one of its documents has whitespace in its ID, "
                "which TREC cannot represent."
            )
        for doc, grade in sorted(grades.items()):
            if grade != int(grade):
                raise TrecError(
                    f"Question {q.query_id!r} grades {doc!r} as {grade}; TREC qrels need integers."
                )
            lines.append(f"{q.query_id} 0 {doc} {int(grade)}")
    return lines, held


# --- import ---------------------------------------------------------------------------------


@dataclass
class TrecRun:
    """A parsed TREC run file."""

    rankings: dict[str, list[tuple[int, float, str]]]
    """Per question: ``(rank, score, docno)`` ordered by rank."""
    tag: str | None
    rows: int
    sha256: str
    rank_disagrees_with_score: list[str] = field(default_factory=list)
    """Questions where a lower-scored document was ranked above a higher-scored one."""


def parse_run(path: str | Path) -> TrecRun:
    """Read a TREC run file, refusing anything ambiguous instead of guessing."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise TrecError(f"Cannot read {p}: {e.strerror or e}") from e
    rankings: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    tags: list[str] = []
    rows = 0
    for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            raise TrecError(
                f"{p.name} line {n} has {len(parts)} fields; a TREC run line is "
                "`qid Q0 docno rank score tag`."
            )
        qid, _, docno, rank_text, score_text, tag = parts
        try:
            rank = int(rank_text)
            score = float(score_text)
        except ValueError as e:
            raise TrecError(
                f"{p.name} line {n}: rank must be an integer and score a number."
            ) from e
        if not math.isfinite(score):
            raise TrecError(f"{p.name} line {n}: the score is not finite.")
        rankings[qid].append((rank, score, docno))
        tags.append(tag)
        rows += 1
    if not rows:
        raise TrecError(f"{p.name} holds no run lines.")
    disagree = []
    for qid, items in rankings.items():
        items.sort(key=lambda r: r[0])
        if len({r[0] for r in items}) != len(items):
            raise TrecError(f"Question {qid!r} has two documents at the same rank.")
        if len({r[2] for r in items}) != len(items):
            raise TrecError(f"Question {qid!r} lists a document twice.")
        scores = [r[1] for r in items]
        if any(a < b for a, b in pairwise(scores)):
            disagree.append(qid)
    return TrecRun(dict(rankings), tags[0], rows, hashlib.sha256(raw).hexdigest(), sorted(disagree))


class _Replay(Connector):
    """Answers each question with the ranking from the file; searches nothing."""

    name = "trec"

    def __init__(self, trec: TrecRun, path: Path, *, dim: int) -> None:
        self.trec, self.path, self.dim = trec, path, dim
        self.current = ""

    def describe(self) -> BackendInfo:
        return BackendInfo(
            "trec",
            Distance.L2,
            self.dim,
            None,
            target={"file": self.path.name},
            snapshot=f"sha256:{self.trec.sha256}",
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(search=True)

    def _search(self, request: SearchRequest) -> list[str]:
        return [d for _, _, d in self.trec.rankings.get(self.current, [])]


def import_run(
    path: str | Path,
    dataset: Dataset,
    *,
    k: int = 10,
    label: str | None = None,
    missing_as_empty: bool = False,
) -> dict[str, Any]:
    """A ``vectraudit.run`` for the TREC file's ranking of every dataset question."""
    p = Path(path)
    trec = parse_run(p)
    options = RunOptions(k=k, warmup=0, evidence=False, label=label or trec.tag)
    record = _skeleton(dataset, options, {"backend": "trec", "file": p.name}, None)
    connector = _Replay(trec, p, dim=1)
    info, caps = connector.describe(), connector.capabilities()
    record["backend"], record["capabilities"] = info.as_dict(), caps.as_dict()
    record["search"]["exact_reference"] = False
    record["timing"].update(
        {
            "scope": "not measured: an imported run's latency is unknown",
            "source": SOURCE,
            "wall_s": 0.0,
        }
    )
    known = {q.query_id for q in dataset.queries}
    in_file = set(trec.rankings)
    rates = PassRates(connector)
    for q in dataset.queries:
        connector.current = q.query_id
        entry = _run_query(connector, q, np.zeros(1, dtype=np.float32), options, False, rates)
        if q.query_id not in in_file and not missing_as_empty:
            entry["ids"] = None
            entry["repeat_ids"] = [[]]
            entry["metrics"] = None
            entry["repeat_metrics"] = [None]
            entry["error"] = {
                "code": "query",
                "message": "The TREC file has no ranking for this question.",
                "hint": "Re-run the other system on every question, or import with "
                "--missing-as-empty to count it as retrieving nothing.",
            }
        entry["latency_ms"] = 0.0  # replaced below: not measured
        record["queries"].append(entry)
    failed = any(q["error"] for q in record["queries"])
    record["status"] = "partial" if failed else "complete"
    record["summary"] = summarize(record["queries"], len(dataset))
    record["summary"]["latency_ms"] = None
    for entry in record["queries"]:
        entry["latency_ms"] = None
    record["import"] = {
        "format": "trec",
        "source": SOURCE,
        "file": p.name,
        "file_sha256": trec.sha256,
        "rows": trec.rows,
        "questions_in_file": len(in_file),
        "questions_not_in_dataset": len(in_file - known),
        "dataset_questions_not_in_file": len(known - in_file),
        "missing_as_empty": missing_as_empty,
        "rank_disagrees_with_score": len(trec.rank_disagrees_with_score),
        "rankings_cut_to_k": k,
        "ordered_by": "the rank column",
    }
    record["run_id"] = uuid.uuid4().hex
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return record


def import_notes(record: Mapping[str, Any]) -> list[str]:
    """Plain-language notes about what an import could not know or had to decide."""
    info = record["import"]
    notes = [
        "Latency, stage evidence and permission checks are not measured for an imported run.",
    ]
    if info["questions_not_in_dataset"]:
        notes.append(
            f"{info['questions_not_in_dataset']} questions in the file are not in the dataset "
            "and were ignored."
        )
    missing = info["dataset_questions_not_in_file"]
    if missing:
        notes.append(
            f"{missing} dataset questions have no ranking in the file: "
            + (
                "counted as retrieving nothing."
                if info["missing_as_empty"]
                else "recorded as errors."
            )
        )
    if info["rank_disagrees_with_score"]:
        notes.append(
            f"{info['rank_disagrees_with_score']} questions rank a lower-scored document above a "
            "higher-scored one; the rank column was used (trec_eval would use the scores)."
        )
    return notes
