"""A reproducible random sample of questions for human review, for judge-corrected estimates.

Prediction-powered inference (:mod:`vectraudit.evaluation.ppi`) is valid only when the
human-reviewed units are a uniform random sample of the units the judge scored. A review queue,
a disputed-first list or hand-picked examples are not: using them would make the corrected
interval look sound and be wrong. So the sample is drawn here, once, by a rule anyone can
re-run:

- the **frame** is every exploratory question family of a run that did not error (held-out
  questions are never in it);
- families are ranked by ``sha256(f"{seed}:{family}")`` and the first ``size`` are chosen. The
  hash makes the draw uniform, independent of input order and identical on every machine and
  numpy version, so :func:`verify_sample` can recompute it and refuse a sample that was edited.

The unit is the **family** (questions sharing a ``family_id``, or a question on its own), so a
family of paraphrases is reviewed together and counts once.

The file holds identifiers only: family and question IDs, the seed and digests; no question,
answer or document text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.families import family_id
from vectraudit.evaluation.ppi import MIN_LABELLED

FORMAT = "vectraudit.review_sample"
FORMAT_VERSION = 1
RULE = "the first `size` frame families ordered by sha256(f'{seed}:{family}'), then sorted"


class ReviewSampleError(ValueError):
    """The sample cannot be drawn or read; ``hint`` says what to change."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def _digest(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def draw(frame: Sequence[str], size: int, seed: int) -> list[str]:
    """The ``size`` families chosen from ``frame`` (sorted): uniform, seeded, reproducible."""
    ranked = sorted(set(frame), key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest())
    return sorted(ranked[:size])


def frame_of(run: Mapping[str, Any], dataset: Dataset) -> dict[str, list[str]]:
    """``{family: [query ids]}`` for exploratory families of the run that did not error.

    A family with any held-out question, any question the run lacks or any question that
    errored is left out whole: the unit must be judged completely or not at all.
    """
    split = {q.query_id: q.split for q in dataset.queries}
    entries = {e["query_id"]: e for e in run.get("queries") or []}
    members: dict[str, list[str]] = {}
    for q in dataset.queries:
        members.setdefault(
            family_id({"query_id": q.query_id, "family_id": q.family_id}), []
        ).append(q.query_id)
    frame = {}
    for family, ids in members.items():
        if all(split[i] == "explore" for i in ids) and all(
            i in entries and not entries[i].get("error") for i in ids
        ):
            frame[family] = sorted(ids)
    return frame


@dataclass(frozen=True)
class ReviewSample:
    """A sample read back from its file."""

    dataset_sha256: str
    run_id: str | None
    seed: int
    size: int
    frame: tuple[str, ...]
    chosen: tuple[str, ...]
    questions: Mapping[str, tuple[str, ...]]
    """The chosen families' question IDs."""
    sha256: str
    """Digest of the file: what an estimate records as its sample."""


def build_sample(
    run: Mapping[str, Any], dataset: Dataset, size: int, seed: int = 0
) -> dict[str, Any]:
    """The sample file's content for ``size`` families of ``run``'s frame."""
    if run["dataset"]["sha256"] != dataset.sha256:
        raise ReviewSampleError(
            "The run was not recorded on this dataset version.",
            "Pass the dataset the run was recorded on.",
        )
    frame = frame_of(run, dataset)
    if size < MIN_LABELLED:
        raise ReviewSampleError(
            f"A review sample of {size} families is too small to correct an estimate.",
            f"Use at least {MIN_LABELLED} (100 or more gives a trustworthy interval).",
        )
    if size >= len(frame):
        raise ReviewSampleError(
            f"The frame has {len(frame)} families; a sample of {size} leaves none to correct.",
            "Review fewer families than the frame holds, or just use the human verdicts.",
        )
    chosen = draw(list(frame), size, seed)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "unit": "family",
        "dataset_sha256": dataset.sha256,
        "run_id": run.get("run_id"),
        "seed": seed,
        "size": size,
        "rule": RULE,
        "frame": sorted(frame),
        "frame_sha256": _digest(list(frame)),
        "chosen": chosen,
        "questions": {f: frame[f] for f in chosen},
        "note": (
            "A uniform random sample of the frame. Judge-corrected estimates use human "
            "verdicts only for these families; verdicts gathered any other way do not qualify."
        ),
    }


def write_sample(content: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p


def read_sample(path: str | Path) -> ReviewSample:
    """Read a sample file, refusing one whose frame, digests or draw do not add up."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        data = json.loads(raw)
    except OSError as e:
        raise ReviewSampleError(f"Cannot read {p}: {e.strerror or e}") from e
    except ValueError as e:
        raise ReviewSampleError(f"{p} is not JSON.") from e
    if (
        not isinstance(data, dict)
        or data.get("format") != FORMAT
        or data.get("format_version") != FORMAT_VERSION
    ):
        raise ReviewSampleError(f"{p} is not a {FORMAT} v{FORMAT_VERSION} file.")
    try:
        frame = [str(f) for f in data["frame"]]
        chosen = [str(f) for f in data["chosen"]]
        seed, size = int(data["seed"]), int(data["size"])
        questions = {str(f): tuple(str(i) for i in ids) for f, ids in data["questions"].items()}
        if _digest(frame) != data["frame_sha256"]:
            raise ReviewSampleError(
                "The frame does not match its digest: the sample file was edited.",
                "Draw the sample again with `workbench review-sample`.",
            )
        if draw(frame, size, seed) != sorted(chosen) or len(set(chosen)) != size:
            raise ReviewSampleError(
                "The chosen families are not what the recorded seed draws from the frame.",
                "Only a sample drawn by `workbench review-sample` is a random sample.",
            )
        if set(questions) != set(chosen):
            raise ReviewSampleError("The sample's questions do not match its chosen families.")
        return ReviewSample(
            dataset_sha256=str(data["dataset_sha256"]),
            run_id=data.get("run_id"),
            seed=seed,
            size=size,
            frame=tuple(sorted(frame)),
            chosen=tuple(sorted(chosen)),
            questions=questions,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except KeyError as e:
        raise ReviewSampleError(f"{p.name} lacks the field {e.args[0]!r}.") from e
    except (TypeError, AttributeError) as e:
        raise ReviewSampleError(f"{p.name} is malformed.") from e


def problems_with(sample: ReviewSample, run: Mapping[str, Any]) -> list[str]:
    """Why this sample cannot support an estimate for ``run`` (empty when it can)."""
    out = []
    if sample.dataset_sha256 != run["dataset"]["sha256"]:
        out.append("The sample was drawn for a different dataset version than this run's.")
    entries = {e["query_id"]: e for e in run.get("queries") or []}
    families: dict[str, list[str]] = {}
    for qid, entry in entries.items():
        families.setdefault(family_id(entry), []).append(qid)
    absent = [
        f
        for f in sample.frame
        if f not in families or any(entries[i].get("error") for i in families[f])
    ]
    if absent:
        out.append(
            f"{len(absent)} frame families are missing from this run or errored in it "
            f"(for example {absent[0]!r}): the frame is not this run's."
        )
    return out
