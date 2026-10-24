"""Attach answer judgments from existing tools to a run, and diagnose each question.

Vectraudit does not grade answers. Teams already do, with human review or tools such as
Ragas, DeepEval, Langfuse scores or Phoenix evaluations. This module reads those judgments
from a JSONL file, one judgment per line::

    {"query_id": "q1", "evaluator": "ragas.answer_correctness", "evaluator_version": "0.2",
     "kind": "model", "score": 0.42}
    {"trace_id": "4bf9…", "evaluator": "support-review", "rubric": "v3", "kind": "human",
     "verdict": "incorrect"}

The file may also be a Langfuse scores export or LangSmith feedback export, read through
:mod:`vectraudit.evaluation.judgment_formats` once the caller names which scores judge
answers. A judgment names its question by ``query_id`` or, for runs imported from traces, by
``trace_id``. ``verdict`` is ``correct``, ``partial`` or ``incorrect``; a judgment with only a
``score`` gets a verdict from a threshold the caller declares for that evaluator, never from
a default. ``outcome`` (``answered``, ``abstained`` or ``handoff``) separates an answer from a
generated abstention or an appropriate human handoff. Explanations and answer text are not
read, so nothing private is copied into the run.

Human judgments decide a question's verdict when present; otherwise model judgments do.
Disagreement among the deciding judgments makes the verdict ``disputed`` instead of picking
one. Each model evaluator's agreement with human review is reported where both judged.

The diagnosis states the first observable loss on the path from retrieval to answer, with
the facts behind it; it is not a root cause. It needs ``required_evidence`` in the
benchmark: without it, which documents suffice to answer is not known.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.evaluation import judgment_formats

ANSWERS_FORMAT_VERSION = 1
VERDICTS = ("correct", "partial", "incorrect")
OUTCOMES = ("answered", "abstained", "handoff")
KINDS = ("human", "model")
MAX_JUDGMENTS = 200_000
_TEXT_LIMIT = 200
DIAGNOSES = {
    "answered_correctly": "Answered correctly.",
    "correct_without_required_evidence": (
        "Answered correctly although no required evidence group reached the model; the model "
        "may have used other knowledge, or the required evidence is incomplete."
    ),
    "answered_wrong_despite_evidence": (
        "Answered wrong although a complete required evidence group reached the model."
    ),
    "lost_in_context": (
        "A complete required evidence group was retrieved and delivered, but did not reach "
        "the model intact."
    ),
    "lost_downstream": (
        "Required evidence appeared in a recorded stage but was dropped before delivery."
    ),
    "not_observed": (
        "Required evidence was not in any recorded stage. Missing instrumentation or an "
        "absent or stale source is still possible."
    ),
    "context_unknown": (
        "Retrieval delivered a complete required group, but whether it reached the model is "
        "unknown."
    ),
    "abstained_despite_evidence": (
        "Abstained although a complete required evidence group reached the model."
    ),
    "abstained": "Abstained; the required evidence did not demonstrably reach the model.",
    "handoff": "Handed off to a person.",
    "disputed": "Judgments disagree; resolve them before diagnosing.",
    "unjudged": "No usable verdict (a score without a declared threshold).",
    "no_required_evidence": "The benchmark declares no required evidence for this question.",
}


class AnswerJudgmentError(ValueError):
    """The judgments file or settings cannot be used; the message says why."""


@dataclass(frozen=True)
class Judgment:
    """One evaluator's judgment of one question's answer."""

    evaluator: str
    kind: str
    verdict: str | None
    score: float | None
    outcome: str
    evaluator_version: str | None
    rubric: str | None

    def record(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "kind": self.kind,
            "verdict": self.verdict,
            "score": self.score,
            "outcome": self.outcome,
            **({"evaluator_version": self.evaluator_version} if self.evaluator_version else {}),
            **({"rubric": self.rubric} if self.rubric else {}),
        }


def _short(value: Any, what: str, where: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > _TEXT_LIMIT:
        raise AnswerJudgmentError(
            f"{where}: {what} must be a non-empty string of at most {_TEXT_LIMIT} characters"
        )
    return value


def _judgment(
    obj: Mapping[str, Any], where: str, thresholds: Mapping[str, float]
) -> tuple[str, str, Judgment]:
    keys = [k for k in ("query_id", "trace_id") if obj.get(k) is not None]
    if len(keys) != 1:
        raise AnswerJudgmentError(f"{where}: give exactly one of query_id or trace_id")
    key = _short(obj[keys[0]], keys[0], where, required=True)
    evaluator = _short(obj.get("evaluator"), "evaluator", where, required=True)
    assert key is not None
    assert evaluator is not None
    kind = obj.get("kind", "model")
    if kind not in KINDS:
        raise AnswerJudgmentError(f"{where}: kind must be human or model")
    outcome = obj.get("outcome", "answered")
    if outcome not in OUTCOMES:
        raise AnswerJudgmentError(f"{where}: outcome must be one of {', '.join(OUTCOMES)}")
    score = obj.get("score")
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
        raise AnswerJudgmentError(f"{where}: score must be a finite number")
    verdict = obj.get("verdict")
    if verdict is not None and verdict not in VERDICTS:
        raise AnswerJudgmentError(f"{where}: verdict must be one of {', '.join(VERDICTS)}")
    if verdict is None and score is not None and evaluator in thresholds:
        verdict = "correct" if score >= thresholds[evaluator] else "incorrect"
    if verdict is None and score is None and outcome == "answered":
        raise AnswerJudgmentError(f"{where}: a judgment needs a verdict or a score")
    return (
        keys[0],
        key,
        Judgment(
            evaluator=evaluator,
            kind=kind,
            verdict=verdict,
            score=None if score is None else float(score),
            outcome=outcome,
            evaluator_version=_short(obj.get("evaluator_version"), "evaluator_version", where),
            rubric=_short(obj.get("rubric"), "rubric", where),
        ),
    )


def _records(text: str) -> list[tuple[str, Any]]:
    """``(where, object)`` for a JSON array, an API page (``{"data": [...]}``) or JSON lines."""
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            whole = json.loads(stripped)
        except ValueError:
            whole = None  # JSON lines, or broken JSON reported line by line below
        if isinstance(whole, dict) and isinstance(whole.get("data"), list):
            whole = whole["data"]
        if isinstance(whole, list):
            return [(f"record {i}", obj) for i, obj in enumerate(whole, 1)]
    out = []
    for line, raw in enumerate(text.splitlines(), 1):
        if raw.strip():
            try:
                out.append((f"line {line}", json.loads(raw)))
            except ValueError as exc:
                raise AnswerJudgmentError(f"line {line}: not JSON") from exc
    return out


@dataclass(frozen=True)
class JudgmentFile:
    """The judgments read from a file, its format, and what was skipped and why."""

    rows: list[tuple[str, str, Judgment]]
    source_format: str
    skipped: dict[str, dict[str, int]] = field(default_factory=dict)


def read_judgment_file(
    path: str | Path,
    thresholds: Mapping[str, float] | None = None,
    *,
    evaluators: Mapping[str, str | None] | None = None,
    labels: Mapping[str, str] | None = None,
) -> JudgmentFile:
    """Judgments from a judgments file, a Langfuse scores export or LangSmith feedback.

    ``evaluators`` names the vendor scores that judge answers, each with the kind (``human``
    or ``model``) to use where the export does not say who judged, or ``None``. ``labels``
    maps categorical values to verdicts. Both apply to vendor exports only.
    """
    thresholds = dict(thresholds or {})
    for name, value in thresholds.items():
        if not math.isfinite(value):
            raise AnswerJudgmentError(f"threshold for {name} must be finite")
    for name, kind in (evaluators or {}).items():
        if kind is not None and kind not in KINDS:
            raise AnswerJudgmentError(f"evaluator {name}: kind must be human or model")
    for label, verdict in (labels or {}).items():
        if verdict not in VERDICTS:
            raise AnswerJudgmentError(
                f"label {label!r}: verdict must be one of {', '.join(VERDICTS)}"
            )
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AnswerJudgmentError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    records = _records(text)
    if len(records) > MAX_JUDGMENTS:
        raise AnswerJudgmentError(f"more than {MAX_JUDGMENTS} judgments; split the file")
    for where, obj in records:
        if not isinstance(obj, dict):
            raise AnswerJudgmentError(f"{where}: expected a JSON object")
    shapes = {judgment_formats.shape_of(obj) or "vectraudit" for _, obj in records}
    if len(shapes) > 1:
        raise AnswerJudgmentError(
            f"{path} mixes formats ({', '.join(sorted(shapes))}); use one per file"
        )
    shape = shapes.pop() if shapes else "vectraudit"
    if shape == "vectraudit" and (evaluators or labels):
        raise AnswerJudgmentError(
            "--evaluator and --label apply to Langfuse and LangSmith exports; a judgments "
            "file names its evaluators and verdicts itself"
        )
    if shape != "vectraudit" and not evaluators:
        names = Counter(
            judgment_formats.evaluator_of(obj, shape) or "(unnamed)" for _, obj in records
        )
        listed = ", ".join(f"{n} ({c})" for n, c in sorted(names.items())[:30])
        raise AnswerJudgmentError(
            f"{path} is a {shape} export with scores named {listed}. Pass --evaluator NAME "
            "for each score that judges answers (NAME=human or NAME=model where the export "
            "does not say who judged); nothing is read by default."
        )
    selection = judgment_formats.Selection(evaluators or {}, labels or {})
    rows = []
    skipped: dict[str, Counter[str]] = defaultdict(Counter)
    for where, obj in records:
        if shape != "vectraudit":
            converted = judgment_formats.convert(obj, shape, selection)
            if isinstance(converted, str):
                skipped[converted][judgment_formats.evaluator_of(obj, shape) or "(unnamed)"] += 1
                continue
            obj = converted
        rows.append(_judgment(obj, where, thresholds))
    summary = {reason: dict(sorted(names.items())) for reason, names in sorted(skipped.items())}
    if not rows:
        why = "; ".join(
            f"{judgment_formats.SKIPPED[r]}: {sum(n.values())}" for r, n in summary.items()
        )
        raise AnswerJudgmentError(f"{path} contains no judgments" + (f" ({why})" if why else ""))
    return JudgmentFile(rows=rows, source_format=shape, skipped=summary)


def read_judgments(
    path: str | Path, thresholds: Mapping[str, float] | None = None
) -> list[tuple[str, str, Judgment]]:
    """``(key kind, key, judgment)`` rows; key kind is ``query_id`` or ``trace_id``."""
    return read_judgment_file(path, thresholds).rows


def _decide(judgments: Sequence[Judgment]) -> tuple[str | None, str | None, str]:
    """``(verdict, decided_by, outcome)``. Humans decide when present; ties are disputes."""
    outcomes = {j.outcome for j in judgments}
    outcome = (
        "handoff"
        if "handoff" in outcomes
        else "abstained"
        if "abstained" in outcomes
        else ("answered")
    )
    for kind in KINDS:
        verdicts = {j.verdict for j in judgments if j.kind == kind and j.verdict is not None}
        if len(verdicts) > 1:
            return "disputed", kind, outcome
        if verdicts:
            return verdicts.pop(), kind, outcome
    return None, None, outcome


def diagnose(query: Mapping[str, Any], verdict: str | None, outcome: str) -> dict[str, Any]:
    """The first observable loss for one question, with the facts that support it."""
    facts: list[str] = []
    sufficiency = query.get("sufficiency")
    evidence = query.get("evidence") or {}
    context = evidence.get("context") if isinstance(evidence, Mapping) else None
    if verdict is not None:
        facts.append(f"answer judged {verdict}")
    if outcome != "answered":
        facts.append(f"outcome: {outcome}")
    if isinstance(context, Mapping):
        for loss in context.get("losses") or []:
            facts.append(f"{loss['doc_id']} {loss['kind']} in the model context")
    if outcome == "handoff":
        return {"class": "handoff", "facts": facts}
    if verdict == "disputed":
        return {"class": "disputed", "facts": facts}
    if verdict is None and outcome == "answered":
        return {"class": "unjudged", "facts": facts}
    if not isinstance(sufficiency, Mapping):
        return {"class": "no_required_evidence", "facts": facts}
    delivered, in_context = sufficiency["delivered"], sufficiency["context"]
    facts.append(f"required evidence delivered: {delivered}; in model context: {in_context}")
    if outcome == "abstained":
        name = "abstained_despite_evidence" if in_context == "sufficient" else "abstained"
        return {"class": name, "facts": facts}
    if verdict == "correct":
        name = (
            "correct_without_required_evidence"
            if in_context == "insufficient"
            else ("answered_correctly")
        )
        return {"class": name, "facts": facts}
    if in_context == "sufficient":
        return {"class": "answered_wrong_despite_evidence", "facts": facts}
    if delivered == "sufficient":
        name = "lost_in_context" if in_context == "insufficient" else "context_unknown"
        return {"class": name, "facts": facts}
    missing = set(sufficiency.get("missing_from_delivered") or [])
    located = [
        f
        for f in (evidence.get("failures") if isinstance(evidence, Mapping) else None) or []
        if f.get("doc_id") in missing and f.get("stage")
    ]
    if located:
        facts += [f"{f['doc_id']} dropped at {f['stage']}" for f in located]
        return {"class": "lost_downstream", "facts": facts}
    facts += [f"{doc} not in any recorded stage" for doc in sorted(missing)]
    return {"class": "not_observed", "facts": facts}


def attach_answers(
    run: Mapping[str, Any],
    rows: Sequence[tuple[str, str, Judgment]],
    *,
    source_sha256: str | None = None,
    source_format: str = "vectraudit",
    skipped: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """A copy of the run with ``answer`` on each judged question and an ``answers`` summary.

    Judgments for questions the run does not contain are counted, never attached. Failed
    questions keep their judgments but get no diagnosis: there is no evidence to read.
    """
    out: dict[str, Any] = json.loads(json.dumps(run))
    by_query = {q["query_id"]: q for q in out["queries"]}
    by_trace = {q["trace_id"]: q for q in out["queries"] if isinstance(q.get("trace_id"), str)}
    grouped: dict[str, list[Judgment]] = defaultdict(list)
    unknown = 0
    for key_kind, key, judgment in rows:
        q = by_query.get(key) if key_kind == "query_id" else by_trace.get(key)
        if q is None:
            unknown += 1
            continue
        grouped[q["query_id"]].append(judgment)
    verdicts: Counter[str] = Counter()
    diagnoses: Counter[str] = Counter()
    pairs: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for qid, judgments in grouped.items():
        q = by_query[qid]
        verdict, decided_by, outcome = _decide(judgments)
        answer: dict[str, Any] = {
            "verdict": verdict,
            "decided_by": decided_by,
            "outcome": outcome,
            "judgments": [j.record() for j in judgments],
        }
        if q.get("error") is None:
            answer["diagnosis"] = diagnose(q, verdict, outcome)
            diagnoses[answer["diagnosis"]["class"]] += 1
        q["answer"] = answer
        verdicts[verdict or "unjudged"] += 1
        human = {j.verdict for j in judgments if j.kind == "human" and j.verdict}
        if len(human) == 1:
            reference = next(iter(human))
            for j in judgments:
                if j.kind == "model" and j.verdict is not None:
                    pairs[j.evaluator].append((qid, j.verdict == reference))
    evaluators = sorted(
        {(j.evaluator, j.kind, j.evaluator_version, j.rubric) for _, _, j in rows},
        key=lambda e: tuple(x or "" for x in e),
    )
    out["answers"] = {
        "format_version": ANSWERS_FORMAT_VERSION,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "skipped_judgments": {r: dict(n) for r, n in (skipped or {}).items()},
        "judged_questions": len(grouped),
        "unmatched_judgments": unknown,
        "evaluators": [
            {"evaluator": e, "kind": k, "evaluator_version": v, "rubric": r}
            for e, k, v, r in evaluators
        ],
        "verdicts": dict(sorted(verdicts.items())),
        "diagnoses": dict(sorted(diagnoses.items())),
        "agreement_with_human": {
            name: {
                "questions": len(results),
                "agreed": sum(ok for _, ok in results),
                "disagreed_on": sorted(qid for qid, ok in results if not ok)[:50],
            }
            for name, results in sorted(pairs.items())
        },
    }
    return out


def attach_file(
    run: Mapping[str, Any],
    path: str | Path,
    thresholds: Mapping[str, float] | None = None,
    *,
    evaluators: Mapping[str, str | None] | None = None,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """:func:`attach_answers` from a judgments file or vendor export, recording its digest."""
    read = read_judgment_file(path, thresholds, evaluators=evaluators, labels=labels)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return attach_answers(
        run, read.rows, source_sha256=digest, source_format=read.source_format, skipped=read.skipped
    )


def render_summary(run: Mapping[str, Any]) -> str:
    """The attached judgments as plain text for the terminal."""
    answers = run["answers"]
    lines = [
        f"Judged {answers['judged_questions']} of {len(run['queries'])} questions"
        + (
            f"; {answers['unmatched_judgments']} judgments matched no question in this run"
            if answers["unmatched_judgments"]
            else ""
        )
        + ".",
        "Verdicts: " + (", ".join(f"{n} {v}" for v, n in answers["verdicts"].items()) or "none"),
    ]
    if answers["diagnoses"]:
        lines += ["", "Diagnosis (first observable loss, not a root cause):"]
        lines += [
            f"  {n}: {name.replace('_', ' ')}. {DIAGNOSES[name]}"
            for name, n in sorted(answers["diagnoses"].items(), key=lambda kv: -kv[1])
        ]
    for name, stats in answers["agreement_with_human"].items():
        lines.append(
            f"{name} (model judgments) agrees with human review on {stats['agreed']} of "
            f"{stats['questions']} questions judged by both."
        )
    hints = {
        "kind_undeclared": "pass --evaluator NAME=human or NAME=model if these are reviews; "
        "end-user ratings are not",
        "label_unmapped": "pass --label LABEL=correct|partial|incorrect",
    }
    for reason, names in (answers.get("skipped_judgments") or {}).items():
        lines.append(
            f"Skipped {sum(names.values())}: {judgment_formats.SKIPPED[reason]} "
            f"({', '.join(sorted(names)[:10])})"
            + (f"; {hints[reason]}" if reason in hints else "")
            + "."
        )
    unjudged = answers["verdicts"].get("unjudged")
    if unjudged:
        lines.append(
            f"{unjudged} questions have scores but no verdict: pass --threshold "
            "EVALUATOR=VALUE for each scored evaluator."
        )
    return "\n".join(lines)
