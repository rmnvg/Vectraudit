"""RAGChecker claim-level results as answer judgments for ``attach-answers``.

Teams that use [RAGChecker](https://github.com/amazon-science/RAGChecker) (Ru et al.,
NeurIPS 2024 Datasets and Benchmarks; Apache-2.0) keep running it; this reads the file its
``RAGResults.to_json()`` writes and keeps what Vectraudit does not measure: **which delivered
document supported the answer, and which one misled it.** For every question RAGChecker
labels claim pairs ``Entailment``, ``Neutral`` or ``Contradiction``:

- ``answer2response``: is each response claim entailed by the reference answer (precision)?
- ``response2answer``: is each reference claim entailed by the response (recall)?
- ``retrieved2answer``: reference claims by chunks, which chunk entails which claim.
- ``retrieved2response``: response claims by chunks.

The counts here follow RAGChecker's own definitions (``ragchecker/computation.py``) and are
re-derived from the labels, not copied from its metrics, so they can be checked against them
(``unsupported_incorrect / response`` is RAGChecker's hallucination, ``reference_in_context
/ reference`` its claim recall, and so on). RAGChecker treats ``Contradiction`` like
``Neutral``; here a delivered document contradicting a reference claim is also counted on its
own, because it often means a stale document.

- **Scores and claim facts, never decisions.** One score-only judgment is written per chosen
  metric (``ragchecker.f1``), with a verdict only from a threshold the user declares. Another
  line per question (``ragchecker.claims``) carries the claim counts and, per document, how
  many reference and response claims it entails, and which reference claims by position
  (so the smallest set of documents covering the reference answer can be proposed). It has
  no verdict.
- **Retrieval metrics are refused.** ``claim_recall`` and ``context_precision`` measure
  retrieval, which Vectraudit measures against the benchmark's labels itself.
- **Chunk evidence needs matching documents.** Each ``retrieved_context[j].doc_id`` must be a
  document the run delivered or put in the model context, once. Otherwise that question's
  chunk evidence is dropped (and counted); its scores are still written, since they do not
  depend on which documents the chunks were.
- **Malformed rows are refused and counted,** never repaired.
- **Nothing private is copied:** claims, responses, reference answers, questions and chunk
  text are never written, only counts, flags, scores and document IDs.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vectraudit.evaluation.deepeval_import import _norm

PREFIX = "ragchecker."
CLAIMS_EVALUATOR = PREFIX + "claims"
ENTAILMENT, NEUTRAL, CONTRADICTION = "Entailment", "Neutral", "Contradiction"
LABELS = (ENTAILMENT, NEUTRAL, CONTRADICTION)
RETRIEVAL_METRICS = ("claim_recall", "context_precision")
"""Measured against the benchmark's labels by Vectraudit itself, so never imported."""
SCORE_METRICS = (
    "precision",
    "recall",
    "f1",
    "faithfulness",
    "context_utilization",
    "noise_sensitivity_in_relevant",
    "noise_sensitivity_in_irrelevant",
    "hallucination",
    "self_knowledge",
)
LOWER_IS_BETTER = frozenset(
    {"hallucination", "noise_sensitivity_in_relevant", "noise_sensitivity_in_irrelevant"}
)
NO_VERDICT = frozenset({"self_knowledge"})
"""Correct answers from the model's own knowledge are neither good nor bad on their own."""
MAX_BYTES = 200 * 2**20
MAX_CHUNKS = 500
MAX_CLAIMS = 1_000


class RagCheckerImportError(ValueError):
    """The RAGChecker file or settings cannot be used; the message says why."""


@dataclass(frozen=True)
class Facts:
    """Claim counts and per-chunk support derived from one question's labels."""

    claims: dict[str, int]
    support: list[dict[str, Any]]
    """Per chunk, in ``retrieved_context`` order. ``reference_claims`` lists the indices
    (never the text) of the reference claims the chunk entails."""
    metrics: dict[str, float]
    """RAGChecker's metrics, re-derived from the labels with its formulas."""


def load_output(path: str | Path) -> list[dict[str, Any]]:
    """The per-question results of a RAGChecker output file."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise RagCheckerImportError(f"{path} is larger than {MAX_BYTES // 2**20} MiB.")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise RagCheckerImportError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise RagCheckerImportError(f"{path} is not JSON.") from exc
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise RagCheckerImportError(
            f'{path} is not a RAGChecker output file: expected {{"results": [...]}} written '
            "by RAGChecker's evaluator."
        )
    return rows


def metric_names(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Metrics RAGChecker recorded, and how many rows have each."""
    found: Counter[str] = Counter()
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            found.update(k for k, v in metrics.items() if _score(v) is not None)
    return found


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _flat(value: Any) -> list[bool]:
    """Entailment flags of a label list; raises ``ValueError`` on anything else."""
    if not isinstance(value, list) or any(label not in LABELS for label in value):
        raise ValueError(f"labels must be a list of {', '.join(LABELS)}")
    return [label == ENTAILMENT for label in value]


def _grid(value: Any, rows: int, cols: int) -> list[list[str]]:
    """A ``rows`` by ``cols`` matrix of labels; raises ``ValueError`` on any other shape."""
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("label matrix does not match the number of claims")
    for line in value:
        if not isinstance(line, list) or len(line) != cols:
            raise ValueError("matrix rows must match the number of chunks")
        if any(label not in LABELS for label in line):
            raise ValueError(f"labels must be {', '.join(LABELS)}")
    return value


def derive(row: Mapping[str, Any]) -> tuple[Facts, list[str | None]]:
    """Facts for one result row, and its chunk document IDs (``None`` where one is missing).

    Raises ``ValueError`` when the labels or their shapes are malformed.
    """
    context = row.get("retrieved_context")
    if not isinstance(context, list) or len(context) > MAX_CHUNKS:
        raise ValueError(f"retrieved_context must be a list of at most {MAX_CHUNKS} chunks")
    ids: list[str | None] = []
    for chunk in context:
        doc = chunk.get("doc_id") if isinstance(chunk, Mapping) else None
        ids.append(doc if isinstance(doc, str) and doc.strip() else None)
    chunks = len(context)
    answer2response = row.get("answer2response")
    response2answer = row.get("response2answer")
    if not isinstance(answer2response, list) or not isinstance(response2answer, list):
        raise ValueError("answer2response and response2answer must be lists")
    if max(len(answer2response), len(response2answer)) > MAX_CLAIMS:
        raise ValueError(f"more than {MAX_CLAIMS} claims")
    correct = _flat(answer2response)
    in_response = _flat(response2answer)
    responses, references = len(correct), len(in_response)
    labelled_response = _grid(row.get("retrieved2response"), responses, chunks)
    labelled_answer = _grid(row.get("retrieved2answer"), references, chunks)
    ret2resp = [[label == ENTAILMENT for label in line] for line in labelled_response]
    ret2ans = [[label == ENTAILMENT for label in line] for line in labelled_answer]
    contradicts = [[label == CONTRADICTION for label in line] for line in labelled_answer]

    recalled = [any(line) for line in ret2ans]
    useful = [any(ret2ans[i][j] for i in range(references)) for j in range(chunks)]
    unfaithful = [not any(line) for line in ret2resp]
    incorrect = [not c for c in correct]
    relevant_faithful = [
        any(useful[j] and ret2resp[s][j] for j in range(chunks)) for s in range(responses)
    ]
    irrelevant_faithful = [
        any(not useful[j] and ret2resp[s][j] for j in range(chunks)) and not relevant_faithful[s]
        for s in range(responses)
    ]
    claims = {
        "reference": references,
        "response": responses,
        "response_incorrect": sum(incorrect),
        "reference_in_context": sum(recalled),
        "reference_in_response": sum(in_response),
        "reference_used": sum(r and u for r, u in zip(recalled, in_response, strict=True)),
        "unsupported_incorrect": sum(u and i for u, i in zip(unfaithful, incorrect, strict=True)),
        "unsupported_correct": sum(u and not i for u, i in zip(unfaithful, incorrect, strict=True)),
        "from_irrelevant_incorrect": sum(
            f and i for f, i in zip(irrelevant_faithful, incorrect, strict=True)
        ),
        "from_relevant_incorrect": sum(
            f and i for f, i in zip(relevant_faithful, incorrect, strict=True)
        ),
        "context_documents": chunks,
        "useful_documents": sum(useful),
    }
    support = [
        {
            "reference": sum(ret2ans[i][j] for i in range(references)),
            "response": sum(ret2resp[s][j] for s in range(responses)),
            "response_incorrect": sum(ret2resp[s][j] and incorrect[s] for s in range(responses)),
            "contradicts_reference": sum(contradicts[i][j] for i in range(references)),
            "reference_claims": [i for i in range(references) if ret2ans[i][j]],
        }
        for j in range(chunks)
    ]
    return Facts(claims, support, _metrics(claims, correct, in_response, chunks)), ids


def _metrics(
    claims: Mapping[str, int], correct: Sequence[bool], in_response: Sequence[bool], chunks: int
) -> dict[str, float]:
    """RAGChecker's metrics from the counts, as ``ragchecker/computation.py`` defines them."""
    responses, references = claims["response"], claims["reference"]
    precision = sum(correct) / responses if responses else 0.0
    recall = sum(in_response) / references if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0
    have_context = chunks > 0 and references > 0
    retrieval = chunks > 0 and references > 0 and responses > 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "claim_recall": claims["reference_in_context"] / references if have_context else 0.0,
        "context_precision": claims["useful_documents"] / chunks if have_context else 0.0,
        "context_utilization": (
            claims["reference_used"] / claims["reference_in_context"]
            if have_context and claims["reference_in_context"]
            else 0.0
        ),
        "faithfulness": (
            (responses - claims["unsupported_incorrect"] - claims["unsupported_correct"])
            / responses
            if chunks and responses
            else 0.0
        ),
        "hallucination": (
            claims["unsupported_incorrect"] / responses if chunks and responses else 0.0
        ),
        "self_knowledge": (
            claims["unsupported_correct"] / responses if chunks and responses else 0.0
        ),
        "noise_sensitivity_in_relevant": (
            claims["from_relevant_incorrect"] / responses if retrieval else 0.0
        ),
        "noise_sensitivity_in_irrelevant": (
            claims["from_irrelevant_incorrect"] / responses if retrieval else 0.0
        ),
    }


def verdict(metric: str, score: float, threshold: float) -> str:
    """``correct`` when the score meets the declared pass mark (lower is better for noise)."""
    ok = score <= threshold if metric in LOWER_IS_BETTER else score >= threshold
    return "correct" if ok else "incorrect"


def to_judgments(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    questions: Mapping[str, str | None],
    *,
    run_documents: Mapping[str, set[str]] | None,
    extractor: str,
    checker: str,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for the chosen metrics, a claims line per question, and what was skipped.

    ``questions`` maps benchmark question IDs to their text; ``run_documents`` maps them to
    the document IDs the run delivered or put in the model context (``None``: no run, so no
    chunk evidence).
    """
    names = [m.removeprefix(PREFIX) for m in metrics]
    refused = sorted({m for m in names if m in RETRIEVAL_METRICS})
    if refused:
        raise RagCheckerImportError(
            f"{', '.join(refused)} measure retrieval; Vectraudit measures retrieval against "
            "the benchmark's labels itself."
        )
    unknown = sorted({m for m in names if m not in SCORE_METRICS})
    if unknown:
        raise RagCheckerImportError(
            f"Unknown metric {', '.join(unknown)}. Answer metrics: {', '.join(SCORE_METRICS)}."
        )
    limits = {k.removeprefix(PREFIX): v for k, v in (thresholds or {}).items()}
    for name, value in limits.items():
        if name not in names:
            raise RagCheckerImportError(f"--threshold names {name}, which is not a chosen metric.")
        if name in NO_VERDICT:
            raise RagCheckerImportError(
                f"{name} is neither good nor bad on its own (a correct claim from the model's "
                "own knowledge); it cannot have a pass mark."
            )
        if not math.isfinite(value):
            raise RagCheckerImportError(f"The threshold for {name} must be finite.")
    for option, model in (("extractor", extractor), ("checker", checker)):
        if not model.strip() or len(model) > 80:
            raise RagCheckerImportError(
                f"--{option} must name the model, in at most 80 characters."
            )
    version = f"ragchecker; extractor={extractor}; checker={checker}"
    by_text: dict[str, list[str]] = defaultdict(list)
    for question_id, question in questions.items():
        if question:
            by_text[_norm(question)].append(question_id)
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        matched = _match(row, questions, by_text, counts)
        if matched is None:
            continue
        qid = matched
        if qid in seen:
            counts["duplicate_question"] += 1
            continue
        try:
            facts, doc_ids = derive(row)
        except ValueError:
            counts["malformed"] += 1
            continue
        seen.add(qid)
        recorded = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        for name in dict.fromkeys(names):
            score = _score(recorded.get(name)) if recorded else None
            score = facts.metrics[name] if score is None else score
            line: dict[str, Any] = {
                "query_id": qid,
                "evaluator": PREFIX + name,
                "kind": "model",
                "evaluator_version": version,
                "score": score,
            }
            if name in limits:
                line["verdict"] = verdict(name, score, limits[name])
            out.append(line)
            counts["judgments"] += 1
        reason = _chunk_problem(doc_ids, qid, run_documents)
        if reason is not None:
            counts[f"chunk_evidence_disabled_{reason}"] += 1
            continue
        out.append(
            {
                "query_id": qid,
                "evaluator": CLAIMS_EVALUATOR,
                "kind": "model",
                "evaluator_version": version,
                "claims": facts.claims,
                "claim_support": {
                    doc: support for doc, support in zip(doc_ids, facts.support, strict=True)
                },
            }
        )
        counts["claim_lines"] += 1
    return out, counts


def _chunk_problem(
    doc_ids: Sequence[str | None], qid: str, run_documents: Mapping[str, set[str]] | None
) -> str | None:
    """Why this question's chunks cannot be tied to documents in the run, or ``None``."""
    if run_documents is None:
        return "no_run"
    if qid not in run_documents:
        return "not_in_run"
    delivered = run_documents[qid]
    if not doc_ids:
        return "no_context"
    if any(doc is None for doc in doc_ids):
        return "missing_doc_id"
    if len(set(doc_ids)) != len(doc_ids):
        return "duplicate_doc_id"
    if any(doc not in delivered for doc in doc_ids):
        return "unknown_doc_id"
    return None


def _match(
    row: Mapping[str, Any],
    questions: Mapping[str, str | None],
    by_text: Mapping[str, list[str]],
    counts: Counter[str],
) -> str | None:
    query_id = row.get("query_id")
    if (
        isinstance(query_id, (str, int))
        and not isinstance(query_id, bool)
        and str(query_id) in questions
    ):
        counts["matched_by_query_id"] += 1
        return str(query_id)
    text = row.get("query")
    found = by_text.get(_norm(text)) if isinstance(text, str) else None
    if found and len(found) == 1:
        counts["matched_by_text"] += 1
        return found[0]
    counts["ambiguous_text" if found else "unmatched"] += 1
    return None


def run_documents(run: Mapping[str, Any]) -> dict[str, set[str]]:
    """Per question, the document IDs a run delivered or put in the model context."""
    out: dict[str, set[str]] = {}
    for q in run.get("queries") or []:
        docs = set(q.get("ids") or [])
        context = q.get("model_context")
        if isinstance(context, Mapping):
            docs.update(i["id"] for i in context.get("items") or [] if isinstance(i, Mapping))
        out[q["query_id"]] = docs
    return out


def render(counts: Mapping[str, int], written: int, out: str, metrics: Sequence[str]) -> str:
    """What the import did, for the terminal."""
    lines = [f"Wrote {written} judgments ({counts.get('claim_lines', 0)} claim lines) to {out}."]
    labels = {
        "matched_by_query_id": "matched by query_id",
        "matched_by_text": "matched by question text",
        "ambiguous_text": "with a question matching several benchmark questions",
        "unmatched": "matching no benchmark question",
        "duplicate_question": "for a question already imported (skipped)",
        "malformed": "malformed (labels or shapes invalid; refused)",
    }
    reasons = {
        "no_run": "no --run",
        "not_in_run": "the question is not in the run",
        "no_context": "no chunks",
        "missing_doc_id": "a chunk has no doc_id",
        "duplicate_doc_id": "a doc_id repeats",
        "unknown_doc_id": "a doc_id is not in the run",
    }
    labels |= {
        f"chunk_evidence_disabled_{key}": f"scored without chunk evidence ({why})"
        for key, why in reasons.items()
    }
    for key, label in labels.items():
        if counts.get(key):
            lines.append(f"  {counts[key]} rows {label}")
    names = dict.fromkeys(PREFIX + m.removeprefix(PREFIX) for m in metrics)
    lines.append(
        "Scores only unless you declared a pass mark; attach with attach-answers "
        "(--threshold " + " --threshold ".join(f"{n}=VALUE" for n in names) + " adds verdicts)."
    )
    return "\n".join(lines)
