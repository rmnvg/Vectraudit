"""Turn a baseline/candidate pair of runs into ranked, evidence-linked findings.

The release check decides *whether* a candidate may ship; triage answers *where to look*.
Each finding names the query, the documents involved, and the recorded stage evidence it
came from. Nothing is inferred beyond what the runs recorded: without stage traces a lost
document is reported as dropped from results, not as a located loss.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from vectraudit.evaluation.metrics import SEMANTIC_METRICS

TRIAGE_FORMAT = "vectraudit.triage"
MAX_ITEMS = 500
EPS = 1e-9

Severity = Literal["blocking", "error", "regression", "info", "improvement"]
# Lower ranks sort first: safety gates, then failures to measure, then quality losses.
_RANK: dict[str, int] = {"blocking": 0, "error": 1, "regression": 2, "info": 3, "improvement": 4}


def _metric(q: Mapping[str, Any], name: str) -> float | None:
    if name == "ann_recall":
        value = q.get("ann_recall")
    else:
        metrics = q.get("metrics")
        value = metrics.get(name) if metrics else None
    return None if value is None else float(value)


def _primary(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> str | None:
    """The comparison's own preference: nDCG, then MRR, precision, recall, ANN recall."""
    for name in (*SEMANTIC_METRICS[::-1], "ann_recall"):
        if any(_metric(b, name) is not None and _metric(c, name) is not None for b, c in pairs):
            return name
    return None


def _failures(q: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]] | None:
    """Relevant documents not delivered, grouped by id; None when no evidence was recorded."""
    evidence = q.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("failures") is None:
        return None
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for failure in evidence["failures"]:
        grouped.setdefault(str(failure["doc_id"]), []).append(failure)
    return grouped


def _context(q: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The query's context analysis (see ``runner.context_analysis``), if it recorded one."""
    evidence = q.get("evidence")
    context = evidence.get("context") if isinstance(evidence, Mapping) else None
    return context if isinstance(context, Mapping) else None


def _context_findings(b: Mapping[str, Any], c: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Relevant documents retrieved but left out of, or cut short in, the model context."""
    bctx, cctx = _context(b), _context(c)
    if cctx is None:
        return []
    base_lost = {loss["doc_id"]: loss for loss in (bctx or {}).get("losses") or []}
    base_reached = set((bctx or {}).get("reached") or [])
    findings: list[dict[str, Any]] = []
    new, still, unknown_before = [], [], []
    for loss in cctx.get("losses") or []:
        doc = loss["doc_id"]
        if doc in base_lost:
            still.append(loss)
        elif doc in base_reached:
            new.append(loss)
        else:
            unknown_before.append(loss)

    def describe(losses: list[Mapping[str, Any]]) -> str:
        return ", ".join(f"{loss['doc_id']} ({loss['kind']})" for loss in losses)

    how = (
        " (inferred by matching document text in the prompt)"
        if (cctx.get("method") == "text_match")
        else ""
    )
    if new:
        findings.append(
            {
                "kind": "context_lost",
                "severity": "regression",
                "doc_ids": [loss["doc_id"] for loss in new],
                "context": new,
                "message": f"{len(new)} relevant document(s) were retrieved and delivered but "
                f"did not fully reach the model, as they did in the baseline{how}: "
                + describe(new),
            }
        )
    if unknown_before:
        findings.append(
            {
                "kind": "context_loss",
                "severity": "info",
                "doc_ids": [loss["doc_id"] for loss in unknown_before],
                "context": unknown_before,
                "message": f"{len(unknown_before)} relevant document(s) were delivered but did "
                f"not fully reach the model{how}; the baseline's context does not show whether "
                "this is new: " + describe(unknown_before),
            }
        )
    if still:
        findings.append(
            {
                "kind": "context_still_lost",
                "severity": "info",
                "doc_ids": [loss["doc_id"] for loss in still],
                "context": still,
                "message": f"{len(still)} relevant document(s) were delivered but did not "
                f"fully reach the model in both runs{how}: " + describe(still),
            }
        )
    lost_now = {loss["doc_id"] for loss in cctx.get("losses") or []}
    recovered = [
        doc for doc in base_lost if doc in set(cctx.get("reached") or []) and doc not in lost_now
    ]
    if recovered:
        findings.append(
            {
                "kind": "context_recovered",
                "severity": "improvement",
                "doc_ids": recovered,
                "message": f"{len(recovered)} relevant document(s) now reach the model in full: "
                + ", ".join(recovered),
            }
        )
    return findings


def _answer_findings(b: Mapping[str, Any], c: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Answer verdicts that changed between the runs, with the candidate's diagnosis."""
    ba, ca = b.get("answer"), c.get("answer")
    if not isinstance(ba, Mapping) or not isinstance(ca, Mapping):
        return []
    before, after = ba.get("verdict"), ca.get("verdict")
    diagnosis = (ca.get("diagnosis") or {}).get("class")
    why = f" First observable loss: {diagnosis.replace('_', ' ')}." if diagnosis else ""
    if before == "correct" and after in ("partial", "incorrect"):
        return [
            {
                "kind": "answer_regressed",
                "severity": "regression",
                "doc_ids": [],
                "diagnosis": diagnosis,
                "message": f"The answer was judged correct in the baseline and {after} in the "
                f"candidate.{why}",
            }
        ]
    if before in ("partial", "incorrect") and after == "correct":
        return [
            {
                "kind": "answer_recovered",
                "severity": "improvement",
                "doc_ids": [],
                "message": f"The answer was judged {before} in the baseline and correct now.",
            }
        ]
    if after in ("partial", "incorrect", "disputed"):
        return [
            {
                "kind": "answer_not_correct",
                "severity": "info",
                "doc_ids": [],
                "diagnosis": diagnosis,
                "message": f"The candidate's answer is judged {after} (baseline: "
                f"{before or 'not judged'}).{why}",
            }
        ]
    return []


def _stages_with(q: Mapping[str, Any], doc: str) -> list[str]:
    evidence = q.get("evidence")
    stages = evidence.get("stages") if isinstance(evidence, Mapping) else None
    return [s["name"] for s in stages or [] if any(h["id"] == doc for h in s["hits"])]


def _diverged_at(b: Mapping[str, Any], c: Mapping[str, Any], doc: str) -> str | None:
    """First stage, in baseline order, that held the document in the baseline but not the
    candidate. Only stages recorded under the same name in both runs are compared."""
    candidate_stages = set(_stage_names(c))
    held = set(_stages_with(c, doc))
    for name in _stages_with(b, doc):
        if name in candidate_stages and name not in held:
            return name
    return None


def _stage_names(q: Mapping[str, Any]) -> list[str]:
    evidence = q.get("evidence")
    stages = evidence.get("stages") if isinstance(evidence, Mapping) else None
    return [s["name"] for s in stages or []]


def _losses(
    q: Mapping[str, Any], docs: Sequence[str], baseline: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    grouped = _failures(q) or {}
    out = []
    for doc in docs:
        located = [f for f in grouped.get(doc, []) if f.get("stage")]
        out.append(
            {
                "doc_id": doc,
                "baseline_stages": _stages_with(baseline, doc) if baseline else None,
                "candidate_stages": _stages_with(q, doc),
                "diverged_at": _diverged_at(baseline, q, doc) if baseline else None,
                "stages": [
                    {
                        "stage": f["stage"],
                        "parents": list(f.get("parents", [])),
                        "evidence": f.get("evidence"),
                        "reason": f["reason"],
                    }
                    for f in located
                ],
                "reason": located[0]["reason"]
                if located
                else "not present in recorded stages; internal cause unknown",
            }
        )
    return out


def _query_findings(
    b: Mapping[str, Any], c: Mapping[str, Any], primary: str | None
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    findings: list[dict[str, Any]] = []
    if c.get("error") or b.get("error"):
        failed = [role for role, q in (("baseline", b), ("candidate", c)) if q.get("error")]
        findings.append(
            {
                "kind": "error",
                "severity": "error",
                "doc_ids": [],
                "runs": failed,
                "message": f"The query failed in the {' and '.join(failed)} run; no score.",
            }
        )
        return findings, None, None

    forbidden = list(c.get("forbidden_ids") or [])
    if forbidden:
        before = set(b.get("forbidden_ids") or [])
        findings.append(
            {
                "kind": "permission",
                "severity": "blocking",
                "doc_ids": forbidden,
                "new_doc_ids": [d for d in forbidden if d not in before],
                "documents": [{"doc_id": d, "stages": _stages_with(c, d)} for d in forbidden],
                "message": f"Delivered {len(forbidden)} forbidden document(s): "
                + ", ".join(forbidden)
                + ("" if before else ". The baseline delivered none."),
            }
        )
    if c.get("no_answer_correct") is False:
        returned = list(c.get("ids") or [])
        findings.append(
            {
                "kind": "no_answer",
                "severity": "blocking",
                "doc_ids": returned,
                "baseline_correct": b.get("no_answer_correct"),
                "message": "No relevant document exists, but the candidate returned "
                f"{len(returned)} result(s)"
                + (
                    "; the baseline correctly returned none." if b.get("no_answer_correct") else "."
                ),
            }
        )

    bv = _metric(b, primary) if primary else None
    cv = _metric(c, primary) if primary else None
    delta = None if bv is None or cv is None else cv - bv
    base_missing, cand_missing = _failures(b), _failures(c)
    if base_missing is not None and cand_missing is not None:
        lost = [d for d in cand_missing if d not in base_missing]
        still = [d for d in cand_missing if d in base_missing]
        recovered = [d for d in base_missing if d not in cand_missing]
        if lost:
            findings.append(
                {
                    "kind": "lost_relevant",
                    "severity": "regression",
                    "doc_ids": lost,
                    "documents": _losses(c, lost, b),
                    "message": f"The baseline delivered {len(lost)} relevant document(s) "
                    "that the candidate lost: " + ", ".join(lost),
                }
            )
        if recovered:
            findings.append(
                {
                    "kind": "recovered_relevant",
                    "severity": "improvement",
                    "doc_ids": recovered,
                    "message": f"The candidate now delivers {len(recovered)} relevant "
                    "document(s) the baseline missed: " + ", ".join(recovered),
                }
            )
        if still:
            findings.append(
                {
                    "kind": "still_missing",
                    "severity": "info",
                    "doc_ids": still,
                    "documents": _losses(c, still),
                    "message": f"{len(still)} relevant document(s) are missing in both runs: "
                    + ", ".join(still),
                }
            )
    findings += _context_findings(b, c)
    findings += _answer_findings(b, c)
    if delta is not None and delta < -EPS:
        dropped = [i for i in b.get("ids") or [] if i not in set(c.get("ids") or [])]
        if not any(f["kind"] == "lost_relevant" for f in findings):
            # Without located losses the ranking changed; say so instead of guessing relevance.
            findings.append(
                {
                    "kind": "ranking_regressed",
                    "severity": "regression",
                    "doc_ids": dropped,
                    "message": f"{primary} fell from {bv:.3f} to {cv:.3f}"
                    + (
                        "; quality averages repeats. Inspect individual executions for losses."
                        if c.get("repeat_metrics") and len(c["repeat_metrics"]) > 1
                        else f"; results no longer returned: {', '.join(dropped)}."
                        if dropped
                        else "; the same documents are ranked lower."
                    ),
                }
            )
    elif delta is not None and delta > EPS and not findings:
        findings.append(
            {
                "kind": "improved",
                "severity": "improvement",
                "doc_ids": [],
                "message": f"{primary} rose from {bv:.3f} to {cv:.3f}.",
            }
        )
    if any(len(q.get("repeat_ids") or []) > 1 for q in (b, c)):
        for finding in findings:
            if finding["kind"] in {
                "lost_relevant",
                "recovered_relevant",
                "still_missing",
                "context_lost",
                "context_loss",
                "context_still_lost",
                "context_recovered",
            }:
                finding["message"] += " This stage evidence describes repeat 1 only."
    return findings, bv, cv


def triage(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Rank per-query findings between two runs, most severe first.

    Works for any two run files, including runs a release check rejects as not comparable:
    queries present in only one run are counted, never paired.
    """
    by_id = {q["query_id"]: q for q in candidate["queries"]}
    pairs = [(q, by_id[q["query_id"]]) for q in baseline["queries"] if q["query_id"] in by_id]
    primary = _primary(pairs)
    items = []
    counts = {severity: 0 for severity in _RANK}
    unchanged = 0
    for b, c in pairs:
        findings, bv, cv = _query_findings(b, c, primary)
        if not findings:
            unchanged += 1
            continue
        severity = min((f["severity"] for f in findings), key=_RANK.__getitem__)
        counts[severity] += 1
        evidence = c.get("evidence") if isinstance(c.get("evidence"), Mapping) else None
        items.append(
            {
                "query_id": c["query_id"],
                "tags": list(b.get("tags") or []),
                "severity": severity,
                "metric": primary,
                "baseline": bv,
                "candidate": cv,
                "delta": None if bv is None or cv is None else cv - bv,
                "stages_recorded": bool(evidence and evidence.get("stages")),
                "findings": sorted(findings, key=lambda f: _RANK[f["severity"]]),
            }
        )
    items.sort(
        key=lambda x: (
            _RANK[x["severity"]],
            x["delta"] if x["delta"] is not None else 0.0,
            x["query_id"],
        )
    )
    candidate_ids = {q["query_id"] for q in candidate["queries"]}
    baseline_ids = {q["query_id"] for q in baseline["queries"]}
    return {
        "format": TRIAGE_FORMAT,
        "baseline_run": baseline.get("run_id"),
        "candidate_run": candidate.get("run_id"),
        "primary_metric": primary,
        "n_paired": len(pairs),
        "counts": {**counts, "unchanged": unchanged},
        "unpaired": {
            "baseline_only": sorted(baseline_ids - candidate_ids)[:100],
            "candidate_only": sorted(candidate_ids - baseline_ids)[:100],
        },
        "truncated": len(items) > MAX_ITEMS,
        "items": items[:MAX_ITEMS],
    }
