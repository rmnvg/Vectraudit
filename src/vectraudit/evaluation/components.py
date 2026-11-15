"""Component scoreboard: where along the pipeline each question first lost its evidence.

A pipeline run records, per question, the stages that handled each relevant document, the
documents that reached the model, and (after ``attach-answers``) whether the answer was judged
wrong despite the evidence. This aggregates those records into one row per component, so a
team sees whether its losses sit at retrieval, a filter, fusion, the reranker, the cut-off,
context packing or the answer:

- **Evidence lost** means the question's required evidence (its ``required_evidence`` groups,
  any one complete group suffices) or, without groups, every relevant document, did not
  survive. With groups, a group breaks at the first drop of any of its documents and the
  question is lost where its last group broke; without them, the question is lost where its
  last relevant document was dropped.
- **First observable loss** places that point on the recorded path. A document no recorded
  stage returned counts at ``retrieve``: the cause may be the corpus, the index, or query
  rewriting (which no stage records) as much as the retriever. A delivered group that did not
  reach the model counts at ``context``, and an answer judged wrong although the evidence
  reached the model counts at ``answer`` (the existing answer diagnosis).
- **Not observed, never assigned.** A question whose evidence was not delivered and whose
  stages were not recorded is counted as not observed; so is one whose context or answer was
  not recorded when that is what would place it. Unlabelled and no-answer questions are not
  applicable; errored questions are counted apart.

Rates are family-weighted: each paraphrase family counts once, split across the components
its questions lost at, with a Wilson interval over families. The dataset header may map the
customer's own tags to the components they are meant to exercise (``component_tags``), and
each such cohort gets its own rows. This is information for the run view and the release
report; it never changes a verdict.

IDs, counts and rates only. Credit: the per-component report follows Giskard's RAG
evaluation toolkit (RAGET), which scores retriever, generator, rewriter and router by mapping
question types to the components they exercise; here the mapping is the customer's tags and
the attribution comes from recorded evidence, never from generated questions or a model
grader. See docs/credits.md.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vectraudit.evaluation.families import family_id
from vectraudit.evaluation.significance import wilson_interval

FORMAT = "vectraudit.component_scoreboard"
FORMAT_VERSION = 1
COMPONENTS = ("retrieve", "filter", "fusion", "rerank", "truncate", "custom", "context", "answer")
STAGE_COMPONENT = {
    "keyword": "retrieve",
    "vector": "retrieve",
    "filter": "filter",
    "fusion": "fusion",
    "rerank": "rerank",
    "truncate": "truncate",
    "custom": "custom",
}
KEPT = "kept"
NOT_OBSERVED = "not_observed"
UNLABELLED = "unlabelled"
NO_ANSWER = "no_answer"
ERRORED = "errored"
_REACHED = 10_000  # path position past every stage: delivered and, where recorded, in context
_CONTEXT = 9_999  # delivered, then lost before the model


def _first_drop(evidence: Mapping[str, Any]) -> dict[str, int | None]:
    """Each undelivered relevant document's first loss: -1 never returned, a stage index, or
    ``None`` when the record cannot place it."""
    order = {s.get("name"): i for i, s in enumerate(evidence.get("stages") or [])}
    out: dict[str, int | None] = {}
    for failure in evidence.get("failures") or []:
        doc = failure.get("doc_id")
        if doc in out and out[doc] is not None:
            continue
        stage = failure.get("stage")
        if stage is None:
            out[doc] = -1 if failure.get("evidence") == "recorded" else None
        else:
            out[doc] = order.get(stage)
    return out


def _context_lost(evidence: Mapping[str, Any]) -> set[str]:
    context = evidence.get("context")
    if not isinstance(context, Mapping):
        return set()
    return {loss["doc_id"] for loss in context.get("losses") or [] if isinstance(loss, Mapping)}


def attribute(entry: Mapping[str, Any]) -> tuple[str, str | None]:
    """``(outcome, stage kind)`` for one question of a pipeline run.

    The outcome is a component name, ``kept``, ``not_observed``, ``unlabelled``,
    ``no_answer`` or ``errored``. The stage kind is set when a stage placed the loss.
    """
    if entry.get("error"):
        return ERRORED, None
    if entry.get("no_answer_correct") is not None:
        return NO_ANSWER, None
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        return UNLABELLED, None
    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    stages = evidence.get("stages") or []
    sufficiency = entry.get("sufficiency")
    if isinstance(sufficiency, Mapping):
        delivered = sufficiency.get("delivered") == "sufficient"
        missing = list(sufficiency.get("missing_from_delivered") or [])
    else:
        delivered = bool(metrics.get("hit_rate"))
        missing = [f.get("doc_id") for f in evidence.get("failures") or []]
    if not delivered:
        drops = _first_drop(evidence)
        positions = [drops.get(doc) for doc in dict.fromkeys(missing)]
        if not stages or not positions or any(p is None for p in positions):
            return NOT_OBSERVED, None
        known = [p for p in positions if p is not None]
        # A group breaks at its first drop; alternatives are lost where the last one went.
        position = min(known) if isinstance(sufficiency, Mapping) else max(known)
        if position < 0:
            return "retrieve", None
        kind = str(stages[position].get("kind"))
        return STAGE_COMPONENT.get(kind, "custom"), kind
    diagnosis = ((entry.get("answer") or {}).get("diagnosis") or {}).get("class")
    if isinstance(sufficiency, Mapping):
        context = sufficiency.get("context")
        if context == "insufficient":
            return "context", None
        if diagnosis == "answered_wrong_despite_evidence":
            return "answer", None
        if diagnosis in ("context_unknown",):
            return NOT_OBSERVED, None
        return KEPT, None
    context = evidence.get("context")
    if isinstance(context, Mapping) and not context.get("reached") and _context_lost(evidence):
        return "context", None
    return KEPT, None


def _rate(shares: Mapping[str, float], families: int) -> dict[str, Any]:
    successes = sum(shares.values())
    low, high = wilson_interval(min(successes, families), families)
    return {
        "families": round(successes, 4),
        "rate": round(successes / families, 4) if families else None,
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
    }


def _table(rows: Sequence[tuple[str, str]], names: Iterable[str]) -> dict[str, Any]:
    """Questions and family-weighted rates per outcome, over the observed questions."""
    observed = [
        (family, outcome) for family, outcome in rows if outcome in COMPONENTS or outcome == KEPT
    ]
    per_family: dict[str, list[str]] = defaultdict(list)
    for family, outcome in observed:
        per_family[family].append(outcome)
    n_families = len(per_family)
    shares: dict[str, dict[str, float]] = defaultdict(dict)
    for family, outcomes in per_family.items():
        for outcome in outcomes:
            shares[outcome][family] = shares[outcome].get(family, 0.0) + 1 / len(outcomes)
    counts: defaultdict[str, int] = defaultdict(int)
    for _, outcome in rows:
        counts[outcome] += 1
    return {
        "observed": {"questions": len(observed), "families": n_families},
        "components": [
            {"component": name, "questions": counts[name], **_rate(shares[name], n_families)}
            for name in names
        ],
        "kept": {"questions": counts[KEPT], **_rate(shares[KEPT], n_families)},
        "not_observed": counts[NOT_OBSERVED],
        "not_applicable": {"unlabelled": counts[UNLABELLED], "no_answer": counts[NO_ANSWER]},
        "errored": counts[ERRORED],
    }


def declared(run: Mapping[str, Any]) -> list[str]:
    """The components this run can observe: its stage kinds, then context and answer."""
    found: set[str] = set()
    for entry in run.get("queries") or []:
        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        for stage in evidence.get("stages") or []:
            found.add(STAGE_COMPONENT.get(str(stage.get("kind")), "custom"))
        if isinstance(evidence.get("context"), Mapping):
            found.add("context")
        if ((entry.get("answer") or {}).get("diagnosis") or {}).get("class"):
            found.add("answer")
    if found:
        found.add("retrieve")  # documents no recorded stage returned count here
    return [c for c in COMPONENTS if c in found]


def component_scoreboard(run: Mapping[str, Any]) -> dict[str, Any]:
    """The ``vectraudit.component_scoreboard`` report for one pipeline run."""
    if run.get("format") != "vectraudit.run":
        raise ValueError("The component scoreboard reads a vectraudit.run.")
    names = declared(run)
    rows: list[tuple[str, str]] = []
    tagged: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_kind: dict[str, int] = defaultdict(int)
    for entry in run.get("queries") or []:
        outcome, kind = attribute(entry)
        if kind is not None:
            by_kind[kind] += 1
        row = (family_id(entry), outcome)
        rows.append(row)
        for tag in entry.get("tags") or []:
            tagged[tag].append(row)
    meta = (run.get("dataset") or {}).get("meta") or {}
    component_tags: Mapping[str, Sequence[str]] = meta.get("component_tags") or {}
    cohorts = []
    for tag, targets in sorted(component_tags.items()):
        table = _table(tagged.get(tag, []), names)
        lost = sum(c["questions"] for c in table["components"] if c["component"] in targets)
        cohorts.append(
            {
                "tag": tag,
                "targets": list(targets),
                "observable": [t for t in targets if t in names],
                "lost_at_targets": lost,
                **table,
            }
        )
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "run_id": run.get("run_id"),
        "label": run.get("label"),
        "evidence": "supported",
        "declared": names,
        **_table(rows, names),
        "by_stage_kind": dict(sorted(by_kind.items())),
        "cohorts": cohorts,
        "note": (
            "Where each question first lost its required evidence, from the stages, context "
            "and answer judgments the run recorded. A document no recorded stage returned counts "
            "at retrieve, though the corpus, the index or query rewriting may be the cause. "
            "Questions whose loss the run cannot place are not observed, never assigned. "
            "Information only; it does not change a verdict."
        ),
    }


def compare_components(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Both runs' component rates side by side, or ``None`` when neither observed a stage."""
    sides = {
        "baseline": component_scoreboard(baseline),
        "candidate": component_scoreboard(candidate),
    }
    if not any(s["declared"] for s in sides.values()):
        return None
    names = [c for c in COMPONENTS if any(c in s["declared"] for s in sides.values())]

    def rates(board: Mapping[str, Any]) -> dict[str, Any]:
        rows = {c["component"]: c for c in board["components"]}
        return {
            name: {k: rows[name][k] for k in ("questions", "rate", "ci_low", "ci_high")}
            for name in names
            if name in rows
        }

    return {
        "components": names,
        "baseline": rates(sides["baseline"]),
        "candidate": rates(sides["candidate"]),
        "observed": {k: s["observed"] for k, s in sides.items()},
        "not_observed": {k: s["not_observed"] for k, s in sides.items()},
        "note": "Where questions first lost their evidence, per run. Information only.",
    }


def render(board: Mapping[str, Any]) -> str:
    """The scoreboard as plain text for the terminal."""
    observed = board["observed"]
    if not board["declared"]:
        return (
            "Component scoreboard: the run recorded no stages, model context or answer "
            f"diagnosis, so no loss can be placed ({board['not_observed']} questions not observed)."
        )
    lines = [
        f"Component scoreboard ({observed['questions']} observed questions in "
        f"{observed['families']} families; first loss of the required evidence):"
    ]
    for row in [*board["components"], {"component": KEPT, **board["kept"]}]:
        rate = "n/a" if row["rate"] is None else f"{row['rate'] * 100:5.1f}%"
        interval = (
            ""
            if row["rate"] is None
            else f" (95%: {row['ci_low'] * 100:.1f} to {row['ci_high'] * 100:.1f})"
        )
        lines.append(f"  {row['component']:<9} {rate}{interval}, {row['questions']} questions")
    applicable = board["not_applicable"]
    lines.append(
        f"  Not observed: {board['not_observed']}; not applicable: {applicable['unlabelled']} "
        f"unlabelled, {applicable['no_answer']} no-answer; errored: {board['errored']}."
    )
    for cohort in board["cohorts"]:
        lines.append(
            f"  Cohort {cohort['tag']} (targets {', '.join(cohort['targets'])}): "
            f"{cohort['lost_at_targets']} of {cohort['observed']['questions']} observed questions "
            "lost their evidence at a target."
        )
    return "\n".join(lines)
