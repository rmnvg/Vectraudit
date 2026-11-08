"""Judge-corrected correct-answer rates for a run: judge only, human only, and corrected.

For each model evaluator that gave verdicts, this combines the judge's verdicts on the whole
frame with human verdicts on a random review sample (:mod:`vectraudit.evaluation.review_sample`)
through prediction-powered inference (:mod:`vectraudit.evaluation.ppi`). The rate is the share
of question families whose questions were **all** judged ``correct``, averaged over families
(a family's value is the mean over its questions, so paraphrases count once).

The corrected estimate is **absent, with the reason**, whenever its preconditions fail; it is
never filled in from weaker data:

- no review sample was given, or the sample is not this run's (dataset, frame);
- the evaluator did not give a usable verdict for every question of every frame family
  (a family is judged completely or not at all, and judging only some would break the
  randomness the interval relies on);
- people did not review every question of every sampled family, or reviewers disagreed on one;
- too few families were reviewed.

The rate is a statement about the population, not about any question: it never changes a
verdict, and nothing in a release policy reads it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.families import family_id
from vectraudit.evaluation.ppi import PPIError, classical_mean, corrected_rate, ppi_mean
from vectraudit.evaluation.review_sample import ReviewSample, problems_with

METRIC = "correct_rate"
DEFINITION = (
    "Share of question families whose questions were all judged correct (a family's value is "
    "the mean over its questions). Partial and incorrect count as not correct."
)

# (evaluator, kind, verdict) for each judgment of a question.
Verdicts = Mapping[str, Sequence[tuple[str, str, str | None]]]


def _value(verdicts: set[str]) -> float | None:
    """1.0 / 0.0 when the lines agree on one verdict (correct or not); ``None`` otherwise."""
    if len(verdicts) != 1:
        return None
    return 1.0 if next(iter(verdicts)) == "correct" else 0.0


def _family_values(
    members: Mapping[str, Sequence[str]],
    per_question: Mapping[str, float | None],
    families: Sequence[str],
) -> dict[str, float]:
    """Family means for the families whose every question has a usable value."""
    out = {}
    for family in families:
        values = [per_question.get(q) for q in members[family]]
        if values and all(v is not None for v in values):
            out[family] = sum(v for v in values if v is not None) / len(values)
    return out


def estimate_rates(
    run: Mapping[str, Any], judgments: Verdicts, sample: ReviewSample | None
) -> dict[str, Any]:
    """The ``estimates`` block of a run's ``answers`` summary."""
    evaluators = sorted(
        {e for rows in judgments.values() for e, kind, v in rows if kind == "model" and v}
    )
    entries = {e["query_id"]: e for e in run.get("queries") or []}
    members: dict[str, list[str]] = defaultdict(list)
    for qid, entry in entries.items():
        members[family_id(entry)].append(qid)
    human: dict[str, float | None] = {}
    for qid, rows in judgments.items():
        mine = {v for _, kind, v in rows if kind == "human" and v}
        human[qid] = _value(mine) if mine else None
    block: dict[str, Any] = {
        "metric": METRIC,
        "definition": DEFINITION,
        "sample": None
        if sample is None
        else {
            "sha256": sample.sha256,
            "size": sample.size,
            "frame": len(sample.frame),
            "frame_families": list(sample.frame),
            "chosen": list(sample.chosen),
        },
        "by_evaluator": {},
    }
    sample_problems = [] if sample is None else problems_with(sample, run)
    for name in evaluators:
        scored: dict[str, float | None] = {}
        for qid, rows in judgments.items():
            mine = {v for e, kind, v in rows if e == name and kind == "model" and v}
            scored[qid] = _value(mine) if mine else None
        everything = sorted(members)
        judge_all = _family_values(members, scored, everything)
        row: dict[str, Any] = {
            "status": "absent",
            "reason": "",
            "judge_only": {
                "estimate": sum(judge_all.values()) / len(judge_all) if judge_all else None,
                "families": len(judge_all),
                "note": "The judge's rate over the families it judged completely; biased by the "
                "judge's own error, and silent about families it did not judge.",
            },
        }
        block["by_evaluator"][name] = row
        if sample is None:
            row["reason"] = (
                "No review sample was given. Draw one with `workbench review-sample`, have "
                "people review exactly those questions, and pass it with --sample."
            )
            continue
        if sample_problems:
            row["reason"] = " ".join(sample_problems)
            continue
        frame = list(sample.frame)
        judged = _family_values(members, scored, frame)
        if len(judged) < len(frame):
            row["reason"] = (
                f"{name} gave a usable verdict for every question of {len(judged)} of "
                f"{len(frame)} frame families. A corrected estimate needs all of them: judging "
                "only some would break the randomness it relies on."
            )
            continue
        reviewed = _family_values(members, human, list(sample.chosen))
        if len(reviewed) < len(sample.chosen):
            row["reason"] = (
                f"People gave one agreed verdict for every question of {len(reviewed)} of "
                f"{len(sample.chosen)} sampled families. Review the rest (or the reviewers "
                "disagreed); a partial sample is not a random one."
            )
            continue
        chosen = sorted(sample.chosen)
        rest = sorted(set(frame) - set(chosen))
        try:
            numbers = corrected_rate(
                [reviewed[f] for f in chosen],
                [judged[f] for f in chosen],
                [judged[f] for f in rest],
            )
        except PPIError as exc:
            row["reason"] = f"{exc} {exc.hint}"
            continue
        row.update(
            {
                "status": "ok",
                "reason": None,
                "judge_only": {**numbers["judge_only"], "families": len(frame)},
                "human_only": numbers["human_only"],
                "corrected": numbers["corrected"],
                "agreement_on_sample": numbers["agreement_on_sample"],
                "sample_families": len(chosen),
                "frame_families": len(frame),
            }
        )
    return block


# --- two runs ---------------------------------------------------------------------------


def _verdicts_of(run: Mapping[str, Any]) -> Verdicts:
    """The judgments attached to a run's questions, as ``(evaluator, kind, verdict)``."""
    out: dict[str, list[tuple[str, str, str | None]]] = {}
    for q in run.get("queries") or []:
        rows = (q.get("answer") or {}).get("judgments") or []
        out[q["query_id"]] = [(j["evaluator"], j["kind"], j.get("verdict")) for j in rows]
    return out


def _values(
    run: Mapping[str, Any], evaluator: str | None, kind: str, families: Sequence[str]
) -> dict[str, float]:
    """Family values of a run for one evaluator (``kind`` model) or the people (``human``)."""
    members: dict[str, list[str]] = defaultdict(list)
    for q in run.get("queries") or []:
        members[family_id(q)].append(q["query_id"])
    per_question: dict[str, float | None] = {}
    for qid, rows in _verdicts_of(run).items():
        mine = {v for e, k, v in rows if k == kind and v and (evaluator is None or e == evaluator)}
        per_question[qid] = _value(mine) if mine else None
    return _family_values(members, per_question, families)


def compare_estimates(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any] | None:
    """The change in correct rate between two runs, judge-corrected, or ``None``.

    Only for runs that both carry estimates from the **same** review sample: the same families
    must have been reviewed on both sides, or the difference is not of one population. For each
    model evaluator whose estimate is ``ok`` in both runs, prediction-powered inference is applied
    to the per-family *differences* (candidate minus baseline), so a family counts once and the
    judge's error on the difference is measured, not assumed. Information only: nothing here
    changes a verdict, and no policy rule reads it.
    """
    be = (base.get("answers") or {}).get("estimates")
    ce = (cand.get("answers") or {}).get("estimates")
    if not be or not ce:
        return None
    block: dict[str, Any] = {
        "metric": METRIC,
        "definition": "Change in " + DEFINITION[0].lower() + DEFINITION[1:],
        "sample": ce.get("sample"),
        "by_evaluator": {},
    }
    bs, cs = be.get("sample"), ce.get("sample")
    if not bs or not cs or bs.get("sha256") != cs.get("sha256"):
        block["note"] = (
            "The two runs were not corrected with the same review sample, so the change in "
            "correct rate is not estimated."
        )
        return block
    frame, chosen = list(cs["frame_families"]), sorted(cs["chosen"])
    rest = sorted(set(frame) - set(chosen))
    for name, brow in be["by_evaluator"].items():
        crow = ce["by_evaluator"].get(name)
        row: dict[str, Any] = {"status": "absent", "reason": ""}
        block["by_evaluator"][name] = row
        if not crow or brow["status"] != "ok" or crow["status"] != "ok":
            row["reason"] = (
                "A corrected estimate is missing for this evaluator in "
                + ("the candidate" if brow["status"] == "ok" else "the baseline")
                + "; see each run's own reason."
            )
            continue
        jb, jc = _values(base, name, "model", frame), _values(cand, name, "model", frame)
        hb, hc = _values(base, None, "human", chosen), _values(cand, None, "human", chosen)
        if (
            len(jb) < len(frame)
            or len(jc) < len(frame)
            or len(hb) < len(chosen)
            or len(hc) < len(chosen)
        ):
            row["reason"] = "A run lacks a complete verdict for some frame or sampled family."
            continue
        judge_diff = {f: jc[f] - jb[f] for f in frame}
        human_diff = [hc[f] - hb[f] for f in chosen]
        try:
            numbers = _difference(
                human_diff, [judge_diff[f] for f in chosen], [judge_diff[f] for f in rest]
            )
        except PPIError as exc:
            row["reason"] = f"{exc} {exc.hint}"
            continue
        row.update(
            {
                "status": "ok",
                "reason": None,
                **numbers,
                "sample_families": len(chosen),
                "frame_families": len(frame),
            }
        )
    return block


def _difference(
    human: Sequence[float], judge_on_sample: Sequence[float], judge_on_rest: Sequence[float]
) -> dict[str, Any]:
    """Judge-only, people-only and corrected means of a difference (not clipped to [0, 1])."""
    pooled = [*judge_on_sample, *judge_on_rest]
    return {
        "judge_only": {"estimate": sum(pooled) / len(pooled), "n": len(pooled)},
        "human_only": classical_mean(human, bounded=False).as_dict(),
        "corrected": ppi_mean(human, judge_on_sample, judge_on_rest, bounded=False).as_dict(),
    }
