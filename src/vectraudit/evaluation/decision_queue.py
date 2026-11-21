"""Which judgments could still change a release verdict, and when none can.

For a baseline and candidate run of one dataset version, every unjudged document either run
returned for an exploratory question is a judgment someone could make. This module:

1. **Bounds** each question's metrics over every way those documents could be judged (any
   grade from 0 to ``max_grade``). The bounds are exact for recall, precision, MRR and hit
   rate, and safe for nDCG: its grade scale cancels, a document's effect on the difference
   between the runs is its gain times the difference of its discounts, and judging documents
   relevant only raises the ideal.
2. **Checks every rule at both extremes** with :func:`compare_runs`, every question at its
   lowest and at its highest (of the change for drop rules, of the candidate's value for
   ``min`` rules). Every metric rule (``max_drop``, with or without a significance
   requirement or tolerance, and ``min``), the regressed-query count and uncontrolled cohort
   rules are monotone in the per-question values, and the bootstrap draws the same resamples
   at both extremes, so when every rule is the same at both no remaining judgment, alone or
   together, can change the verdict. That is the **stop rule**. It is conservative: it may say
   "could still change" when no actual set of judgments would.
3. **Orders** the documents by swing: how far the question's difference could move on a
   rule that is not yet settled if the document alone were judged relevant at
   ``max_grade``. Documents both runs return at the same rank move no difference.

Ordering by swing settled verdicts with 56 to 68% of the judgments random order needed in a
synthetic study (``scripts/decision_queue_spike.py``, ``docs/DECISIONS.md`` 46.0): about a
third fewer, not half. Nothing here judges a document; the queue is for a reviewer.

Not certified (the queue is still ordered): rules on MAP, which has no tight bound here, and
cohort rules under Bonferroni or Holm control, whose normal approximation uses the spread of
the changes and is not monotone in them. A labelled question is required to stay labelled;
an exploratory question with no relevant label yet could join the check when judged, so it
keeps the verdict open. Held-out questions are never queued, and their unjudged documents
stay not relevant (as every metric counts them).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.families import family_id
from vectraudit.evaluation.metrics import LABEL_METRICS, repeat_quality
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy

FORMAT = "vectraudit.decision_queue"
FORMAT_VERSION = 1
BOUNDED = ("recall", "precision", "mrr", "hit_rate", "ndcg")
"""Label metrics with a bound; MAP has none and rules on it are not certified."""
DEFAULT_MAX_GRADE = 3.0
EPS = 1e-9
SAVING = (
    "Ordered by swing, a synthetic study settled verdicts with 56 to 68% of the judgments "
    "random order needed: about a third fewer, not half (docs/DECISIONS.md, 46.0)."
)


class DecisionQueueError(ValueError):
    """The runs, dataset or options cannot be used together."""


@dataclass(frozen=True)
class _Rule:
    key: str
    metric: str
    kind: str  # "delta" or "candidate"
    scale: float
    members: frozenset[str] | None  # query ids in scope, None for all paired questions


@dataclass
class _Question:
    query: DatasetQuery
    base: list[list[str]]
    cand: list[list[str]]
    labels: dict[str, float]
    unjudged: list[str]
    measured: bool


def decision_queue(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    dataset: Dataset,
    policy: Policy = DEFAULT_POLICY,
    *,
    irrelevant: Mapping[str, Sequence[str]] | None = None,
    max_grade: float | None = None,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """The ``vectraudit.decision_queue`` report for two runs of ``dataset``.

    ``irrelevant`` maps a question to documents a reviewer already marked not relevant
    outside the dataset (a workbench draft's judgments); they are judged, at grade 0.
    ``max_grade`` is the highest grade a reviewer can give (default: 3, or the dataset's
    highest grade if larger). ``seed`` and ``n_bootstrap`` must match the check's.
    """
    for run in (baseline, candidate):
        if run["dataset"]["sha256"] != dataset.sha256:
            raise DecisionQueueError(
                "Both runs must be of this exact dataset version; remeasure after changing labels."
            )
    top = max((g for q in dataset.queries for g in q.relevant.values()), default=0.0)
    grade = float(max_grade if max_grade is not None else max(DEFAULT_MAX_GRADE, top))
    if not math.isfinite(grade) or grade <= 0 or grade < top:
        raise DecisionQueueError(
            f"max_grade must be positive and at least the dataset's highest grade ({top:g})."
        )
    k = int(candidate["search"]["k"])
    current = compare_runs(baseline, candidate, policy, seed=seed, n_bootstrap=n_bootstrap)
    questions = _questions(baseline, candidate, dataset, irrelevant or {})
    rules = _rules(policy, current, questions)
    out: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "dataset_sha256": dataset.sha256,
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "verdict": current["verdict"],
        "assumptions": {
            "max_grade": grade,
            "k": k,
            "scope": "unjudged documents either run returned for exploratory questions",
            "unjudged": "not relevant until judged, as every metric counts them",
            "seed": seed,
            "n_bootstrap": n_bootstrap,
        },
        "saving": SAVING,
    }
    pool = [(q, doc) for q in questions.values() for doc in q.unjudged]
    joining = sorted(q.query.query_id for q in questions.values() if q.unjudged and not q.measured)
    out["n_pool"] = len(pool)
    out["unlabelled_questions"] = joining
    if current["verdict"] not in (Verdict.PASS.value, Verdict.FAIL.value):
        state, message, unsettled = (
            "not_applicable",
            f"The check is {current['verdict']}: settle that before judging documents for it.",
            set(rules),
        )
        out["rules"] = []
    else:
        state, message, unsettled, out["rules"] = _stop_rule(
            baseline, candidate, policy, current, questions, rules, k, grade, seed, n_bootstrap
        )
    items = _items(questions, rules, unsettled, k, grade)
    could = [i for i in items if i["could_change"]]
    if state == "open":
        message = f"About {len(could)} more judgments could still change the verdict" + (
            f"; {len(joining)} unlabelled exploratory questions could join the check."
            if joining
            else "."
        )
    out.update(
        {
            "state": state,
            "settled": state == "settled",
            "message": message,
            "n_could_change": len(could) if state != "settled" else 0,
            "items": items,
        }
    )
    return out


# --- inputs -----------------------------------------------------------------------------------


def _rankings(entry: Mapping[str, Any], k: int) -> list[list[str]]:
    repeats = entry.get("repeat_ids") or [entry.get("ids") or []]
    return [list(r)[:k] for r in repeats]


def _questions(
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    dataset: Dataset,
    irrelevant: Mapping[str, Sequence[str]],
) -> dict[str, _Question]:
    k = int(cand["search"]["k"])
    by_id = {q.query_id: q for q in dataset.queries}
    cand_by = {q["query_id"]: q for q in cand["queries"]}
    out: dict[str, _Question] = {}
    for b in base["queries"]:
        c = cand_by.get(b["query_id"])
        q = by_id.get(b["query_id"])
        if c is None or q is None or q.split != "explore" or q.no_answer:
            continue
        if b.get("error") or c.get("error") or b.get("ids") is None or c.get("ids") is None:
            continue
        base_r, cand_r = _rankings(b, k), _rankings(c, k)
        judged = set(q.relevant) | {s.doc_id for s in q.relevant_spans}
        judged |= set(irrelevant.get(q.query_id, ())) | set(q.forbidden)
        returned = dict.fromkeys(d for r in (*base_r, *cand_r) for d in r)
        out[q.query_id] = _Question(
            query=q,
            base=base_r,
            cand=cand_r,
            labels=dict(q.relevant),
            unjudged=[d for d in returned if d not in judged],
            measured=b.get("metrics") is not None and c.get("metrics") is not None,
        )
    return out


def _rules(
    policy: Policy, current: Mapping[str, Any], questions: Mapping[str, _Question]
) -> dict[str, _Rule]:
    """The policy's label-dependent rules, keyed as the check names them."""
    measured = current.get("metrics") or {}
    rules: dict[str, _Rule] = {}
    for name, rule in policy.metrics.items():
        if name not in LABEL_METRICS or name not in measured:
            continue
        if rule.max_drop is not None:
            rules[f"metric:{name}.max_drop"] = _Rule(
                f"metric:{name}.max_drop", name, "delta", max(rule.max_drop, 0.01), None
            )
        if rule.min is not None:
            rules[f"metric:{name}.min"] = _Rule(f"metric:{name}.min", name, "candidate", 0.01, None)
    primary = (current.get("queries") or {}).get("primary_metric")
    if policy.max_regressed_queries is not None and primary in LABEL_METRICS:
        rules["regressed_queries"] = _Rule("regressed_queries", primary, "delta", 0.01, None)
    for tag, cohort in (current.get("cohorts") or {}).items():
        members = frozenset(
            qid for qid, q in questions.items() if tag in q.query.cohort_tags and q.measured
        )
        for name in cohort.get("judged_metrics") or []:
            if name not in LABEL_METRICS:
                continue
            rule = policy.cohorts.metrics[name]
            if rule.max_drop is not None:
                key = f"cohort:{tag}:{name}.max_drop"
                rules[key] = _Rule(key, name, "delta", max(rule.max_drop, 0.01), members)
            if rule.min is not None:
                key = f"cohort:{tag}:{name}.min"
                rules[key] = _Rule(key, name, "candidate", 0.01, members)
    return rules


# --- the stop rule ----------------------------------------------------------------------------


def _uncertifiable(policy: Policy, rules: Mapping[str, _Rule]) -> list[str]:
    why = []
    if any(r.metric not in BOUNDED for r in rules.values()):
        names = sorted({r.metric for r in rules.values() if r.metric not in BOUNDED})
        why.append(f"rules on {', '.join(names)} have no bound here")
    controlled = [r for r in rules.values() if r.key.startswith("cohort:") and r.kind == "delta"]
    if policy.cohorts.control in ("bonferroni", "holm") and len(controlled) > 1:
        why.append(
            f"cohort rules under {policy.cohorts.control} control are not monotone in the "
            "changes they judge"
        )
    return why


def _stop_rule(
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    policy: Policy,
    current: Mapping[str, Any],
    questions: Mapping[str, _Question],
    rules: Mapping[str, _Rule],
    k: int,
    grade: float,
    seed: int,
    n_bootstrap: int,
) -> tuple[str, str, set[str], list[dict[str, Any]]]:
    bounds = {
        qid: _bounds(q, k, grade) for qid, q in questions.items() if q.measured and q.unjudged
    }
    extremes = {
        (what, side): compare_runs(
            *_at_extreme(base, cand, bounds, what, side),
            policy,
            seed=seed,
            n_bootstrap=n_bootstrap,
        )
        for what in ("delta", "candidate")
        for side in ("low", "high")
    }

    def failing(result: Mapping[str, Any]) -> set[str]:
        return {
            str(r["rule"])
            for r in result["reasons"]
            if r["verdict"] == Verdict.FAIL.value and r.get("rule") in rules
        }

    now = failing(current)
    fails = {key: failing(result) for key, result in extremes.items()}
    rows, unsettled = [], set()
    for key in sorted(rules):
        what = rules[key].kind
        at_low, at_high = key in fails[(what, "low")], key in fails[(what, "high")]
        if at_low != at_high:
            unsettled.add(key)
        rows.append(
            {
                "rule": key,
                "metric": rules[key].metric,
                "reads": "change" if what == "delta" else "candidate",
                "now": "fail" if key in now else "pass",
                "at_lowest": "fail" if at_low else "pass",
                "at_highest": "fail" if at_high else "pass",
                "settled": at_low == at_high,
            }
        )
    # Every extreme is a valid comparison of the same questions; anything else means the
    # bound could not be checked, never that it holds.
    valid = all(r["verdict"] in (Verdict.PASS.value, Verdict.FAIL.value) for r in extremes.values())
    verdict = current["verdict"]
    why = _uncertifiable(policy, rules) if any(q.unjudged for q in questions.values()) else []
    joining = any(q.unjudged and not q.measured for q in questions.values())
    # A failure no label can touch (latency, a critical document, a rule without labels)
    # keeps a fail whatever is judged; so does a rule that fails at both of its extremes.
    fixed_fail = [
        str(r.get("rule"))
        for r in current["reasons"]
        if r["verdict"] == Verdict.FAIL.value and r.get("rule") not in rules
    ]
    held = [row["rule"] for row in rows if row["at_lowest"] == row["at_highest"] == "fail"]
    settled = (
        verdict == Verdict.FAIL.value
        and (fixed_fail or (valid and held and not joining and not why))
    ) or (verdict == Verdict.PASS.value and valid and not unsettled and not joining and not why)
    if settled:
        because = fixed_fail or held
        return (
            "settled",
            f"No remaining judgment of an exploratory question can change this verdict "
            f"({verdict})" + (f": {because[0]} fails whatever is judged." if because else "."),
            set(),
            rows,
        )
    if why:
        reason = "Cannot say when judging can stop: " + "; ".join(why) + "."
        return "not_certifiable", reason, set(rules), rows
    return "open", "", unsettled or set(rules), rows


def _at_extreme(
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    bounds: Mapping[str, Mapping[str, tuple[float, float, float, float]]],
    what: str,
    side: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copies of the runs with every bounded question at one extreme of one quantity.

    Every rule reads either the change (``max_drop``, regressed-query and cohort drop rules)
    or the candidate's value (``min`` rules), never both, so each is judged from its own
    pair of extremes. ``what="delta"`` sets each question to a valid pair with the extreme
    change; ``what="candidate"`` sets both runs to the candidate's extreme value.
    """
    b, c = copy.deepcopy(dict(base)), copy.deepcopy(dict(cand))
    cand_by = {q["query_id"]: q for q in c["queries"]}
    for bq in b["queries"]:
        found = bounds.get(bq["query_id"])
        if not found:
            continue
        cq = cand_by[bq["query_id"]]
        for name, (d_lo, d_hi, c_lo, c_hi) in found.items():
            if name not in (bq.get("metrics") or {}) or name not in (cq.get("metrics") or {}):
                continue
            if what == "delta":
                change = d_lo if side == "low" else d_hi
                cq["metrics"][name], bq["metrics"][name] = max(change, 0.0), max(-change, 0.0)
            else:
                value = c_lo if side == "low" else c_hi
                cq["metrics"][name] = bq["metrics"][name] = value
        # A run's metrics must equal the mean of its repeats' (the check validates it).
        for q in (bq, cq):
            if q.get("repeat_metrics") is not None:
                q["repeat_metrics"] = [dict(q["metrics"]) for _ in q["repeat_metrics"]]
    return b, c


# --- bounds -----------------------------------------------------------------------------------


def _first_ranks(ranking: Sequence[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, doc in enumerate(ranking, 1):
        ranks.setdefault(doc, rank)
    return ranks


def _disc(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def _ideal(grades: Sequence[float], k: int) -> float:
    return sum(g * _disc(r) for r, g in enumerate(sorted(grades, reverse=True)[:k], 1))


def _side(
    ranking: Sequence[str], labels: Mapping[str, float], unjudged: Sequence[str], k: int, g: float
) -> dict[str, tuple[float, float]]:
    """Lowest and highest value of each bounded metric for one ranking."""
    ranks = _first_ranks(ranking[:k])
    relevant = {d for d, x in labels.items() if x > 0}
    hits = len(relevant & ranks.keys())
    mine = [u for u in unjudged if u in ranks]
    others = len(unjudged) - len(mine)
    first = min((ranks[d] for d in relevant if d in ranks), default=None)
    rr = 1 / first if first else 0.0
    best = min((ranks[u] for u in mine), default=None)
    i0 = _ideal(list(labels.values()), k)
    i_max = _ideal([*labels.values(), *[g] * len(unjudged)], k)
    gain = sum(labels.get(d, 0.0) * _disc(r) for d, r in ranks.items())
    extra = g * sum(_disc(ranks[u]) for u in mine)
    return {
        "precision": (hits / k, (hits + len(mine)) / k),
        "hit_rate": (float(hits > 0), float(hits > 0 or bool(mine))),
        "recall": (
            hits / (len(relevant) + others),
            (hits + len(mine)) / (len(relevant) + len(mine)),
        ),
        "mrr": (rr, max(rr, 1 / best) if best else rr),
        "ndcg": (gain / i_max, min(1.0, (gain + extra) / i0)),
    }


def _joint(
    base: Sequence[str],
    cand: Sequence[str],
    labels: Mapping[str, float],
    unjudged: Sequence[str],
    k: int,
    g: float,
) -> dict[str, tuple[float, float]]:
    """Lowest and highest candidate-minus-baseline change, both runs judged together."""
    rb, rc = _first_ranks(base[:k]), _first_ranks(cand[:k])
    relevant = {d for d, x in labels.items() if x > 0}
    hb, hc = len(relevant & rb.keys()), len(relevant & rc.keys())
    shared = [u for u in unjudged if u in rb and u in rc]
    only_c = [u for u in unjudged if u in rc and u not in rb]
    only_b = [u for u in unjudged if u in rb and u not in rc]
    n_rel = len(relevant)
    recalls = [
        (hc + xs + xc - hb - xs - xb) / (n_rel + xs + xc + xb)
        for xs in {0, len(shared)}
        for xc in range(len(only_c) + 1)
        for xb in range(len(only_b) + 1)
    ]
    hits = [
        float(hc > 0 or xs or xc) - float(hb > 0 or xs or xb)
        for xs in {False, bool(shared)}
        for xc in {False, bool(only_c)}
        for xb in {False, bool(only_b)}
    ]

    def rr(ranks: Mapping[str, int]) -> float:
        first = min((ranks[d] for d in relevant if d in ranks), default=None)
        return 1 / first if first else 0.0

    rr_b, rr_c = rr(rb), rr(rc)
    raise_c = [
        (1 / rc[u]) - max(rr_b, 1 / rb[u] if u in rb else 0.0)
        for u in unjudged
        if u in rc and 1 / rc[u] > rr_c
    ]
    raise_b = [
        max(rr_c, 1 / rc[u] if u in rc else 0.0) - (1 / rb[u])
        for u in unjudged
        if u in rb and 1 / rb[u] > rr_b
    ]
    i0 = _ideal(list(labels.values()), k)
    i_max = _ideal([*labels.values(), *[g] * len(unjudged)], k)
    gains = sum(labels.get(d, 0.0) * _disc(r) for d, r in rc.items())
    gains -= sum(labels.get(d, 0.0) * _disc(r) for d, r in rb.items())
    weights = [_disc(rc.get(u)) - _disc(rb.get(u)) for u in unjudged]
    up = gains + g * sum(w for w in weights if w > 0)
    down = gains + g * sum(w for w in weights if w < 0)
    return {
        "precision": ((hc - hb - len(only_b)) / k, (hc - hb + len(only_c)) / k),
        "hit_rate": (min(hits), max(hits)),
        "recall": (min(recalls), max(recalls)),
        "mrr": (min([rr_c - rr_b, *raise_b]), max([rr_c - rr_b, *raise_c])),
        "ndcg": (
            max(-1.0, down / (i_max if down >= 0 else i0)),
            min(1.0, up / (i0 if up > 0 else i_max)),
        ),
    }


def _bounds(q: _Question, k: int, g: float) -> dict[str, tuple[float, float, float, float]]:
    """Per bounded metric: (lowest change, highest change, lowest and highest candidate)."""

    def mean_side(rankings: Sequence[Sequence[str]]) -> dict[str, tuple[float, float]]:
        sides = [_side(r, q.labels, q.unjudged, k, g) for r in rankings]
        return {
            m: (
                sum(s[m][0] for s in sides) / len(sides),
                sum(s[m][1] for s in sides) / len(sides),
            )
            for m in BOUNDED
        }

    cand, base = mean_side(q.cand), mean_side(q.base)
    single = all(r == q.base[0] for r in q.base) and all(r == q.cand[0] for r in q.cand)
    joint = _joint(q.base[0], q.cand[0], q.labels, q.unjudged, k, g) if single else None
    out = {}
    for m in BOUNDED:
        d_lo, d_hi = joint[m] if joint else (cand[m][0] - base[m][1], cand[m][1] - base[m][0])
        out[m] = (d_lo, d_hi, cand[m][0], cand[m][1])
    return out


# --- ordering ---------------------------------------------------------------------------------


def _values(q: _Question, labels: Mapping[str, float], k: int) -> tuple[dict[str, float], ...]:
    def score(rankings: Sequence[Sequence[str]]) -> dict[str, float]:
        if not any(x > 0 for x in labels.values()):
            return dict.fromkeys(LABEL_METRICS, 0.0)
        metrics = repeat_quality(rankings, labels, k)["metrics"]
        return {m: float(v) for m, v in metrics.items()}

    return score(q.base), score(q.cand)


def _items(
    questions: Mapping[str, _Question],
    rules: Mapping[str, _Rule],
    unsettled: set[str],
    k: int,
    grade: float,
) -> list[dict[str, Any]]:
    """Every queued document with its swing, the largest first."""
    families: dict[frozenset[str] | None, dict[str, int]] = {}

    def weight(rule: _Rule, q: _Question) -> float:
        if rule.members not in families:
            counts: dict[str, int] = defaultdict(int)
            for other in questions.values():
                inside = rule.members is None or other.query.query_id in rule.members
                if inside and (other.measured or other.unjudged):
                    counts[family_id({"query_id": other.query.query_id, **_family(other)})] += 1
            families[rule.members] = counts
        counts = families[rule.members]
        fam = family_id({"query_id": q.query.query_id, **_family(q)})
        return 1.0 / (counts[fam] * len(counts)) if counts.get(fam) else 0.0

    items: list[dict[str, Any]] = []
    for qid, q in questions.items():
        if not q.unjudged:
            continue
        base_now, cand_now = _values(q, q.labels, k)
        base_ranks, cand_ranks = _first_ranks(q.base[0]), _first_ranks(q.cand[0])
        for doc in q.unjudged:
            base_then, cand_then = _values(q, {**q.labels, doc: grade}, k)
            moves: dict[str, float] = {}
            for key, rule in rules.items():
                if rule.members is not None and qid not in rule.members:
                    continue
                m = rule.metric
                change = cand_then[m] - cand_now[m]
                if rule.kind == "delta":
                    change -= base_then[m] - base_now[m]
                moves[key] = abs(change) * weight(rule, q) / rule.scale
            open_moves = [v for key, v in moves.items() if key in unsettled]
            swing = max(open_moves, default=0.0)
            items.append(
                {
                    "query_id": qid,
                    "doc_id": doc,
                    "baseline_rank": base_ranks.get(doc),
                    "candidate_rank": cand_ranks.get(doc),
                    "joins_check": not q.measured,
                    "swing": swing,
                    "could_change": swing > EPS or (not q.measured and bool(unsettled)),
                    "rules": sorted(key for key in moves if key in unsettled and moves[key] > EPS),
                }
            )
    items.sort(
        key=lambda i: (
            -float(i["swing"]),
            not i["joins_check"],
            str(i["query_id"]),
            str(i["doc_id"]),
        )
    )
    return items


def _family(q: _Question) -> dict[str, Any]:
    return {"family_id": q.query.family_id} if q.query.family_id is not None else {}
