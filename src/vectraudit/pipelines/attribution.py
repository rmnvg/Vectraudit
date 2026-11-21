"""Which of several changed settings caused a difference between two configurations?

A release often bundles changes: an embedding model, a chunk size, a reranker version. Given
the baseline and the candidate configuration of one pipeline, :func:`plan_attribution` lists
the parameters that differ and the intermediate combinations to run within a declared budget,
and :func:`attribute` runs them, interleaved, on the exploratory questions and splits the
metric difference between the settings:

- **Contribution**: each setting's Shapley value over the measured combinations, computed
  per query family and averaged, so the contributions add up to the measured difference
  exactly. Intervals are 95% paired bootstraps over families.
- **Interactions**: a pair is reported when its interaction (changing both minus changing
  each, averaged over the other settings) is shown to be larger than a minimum effect: a
  sign-flip test of the interaction moved that minimum toward zero, Holm-adjusted across the
  pairs. A Shapley value splits an interaction equally between its two settings, so the pair
  is always named beside their shares.
- **Designs**: every combination (``2^n`` runs) when the budget allows; otherwise a screen:
  each setting alone from the baseline and each setting reverted from the candidate (``2n + 2``
  runs), whose average effect is the Shapley value exactly when no three settings interact,
  plus one run per pair of settings whose two effects differ, while the budget lasts.
- **Confounded settings**: parameters declared to change together (an embedding model and the
  index built from it) are attributed as one group, never split.

Exploratory questions only; held-out questions stay for validation. Only parameters vary: two
configurations whose adapter or pipeline version differ cannot be mixed, and are refused. The
method and its limits come from the Phase 47.0 spike (``scripts/attribution_spike.py``,
``docs/DECISIONS.md``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.families import family_id
from vectraudit.evaluation.metrics import LABEL_METRICS
from vectraudit.evaluation.significance import holm_adjust, paired_randomization
from vectraudit.pipelines.budget import combine
from vectraudit.pipelines.manifest import Candidate
from vectraudit.pipelines.runner import (
    PairedSide,
    PipelineOptions,
    RetryPolicy,
    execute_interleaved,
    execute_pipeline,
)
from vectraudit.pipelines.runtime import Pipeline

FORMAT = "vectraudit.attribution"
DEFAULT_MAX_RUNS = 16
"""Decision D16: every combination of up to four settings; more needs ``max_runs``."""
MAX_SETTINGS = 12
MIN_EFFECT = 0.01
BOOTSTRAP = 2000
ALPHA = 0.05
EPS = 1e-12
ASSUMPTIONS = (
    "Measured on exploratory questions only; validate a revert on held-out questions under "
    "the release policy before acting on it.",
    "A Shapley value splits an interaction equally between its settings: a setting that is a "
    "gain alone can carry a loss it causes only together with another.",
    "Contributions describe these questions, this corpus and these two configurations; "
    "another baseline would split the difference differently.",
)
SCREEN_ASSUMPTION = (
    "Screen design: each contribution is exact only if no three settings interact; three-way "
    "interactions are not tested."
)


class AttributionError(ValueError):
    """The two configurations cannot be attributed, or the budget does not allow it."""


Value = str | int | float | bool
Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Setting:
    """One attributed unit: a parameter, or parameters declared to change together."""

    name: str
    keys: tuple[str, ...]
    baseline: dict[str, Value | None]
    candidate: dict[str, Value | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": list(self.keys),
            "baseline": self.baseline,
            "candidate": self.candidate,
            "confounded": len(self.keys) > 1,
        }


@dataclass
class Plan:
    """What :func:`attribute` will run."""

    settings: list[Setting]
    design: str
    max_runs: int
    combinations: list[frozenset[str]] = field(default_factory=list)

    @property
    def runs(self) -> int:
        """Runs of the first phase; a screen may add pair runs up to ``max_runs``."""
        return len(self.combinations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "settings": [s.as_dict() for s in self.settings],
            "runs": self.runs,
            "max_runs": self.max_runs,
        }


def plan_attribution(
    baseline: Candidate,
    candidate: Candidate,
    *,
    together: Sequence[Sequence[str]] = (),
    max_runs: int | None = None,
) -> Plan:
    """The differing settings and the combinations to run, within the budget (D16).

    Without ``max_runs`` the budget is 16 runs and more than four settings are refused.
    With it, every combination is run when ``2^n`` fits, else the ``2n + 2`` screen, else
    the plan is refused with the budget each design needs.
    """
    for what, a, b in (
        ("adapter", baseline.adapter.model_dump(), candidate.adapter.model_dump()),
        ("pipeline_version", baseline.pipeline_version, candidate.pipeline_version),
    ):
        if a != b:
            raise AttributionError(
                f"the two configurations differ in {what}, which is not a parameter: "
                "attribution mixes parameters of one pipeline, so both must use the same "
                "adapter and pipeline version"
            )
    differing = [
        key
        for key in sorted(set(baseline.params) | set(candidate.params))
        if baseline.params.get(key) != candidate.params.get(key)
        or (key in baseline.params) != (key in candidate.params)
    ]
    if not differing:
        raise AttributionError("the two configurations have the same parameters")
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for group in together:
        keys = tuple(dict.fromkeys(group))
        unknown = [k for k in keys if k not in differing]
        if unknown:
            raise AttributionError(
                f"--together names {', '.join(unknown)}, which the two configurations do not "
                f"change; changed: {', '.join(differing)}"
            )
        if len(keys) < 2:
            raise AttributionError("--together needs at least two parameters")
        if seen & set(keys):
            raise AttributionError("a parameter can be in only one --together group")
        seen |= set(keys)
        groups.append(keys)
    groups += [(key,) for key in differing if key not in seen]
    settings = [
        Setting(
            "+".join(keys),
            keys,
            {k: baseline.params.get(k) for k in keys},
            {k: candidate.params.get(k) for k in keys},
        )
        for keys in groups
    ]
    n = len(settings)
    if n > MAX_SETTINGS:
        raise AttributionError(
            f"{n} settings differ; attribution handles at most {MAX_SETTINGS}. Group "
            "parameters that change together with --together"
        )
    full, screen = 2**n, 2 * n + 2
    if max_runs is None:
        if full > DEFAULT_MAX_RUNS:
            raise AttributionError(
                f"{n} settings differ: every combination is {full} runs and the screen design "
                f"{screen} or more, beyond the default budget of {DEFAULT_MAX_RUNS} runs "
                "(decision D16). Pass --max-runs to allow more, or group settings with "
                "--together"
            )
        budget = DEFAULT_MAX_RUNS
    else:
        budget = max_runs
    names = [s.name for s in settings]
    if full <= budget:
        combos = [frozenset(c) for r in range(n + 1) for c in itertools.combinations(names, r)]
        return Plan(settings, "full", budget, combos)
    if screen <= budget:
        everything = frozenset(names)
        combos = [frozenset(), everything]
        combos += [frozenset((p,)) for p in names] + [everything - {p} for p in names]
        return Plan(settings, "screen", budget, list(dict.fromkeys(combos)))
    raise AttributionError(
        f"{n} settings differ: every combination is {full} runs and the screen design "
        f"{screen}; --max-runs {budget} allows neither"
    )


def combination_params(
    plan: Plan, base: Mapping[str, Value], on: frozenset[str]
) -> dict[str, Value]:
    """The baseline's parameters with the settings in ``on`` changed to the candidate's."""
    params: dict[str, Value] = dict(base)
    for setting in plan.settings:
        if setting.name not in on:
            continue
        for key, value in setting.candidate.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
    return params


def label(on: frozenset[str], plan: Plan) -> str:
    if not on:
        return "baseline"
    if len(on) == len(plan.settings):
        return "candidate"
    return " + ".join(s.name for s in plan.settings if s.name in on)


def _exploratory(dataset: Dataset) -> Dataset:
    rows = [r for r in dataset_rows(dataset) if r.get("split", "explore") == "explore"]
    if not rows:
        raise AttributionError("the benchmark has no exploratory questions to attribute on")
    return parse_dataset(render_dataset(rows, dataset.meta))


def attribute(
    dataset: Dataset,
    pipeline: Pipeline,
    baseline: Candidate,
    plan: Plan,
    *,
    k: int,
    metric: str = "ndcg",
    repeats: int = 1,
    min_effect: float = MIN_EFFECT,
    seed: int = 0,
    retry: RetryPolicy | None = None,
    config: dict[str, Any] | None = None,
    corpus_version: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the plan's combinations on exploratory questions and attribute the difference."""
    if metric not in LABEL_METRICS:
        raise AttributionError(f"unknown metric {metric!r}; use one of {', '.join(LABEL_METRICS)}")
    if min_effect < 0:
        raise AttributionError("min_effect must be >= 0")
    suite = _exploratory(dataset)

    def run(combos: Sequence[frozenset[str]]) -> dict[frozenset[str], dict[str, Any]]:
        sides = [
            PairedSide(
                pipeline,
                PipelineOptions(
                    k=k,
                    params=combination_params(plan, baseline.params, c),
                    repeats=repeats,
                    warmup=0,
                    label=label(c, plan),
                    retry=retry or RetryPolicy(),
                ),
                {**(config or {}), "attribution": sorted(c)},
                corpus_version,
            )
            for c in combos
        ]
        if len(sides) == 1:
            side = sides[0]
            records = [
                execute_pipeline(
                    suite,
                    side.pipeline,
                    side.options,
                    config=side.config,
                    corpus_version=side.corpus_version,
                )
            ]
        else:
            records = execute_interleaved(suite, sides, progress=progress)
        return dict(zip(combos, records, strict=True))

    runs = run(plan.combinations)
    families, values, excluded = _values(suite, runs, metric)
    names = [s.name for s in plan.settings]
    everything = frozenset(names)
    flagged_gaps: dict[str, Array] = {}
    untested: list[list[str]] = []
    if plan.design == "full":
        contributions = _shapley(values, names)
        pair_effects = _interactions(values, names)
    else:
        contributions = {p: (_alone(values, p) + _last(values, p, everything)) / 2 for p in names}
        gaps = {p: _last(values, p, everything) - _alone(values, p) for p in names}
        flagged = [p for p in names if p in _material(gaps, min_effect, seed)]
        flagged_gaps = {p: gaps[p] for p in flagged}
        pairs = sorted(
            itertools.combinations(flagged, 2),
            key=lambda ab: -abs(float(gaps[ab[0]].mean())) - abs(float(gaps[ab[1]].mean())),
        )
        room = plan.max_runs - plan.runs
        tested, rest = pairs[:room], pairs[room:]
        untested = [list(ab) for ab in rest]
        if tested:
            more = run([frozenset(ab) for ab in tested])
            runs.update(more)
            families, values, excluded = _values(suite, runs, metric)
        pair_effects = {
            ab: values[frozenset(ab)]
            - values[frozenset(ab[:1])]
            - values[frozenset(ab[1:])]
            + values[frozenset()]
            for ab in tested
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(families), size=(BOOTSTRAP, len(families)))
    difference = values[everything] - values[frozenset()]
    total = float(difference.mean())
    rows = []
    for setting in plan.settings:
        p = setting.name
        c = contributions[p]
        rows.append(
            {
                **setting.as_dict(),
                "contribution": float(c.mean()),
                "interval": _interval(c, idx),
                "share": float(c.mean()) / total if abs(total) > EPS else None,
                "alone": float(_alone(values, p).mean()),
                "reverted": float(_last(values, p, everything).mean()),
            }
        )
    rows.sort(key=lambda r: r["contribution"])
    material = _material(pair_effects, min_effect, seed)
    interactions = [
        {
            "settings": list(ab),
            "interaction": float(x.mean()),
            "interval": _interval(x, idx),
            "material": ab in material,
        }
        for ab, x in pair_effects.items()
    ]
    interactions.sort(key=lambda r: (not r["material"], r["interaction"]))
    leader = _leader(rows, contributions, idx)
    report: dict[str, Any] = {
        "format": FORMAT,
        "format_version": 1,
        "status": "exploratory",
        "status_label": "Exploratory: attributes the difference on exploratory questions; not a "
        "release decision. Validate a revert on held-out questions under the policy.",
        "metric": metric,
        "k": k,
        "dataset_sha256": dataset.sha256,
        **plan.as_dict(),
        "runs": len(runs),
        "questions": sum(1 for q in suite.queries if q.query_id not in excluded),
        "families": len(families),
        "excluded_questions": sorted(excluded),
        "baseline_value": float(values[frozenset()].mean()),
        "candidate_value": float(values[everything].mean()),
        "difference": total,
        "difference_interval": _interval(difference, idx),
        "contributions": rows,
        "interactions": interactions,
        "flagged_without_pair": sorted(set(flagged_gaps) - {p for ab in material for p in ab}),
        "untested_pairs": untested,
        "leader": leader,
        "suggested_revert": _revert(plan, values, leader, idx),
        "combinations": [
            {
                "settings": sorted(c),
                "label": label(c, plan),
                "run_id": r["run_id"],
                "status": r["status"],
                "value": float(values[c].mean()),
            }
            for c, r in runs.items()
        ],
        "min_effect": min_effect,
        "seed": seed,
        "assumptions": [*ASSUMPTIONS, *([SCREEN_ASSUMPTION] if plan.design == "screen" else [])],
        "usage": combine([r["usage"] for r in runs.values()]),
    }
    report["message"] = summary(report)
    report["run_records"] = {label(c, plan): r for c, r in runs.items()}
    return report


def _values(
    suite: Dataset, runs: Mapping[frozenset[str], Mapping[str, Any]], metric: str
) -> tuple[list[str], dict[frozenset[str], Array], set[str]]:
    """Per-family means of ``metric`` for each combination, over questions every run measured."""
    by_run = {c: {q["query_id"]: q for q in r["queries"]} for c, r in runs.items()}
    excluded: set[str] = set()
    for q in suite.queries:
        for rows in by_run.values():
            row = rows.get(q.query_id)
            if row is None or row.get("error") or not row.get("metrics"):
                excluded.add(q.query_id)
                break
    kept = [q for q in suite.queries if q.query_id not in excluded]
    if not kept:
        failed = sorted(
            "+".join(sorted(c)) or "baseline"
            for c, rows in by_run.items()
            if all(r.get("error") or not r.get("metrics") for r in rows.values())
        )
        raise AttributionError(
            "no question was measured by every combination"
            + (f"; failed: {', '.join(failed)}" if failed else "")
            + ". If some settings cannot be combined, declare them with --together"
        )
    order: list[str] = []
    for q in kept:
        fam = family_id({"query_id": q.query_id, "family_id": q.family_id})
        if fam not in order:
            order.append(fam)
    position = {fam: i for i, fam in enumerate(order)}
    counts = np.zeros(len(order))
    for q in kept:
        counts[position[family_id({"query_id": q.query_id, "family_id": q.family_id})]] += 1
    values = {}
    for c, rows in by_run.items():
        sums = np.zeros(len(order))
        for q in kept:
            fam = family_id({"query_id": q.query_id, "family_id": q.family_id})
            sums[position[fam]] += float(rows[q.query_id]["metrics"][metric])
        values[c] = sums / counts
    return order, values, excluded


def _alone(values: Mapping[frozenset[str], Array], p: str) -> Array:
    return np.asarray(values[frozenset((p,))] - values[frozenset()])


def _last(values: Mapping[frozenset[str], Array], p: str, everything: frozenset[str]) -> Array:
    return np.asarray(values[everything] - values[everything - {p}])


def _shapley(values: Mapping[frozenset[str], Array], names: Sequence[str]) -> dict[str, Array]:
    n = len(names)
    out = {}
    for p in names:
        others = [x for x in names if x != p]
        total = np.zeros_like(values[frozenset()])
        for r in range(n):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for subset in itertools.combinations(others, r):
                s = frozenset(subset)
                total = total + weight * (values[s | {p}] - values[s])
        out[p] = total
    return out


def _interactions(
    values: Mapping[frozenset[str], Array], names: Sequence[str]
) -> dict[tuple[str, str], Array]:
    out = {}
    for a, b in itertools.combinations(names, 2):
        others = [x for x in names if x not in (a, b)]
        terms = [
            values[s | {a, b}] - values[s | {a}] - values[s | {b}] + values[s]
            for r in range(len(others) + 1)
            for s in map(frozenset, itertools.combinations(others, r))
        ]
        out[(a, b)] = np.mean(terms, axis=0)
    return out


def _material(effects: Mapping[Any, Array], min_effect: float, seed: int) -> set[Any]:
    """Keys whose mean effect is shown larger than ``min_effect`` (Holm across the keys)."""
    keys = list(effects)
    if not keys:
        return set()
    ps = []
    for key in keys:
        x = effects[key]
        moved = x - np.sign(x.mean()) * min_effect
        ps.append(paired_randomization(moved.tolist(), seed=seed).p_value if x.any() else 1.0)
    return {
        key
        for key, p in zip(keys, holm_adjust(ps), strict=True)
        if p < ALPHA and abs(float(effects[key].mean())) > min_effect
    }


def _interval(x: Array, idx: Array) -> list[float]:
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, ALPHA / 2)), float(np.quantile(means, 1 - ALPHA / 2))]


def _leader(
    rows: Sequence[Mapping[str, Any]],
    contributions: Mapping[str, Array],
    idx: Array,
) -> dict[str, Any] | None:
    """The most harmful setting, and whether it is clearly ahead of the next one."""
    first = rows[0]
    if first["contribution"] >= 0:
        return None
    if len(rows) == 1:
        return {"setting": first["name"], "clear": True, "tied_with": []}
    gap = contributions[rows[1]["name"]] - contributions[first["name"]]
    low = _interval(gap, idx)[0]
    tied = [] if low > 0 else [rows[1]["name"]]
    return {"setting": first["name"], "clear": not tied, "tied_with": tied}


def _revert(
    plan: Plan,
    values: Mapping[frozenset[str], Array],
    leader: Mapping[str, Any] | None,
    idx: Array,
) -> dict[str, Any] | None:
    """The single setting whose revert from the candidate recovers the most, if one is clear.

    This is the leave-one-out effect, measured directly, not the Shapley value: a setting
    that contributes less can still be the better revert when reverting it also removes an
    interaction. ``None`` when no revert recovers anything shown, or two cannot be told apart.
    """
    everything = frozenset(s.name for s in plan.settings)
    gains = {s.name: -_last(values, s.name, everything) for s in plan.settings}
    ranked = sorted(gains, key=lambda p: -float(gains[p].mean()))
    best = ranked[0]
    lo, hi = _interval(gains[best], idx)
    if lo <= 0:
        return None
    tied = [p for p in ranked[1:2] if _interval(gains[best] - gains[p], idx)[0] <= 0]
    setting = next(s for s in plan.settings if s.name == best)
    if tied:
        note = (
            f"Reverting {tied[0]} instead recovers about as much on these questions; validate both."
        )
    elif leader is not None and leader["setting"] != best:
        note = (
            f"{leader['setting']} contributes the most, but reverting {best} recovers more, "
            "because it also removes an interaction."
        )
    else:
        note = "Validate the revert on held-out questions under the release policy."
    return {
        "setting": best,
        "params": setting.baseline,
        "recovers": float(gains[best].mean()),
        "interval": [lo, hi],
        "tied_with": tied,
        "note": note,
    }


def _fmt(x: float) -> str:
    return f"{x:+.3f}"


def summary(report: Mapping[str, Any]) -> str:
    """One paragraph a reader can act on."""
    metric = f"{report['metric']}@{report['k']}"
    total = report["difference"]
    lo, hi = report["difference_interval"]
    parts = [
        f"{metric} changes by {_fmt(total)} (95% interval {_fmt(lo)} to {_fmt(hi)}) from the "
        f"baseline to the candidate over {report['questions']} exploratory questions."
    ]
    leader = report["leader"]
    rows = {r["name"]: r for r in report["contributions"]}
    if leader is None:
        parts.append("No setting lowers it.")
    elif leader["clear"]:
        row = rows[leader["setting"]]
        share = f" ({row['share']:.0%} of the change)" if row["share"] is not None else ""
        parts.append(
            f"{leader['setting']} contributes the most: {_fmt(row['contribution'])}{share}."
        )
    else:
        names = [leader["setting"], *leader["tied_with"]]
        parts.append(
            f"{' and '.join(names)} contribute the most and cannot be told apart on these "
            "questions."
        )
    for inter in report["interactions"]:
        if inter["material"]:
            a, b = inter["settings"]
            parts.append(
                f"{a} and {b} interact: together they change it by "
                f"{_fmt(inter['interaction'])} beyond their separate effects, split equally "
                "between their contributions."
            )
    unfound = sorted(
        set(report["flagged_without_pair"]) | {p for pair in report["untested_pairs"] for p in pair}
    )
    if unfound:
        parts.append(
            "These settings interact with another setting that this design did not identify: "
            + ", ".join(unfound)
            + "."
        )
    confounded = [r["name"] for r in report["contributions"] if r["confounded"]]
    if confounded:
        parts.append("Changed together and attributed as one: " + ", ".join(confounded) + ".")
    return " ".join(parts)
