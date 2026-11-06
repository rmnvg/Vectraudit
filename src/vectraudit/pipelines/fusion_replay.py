"""Offline fusion replay: re-fuse a run's recorded stage hits under other fusion settings.

A recorded run keeps every stage's hits for each question (``queries[].evidence.stages``).
When a relevant document sat in one input stage but was ranked out by fusion, trying
another fusion method or weight needs no new pipeline calls: fuse the recorded inputs again
and score the result with the same metric contract. Fusion methods follow ranx (MIT); see
:mod:`vectraudit.core.fusion`.

A replay is a hypothesis, never a measurement of the real pipeline, and it says where it
cannot be trusted. Every limit below is counted in the summary and shown per question:

- A question with no recorded stage trace, no labels, a failed execution or a held-out
  split is excluded. Held-out questions are found with the optimiser's identity guard, which
  also refuses a held-out question reused under a new id.
- ``truncated_inputs``: an input stage recorded fewer than k hits, so a deeper fusion
  candidate may exist that the replay cannot see.
- Stages after fusion are not replayed. With nothing after fusion but the final truncation
  the replay is exact. With only filters, it is bounded: the optimistic ranking keeps every
  replayed document the recorded filters did not drop, and the pessimistic one keeps only
  documents they are known to have kept (a filter is assumed to judge each document on its
  own). A reranker, a custom stage or a branch makes it ``downstream_unknown``: the metrics
  are for the ranking before those stages and are not compared with the recorded one.
- A forbidden document in a replayed top-k is reported, and its absence is never a pass:
  later stages were not replayed.

Only IDs, stage names and numbers are read and written, never document text.
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from vectraudit.core.fusion import (
    FUSION_METHODS,
    NORM_METHODS,
    RANK_METHODS,
    FusionMethod,
    NormMethod,
    fuse,
)
from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.families import family_means
from vectraudit.evaluation.metrics import LABEL_METRICS, METRIC_CONTRACT_VERSION, semantic_metrics
from vectraudit.pipelines.optimize import question_identities

REPLAY_FORMAT = "vectraudit.fusion_replay"
REPLAY_FORMAT_VERSION = 1
MAX_CONFIGURATIONS = 200
GRID_KEYS = ("fusion_k", "fusion_norm", "depth")
"""Grid keys besides ``<input>_weight``."""

_FILTER_KINDS = {"filter"}


class ReplayError(ValueError):
    """The replay cannot be run on these inputs (the message says how to fix them)."""


@dataclass(frozen=True)
class Config:
    """One fusion setting to replay."""

    method: FusionMethod
    weights: tuple[tuple[str, float], ...] = ()
    rrf_k: int = 60
    norm: NormMethod = "min_max"
    depth: int | None = None

    @property
    def label(self) -> str:
        parts = [f"{name}_weight={w:g}" for name, w in self.weights]
        if self.method == "rrf":
            parts.append(f"k={self.rrf_k}")
        if self.method not in RANK_METHODS:
            parts.append(f"norm={self.norm}")
        if self.depth is not None:
            parts.append(f"depth={self.depth}")
        return f"{self.method}({', '.join(parts)})"


def parse_grid(text: str | None) -> dict[str, list[str]]:
    """``"vector_weight=0.3,0.5;fusion_k=20,60"`` as ``{key: [values]}``."""
    grid: dict[str, list[str]] = {}
    for part in (text or "").split(";"):
        if not part.strip():
            continue
        key, sep, values = part.partition("=")
        key = key.strip()
        items = [v.strip() for v in values.split(",") if v.strip()]
        if not sep or not key or not items:
            raise ReplayError(f"Grid entry {part.strip()!r} must look like name=v1,v2")
        if key in grid:
            raise ReplayError(f"Grid names {key!r} twice")
        grid[key] = items
    return grid


def expand_configs(
    methods: Sequence[str],
    grid: Mapping[str, Sequence[str]],
    inputs: Sequence[str],
    *,
    limit: int = MAX_CONFIGURATIONS,
) -> list[Config]:
    """Every method crossed with the grid; settings a method ignores are not repeated."""
    for method in methods:
        if method not in FUSION_METHODS:
            raise ReplayError(f"Unknown fusion method {method!r}; choose from {FUSION_METHODS}")
    if not methods:
        raise ReplayError("Name at least one fusion method")
    weight_keys = {k for k in grid if k.endswith("_weight")}
    for key in grid:
        if key not in GRID_KEYS and key not in weight_keys:
            raise ReplayError(
                f"Unknown grid key {key!r}; use <input>_weight, {', '.join(GRID_KEYS)}"
            )
    if weight_keys:
        if len(inputs) != 2:
            raise ReplayError("Weight grids need exactly two inputs; the other gets 1 - weight")
        if len(weight_keys) > 1:
            raise ReplayError("Give one weight key: the other input's weight is 1 - weight")
        (wkey,) = weight_keys
        if wkey[: -len("_weight")] not in inputs:
            raise ReplayError(f"{wkey!r} does not name an input; inputs are {', '.join(inputs)}")
    configs: list[Config] = []
    seen: set[Config] = set()
    for method in methods:
        axes = {key: list(values) for key, values in grid.items()}
        if method != "rrf":
            axes.pop("fusion_k", None)
        if method in RANK_METHODS:
            axes.pop("fusion_norm", None)
        names = list(axes)
        for combo in itertools.product(*(axes[n] for n in names)):
            chosen = dict(zip(names, combo, strict=True))
            config = _config(cast(FusionMethod, method), chosen, inputs)
            if config not in seen:
                seen.add(config)
                configs.append(config)
        if len(configs) > limit:
            raise ReplayError(f"The grid expands to more than {limit} configurations; narrow it.")
    return configs


def _config(method: FusionMethod, chosen: Mapping[str, str], inputs: Sequence[str]) -> Config:
    config = Config(method)
    for key, raw in chosen.items():
        try:
            if key == "fusion_k":
                config = replace(config, rrf_k=_whole(raw, 0, 10000, key))
            elif key == "depth":
                config = replace(config, depth=_whole(raw, 1, 10000, key))
            elif key == "fusion_norm":
                if raw not in NORM_METHODS:
                    raise ValueError(f"fusion_norm must be one of {', '.join(NORM_METHODS)}")
                config = replace(config, norm=cast(NormMethod, raw))
            else:
                weight = float(raw)
                if not 0 <= weight <= 1:
                    raise ValueError(f"{key} must be between 0 and 1")
                name = key[: -len("_weight")]
                other = next(i for i in inputs if i != name)
                config = replace(config, weights=((name, weight), (other, 1 - weight)))
        except ValueError as exc:
            raise ReplayError(f"Grid value {raw!r} for {key}: {exc}") from exc
    return config


def _whole(raw: str, low: int, high: int, key: str) -> int:
    value = int(raw)
    if not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def question_checks(dataset: Dataset, run: Mapping[str, Any]) -> None:
    """Refuse a run of another dataset, and a held-out question reused as an exploratory one."""
    recorded = (run.get("dataset") or {}).get("sha256")
    if recorded != dataset.sha256:
        raise ReplayError(
            "The run was recorded on a different dataset version than --dataset "
            "(the labels may differ). Pass the dataset the run used."
        )

    def identities(split_is_heldout: bool) -> set[str]:
        return question_identities(
            [
                {"text": q.text, "vector": None if q.vector is None else q.vector.tolist()}
                for q in dataset.queries
                if (q.split == "heldout") is split_is_heldout
            ]
        )

    heldout, explore = identities(True), identities(False)
    if heldout & explore:
        raise ReplayError("duplicate questions cross explore/heldout boundary")


def replay(
    run: Mapping[str, Any],
    dataset: Dataset,
    configs: Sequence[Config],
    *,
    inputs: Sequence[str] | None = None,
    fusion_stage: str | None = None,
    metric: str = "ndcg",
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Replay ``configs`` over the run's recorded stages; the report is a hypothesis."""
    if metric not in LABEL_METRICS:
        raise ReplayError(f"metric must be one of {', '.join(LABEL_METRICS)}")
    if run.get("status") != "complete":
        raise ReplayError(
            f"The run is {run.get('status')!r}; replay needs a complete run. Resume or re-run it."
        )
    if not configs:
        raise ReplayError("Nothing to replay: no configurations")
    question_checks(dataset, run)
    k = int((run.get("search") or {}).get("k") or 0)
    if k < 1:
        raise ReplayError("The run does not record its k")
    by_id = {q.query_id: q for q in dataset.queries}
    fusion_name, input_names = fusion_topology(run, fusion_stage, inputs)
    unknown = [c.label for c in configs if {n for n, _ in c.weights} - set(input_names)]
    if unknown:
        raise ReplayError(f"Weights name stages that are not inputs: {unknown[0]}")

    excluded: dict[str, int] = defaultdict(int)
    questions: list[dict[str, Any]] = []
    rows: list[_Row] = []
    for record in run["queries"]:
        query = by_id.get(record["query_id"])
        if query is None:
            raise ReplayError(f"The run has question {record['query_id']!r}, not in --dataset")
        reason = _exclusion(record, query)
        stages = _stages(record) if reason is None else {}
        if reason is None and (
            fusion_name not in stages or any(name not in stages for name in input_names)
        ):
            reason = "no_stage_trace"
        if reason is not None:
            excluded[reason] += 1
            questions.append({"query_id": record["query_id"], "excluded": reason})
            continue
        rows.append(_row(record, query, stages, fusion_name, input_names, k))
    if not rows:
        raise ReplayError(
            "No question could be replayed: "
            + (", ".join(f"{n} {why}" for why, n in sorted(excluded.items())) or "none in run")
        )

    results = [_run_config(c, rows, k, metric, seed, n_bootstrap) for c in configs]
    for row, shown in zip(rows, _question_rows(rows, configs, k), strict=True):
        questions.append(shown | _row_flags(row))
    questions.sort(key=lambda q: q["query_id"])
    downstream: dict[str, int] = defaultdict(int)
    for row in rows:
        downstream[row.downstream] += 1
    best = _best(results, metric)
    return {
        "format": REPLAY_FORMAT,
        "format_version": REPLAY_FORMAT_VERSION,
        "evidence": "hypothesis",
        "note": (
            "A replay re-fuses recorded stage hits. It is a hypothesis about the pipeline, "
            "not a measurement, and held-out questions were never used."
        ),
        "run": {
            "run_id": run.get("run_id"),
            "label": run.get("label"),
            "dataset_sha256": dataset.sha256,
            "k": k,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        },
        "fusion_stage": fusion_name,
        "inputs": list(input_names),
        "metric": metric,
        "seed": seed,
        "summary": {
            "questions_in_run": len(run["queries"]),
            "replayed": len(rows),
            "excluded": dict(sorted(excluded.items())),
            "truncated_inputs": sum(r.truncated_inputs for r in rows),
            "downstream": {key: downstream.get(key, 0) for key in ("none", "filter", "unknown")},
        },
        "configurations": results,
        "best": best,
        "questions": questions,
    }


@dataclass(frozen=True)
class _Row:
    query_id: str
    family: str
    labels: Mapping[str, float]
    forbidden: frozenset[str]
    recorded_ids: tuple[str, ...]
    recorded_fused: tuple[str, ...]
    inputs: Mapping[str, tuple[tuple[str, float | None], ...]]
    truncated_inputs: bool
    downstream: str
    seen: tuple[tuple[frozenset[str], frozenset[str]], ...]
    """Per downstream filter: (documents it received, documents it kept)."""


def fusion_topology(
    run: Mapping[str, Any], fusion_stage: str | None, inputs: Sequence[str] | None
) -> tuple[str, tuple[str, ...]]:
    """The recorded fusion stage and the input stages to fuse (all of its parents by default)."""
    for record in run["queries"]:
        stages = _stages(record)
        if not stages:
            continue
        fusion = [n for n, s in stages.items() if s["kind"] == "fusion"]
        if fusion_stage is not None:
            if fusion_stage not in stages or stages[fusion_stage]["kind"] != "fusion":
                raise ReplayError(f"{fusion_stage!r} is not a fusion stage in the recorded run")
            name = fusion_stage
        elif len(fusion) == 1:
            name = fusion[0]
        elif not fusion:
            raise ReplayError("The run records no stage of kind 'fusion'; nothing to replay.")
        else:
            raise ReplayError(f"Several fusion stages ({', '.join(fusion)}); pass --fusion-stage")
        parents = tuple(stages[name]["parents"])
        if not parents:
            raise ReplayError(f"Fusion stage {name!r} records no inputs")
        chosen = tuple(inputs) if inputs else parents
        if not set(chosen) <= set(parents):
            raise ReplayError(
                f"--inputs must be inputs of {name!r}: {', '.join(parents)} "
                f"(got {', '.join(chosen)})"
            )
        if len(set(chosen)) != len(chosen):
            raise ReplayError("--inputs names a stage twice")
        return name, chosen
    raise ReplayError(
        "The run records no stage traces (queries[].evidence.stages); "
        "record it with a pipeline that reports its stages."
    )


def _stages(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    evidence = record.get("evidence")
    raw = evidence.get("stages") if isinstance(evidence, Mapping) else None
    return {s["name"]: s for s in raw or [] if isinstance(s, Mapping) and "name" in s}


def _exclusion(record: Mapping[str, Any], query: DatasetQuery) -> str | None:
    if query.split == "heldout":
        return "heldout"
    if record.get("error") or record.get("ids") is None:
        return "errored"
    if not _stages(record):
        return "no_stage_trace"
    if query.no_answer or not query.relevant:
        return "unlabelled"
    return None


def _row(
    record: Mapping[str, Any],
    query: DatasetQuery,
    stages: Mapping[str, Mapping[str, Any]],
    fusion_name: str,
    input_names: Sequence[str],
    k: int,
) -> _Row:
    inputs = {
        name: tuple((h["id"], h.get("score")) for h in stages[name]["hits"]) for name in input_names
    }
    downstream, filters = _downstream(stages, fusion_name)
    return _Row(
        query_id=record["query_id"],
        family=query.family_id or query.query_id,
        labels=query.relevant,
        forbidden=frozenset(query.forbidden),
        recorded_ids=tuple(record["ids"]),
        recorded_fused=tuple(h["id"] for h in stages[fusion_name]["hits"]),
        inputs=inputs,
        truncated_inputs=any(len(hits) < k for hits in inputs.values()),
        downstream=downstream,
        seen=_filter_views(stages, filters),
    )


def _filter_views(
    stages: Mapping[str, Mapping[str, Any]], filters: Sequence[str]
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    views = []
    for name in filters:
        received: set[str] = set()
        for parent in stages[name]["parents"]:
            received.update(h["id"] for h in stages[parent]["hits"])
        views.append((frozenset(received), frozenset(h["id"] for h in stages[name]["hits"])))
    return tuple(views)


def _downstream(stages: Mapping[str, Mapping[str, Any]], fusion_name: str) -> tuple[str, list[str]]:
    """``none``, ``filter`` (with the filters, in order) or ``unknown``."""
    order = list(stages)
    after = order[order.index(fusion_name) + 1 :]
    if not after:
        return "none", []
    chain = []
    previous = fusion_name
    for name in after:
        if list(stages[name]["parents"]) != [previous]:
            return "unknown", []
        chain.append(name)
        previous = name
    kinds = [stages[n]["kind"] for n in chain]
    if kinds[-1] != "truncate" or any(k != "filter" for k in kinds[:-1]):
        return "unknown", []
    filters = [n for n in chain if stages[n]["kind"] in _FILTER_KINDS]
    return ("filter" if filters else "none"), filters


def _fused_ids(config: Config, row: _Row) -> list[str]:
    weights = dict(config.weights) or None
    ranking = fuse(
        {name: list(hits) for name, hits in row.inputs.items()},
        config.method,
        weights=weights,
        rrf_k=config.rrf_k,
        norm=config.norm,
        depth=config.depth,
    )
    return [doc for doc, _ in ranking]


def _views(config: Config, row: _Row, k: int) -> dict[str, list[str]]:
    """The replayed top-k ranking, or its optimistic and pessimistic bounds."""
    fused = _fused_ids(config, row)
    if row.downstream != "filter":
        return {"exact": fused[:k]}
    optimistic = [d for d in fused if not _dropped(d, row)]
    pessimistic = [d for d in fused if _kept(d, row)]
    return {"optimistic": optimistic[:k], "pessimistic": pessimistic[:k]}


def _dropped(doc: str, row: _Row) -> bool:
    return any(doc in received and doc not in kept for received, kept in row.seen)


def _kept(doc: str, row: _Row) -> bool:
    return all(doc in kept for _, kept in row.seen)


def _score(found: Sequence[str], row: _Row, k: int) -> dict[str, float]:
    return semantic_metrics(found, row.labels, k)


def _forbidden(found: Sequence[str], row: _Row) -> list[str]:
    return sorted(set(found) & row.forbidden)


def _run_config(
    config: Config, rows: Sequence[_Row], k: int, metric: str, seed: int, n_bootstrap: int
) -> dict[str, Any]:
    groups: dict[str, list[tuple[_Row, dict[str, float], dict[str, float]]]] = defaultdict(list)
    forbidden = 0
    for row in rows:
        if row.downstream == "unknown":
            found = _fused_ids(config, row)[:k]
            forbidden += bool(_forbidden(found, row))
            continue
        base = _score(row.recorded_ids, row, k)
        for view, found in _views(config, row, k).items():
            groups[view].append((row, _score(found, row, k), base))
            forbidden += bool(_forbidden(found, row)) and view != "pessimistic"
    summary = {
        view: _summarise(items, metric, seed, n_bootstrap) for view, items in sorted(groups.items())
    }
    return {
        "id": config.label,
        "method": config.method,
        "weights": dict(config.weights),
        "fusion_k": config.rrf_k if config.method == "rrf" else None,
        "fusion_norm": config.norm if config.method not in RANK_METHODS else None,
        "depth": config.depth,
        "results": summary,
        "forbidden_questions": forbidden,
    }


def _summarise(
    items: Sequence[tuple[_Row, dict[str, float], dict[str, float]]],
    metric: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in LABEL_METRICS:
        replay_means = family_means((row.family, ours[name]) for row, ours, _ in items)
        base_means = family_means((row.family, base[name]) for row, _, base in items)
        deltas = np.asarray(replay_means) - np.asarray(base_means)
        low, high = _interval(deltas, seed, n_bootstrap)
        metrics[name] = {
            "recorded": statistics.fmean(base_means),
            "replay": statistics.fmean(replay_means),
            "delta": float(deltas.mean()),
            "ci_low": low,
            "ci_high": high,
        }
    return {
        "n": len(items),
        "n_families": len({row.family for row, _, _ in items}),
        "metrics": metrics,
        "primary": metric,
    }


def _interval(deltas: np.ndarray[Any, Any], seed: int, n: int) -> tuple[float, float]:
    """Seeded 95% bootstrap interval of the mean family delta, as in the release comparison."""
    if len(deltas) == 1:
        return float(deltas[0]), float(deltas[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _question_rows(rows: Sequence[_Row], configs: Sequence[Config], k: int) -> list[dict[str, Any]]:
    shown = []
    for row in rows:
        relevant = {d for d, g in row.labels.items() if g > 0}
        recorded = set(row.recorded_ids[:k])
        per_config = {}
        for config in configs:
            views = (
                _views(config, row, k)
                if row.downstream != "unknown"
                else {"before_downstream": _fused_ids(config, row)[:k]}
            )
            per_config[config.label] = {
                view: {
                    "metrics": _score(found, row, k) if view != "before_downstream" else None,
                    "recovered": sorted((set(found) & relevant) - recorded),
                    "lost": sorted((recorded & relevant) - set(found)),
                    "forbidden": _forbidden(found, row),
                }
                for view, found in views.items()
            }
        shown.append({"query_id": row.query_id, "configurations": per_config})
    return shown


def _row_flags(row: _Row) -> dict[str, Any]:
    return {
        "downstream": row.downstream,
        "truncated_inputs": row.truncated_inputs,
        "recorded_ids": list(row.recorded_ids),
    }


def _best(results: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any] | None:
    """The configuration with the best mean delta on replays that are exact or bounded.

    Bounded replays count by their pessimistic ranking, so a configuration is preferred only
    when it helps even if every document the recorded filters never saw is dropped.
    """
    scored = []
    for result in results:
        views = result["results"]
        basis = "exact" if "exact" in views else "pessimistic"
        if basis in views:
            stats = views[basis]["metrics"][metric]
            scored.append((stats["delta"], result["id"], basis, stats))
    if not scored:
        return None
    delta, config_id, basis, stats = max(scored, key=lambda t: (t[0], t[1]))
    return {
        "configuration": config_id,
        "metric": metric,
        "basis": "exact" if basis == "exact" else "pessimistic filter bound",
        "delta": delta,
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
    }


def render_replay(report: Mapping[str, Any]) -> str:
    """A short terminal summary; the JSON report is the record."""
    s = report["summary"]
    lines = [
        f"Fusion replay of {report['fusion_stage']!r} over {', '.join(report['inputs'])} "
        f"(k = {report['run']['k']}, {report['evidence']})",
        f"  {s['replayed']} of {s['questions_in_run']} questions replayed; excluded: "
        + (", ".join(f"{n} {why}" for why, n in s["excluded"].items()) or "none"),
        f"  downstream stages: {s['downstream']['none']} exact, {s['downstream']['filter']} "
        f"filter-bounded, {s['downstream']['unknown']} unknown; "
        f"{s['truncated_inputs']} with shallow inputs",
    ]
    metric = report["metric"]
    for result in report["configurations"]:
        for view, data in result["results"].items():
            m = data["metrics"][metric]
            lines.append(
                f"  {result['id']} [{view}]: {metric} {m['replay']:.3f} vs recorded "
                f"{m['recorded']:.3f} ({m['delta']:+.3f}, 95% CI {m['ci_low']:+.3f} to "
                f"{m['ci_high']:+.3f}, {data['n_families']} families)"
            )
        if result["forbidden_questions"]:
            lines.append(
                f"    forbidden documents in the replayed top-k for "
                f"{result['forbidden_questions']} question(s)"
            )
    if report.get("next_step"):
        lines.append("  Next: " + report["next_step"]["text"])
    return "\n".join(lines)


def baseline_params(config: Config, inputs: Sequence[str]) -> dict[str, str | int | float]:
    """``config`` as the hybrid baseline's candidate params (``vector_weight``, ``fusion``...)."""
    params: dict[str, str | int | float] = {"fusion": config.method}
    if config.method == "rrf":
        params["fusion_k"] = config.rrf_k
    if config.method not in RANK_METHODS:
        params["fusion_norm"] = config.norm
    for name, weight in config.weights:
        if name in inputs:
            params[f"{name}_weight"] = weight
            break
    return params


def next_step(
    report: dict[str, Any],
    configs: Sequence[Config],
    fixes: Sequence[Any] = (),
    manifest: str | None = None,
    candidate: str | None = None,
) -> dict[str, Any] | None:
    """What to do about the best configuration: a declared ``try-fix``, or the ``Fix`` to add.

    A ``try-fix`` is named only when the candidate already declares a fix whose params are
    exactly the replayed settings; otherwise the fix to declare is shown. Parameter names are
    the built-in baseline's, so a custom pipeline maps them to its own.
    """
    best = report.get("best")
    if best is None:
        return None
    config = next(c for c in configs if c.label == best["configuration"])
    params = baseline_params(config, report["inputs"])
    for fix in fixes:
        if dict(fix.params) == params:
            command = (
                f"vectraudit workbench try-fix {manifest or 'MANIFEST'} "
                f"--fix {json.dumps(fix.name)}"
                + (f" --candidate {candidate}" if candidate else "")
                + " --query FAILING_QUESTION_ID"
            )
            return {
                "kind": "try_fix",
                "fix": fix.name,
                "command": command,
                "text": f"run `{command}` to measure it on the real pipeline.",
            }
    proposed = {
        "name": f"fuse with {config.label}",
        "addresses": ["not_observed", "lost_downstream:fusion"],
        "params": params,
    }
    return {
        "kind": "declare_fix",
        "fix": proposed,
        "text": (
            "declare this fix on the candidate, then run try-fix to measure it: "
            + json.dumps(proposed)
        ),
    }
