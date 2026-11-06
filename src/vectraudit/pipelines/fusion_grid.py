"""Fusion settings as optimiser candidates, optionally shortlisted by an offline replay.

``workbench optimize --fusion-grid "fusion=rrf,wsum;vector_weight=0.3,0.5,0.7"`` crosses the
named settings of the built-in hybrid baseline into extra candidates that share the base
candidate's adapter. They are ordinary candidates: each runs on exploratory questions only,
and only the chosen one is validated once on held-out questions, as for any candidate.

``--shortlist-from replay.json`` keeps only the grid points a fusion replay
(:mod:`vectraudit.pipelines.fusion_replay`) ranked highest, so the real runs go to the
settings the recorded stage hits already favour. A replay is a hypothesis: the shortlist
saves runs, it does not replace them, and the optimisation report records that a replay
chose which points ran.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.core.fusion import FUSION_METHODS, NORM_METHODS, RANK_METHODS
from vectraudit.pipelines.fusion_replay import (
    REPLAY_FORMAT,
    REPLAY_FORMAT_VERSION,
    ReplayError,
    parse_grid,
)
from vectraudit.pipelines.manifest import Candidate

GRID_KEYS = ("fusion", "fusion_k", "fusion_norm", "vector_weight")
"""The hybrid baseline's fusion params; any other key would be silently ignored by it."""
DEFAULT_TOP = 3


def expand_fusion_grid(
    base: Candidate, grid_text: str, taken: Sequence[str], *, room: int
) -> list[Candidate]:
    """One candidate per grid point, named ``<base>+fusion=wsum,vector_weight=0.3``.

    ``taken`` are the manifest's candidate names and ``room`` how many more candidates
    ``max_candidates`` allows. Settings the effective method ignores are not repeated
    (``fusion_k`` for anything but ``rrf``, ``fusion_norm`` for rank methods), and a point
    equal to the base candidate itself is dropped.
    """
    if base.adapter.kind != "hybrid":
        raise ReplayError(
            f"--fusion-grid sets the built-in hybrid baseline's params, but {base.name!r} is "
            f"a {base.adapter.kind} candidate; choose a hybrid one with --grid-base."
        )
    grid = parse_grid(grid_text)
    if not grid:
        raise ReplayError("--fusion-grid is empty; use name=v1,v2;name=v1,v2")
    for key in grid:
        if key not in GRID_KEYS:
            raise ReplayError(f"Unknown fusion-grid key {key!r}; use {', '.join(GRID_KEYS)}")
    typed = {key: [_typed(key, raw) for raw in values] for key, values in grid.items()}
    points: list[dict[str, str | int | float]] = []
    names = list(typed)
    for combo in itertools.product(*(typed[n] for n in names)):
        chosen = dict(zip(names, combo, strict=True))
        method = chosen.get("fusion", base.params.get("fusion", "rrf"))
        if method != "rrf":
            chosen.pop("fusion_k", None)
        if method in RANK_METHODS:
            chosen.pop("fusion_norm", None)
        if all(base.params.get(k) == v for k, v in chosen.items()):
            continue
        if chosen not in points:
            points.append(chosen)
    if not points:
        raise ReplayError("Every fusion-grid point is the base candidate's own settings.")
    if len(points) > room:
        raise ReplayError(
            f"The fusion grid makes {len(points)} candidates, but only {room} more fit in "
            f"max_candidates with the {len(taken)} already in the manifest. Narrow the grid "
            "or raise max_candidates."
        )
    candidates = []
    for chosen in points:
        name = f"{base.name}+" + ",".join(f"{k}={v}" for k, v in chosen.items())
        if name in taken:
            raise ReplayError(f"A candidate named {name!r} already exists.")
        candidates.append(
            base.model_copy(update={"name": name, "params": {**base.params, **chosen}, "fixes": []})
        )
    return candidates


def _typed(key: str, raw: str) -> str | int | float:
    try:
        if key == "fusion":
            if raw not in FUSION_METHODS:
                raise ValueError(f"fusion must be one of {', '.join(FUSION_METHODS)}")
            return raw
        if key == "fusion_norm":
            if raw not in NORM_METHODS:
                raise ValueError(f"fusion_norm must be one of {', '.join(NORM_METHODS)}")
            return raw
        if key == "fusion_k":
            value = int(raw)
            if not 0 <= value <= 10000:
                raise ValueError("fusion_k must be an integer from 0 to 10000")
            return value
        weight = float(raw)
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        return weight
    except ValueError as exc:
        raise ReplayError(f"Fusion-grid value {raw!r} for {key}: {exc}") from exc


def shortlist(
    points: Sequence[Candidate],
    base: Candidate,
    replay: Mapping[str, Any],
    dataset_sha256: str,
    *,
    top: int = DEFAULT_TOP,
) -> tuple[list[Candidate], dict[str, Any]]:
    """The grid points among the replay's ``top`` configurations, and what to record.

    Replay configurations rank by mean delta on exact replays (or the pessimistic bound for
    filtered ones). A grid point is kept when its fusion params equal one of them, so the
    replay must have covered the same settings. Raises when the replay is for another
    dataset or matches no grid point.
    """
    if top < 1:
        raise ReplayError("--top must be at least 1")
    if replay.get("format") != REPLAY_FORMAT or replay.get("format_version") != (
        REPLAY_FORMAT_VERSION
    ):
        raise ReplayError("--shortlist-from is not a fusion replay report")
    run = replay.get("run") or {}
    if run.get("dataset_sha256") != dataset_sha256:
        raise ReplayError(
            "The replay was run on a different dataset version than this manifest's; "
            "replay again on the manifest's dataset."
        )
    metric = str(replay.get("metric"))
    inputs = list(replay.get("inputs") or [])
    ranked = []
    for entry in replay.get("configurations") or []:
        views = entry.get("results") or {}
        basis = "exact" if "exact" in views else "pessimistic"
        if basis not in views or entry.get("depth") is not None:
            continue
        delta = views[basis]["metrics"][metric]["delta"]
        ranked.append((-delta, entry["id"], _entry_params(entry, inputs)))
    if not ranked:
        raise ReplayError(
            "The replay has no configuration that was compared with the recorded ranking."
        )
    chosen = sorted(ranked, key=lambda t: (t[0], t[1]))[:top]
    wanted = [p for _, _, p in chosen]
    kept = [c for c in points if _point_params(c) in wanted]
    if not kept:
        raise ReplayError(
            f"None of the replay's top {top} configurations ({', '.join(i for _, i, _ in chosen)}) "
            "is a point of --fusion-grid. Replay the same settings, or widen --top."
        )
    return kept, {
        "source": "fusion_replay",
        "replay_run_id": run.get("run_id"),
        "metric": metric,
        "top": top,
        "grid_points": len(points),
        "shortlisted": [c.name for c in kept],
        "evidence": "hypothesis",
        "note": (
            "A replay of recorded stage hits chose which grid points ran. Each still ran on "
            "exploratory questions, and only the chosen candidate saw held-out questions."
        ),
    }


def _entry_params(entry: Mapping[str, Any], inputs: Sequence[str]) -> dict[str, Any]:
    method = entry["method"]
    params: dict[str, Any] = {"fusion": method}
    if method == "rrf":
        params["fusion_k"] = entry["fusion_k"]
    if method not in RANK_METHODS:
        params["fusion_norm"] = entry["fusion_norm"]
    weight = (entry.get("weights") or {}).get("vector")
    if weight is not None and "vector" in inputs:
        params["vector_weight"] = round(weight, 9)
    return params


def _point_params(candidate: Candidate) -> dict[str, Any]:
    params = candidate.params
    method = params.get("fusion", "rrf")
    out: dict[str, Any] = {"fusion": method}
    if method == "rrf":
        out["fusion_k"] = params.get("fusion_k", 60)
    if method not in RANK_METHODS:
        out["fusion_norm"] = params.get("fusion_norm", "min_max")
    if "vector_weight" in params:
        out["vector_weight"] = round(float(params["vector_weight"]), 9)
    return out
