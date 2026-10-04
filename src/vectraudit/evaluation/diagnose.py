"""Explain a comparison: findings with an explicit evidence level.

- ``verified``: directly measured in these runs (a metric change, evidence counts).
- ``supported``: consistent with a measured change, e.g. a configuration difference that
  coincides with the measured effect. Plausible cause, not proof.
- ``hypothesis``: worth checking; nothing in the runs confirms it.

Findings never claim more than the runs show; each lists the facts it rests on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vectraudit.evaluation.evidence import REASON_TEXT, plan_uses_index

LEVELS = ("verified", "supported", "hypothesis")
MATERIAL = 0.005
"""Changes smaller than this (absolute, on a 0-1 metric) are not called a change."""

Pair = tuple[Mapping[str, Any], Mapping[str, Any]]


def _finding(
    fid: str,
    level: str,
    title: str,
    facts: Sequence[str],
    suggestion: str | None = None,
) -> dict[str, Any]:
    assert level in LEVELS
    return {
        "id": fid,
        "level": level,
        "title": title,
        "facts": list(facts),
        "suggestion": suggestion,
    }


def _dropped(stats: Mapping[str, Any] | None) -> bool:
    """A material drop whose 95% interval excludes zero."""
    return bool(stats) and stats["delta"] < -MATERIAL and stats["ci_high"] < 0  # type: ignore[index]


def _stable(stats: Mapping[str, Any] | None) -> bool:
    return bool(stats) and abs(stats["delta"]) <= MATERIAL  # type: ignore[index]


def _fmt(stats: Mapping[str, Any]) -> str:
    return f"{stats['baseline']:.3f} → {stats['candidate']:.3f} ({stats['delta']:+.3f})"


def diagnose(
    result: Mapping[str, Any],
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    pairs: Sequence[Pair],
) -> list[dict[str, Any]]:
    """Findings for a comparison that got as far as measuring quality."""
    metrics = result.get("metrics") or {}
    diff = {d["path"]: d for d in result.get("config_diff") or []}
    findings: list[dict[str, Any]] = []
    semantic = next((metrics[m] for m in ("ndcg", "recall", "mrr") if m in metrics), None)
    semantic_name = next((m for m in ("ndcg", "recall", "mrr") if m in metrics), None)
    ann = metrics.get("ann_recall")

    findings += _approximation(ann, diff, cand, pairs)
    findings += _relevance(semantic_name, semantic, ann, diff, base, cand)
    findings += _plans(base, cand, pairs, result)
    findings += _short_results(cand, pairs)
    findings += _concentration(result)
    findings += _latency(result, diff)
    findings += _nondeterminism(cand)
    unchanged = all(abs(m["delta"]) <= MATERIAL for m in metrics.values())
    if not findings and result.get("verdict") == "pass" and not diff and unchanged:
        findings.append(
            _finding(
                "no-change",
                "verified",
                "No measurable change in quality",
                [f"{len(pairs)} paired queries; every metric moved by ≤ {MATERIAL}."],
            )
        )
    order = {level: i for i, level in enumerate(LEVELS)}
    return sorted(findings, key=lambda f: order[f["level"]])


def _search_changes(diff: Mapping[str, Any]) -> list[str]:
    return [
        f"{path.removeprefix('search.params.')}: {d['baseline']} → {d['candidate']}"
        for path, d in diff.items()
        if path.startswith("search.params.")
    ]


def _approximation(
    ann: Mapping[str, Any] | None,
    diff: Mapping[str, Any],
    cand: Mapping[str, Any],
    pairs: Sequence[Pair],
) -> list[dict[str, Any]]:
    if not _dropped(ann):
        return []
    assert ann is not None
    out = [
        _finding(
            "approximation-loss",
            "verified",
            "The index finds fewer of its exact nearest neighbours",
            [f"ANN recall@k {_fmt(ann)} over {ann['n']} queries."],
        )
    ]
    changes = _search_changes(diff)
    index_changes = [p for p in diff if p.startswith("backend.config.")]
    if changes or index_changes:
        what = changes + [f"{p.removeprefix('backend.config.')} changed" for p in index_changes]
        out.append(
            _finding(
                "approximation-config",
                "supported",
                "The approximation loss coincides with a search or index configuration change",
                ["Changed: " + "; ".join(what[:6]) + "."],
            )
        )
    reasons: dict[str, int] = {}
    min_nprobe: list[int] = []
    nprobe = None
    for _, c in pairs:
        ev = c.get("evidence") or {}
        if ev.get("kind") != "faiss_ivf":
            continue
        nprobe = ev.get("nprobe")
        min_nprobe.append(int(ev["min_nprobe_for_all"]))
        for reason, n in (ev.get("missed_by_reason") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(n)
    total = sum(reasons.values())
    if total:
        unprobed = reasons.get("CELL_NOT_PROBED", 0)
        facts = [
            f"{total} exact neighbours missed across the candidate's queries: "
            + ", ".join(
                f"{n} {REASON_TEXT.get(r, r)}"
                for r, n in sorted(reasons.items(), key=lambda x: -x[1])
            )
            + " (replayed from the IVF probe order)."
        ]
        if unprobed >= total / 2:
            p95 = math.ceil(float(np.percentile(min_nprobe, 95)))
            out.append(
                _finding(
                    "ivf-unprobed",
                    "verified",
                    "Most missed neighbours were in inverted lists the search did not probe",
                    [
                        *facts,
                        f"nprobe {nprobe}; nprobe {p95} reaches every exact neighbour "
                        "for 95% of queries.",
                    ],
                    f"Raise nprobe towards {p95} and measure the latency cost "
                    "(vectraudit sweep shows the curve).",
                )
            )
        else:
            out.append(
                _finding(
                    "ivf-quantization",
                    "verified",
                    "Most missed neighbours were in probed lists but ranked out by compressed "
                    "distances",
                    facts,
                    "Raising nprobe will not recover these. Re-rank candidates with exact "
                    "distances, or use more bytes per vector (PQ m / SQ type).",
                )
            )
    return out


def _relevance(
    name: str | None,
    semantic: Mapping[str, Any] | None,
    ann: Mapping[str, Any] | None,
    diff: Mapping[str, Any],
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _dropped(semantic):
        return []
    assert semantic is not None
    assert name is not None
    facts = [f"{name}@k {_fmt(semantic)} over {semantic['n']} labelled queries."]
    if ann is not None and _stable(ann):
        facts.append(
            f"ANN recall@k is stable ({_fmt(ann)}): the index still finds its exact neighbours."
        )
        inputs = [
            p
            for p in diff
            if p.startswith(
                ("labels.", "search.embedder", "backend.snapshot", "backend.target", "connection.")
            )
        ]
        if inputs:
            facts.append(
                "Changed inputs: "
                + "; ".join(
                    f"{p} ({diff[p]['baseline']} → {diff[p]['candidate']})" for p in inputs[:5]
                )
                + "."
            )
            return [
                _finding(
                    "relevance-not-approximation",
                    "supported",
                    "Relevance fell although the index approximation did not: the change is in "
                    "what is retrieved from (embeddings, chunks or corpus), not in search tuning",
                    facts,
                    "Compare the embedding or chunking versions on the labelled set before "
                    "tuning the index; index parameters cannot recover this loss.",
                )
            ]
        return [
            _finding(
                "relevance-drift",
                "hypothesis",
                "Relevance fell with no configuration change: the corpus or labels may have "
                "drifted",
                facts,
                "Check whether documents were added, removed or re-chunked between the runs; "
                "freeze or version the corpus for evaluation.",
            )
        ]
    if ann is None:
        facts.append(
            "No exact reference in both runs, so approximation loss can't be separated "
            "from relevance changes."
        )
        return [
            _finding(
                "relevance-drop",
                "verified",
                f"Labelled relevance dropped ({name})",
                facts,
                "Record with an exact reference (raw vectors for FAISS) to separate index "
                "approximation from embedding or corpus effects.",
            )
        ]
    if _dropped(ann):
        assert ann is not None
        facts.append(
            f"ANN recall@k also fell ({_fmt(ann)}): consistent with index approximation "
            "accounting for at least part of the relevance loss."
        )
    return [_finding("relevance-drop", "verified", f"Labelled relevance dropped ({name})", facts)]


def _plans(
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    pairs: Sequence[Pair],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flags: list[tuple[bool, bool]] = []
    for b, c in pairs:
        before, after = plan_uses_index(b.get("evidence")), plan_uses_index(c.get("evidence"))
        if before is not None and after is not None:
            flags.append((before, after))
    if not flags:
        return []
    lost = sum(b and not c for b, c in flags)
    gained = sum(c and not b for b, c in flags)
    out = []
    latency = result.get("latency") or {}
    if lost:
        facts = [
            f"{lost} of {len(flags)} queries used the vector index in the baseline plan "
            "but not in the candidate's (PostgreSQL EXPLAIN)."
        ]
        if (latency.get("p95_change_pct") or 0) > 20:
            facts.append(f"p95 latency rose {latency['p95_change_pct']:.0f}%.")
        out.append(
            _finding(
                "plan-lost-index",
                "verified",
                "The candidate's query plans stopped using the vector index",
                facts,
                "Check the index exists for the configured distance operator class, that "
                "filters don't push the planner to a sequential scan, and table statistics "
                "(ANALYZE).",
            )
        )
    if gained:
        out.append(
            _finding(
                "plan-gained-index",
                "verified",
                "The candidate's query plans now use a vector index",
                [
                    f"{gained} of {len(flags)} queries switched from a sequential scan to an "
                    "index scan. Index scans are approximate: expect lower recall, lower latency."
                ],
            )
        )
    return out


def _short_results(cand: Mapping[str, Any], pairs: Sequence[Pair]) -> list[dict[str, Any]]:
    k = cand["search"]["k"]
    short = [(b, c) for b, c in pairs if len(c["ids"]) < k <= len(b["ids"])]
    if not short:
        return []
    filtered = sum(bool(c.get("filtered")) for _, c in short)
    facts = [
        f"{len(short)} queries returned fewer than k = {k} results in the candidate "
        "but a full list in the baseline."
    ]
    suggestion = None
    if filtered == len(short):
        facts.append("All of them are filtered queries.")
        suggestion = (
            "Approximate indexes filter after the index scan, so selective filters can starve "
            "results: raise ef_search / probes, enable iterative scans (pgvector 0.8+), or "
            "use a partial index or Qdrant payload index for the filter."
        )
    return [
        _finding(
            "short-results",
            "verified",
            "Some queries returned fewer than k results",
            facts,
            suggestion,
        )
    ]


def _concentration(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries = result.get("queries") or {}
    regressed = queries.get("regressed") or []
    n = queries.get("regressed_count") or 0
    total = result.get("n_queries") or 0
    if n < 3 or not total:
        return []
    out = []
    counts: dict[str, int] = {}
    for q in regressed:
        for tag in q.get("tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    cohorts = result.get("cohorts") or {}
    for tag, hits in sorted(counts.items(), key=lambda x: -x[1]):
        size = cohorts.get(tag, {}).get("n", 0)
        if size and hits / n >= 0.6 and size / total <= 0.4:
            out.append(
                _finding(
                    "cohort-concentration",
                    "verified",
                    f"Regressions concentrate in cohort {tag!r}",
                    [
                        f"{hits} of {n} regressed queries are tagged {tag!r}, which is "
                        f"{size} of {total} queries."
                    ],
                )
            )
            break
    filtered = sum(bool(q.get("filtered")) for q in regressed)
    if filtered / n >= 0.8 and len(regressed) == n:
        out.append(
            _finding(
                "filtered-concentration",
                "verified",
                "Regressions are almost all filtered queries",
                [f"{filtered} of {n} regressed queries use a filter."],
                "Look at filter selectivity and how the backend combines filters with "
                "approximate search.",
            )
        )
    return out


def _latency(result: Mapping[str, Any], diff: Mapping[str, Any]) -> list[dict[str, Any]]:
    latency = result.get("latency") or {}
    change = latency.get("p95_change_pct")
    if not latency.get("compared") or change is None or abs(change) < 20:
        return []
    direction = "rose" if change > 0 else "fell"
    facts = [
        f"p95 latency {direction} {abs(change):.0f}% "
        f"({latency['baseline']['p95']:.2f} → {latency['candidate']['p95']:.2f} ms) on "
        "matching runner environments."
    ]
    changes = _search_changes(diff)
    if changes:
        facts.append("Search parameters changed: " + "; ".join(changes) + ".")
        return [
            _finding(
                "latency-params",
                "supported",
                f"Latency {direction} with the search parameter change",
                facts,
            )
        ]
    return [_finding("latency-change", "verified", f"Latency {direction}", facts)]


def _nondeterminism(cand: Mapping[str, Any]) -> list[dict[str, Any]]:
    n = sum(bool(q.get("nondeterministic")) for q in cand["queries"])
    if not n:
        return []
    return [
        _finding(
            "nondeterministic",
            "verified",
            "Some candidate queries are not deterministic",
            [f"{n} queries returned different results across repeats."],
            "Per-query deltas on these queries may be noise; check for concurrent writes, "
            "replica routing or randomized search parameters.",
        )
    ]
