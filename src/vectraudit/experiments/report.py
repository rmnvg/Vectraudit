"""Rank experiment candidates and render the comparison report.

Candidates are ranked by a primary metric (nDCG by default). Each is compared with the
leader on the same queries: the paired difference and its seeded bootstrap 95% interval.
A candidate whose interval includes zero is *statistically tied* with the leader: the
data can't separate them, so the report says so instead of declaring a false winner.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from vectraudit import __version__
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.metrics import SEMANTIC_METRICS
from vectraudit.experiments.runner import CandidateResult

N_BOOTSTRAP = 2000
NONE = "\u2013"  # en dash for missing values


def _per_query(run: dict[str, Any], metric: str) -> dict[str, float]:
    return {q["query_id"]: float(q["metrics"][metric]) for q in run["queries"] if q.get("metrics")}


def _interval(deltas: np.ndarray[Any, Any], seed: int) -> tuple[float, float]:
    if len(deltas) < 2:
        v = float(deltas[0]) if len(deltas) else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(N_BOOTSTRAP, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank(
    summary: dict[str, Any],
    results: Sequence[CandidateResult],
    primary: str = "ndcg",
    seed: int = 0,
) -> dict[str, Any]:
    """Add a ``ranking`` (best first), cohort winners and leader diagnoses to the summary."""
    if primary not in SEMANTIC_METRICS:
        raise ValueError(f"primary metric must be one of {', '.join(SEMANTIC_METRICS)}")
    scored = [r for r in results if r.run["summary"].get("metrics")]
    if not scored:
        raise ValueError("No candidate has labelled metrics: add relevance labels.")
    scored.sort(key=lambda r: -r.run["summary"]["metrics"][primary])
    leader = scored[0]
    lead_scores = _per_query(leader.run, primary)
    ranking = []
    for position, r in enumerate(scored, 1):
        scores = _per_query(r.run, primary)
        shared = sorted(set(scores) & set(lead_scores))
        deltas = np.asarray([scores[q] - lead_scores[q] for q in shared])
        low, high = _interval(deltas, seed)
        tied = r is leader or (low <= 0 <= high)
        findings = []
        if r is not leader:
            check = compare_runs(leader.run, r.run)
            findings = [{"level": f["level"], "title": f["title"]} for f in check["findings"][:3]]
        ranking.append(
            {
                "rank": position,
                "name": r.spec.name,
                "metrics": r.run["summary"]["metrics"],
                "ann_recall": r.run["summary"].get("ann_recall"),
                "span_metrics": (r.span_summary or {}).get("metrics"),
                "delta_vs_leader": float(deltas.mean()) if len(deltas) else 0.0,
                "ci_vs_leader": [low, high],
                "tied_with_leader": tied,
                "latency_ms": r.run["summary"].get("latency_ms"),
                "stats": r.stats,
                "config": r.spec.describe(),
                "findings_vs_leader": findings,
                "run_id": r.run["run_id"],
            }
        )
    cohorts: dict[str, Any] = {}
    tags = sorted({t for r in scored for q in r.run["queries"] for t in q.get("tags", [])})
    for tag in tags:
        best = None
        for r in scored:
            vals = [
                q["metrics"][primary]
                for q in r.run["queries"]
                if tag in q.get("tags", []) and q.get("metrics")
            ]
            if vals and (best is None or float(np.mean(vals)) > best[1]):
                best = (r.spec.name, float(np.mean(vals)), len(vals))
        if best is not None:
            cohorts[tag] = {"winner": best[0], primary: best[1], "n": best[2]}
    tied_names = [x["name"] for x in ranking if x["tied_with_leader"]]
    return {
        **summary,
        "primary_metric": primary,
        "leader": leader.spec.name,
        "tied_with_leader": tied_names,
        "ranking": ranking,
        "cohorts": cohorts,
    }


# --- rendering ---------------------------------------------------------------------------


def _n(x: Any, digits: int = 3, signed: bool = False) -> str:
    if x is None:
        return NONE
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _mb(n: Any) -> str:
    return NONE if n is None else f"{n / 1_048_576:.1f} MB"


def _ci(r: dict[str, Any]) -> str:
    lo, hi = r["ci_vs_leader"]
    delta = _n(r["delta_vs_leader"], signed=True)
    return f"{delta} [{_n(lo, signed=True)}, {_n(hi, signed=True)}]"


def _s(x: Any) -> str:
    return "cached" if x is None else f"{x:.1f} s"


def verdict_sentence(ranked: dict[str, Any]) -> str:
    leader, tied = (
        ranked["leader"],
        [t for t in ranked["tied_with_leader"] if t != ranked["leader"]],
    )
    metric = ranked["primary_metric"]
    if tied:
        return (
            f"{leader} leads on {metric}, but {', '.join(tied)} "
            f"{'is' if len(tied) == 1 else 'are'} statistically tied with it on these "
            f"{ranked['dataset']['n_labelled']} labelled queries: choose on cost, or add "
            "queries to separate them."
        )
    return (
        f"{leader} is best on {metric}; every other candidate is measurably worse on "
        f"these {ranked['dataset']['n_labelled']} labelled queries."
    )


def render_markdown(ranked: dict[str, Any]) -> str:
    m = ranked["primary_metric"]
    lines = [
        f"## Experiment {ranked['name']}: {ranked['leader']} leads",
        "",
        verdict_sentence(ranked),
        "",
        f"| Rank | Candidate | {m} | Δ vs leader (95% CI) | recall | MRR | p95 ms | Index |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked["ranking"]:
        ci = NONE if r["rank"] == 1 else _ci(r)
        tie = " (tied)" if r["tied_with_leader"] and r["rank"] > 1 else ""
        lines.append(
            f"| {r['rank']} | {r['name']}{tie} "
            f"| {_n(r['metrics'][m])} | {ci} | {_n(r['metrics']['recall'])} "
            f"| {_n(r['metrics']['mrr'])} | {_n((r['latency_ms'] or {}).get('p95'), 2)} "
            f"| {_mb(r['stats']['index_bytes'])} |"
        )
    lines.append("")
    return "\n".join(lines)


CSS = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b636d;--line:#d9dde3;--card:#f6f8fa;--good:#1a7f37;
--tie:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--line:#30363d;--card:#161b22;--good:#3fb950;--tie:#d29922}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1150px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px}.muted{color:var(--muted)}
.lead{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;
font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;
border:1px solid currentColor}.best{color:var(--good)}.tie{color:var(--tie)}
td.small{font-size:12px;color:var(--muted)}
"""


def _e(x: Any) -> str:
    return html.escape(NONE if x is None else str(x))


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def render_html(ranked: dict[str, Any]) -> str:
    m = ranked["primary_metric"]
    quality_rows = []
    for r in ranked["ranking"]:
        if r["rank"] == 1:
            badge, ci = '<span class="tag best">leader</span>', NONE
        else:
            badge = '<span class="tag tie">tied</span>' if r["tied_with_leader"] else ""
            ci = _ci(r)
        quality_rows.append(
            [
                str(r["rank"]),
                f"<strong>{_e(r['name'])}</strong> {badge}",
                _e(_n(r["metrics"][m])),
                _e(ci),
                _e(_n(r["metrics"]["recall"])),
                _e(_n(r["metrics"]["mrr"])),
                _e(_n(r["metrics"]["precision"])),
                _e(_n(r["ann_recall"])),
                _e(_n((r["span_metrics"] or {}).get("recall"))),
            ]
        )
    cost_rows = []
    for r in ranked["ranking"]:
        c, s, lat = r["config"], r["stats"], r["latency_ms"] or {}
        cost_rows.append(
            [
                f"<strong>{_e(r['name'])}</strong>",
                _e(c["embedder"]["label"]),
                _e(c["chunking"]["label"]),
                _e(c["index"]["factory"]),
                _e(s["chunks"]),
                _e(s["dim"]),
                _e(_mb(s["index_bytes"])),
                _e(_s(s["embed_documents_s"])),
                _e(_n(lat.get("p50"), 2)),
                _e(_n(lat.get("p95"), 2)),
            ]
        )
    cohort_rows = [
        [_e(tag), _e(c["winner"]), _e(_n(c[m])), _e(c["n"])] for tag, c in ranked["cohorts"].items()
    ]
    finding_rows = [
        [
            f"<strong>{_e(r['name'])}</strong>",
            "<br>".join(f"{_e(f['level'])}: {_e(f['title'])}" for f in r["findings_vs_leader"])
            or '<span class="muted">no findings</span>',
        ]
        for r in ranked["ranking"][1:]
    ]
    body = (
        f"<h1>Experiment {_e(ranked['name'])}</h1>"
        f'<p class="muted">{_e(ranked["dataset"]["n_labelled"])} labelled queries · '
        f"{_e(ranked['corpus']['documents'])} documents · k = {_e(ranked['k'])} · scored at "
        f"document level · vectraudit {_e(__version__)}</p>"
        f'<div class="lead">{_e(verdict_sentence(ranked))}</div>'
        "<h2>Quality</h2>"
        + _table(
            [
                "#",
                "Candidate",
                f"{m}@k",
                "Δ vs leader (95% CI)",
                "Recall@k",
                "MRR@k",
                "Precision@k",
                "ANN recall",
                "Span recall",
            ],
            quality_rows,
        )
        + "<h2>Cost</h2>"
        + _table(
            [
                "Candidate",
                "Embedder",
                "Chunking",
                "Index",
                "Chunks",
                "Dim",
                "Index size",
                "Embed corpus",
                "Search p50 ms",
                "Search p95 ms",
            ],
            cost_rows,
        )
        + (
            "<h2>Best per cohort</h2>"
            + _table(["Cohort", "Winner", f"{m}@k", "Queries"], cohort_rows)
            if cohort_rows
            else ""
        )
        + (
            "<h2>Compared with the leader</h2>" + _table(["Candidate", "Findings"], finding_rows)
            if finding_rows
            else ""
        )
        + "<h2>Next</h2><p>Each candidate has a run file in <code>runs/</code>. Make the "
        "chosen one your baseline: <code>vectraudit cloud push runs/NAME.json</code>, then "
        "<code>vectraudit cloud pin RUN_ID</code>, or keep the file and pass it to "
        "<code>vectraudit check</code>.</p>"
    )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'\">"
        f"<title>Experiment: {_e(ranked['name'])}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )


def to_json(ranked: dict[str, Any]) -> str:
    return json.dumps(ranked, indent=2, allow_nan=False, default=str)
