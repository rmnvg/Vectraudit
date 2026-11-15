"""Generate trec_eval reference values for the queries of the ranx metric fixture.

Runs only in a throwaway environment with ``pytrec_eval`` and ``numpy`` installed (see
``../README.md``); neither is a Vectraudit dependency. It reads ``tests/fixtures/ranx/metrics.json``
(labels and rankings of 200 seeded queries) so both reference implementations score the same
data, and writes ``tests/fixtures/trec_eval/metrics.json``. Rerunning reproduces it exactly.

Mapping to trec_eval's measures (relevance level 1, so a label of 1 or more is relevant):

- ``recall@k``  = ``recall_k``       - ``precision@k`` = ``P_k``
- ``ndcg@k``    = ``ndcg_cut_k``     - ``map@k``       = ``map_cut_k``
- ``hit_rate@k`` = ``success_k``     - ``mrr@k``       = ``recip_rank`` on the ranking cut to k
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pytrec_eval

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "tests/fixtures/ranx/metrics.json"
OUT = ROOT / "tests/fixtures/trec_eval/metrics.json"
MEASURES = {
    "recall": "recall.{k}",
    "precision": "P.{k}",
    "ndcg": "ndcg_cut.{k}",
    "map": "map_cut.{k}",
    "hit_rate": "success.{k}",
}
KEYS = {
    "recall": "recall_{k}",
    "precision": "P_{k}",
    "ndcg": "ndcg_cut_{k}",
    "map": "map_cut_{k}",
    "hit_rate": "success_{k}",
}


def descending(ranking: list[str]) -> dict[str, float]:
    """Strictly decreasing scores, so the order is exactly the ranking (no tie-breaking)."""
    return {d: float(len(ranking) - i) for i, d in enumerate(ranking)}


def alignment(queries: list[dict]) -> str:
    blob = json.dumps(
        [[q["id"], q["labels"], q["ranking"]] for q in queries], sort_keys=True
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text())
    queries = source["queries"]
    ks = source["ks"]
    qrels = {q["id"]: {d: int(g) for d, g in q["labels"].items()} for q in queries}
    full = {q["id"]: descending(q["ranking"]) for q in queries}
    wanted = {m.format(k=k) for m in MEASURES.values() for k in ks}
    scored = pytrec_eval.RelevanceEvaluator(qrels, wanted).evaluate(full)
    rr = {}
    for k in ks:
        cut = {q["id"]: descending(q["ranking"][:k]) for q in queries}
        got = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(cut)
        rr[k] = {qid: v["recip_rank"] for qid, v in got.items()}
    rows = {}
    for q in queries:
        values = {}
        for k in ks:
            for ours, key in KEYS.items():
                values[f"{ours}@{k}"] = float(scored[q["id"]][key.format(k=k)])
            values[f"mrr@{k}"] = float(rr[k][q["id"]])
        rows[q["id"]] = values
    OUT.write_text(
        json.dumps(
            {
                "generator": "tests/fixtures/trec_eval/generate/make_fixtures.py",
                "pytrec_eval_version": version("pytrec_eval"),
                "source": "tests/fixtures/ranx/metrics.json",
                "source_alignment_sha256": alignment(queries),
                "relevance_level": 1,
                "ks": ks,
                "queries": rows,
            },
            indent=1,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(rows)} queries, pytrec_eval {version('pytrec_eval')})")


if __name__ == "__main__":
    main()
