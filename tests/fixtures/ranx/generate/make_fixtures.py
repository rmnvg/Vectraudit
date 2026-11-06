"""Generate ranx reference outputs that Vectraudit's metrics are tested against.

Runs only in a throwaway environment with ``ranx==0.3.21`` installed (see ``../README.md``);
ranx is not a Vectraudit dependency. Everything is seeded, so rerunning reproduces the
committed JSON exactly.
"""

from __future__ import annotations

import json
import random
import sys
from importlib.metadata import version
from pathlib import Path

from ranx import Qrels, Run, evaluate

RANX_COMMIT = "7363db0c35e9"
SEED = 0
N_QUERIES = 200
KS = (1, 5, 10, 100)
METRICS = ("recall", "precision", "mrr", "ndcg", "map", "hit_rate")


def synthetic_queries(rng: random.Random) -> list[dict]:
    """Graded labels 0-3, tied scores, short result lists, single-relevant queries."""
    queries = []
    for n in range(N_QUERIES):
        pool = [f"d{n}-{i}" for i in range(rng.randint(5, 150))]
        judged = rng.sample(pool, rng.randint(1, min(12, len(pool))))
        if n % 5 == 0:
            labels = {judged[0]: rng.randint(1, 3)} | {d: 0 for d in judged[1:]}
        else:
            labels = {d: rng.randint(0, 3) for d in judged}
            if not any(labels.values()):
                labels[judged[0]] = rng.randint(1, 3)
        # Short lists (fewer than k) are common; the run is drawn from the whole pool, so
        # unjudged documents appear too.
        retrieved = rng.sample(pool, rng.randint(1, len(pool)))
        coarse = n % 3 == 0  # coarse scores create ties
        scores = {
            d: round(rng.random(), 1) if coarse else round(rng.random(), 9) for d in retrieved
        }
        queries.append({"id": f"q{n}", "labels": labels, "scores": scores})
    return queries


def resolve_ties(scores: dict[str, float]) -> list[str]:
    """Order by (-score, id). ranx breaks ties with an unstable argsort, so its tie order is
    arbitrary; Vectraudit keeps the backend's order. Fixing the order here and handing ranx
    strictly decreasing scores means both sides score the same ranking."""
    return sorted(scores, key=lambda d: (-scores[d], d))


def metric_fixture(queries: list[dict]) -> dict:
    rankings = {q["id"]: resolve_ties(q["scores"]) for q in queries}
    qrels = Qrels({q["id"]: q["labels"] for q in queries})
    run = Run({qid: {d: float(len(r) - i) for i, d in enumerate(r)} for qid, r in rankings.items()})
    names = [f"{m}@{k}" for m in METRICS for k in KS]
    evaluate(qrels, run, names, return_mean=False, make_comparable=False)
    rows = []
    for q in queries:
        rows.append(
            {
                "id": q["id"],
                "labels": q["labels"],
                "ranking": rankings[q["id"]],
                "tied_scores": len(set(q["scores"].values())) < len(q["scores"]),
                "ranx": {name: float(run.scores[name][q["id"]]) for name in names},
            }
        )
    return {
        "generator": "tests/fixtures/ranx/generate/make_fixtures.py",
        "ranx_version": version("ranx"),
        "ranx_commit": RANX_COMMIT,
        "seed": SEED,
        "ks": list(KS),
        "metrics": list(METRICS),
        "queries": rows,
    }


RANX_METHOD = {
    "rrf": "rrf",
    "isr": "isr",
    "bordafuse": "bordafuse",
    "wsum": "wsum",
    "combsum": "sum",
    "combmnz": "mnz",
    "combmax": "max",
}
RANX_NORM = {
    "none": None,
    "min_max": "min-max",
    "max": "max",
    "sum": "sum",
    "zmuv": "zmuv",
    "rank": "rank",
    "borda": "borda",
}
STAGES = ("keyword", "vector", "sparse")


def fusion_queries(rng: random.Random, n: int = 40) -> list[dict]:
    """Two or three overlapping stages per query with distinct scores (ranx order is then
    unambiguous); some scores are negative, and some stages return a single hit."""
    out = []
    for q in range(n):
        pool = [f"d{q}-{i}" for i in range(rng.randint(2, 25))]
        stages = {}
        for name in STAGES[: rng.choice((2, 3))]:
            docs = rng.sample(pool, 1 if q % 7 == 0 else rng.randint(1, len(pool)))
            offset = -5.0 if q % 4 == 0 else 0.0
            scores = sorted(rng.sample(range(1, 10**6), len(docs)), reverse=True)
            stages[name] = [[d, s / 1e5 + offset] for d, s in zip(docs, scores, strict=True)]
        out.append({"id": f"q{q}", "stages": stages})
    return out


def fusion_cases() -> list[dict]:
    cases = [{"method": m, "norm": "none"} for m in ("isr", "bordafuse")]
    cases += [{"method": "rrf", "norm": "none", "rrf_k": k} for k in (20, 60)]
    for method in ("wsum", "combsum", "combmnz", "combmax"):
        cases += [{"method": method, "norm": norm} for norm in RANX_NORM]
    return cases


def fusion_fixture(queries: list[dict]) -> dict:
    from ranx import fuse

    weights = {"keyword": 0.3, "vector": 0.7, "sparse": 0.5}
    results = []
    for case in fusion_cases():
        by_query: dict[str, dict[str, float]] = {}
        # ranx fuses whole runs with the same stages, so group queries by stage count.
        for count in (2, 3):
            group = [q for q in queries if len(q["stages"]) == count]
            names = STAGES[:count]
            runs = [
                Run({q["id"]: {d: s for d, s in q["stages"][name]} for q in group})
                for name in names
            ]
            params = {}
            if case["method"] == "rrf":
                params = {"k": case["rrf_k"]}
            elif case["method"] == "wsum":
                params = {"weights": [weights[name] for name in names]}
            fused = fuse(
                runs,
                norm=RANX_NORM[case["norm"]],
                method=RANX_METHOD[case["method"]],
                params=params,
            )
            for q in group:
                by_query[q["id"]] = {d: float(s) for d, s in fused[q["id"]].items()}
        results.append({**case, "weighted": case["method"] == "wsum", "fused": by_query})
    return {
        "generator": "tests/fixtures/ranx/generate/make_fixtures.py",
        "ranx_version": version("ranx"),
        "ranx_commit": RANX_COMMIT,
        "seed": SEED,
        "weights": weights,
        "queries": queries,
        "cases": results,
    }


def main(out_dir: str) -> None:
    out = Path(out_dir)
    queries = synthetic_queries(random.Random(SEED))
    (out / "metrics.json").write_text(json.dumps(metric_fixture(queries), indent=1) + "\n")
    fusion = fusion_fixture(fusion_queries(random.Random(SEED + 1)))
    (out / "fusion.json").write_text(json.dumps(fusion, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent))
