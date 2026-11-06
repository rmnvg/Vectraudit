"""The hybrid baseline's stage outputs must not change when its internals are refactored.

``fixtures/hybrid_golden.json`` was captured from the baseline before its fusion stage moved
to ``vectraudit.core.fusion`` (commit c604521). It holds the SHA-256 of every stage's hits,
order and scores for every question and parameter set (a byte-for-byte check), and the
final hits in readable form so a failure shows where the output moved. Two scores vary in
their last bits outside Vectraudit's control: vector scores (a BLAS matrix product, which
differs between platforms) are recorded by order only, and BM25 scores (``sum()`` of
floats, compensated since Python 3.12) to 12 significant digits. Fusion uses ranks, so it
and everything downstream are exact. Regenerate
only for an intended behaviour change, with ``uv run python tests/test_hybrid_golden.py``,
and say why in the commit.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.models import PipelineRequest

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "fixtures/hybrid_golden.json"
WORKBENCH = ROOT.parent / "examples/workbench"
WORDS = [
    "invoice",
    "export",
    "billing",
    "reset",
    "password",
    "account",
    "tenant",
    "refund",
    "policy",
    "audit",
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _synthetic() -> tuple[list[dict], list[dict]]:
    rng = random.Random(7)
    docs = [
        {
            "doc_id": f"doc-{i}",
            "text": " ".join(rng.choices(WORDS, k=rng.randint(3, 12))),
            "metadata": {"tenant": rng.choice(["a", "b"])},
            "vector": [rng.uniform(-1, 1) for _ in range(4)],
        }
        for i in range(80)
    ]
    queries = [
        {
            "query_id": f"q{i}",
            "text": " ".join(rng.sample(WORDS, rng.randint(1, 3))),
            "vector": [rng.uniform(-1, 1) for _ in range(4)],
            "context": {"tenant": "a"} if i % 2 else {},
        }
        for i in range(20)
    ]
    return docs, queries


PARAMS = [
    {"vector_weight": w, "reranker": r, "depth": d}
    for w in (0, 0.3, 0.5, 1)
    for r in ("none", "phrase")
    for d in (10, 50)
]


def _portable(kind: str, score: float | None) -> float | None:
    if kind == "vector" or score is None:
        return None
    return float(f"{score:.12g}") if kind == "keyword" else score


def stage_outputs() -> str:
    corpora = {
        "workbench": (_jsonl(WORKBENCH / "corpus.jsonl"), _jsonl(WORKBENCH / "queries.jsonl")),
        "synthetic": _synthetic(),
    }
    out: dict[str, list] = {}
    for name, (docs, queries) in corpora.items():
        pipeline = hybrid_pipeline(docs)
        has_vectors = all("vector" in d for d in docs) and all("vector" in q for q in queries)
        rows = []
        # Not every examples/workbench document or question has a vector: keyword-only there.
        for params in [p for p in PARAMS if has_vectors or p["vector_weight"] == 0]:
            for q in queries:
                request = PipelineRequest(
                    query_id=q["query_id"],
                    text=q.get("text"),
                    vector=q.get("vector"),
                    k=5,
                    filters=q.get("filter", {}),
                    context=q.get("context", {}),
                    params=params,
                )
                result = pipeline(request)
                rows.append(
                    {
                        "query_id": q["query_id"],
                        "params": params,
                        "stages": {
                            s.name: [[h.id, _portable(s.kind, h.score)] for h in s.hits]
                            for s in result.stages
                        },
                    }
                )
        out[name] = rows
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def golden() -> dict:
    outputs = stage_outputs()
    final = {
        name: [[r["query_id"], r["params"], r["stages"]["final"]] for r in rows]
        for name, rows in json.loads(outputs).items()
    }
    return {"sha256": hashlib.sha256(outputs.encode()).hexdigest(), "final": final}


def test_hybrid_baseline_matches_the_golden_output():
    expected = json.loads(GOLDEN.read_text())
    ours = golden()
    assert ours["final"] == expected["final"]
    assert ours["sha256"] == expected["sha256"]


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(golden(), indent=1) + "\n")
