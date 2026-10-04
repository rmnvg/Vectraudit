"""Plain-language evidence summaries."""

from __future__ import annotations

from vectraudit.evaluation.evidence import describe_evidence, plan_uses_index


def test_faiss_ivf_summaries() -> None:
    ev = {
        "kind": "faiss_ivf",
        "nprobe": 2,
        "min_nprobe_for_all": 9,
        "missed_by_reason": {"CELL_NOT_PROBED": 3, "QUANTIZATION": 1},
    }
    assert describe_evidence(ev) == (
        "nprobe 2: 3 list not probed, 1 ranked out by quantization; all found from nprobe 9"
    )
    ev = {"kind": "faiss_ivf", "nprobe": 8, "min_nprobe_for_all": 3, "missed_by_reason": {}}
    assert describe_evidence(ev) == "nprobe 8: no exact neighbour missed"


def test_plan_summaries() -> None:
    indexed = {"kind": "pg_plan", "nodes": ["Limit", "Index Scan"], "indexes": ["ix"]}
    seq = {"kind": "pg_plan", "nodes": ["Limit", "Sort", "Seq Scan"], "indexes": []}
    other = {"kind": "pg_plan", "nodes": ["Limit", "Bitmap Heap Scan"], "indexes": []}
    assert describe_evidence(indexed) == "plan uses index ix"
    assert describe_evidence(seq) == "plan scans the table sequentially (no vector index)"
    assert describe_evidence(other) == "plan: Limit → Bitmap Heap Scan"
    assert plan_uses_index(indexed) is True
    assert plan_uses_index(seq) is False
    assert plan_uses_index({"kind": "faiss_ivf"}) is None
    assert plan_uses_index(None) is None


def test_other_evidence() -> None:
    assert describe_evidence(None) is None
    assert describe_evidence({"kind": "mystery"}) is None
    unavailable = {"kind": "unavailable", "error": {"message": "no plan"}}
    assert describe_evidence(unavailable) == "evidence unavailable: no plan"
