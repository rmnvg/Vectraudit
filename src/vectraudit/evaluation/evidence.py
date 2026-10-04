"""Read per-query backend evidence (see ``Connector.evidence``) for reports and diagnoses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_REASONS = {
    "CELL_NOT_PROBED": "list not probed",
    "QUANTIZATION": "ranked out by quantization",
    "TRANSFORM": "ranked out by the PreTransform",
    "RANKED_OUT": "ranked out",
}
SEQUENTIAL_NODES = frozenset({"Seq Scan", "Parallel Seq Scan"})


def describe_evidence(evidence: Mapping[str, Any] | None) -> str | None:
    """One line about what the backend did for this query, or ``None`` if nothing to say."""
    if not evidence:
        return None
    kind = evidence.get("kind")
    if kind == "faiss_ivf":
        missed = evidence.get("missed_by_reason") or {}
        if not missed:
            return f"nprobe {evidence['nprobe']}: no exact neighbour missed"
        parts = ", ".join(f"{n} {_REASONS.get(r, r.lower())}" for r, n in sorted(missed.items()))
        hint = ""
        if "CELL_NOT_PROBED" in missed:
            hint = f"; all found from nprobe {evidence['min_nprobe_for_all']}"
        return f"nprobe {evidence['nprobe']}: {parts}{hint}"
    if kind == "pg_plan":
        indexes = evidence.get("indexes") or []
        if indexes:
            return f"plan uses index {', '.join(indexes)}"
        if SEQUENTIAL_NODES & set(evidence.get("nodes") or []):
            return "plan scans the table sequentially (no vector index)"
        return "plan: " + " → ".join(evidence.get("nodes") or [])
    if kind == "unavailable":
        error = evidence.get("error") or {}
        return f"evidence unavailable: {error.get('message')}"
    return None


def plan_uses_index(evidence: Mapping[str, Any] | None) -> bool | None:
    """For pgvector plans: True / False, or ``None`` when the evidence is not a plan."""
    if not evidence or evidence.get("kind") != "pg_plan":
        return None
    return bool(evidence.get("indexes"))
