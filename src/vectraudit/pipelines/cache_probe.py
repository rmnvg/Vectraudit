"""Cache isolation between principals: does one user's question warm another's answer?

A cache keyed on the question text alone (results, embeddings with filtered candidates,
reranked lists, packed prompts) serves one user's documents to the next user who asks the
same thing. A normal evaluation sees this only by accident of order: concurrent runs
reorder questions, and a restricted question asked before its peers caches its own clean
answer.

:func:`cache_probe` uses questions the benchmark already has: the same question (same
``family_id``, otherwise the same text) asked under different contexts, at least one of
them with ``forbidden`` documents. Permission scenarios produce exactly these. For each
such *target* and each *primer* with another context it calls the primer, warming any
cache, then the target. A forbidden document the target received is a leak.

The primer must go first: asking the target first would cache its own, clean answer under
a text-only key, and the defect would hide. For the same reason a normally recorded run
is no clean reference: in benchmark order the cache leaks there too. Each leak instead
records what the probe itself observed: ``same_as_primer`` when the target received
exactly the primer's ranking (what a shared cache entry produces, though an access check
that ignores this context looks the same), or ``differs_from_primer`` when it did not, so
the access check itself let the document through. Every leak fails. Expected access is the
benchmark's ``forbidden`` list, never the application's own decision.

The probe calls the pipeline serially, so order is real. It cannot clear the application's
caches: an answer cached before the probe can mask a leak, so probe a fresh deployment or
one whose cache has expired. A clean probe is evidence about these questions and orderings,
not proof of isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.models import PipelineRequest, PipelineResult
from vectraudit.pipelines.runner import forbidden_delivered, permission_path
from vectraudit.pipelines.runtime import Pipeline

MAX_PAIRS = 500


def _request(q: DatasetQuery, k: int, params: Mapping[str, Any]) -> PipelineRequest:
    return PipelineRequest(
        query_id=q.query_id,
        text=q.text,
        vector=None if q.vector is None else q.vector.tolist(),
        k=k,
        filters=q.filter,
        context=q.context,
        history=list(q.history),
        params=dict(params),
    )


def probe_pairs(dataset: Dataset) -> list[tuple[DatasetQuery, DatasetQuery]]:
    """``(target, primer)`` pairs: the same question under another context."""
    groups: dict[str, list[DatasetQuery]] = {}
    for q in dataset.queries:
        key = q.family_id or (f"text:{q.text}" if q.text else None)
        if key is not None:
            groups.setdefault(key, []).append(q)
    pairs = []
    for members in groups.values():
        for target in members:
            if not target.forbidden:
                continue
            pairs += [(target, p) for p in members if p.context != target.context]
    return pairs


def _leaks(result: PipelineResult, forbidden: Sequence[str]) -> dict[str, list[str]]:
    """Each forbidden document the result exposed, and where."""
    out: dict[str, list[str]] = {}
    for doc in permission_path([result], forbidden)["documents"]:
        where = [f"stage {s['name']} ({s['kind']})" for s in doc["stages"]]
        where += ["results"] if doc["results"] else []
        where += ["model context"] if doc["context"] else []
        out[doc["doc_id"]] = where
    for doc in forbidden_delivered(result, forbidden):
        out.setdefault(doc, ["results"])
    return out


def cache_probe(
    dataset: Dataset,
    pipeline: Pipeline,
    *,
    k: int = 10,
    params: Mapping[str, Any] | None = None,
    max_pairs: int = MAX_PAIRS,
) -> dict[str, Any]:
    """Probe every target/primer pair in order (at most ``max_pairs``; 2 calls each)."""
    pairs = probe_pairs(dataset)
    if len(pairs) > max_pairs:
        raise ValueError(
            f"{len(pairs)} cache probe pairs exceed the limit of {max_pairs} "
            f"({2 * len(pairs)} calls); narrow the benchmark or raise max_pairs"
        )
    settings = dict(params or {})
    rows: list[dict[str, Any]] = []
    leaks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target, primer in pairs:
        row: dict[str, Any] = {"target": target.query_id, "primer": primer.query_id}
        try:
            warmed = pipeline(_request(primer, k, settings))
            primed = pipeline(_request(target, k, settings))
        except PipelineCallError as exc:
            row["error"] = {"code": exc.code, "message": str(exc)}
            errors.append(row)
            rows.append(row)
            continue
        found = _leaks(PipelineResult.model_validate(primed.model_dump()), target.forbidden)
        row["leaked"] = found
        same = [h.id for h in primed.hits] == [h.id for h in warmed.hits]
        for doc, where in found.items():
            leaks.append(
                {
                    "target": target.query_id,
                    "primer": primer.query_id,
                    "doc_id": doc,
                    "reached": where,
                    "observed": "same_as_primer" if same else "differs_from_primer",
                }
            )
        rows.append(row)
    return {
        "format": "vectraudit.cache_probe",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": dataset.sha256,
        "pairs": len(pairs),
        "calls": 2 * len(pairs),
        "complete": not errors,
        "leaks": leaks,
        "errors": errors,
        "rows": rows,
        "failing": len(leaks),
    }


def render(report: Mapping[str, Any]) -> str:
    """The probe as plain text for the terminal."""
    lines = [
        f"{report['pairs']} target/primer pair(s), {report['calls']} calls: "
        f"{len(report['leaks'])} leak(s)"
    ]
    for hit in report["leaks"][:50]:
        why = (
            "identical to the primer's answer: a shared cache entry, or an access check "
            "that ignores this context"
            if hit["observed"] == "same_as_primer"
            else "not the primer's answer: the access check let it through"
        )
        lines.append(
            f"  {hit['target']} after {hit['primer']}: {hit['doc_id']} reached "
            + ", ".join(hit["reached"])
            + f" ({why})"
        )
    for row in report["errors"][:20]:
        lines.append(f"  {row['target']} after {row['primer']}: {row['error']['code']}")
    if not report["pairs"]:
        lines.append(
            "No question is asked under two contexts with forbidden documents; add permission "
            "scenarios to probe cache isolation."
        )
    elif not report["complete"]:
        lines.append("The probe is incomplete: failed calls prove nothing about isolation.")
    elif not report["failing"]:
        lines.append("No forbidden document crossed between contexts in these orderings.")
    return "\n".join(lines)
