"""Migration assurance: compare two user-configured systems on one labelled suite.

A migration (pgvector to Qdrant, a new embedding model, a forced provider change) is judged
by the same release policy as any other change, plus checks a migration needs because the
two systems are built separately:

- **Corpus equivalence.** Each system's *inventory* (one document per line:
  ``{"id": ..., "sha256": ..., "metadata": {...}}``, exported by the user; no text)
  should hold the same documents with the same content. Labelled documents missing from
  the proposed system explain quality drops that no retrieval change can recover.
- **ID mapping.** Results are compared by ID, so both systems must return the benchmark's
  IDs. A system returning IDs absent from its inventory, or mostly IDs no label or the
  other system knows, points at a mapping error rather than worse retrieval.
- **Filter semantics.** For questions with filters, every returned document's inventory
  metadata must satisfy the filter under the portable semantics (equality, list
  membership, numeric ranges). A backend that ignores or reinterprets a filter shows up
  here, not just as a metric change.
- **Environment.** Hardware, concurrency, pairing, backend configuration and latency source
  are listed side by side: latency is only comparable where they match.
- **Cost.** Measured spend from the runs is kept apart from costs the team declares (for
  example re-indexing or hosting), which are labelled as estimates, never measured.

Nothing here copies, re-indexes or reads production data; the runs and inventories are the
user's own exports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vectraudit.contracts import matches_filter
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.policy import Policy
from vectraudit.evaluation.report import render_markdown

UNKNOWN_IDS_WARNING = 0.8
"""Share of returned IDs unknown to labels and the other system above which a mapping error
is suspected. Deliberately high: a different embedding model legitimately returns many
documents the other system never did, while a mapping error shares almost none. An
inventory gives the direct check (IDs the system returned but does not hold)."""


class MigrationError(ValueError):
    """The migration inputs cannot be compared."""


def read_inventory(path: str | Path) -> dict[str, dict[str, Any]]:
    """A system's document inventory: ID to optional ``sha256`` and ``metadata``."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MigrationError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    docs: dict[str, dict[str, Any]] = {}
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"{path} line {line}: not JSON ({exc.msg})") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise MigrationError(f"{path} line {line}: each line needs a string id")
        if set(row) - {"id", "sha256", "metadata"}:
            extra = ", ".join(sorted(set(row) - {"id", "sha256", "metadata"}))
            raise MigrationError(f"{path} line {line}: unexpected field(s) {extra}; no text")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise MigrationError(f"{path} line {line}: metadata must be an object")
        if row["id"] in docs:
            raise MigrationError(f"{path} line {line}: {row['id']} appears twice")
        docs[row["id"]] = {k: v for k, v in row.items() if k != "id"}
    if not docs:
        raise MigrationError(f"{path} lists no documents")
    return docs


def _labelled(dataset: Dataset) -> set[str]:
    out: set[str] = set()
    for q in dataset.queries:
        out |= {d for d, g in q.relevant.items() if g > 0}
        out |= {s.doc_id for s in q.relevant_spans}
        out |= set(q.forbidden)
    return out


def _returned(run: Mapping[str, Any]) -> dict[str, list[str]]:
    return {q["query_id"]: list(q.get("ids") or []) for q in run["queries"] if not q.get("error")}


def corpus_equivalence(
    current: Mapping[str, Mapping[str, Any]],
    proposed: Mapping[str, Mapping[str, Any]],
    dataset: Dataset,
) -> dict[str, Any]:
    """Documents missing, added or changed between the two inventories."""
    shared = set(current) & set(proposed)
    changed = sorted(
        d
        for d in shared
        if current[d].get("sha256")
        and proposed[d].get("sha256")
        and current[d]["sha256"] != proposed[d]["sha256"]
    )
    unhashed = sum(
        1 for d in shared if not (current[d].get("sha256") and proposed[d].get("sha256"))
    )
    relevant = {d for q in dataset.queries for d, g in q.relevant.items() if g > 0}
    relevant |= {s.doc_id for q in dataset.queries for s in q.relevant_spans}
    missing = sorted(set(current) - set(proposed))
    return {
        "current_documents": len(current),
        "proposed_documents": len(proposed),
        "missing_in_proposed": missing[:200],
        "n_missing_in_proposed": len(missing),
        "added_in_proposed": sorted(set(proposed) - set(current))[:200],
        "n_added_in_proposed": len(set(proposed) - set(current)),
        "content_changed": changed[:200],
        "n_content_changed": len(changed),
        "content_unverified": unhashed,
        "labelled_missing_in_proposed": sorted(relevant - set(proposed)),
        "equivalent": not missing
        and not (set(proposed) - set(current))
        and not changed
        and not unhashed,
    }


def id_mapping(
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
    dataset: Dataset,
    inventories: Mapping[str, Mapping[str, Mapping[str, Any]] | None],
) -> dict[str, Any]:
    """Whether each system returns IDs the benchmark and its own inventory know."""
    labelled = _labelled(dataset)
    runs = {"current": _returned(current), "proposed": _returned(proposed)}
    seen = {side: {d for ids in r.values() for d in ids} for side, r in runs.items()}
    out: dict[str, Any] = {}
    for side, other in (("current", "proposed"), ("proposed", "current")):
        returned = [d for ids in runs[side].values() for d in ids]
        unknown = [d for d in returned if d not in labelled and d not in seen[other]]
        inventory = inventories.get(side)
        outside = sorted({d for d in returned if inventory is not None and d not in inventory})
        rate = len(unknown) / len(returned) if returned else 0.0
        out[side] = {
            "returned": len(returned),
            "unknown_to_labels_and_other_system": len(unknown),
            "unknown_rate": round(rate, 4),
            "not_in_inventory": outside[:200],
            "n_not_in_inventory": len(outside),
            "suspect": bool(outside) or rate >= UNKNOWN_IDS_WARNING,
        }
    return out


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    value: Any = metadata
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def filter_conformance(
    run: Mapping[str, Any],
    dataset: Dataset,
    inventory: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Returned documents whose metadata does not satisfy the question's filter."""
    filtered = [q for q in dataset.queries if q.filter]
    returned = _returned(run)
    if inventory is None:
        return {"checked": False, "filtered_questions": len(filtered)}
    violations, unverifiable, empty = [], 0, 0
    for q in filtered:
        ids = returned.get(q.query_id)
        if ids is None:
            continue
        if not ids:
            empty += 1
        for doc in ids:
            meta = (inventory.get(doc) or {}).get("metadata")
            if meta is None:
                unverifiable += 1
                continue
            failed = [
                key
                for key, condition in q.filter.items()
                if not matches_filter(_metadata_value(meta, key), condition)
            ]
            if failed:
                violations.append({"query_id": q.query_id, "doc_id": doc, "fields": failed})
    return {
        "checked": True,
        "filtered_questions": len(filtered),
        "empty_results": empty,
        "violations": violations[:200],
        "n_violations": len(violations),
        "unverifiable_results": unverifiable,
    }


def _runner(run: Mapping[str, Any]) -> str | None:
    env = (run.get("provenance") or {}).get("runner")
    if not isinstance(env, Mapping):
        return None
    return (
        f"{env.get('system')}/{env.get('machine')}, {env.get('cpu_count')} CPUs, "
        f"Python {env.get('python')}, vectraudit {env.get('vectraudit')}"
        + (", CI" if env.get("ci") else "")
    )


_ENV_FIELDS = (
    ("runner", _runner),
    ("backend", lambda r: {k: (r.get("backend") or {}).get(k) for k in ("name", "backend")}),
    ("configuration", lambda r: (r.get("backend") or {}).get("config")),
    ("corpus version", lambda r: (r.get("backend") or {}).get("corpus_version")),
    ("k", lambda r: (r.get("search") or {}).get("k")),
    ("embedder", lambda r: (r.get("search") or {}).get("embedder")),
    ("concurrency", lambda r: (r.get("timing") or {}).get("concurrency", 1)),
    ("repeats", lambda r: (r.get("timing") or {}).get("repeats")),
    ("latency source", lambda r: (r.get("timing") or {}).get("source", "measured")),
    ("order", lambda r: (r.get("timing") or {}).get("order", "separate")),
)


def environment(current: Mapping[str, Any], proposed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each environment field of both runs, marked when they differ."""
    rows = []
    for name, get in _ENV_FIELDS:
        a, b = get(current), get(proposed)
        rows.append({"field": name, "current": a, "proposed": b, "differs": a != b})
    return rows


def costs(
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
    declared: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Measured spend from each run beside the team's declared estimates."""

    def measured(run: Mapping[str, Any]) -> dict[str, Any]:
        usage = run.get("usage") or {}
        calls = usage.get("calls") or 0
        unpriced = usage.get("unpriced_calls") or 0
        spend = usage.get("spend_usd")
        return {
            "calls": calls,
            "spend_usd": spend if calls and not unpriced else None,
            "per_call_usd": spend / calls if calls and not unpriced and spend is not None else None,
            "unpriced_calls": unpriced,
        }

    declared = declared or {}
    for side, values in declared.items():
        if side not in ("current", "proposed") or not isinstance(values, Mapping):
            raise MigrationError("declared costs map current and proposed to named USD amounts")
        for key, value in values.items():
            if type(value) not in (int, float) or value < 0:
                raise MigrationError(f"declared cost {side}.{key} must be a non-negative number")
    return {
        "measured": {"current": measured(current), "proposed": measured(proposed)},
        "declared": {side: dict(declared.get(side) or {}) for side in ("current", "proposed")},
        "note": "Measured spend comes from adapter-reported call costs; declared amounts are "
        "the team's estimates and are not measured.",
    }


def migration_report(
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
    dataset: Dataset,
    policy: Policy,
    *,
    current_inventory: Mapping[str, Mapping[str, Any]] | None = None,
    proposed_inventory: Mapping[str, Mapping[str, Any]] | None = None,
    declared_costs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The migration decision under ``policy`` and the checks a migration needs."""
    for side, run in (("current", current), ("proposed", proposed)):
        if (run.get("dataset") or {}).get("sha256") != dataset.sha256:
            raise MigrationError(
                f"the {side} run was not recorded on this benchmark (dataset hash differs); "
                "both systems must answer the same labelled suite"
            )
    check = compare_runs(current, proposed, policy)
    inventories = {"current": current_inventory, "proposed": proposed_inventory}
    corpus = (
        corpus_equivalence(current_inventory, proposed_inventory, dataset)
        if current_inventory is not None and proposed_inventory is not None
        else None
    )
    ids = id_mapping(current, proposed, dataset, inventories)
    filters = {
        side: filter_conformance(run, dataset, inventories[side])
        for side, run in (("current", current), ("proposed", proposed))
    }
    concerns = []
    if corpus is None:
        concerns.append("Corpus equivalence is unverified: provide both inventories.")
    elif not corpus["equivalent"]:
        concerns.append(
            f"The corpora differ: {corpus['n_missing_in_proposed']} missing, "
            f"{corpus['n_added_in_proposed']} added, {corpus['n_content_changed']} changed, "
            f"{corpus['content_unverified']} without hashes to compare."
        )
    if corpus and corpus["labelled_missing_in_proposed"]:
        concerns.append(
            f"{len(corpus['labelled_missing_in_proposed'])} labelled document(s) are missing "
            "from the proposed system; quality lost on them is a corpus gap, not retrieval."
        )
    for side in ("current", "proposed"):
        if ids[side]["suspect"]:
            concerns.append(
                f"The {side} system returns IDs that look unmapped "
                f"({ids[side]['n_not_in_inventory']} not in its inventory; "
                f"{ids[side]['unknown_rate']:.0%} unknown to labels and the other system)."
            )
        if filters[side].get("n_violations"):
            concerns.append(
                f"The {side} system returned {filters[side]['n_violations']} document(s) that "
                "do not satisfy the question's filter."
            )
        if filters[side]["filtered_questions"] and not filters[side]["checked"]:
            concerns.append(
                f"Filter semantics on the {side} system are unverified: its inventory has no "
                "metadata to check against."
            )
    env = environment(current, proposed)
    differing = [row["field"] for row in env if row["differs"]]
    return {
        "format": "vectraudit.migration_report",
        "format_version": 1,
        "current": {"run_id": current.get("run_id"), "label": current.get("label")},
        "proposed": {"run_id": proposed.get("run_id"), "label": proposed.get("label")},
        "dataset_sha256": dataset.sha256,
        "verdict": check["verdict"],
        "check": check,
        "corpus": corpus,
        "id_mapping": ids,
        "filters": filters,
        "environment": env,
        "environment_differs": differing,
        "costs": costs(current, proposed, declared_costs),
        "concerns": concerns,
        "migration_ready": check["verdict"] == "pass" and not concerns,
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)[:80].replace("|", "\\|")
    return str(value).replace("|", "\\|")


def render_migration(report: Mapping[str, Any]) -> str:
    """The migration report as Markdown, for the decision record or a client deliverable."""
    cur, pro = report["current"], report["proposed"]
    ready = report["migration_ready"]
    lines = [
        f"# Migration report: {cur['label'] or cur['run_id']} → {pro['label'] or pro['run_id']}",
        "",
        f"**{'Ready to migrate under the policy' if ready else 'Not ready to migrate'}.** "
        f"Release policy verdict: {report['verdict']}; "
        f"{len(report['concerns'])} migration concern(s).",
        "",
    ]
    if report["concerns"]:
        lines += ["## Migration concerns", ""] + [f"- {c}" for c in report["concerns"]] + [""]
    lines += ["## Decision under the release policy", ""]
    lines += [
        line.replace("## ", "### ", 1) if line.startswith("## ") else line
        for line in render_markdown(report["check"]).splitlines()
    ]
    corpus = report["corpus"]
    lines += ["", "## Corpus equivalence", ""]
    if corpus is None:
        lines.append("Not checked: provide an inventory for each system.")
    else:
        lines += [
            "| | Documents |",
            "|---|---|",
            f"| Current | {corpus['current_documents']} |",
            f"| Proposed | {corpus['proposed_documents']} |",
            f"| Missing in proposed | {corpus['n_missing_in_proposed']} |",
            f"| Added in proposed | {corpus['n_added_in_proposed']} |",
            f"| Content changed | {corpus['n_content_changed']} |",
            f"| Without hashes to compare | {corpus['content_unverified']} |",
            f"| Labelled but missing in proposed | {len(corpus['labelled_missing_in_proposed'])} |",
        ]
    lines += ["", "## ID mapping", "", "| | Returned | Unknown rate | Not in inventory |"]
    lines.append("|---|---|---|---|")
    for side in ("current", "proposed"):
        row = report["id_mapping"][side]
        lines.append(
            f"| {side.title()} | {row['returned']} | {row['unknown_rate']:.0%} | "
            f"{row['n_not_in_inventory']} |"
        )
    lines += ["", "## Filter semantics", ""]
    for side in ("current", "proposed"):
        f = report["filters"][side]
        if not f["checked"]:
            lines.append(
                f"- {side.title()}: not checked ({f['filtered_questions']} filtered questions; "
                "no inventory metadata)."
            )
        else:
            lines.append(
                f"- {side.title()}: {f['n_violations']} filter violation(s) across "
                f"{f['filtered_questions']} filtered questions; {f['empty_results']} returned "
                f"nothing; {f['unverifiable_results']} result(s) without metadata."
            )
    lines += ["", "## Environment", "", "| Field | Current | Proposed |", "|---|---|---|"]
    for row in report["environment"]:
        mark = " **(differs)**" if row["differs"] else ""
        lines.append(
            f"| {row['field']}{mark} | {_cell(row['current'])} | {_cell(row['proposed'])} |"
        )
    lines += [
        "",
        "Latency is only comparable where the environment matches; differences are disclosed, "
        "not corrected for.",
        "",
        "## Cost",
        "",
    ]
    measured = report["costs"]["measured"]
    for side in ("current", "proposed"):
        m = measured[side]
        spend = "unknown (unpriced calls)" if m["spend_usd"] is None else f"${m['spend_usd']:.4f}"
        lines.append(f"- {side.title()}, measured: {m['calls']} calls, {spend}.")
    for side in ("current", "proposed"):
        for key, value in report["costs"]["declared"][side].items():
            lines.append(f"- {side.title()}, declared estimate: {key} ${value:,.2f}.")
    lines += ["", report["costs"]["note"], ""]
    return "\n".join(lines)
