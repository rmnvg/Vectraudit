"""Versioned, allowlisted hosted run projections; raw artifacts are never mutated.

Profile 1 shares release measurements, IDs, tags, run labels, search parameters and pipeline
identity (adapter kind, name, configuration hash and versions), so a hosted report can say
what changed. Numeric and boolean parameters are shared as values; text parameters (which
may hold prompts or names) only as SHA-256 fingerprints. Other adapter configuration
(endpoint hosts, module paths), raw evidence, request context, vectors and arbitrary
extension fields stay local. Query text/history have a separate explicit opt-in; they are
never needed for release verdicts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from vectraudit.contracts import HISTORY
from vectraudit.evaluation.runfile import check_run
from vectraudit.evaluation.runner import summarize

MAX_BYTES = 64 * 1024 * 1024
PROFILE = "release-metadata-v1"


def _json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _scalars(value: Any, keys: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in keys.split()
        if key in value and (value[key] is None or type(value[key]) in (str, bool, int, float))
    }


FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{16}$")


def _param(value: str | int | float | bool, projected: bool) -> str | int | float | bool:
    if not isinstance(value, str):
        return value
    if projected:
        # Re-validating a projection: text must already be a fingerprint, never raw text.
        if not FINGERPRINT.fullmatch(value):
            raise ValueError("Projected text parameters must be SHA-256 fingerprints")
        return value
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _error(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    # Imported lazily: importing the pipelines package while loading the runner cycles.
    from vectraudit.pipelines.errors import CALL_FAILURES

    code = value.get("code") if isinstance(value, dict) else None
    known = {
        *CALL_FAILURES,
        "config",
        "dependency",
        "connection",
        "auth",
        "unsupported",
        "query",
        "internal",
        "pipeline_error",
    }
    return {
        "code": code if isinstance(code, str) and code in known else "pipeline_error",
        "message": "Error details retained locally.",
        "hint": None,
    }


def _query(
    query: dict[str, Any], share_query_text: bool, *, attempt: bool = False
) -> dict[str, Any]:
    out = _scalars(
        query,
        "query_id family_id filtered latency_ms ann_recall nondeterministic "
        "no_answer_correct permission_checked abstained cost_usd memory_bytes "
        "attempt_count elapsed_ms",
    )
    for key in ("tags", "ids", "exact_ids", "forbidden_ids", "latency_samples_ms"):
        if key in query:
            value = query[key]
            if value is None or (
                isinstance(value, list) and all(type(v) in (str, int, float, bool) for v in value)
            ):
                out[key] = value
    if "repeat_ids" in query:
        out["repeat_ids"] = query["repeat_ids"]  # validated by check_run, including attempts
    if "repeat_metrics" in query:
        out["repeat_metrics"] = [
            _scalars(sample, "recall precision mrr ndcg") if sample is not None else None
            for sample in query["repeat_metrics"]
        ]
    if "repeat_ann_recall" in query:
        out["repeat_ann_recall"] = query["repeat_ann_recall"]
    out["error"] = _error(query.get("error"))
    out.setdefault("ann_recall", None)
    out.setdefault("nondeterministic", False)
    out["metrics"] = (
        _scalars(query["metrics"], "recall precision mrr ndcg")
        if query.get("metrics") is not None
        else None
    )
    out["evidence"] = None
    if "repeat_safety" in query:
        out["repeat_safety"] = [
            {
                "forbidden_ids": [v for v in row.get("forbidden_ids", []) if isinstance(v, str)],
                **_scalars(row, "no_answer_correct"),
            }
            for row in query["repeat_safety"]
            if isinstance(row, dict)
        ]
    if "retries" in query:
        out["retries"] = [
            {**_scalars(row, "call attempt delay_s"), "code": (_error(row) or {})["code"]}
            for row in query["retries"]
            if isinstance(row, dict)
        ]
    if not attempt and "attempts" in query:
        out["attempts"] = [
            {
                **_scalars(row, "attempt status started_at finished_at"),
                "result": _query(row["result"], False, attempt=True) if row.get("result") else None,
            }
            for row in query["attempts"]
        ]
    if share_query_text:
        if "text" in query:
            text = query["text"]
            if not isinstance(text, str) or len(text.encode()) > 65536:
                raise ValueError("Shared query text must be a string of at most 64 KiB")
            out["text"] = text
        if "history" in query:
            out["history"] = [
                turn.model_dump() for turn in HISTORY.validate_python(query["history"])
            ]
    return out


def _payload(
    record: dict[str, Any], share_query_text: bool, *, projected: bool = False
) -> dict[str, Any]:
    out = _scalars(record, "format format_version run_id created_at label status")
    out["error"] = _error(record.get("error"))
    out["dataset"] = _scalars(
        record["dataset"], "sha256 n_queries n_families n_labelled n_filtered dim"
    )
    out["backend"] = (
        _scalars(
            record["backend"],
            "backend name distance dim count snapshot corpus_version "
            "backend_version client_version",
        )
        if record.get("backend") is not None
        else None
    )
    if out["backend"] is not None and isinstance(record["backend"].get("config"), dict):
        # Identity and versions only; hosts, module paths and other settings stay local.
        out["backend"]["config"] = _scalars(
            record["backend"]["config"], "adapter name sha256 pipeline_version corpus_sha256"
        )
    out["connection"] = {}
    out["capabilities"] = _scalars(record.get("capabilities"), "exact_reference filters search")
    search = record["search"]
    out["search"] = {
        **_scalars(search, "k exact_reference evidence embedder"),
        # Tuning parameters are what a release report compares. Text values can hold prompts,
        # so only their fingerprint leaves: a report can still show that one changed.
        "params": {
            key: _param(value, projected)
            for key, value in (search.get("params") or {}).items()
            if isinstance(key, str) and type(value) in (str, int, float, bool)
        },
    }
    out["timing"] = _scalars(
        record.get("timing"),
        "warmup repeats concurrency max_rps overlapped_runs wall_s embedding_s",
    )
    out["metric_contract"] = _scalars(record["metric_contract"], "version k repeat_aggregation")
    provenance = record.get("provenance") or {}
    out["provenance"] = {
        **_scalars(provenance, "commit_sha"),
        "runner": _scalars(provenance.get("runner"), "system machine cpu_count python"),
    }
    out["queries"] = [_query(q, share_query_text) for q in record["queries"]]
    out["summary"] = summarize(out["queries"], out["dataset"]["n_queries"])
    return check_run(out)


def project_run(record: Mapping[str, Any], *, share_query_text: bool = False) -> dict[str, Any]:
    """Produce/validate a hosted projection, with distinct source and payload digests.

    Existing projections are validated and kept unchanged. Changing sharing settings for
    the same uploaded run ID conflicts on the server rather than rewriting its history.
    """
    raw = _json(record)
    if len(raw) > MAX_BYTES:
        raise ValueError("Run exceeds the 64 MiB sharing limit")
    copied = check_run(json.loads(raw))
    sharing = copied.get("sharing")
    if sharing is not None:
        if not isinstance(sharing, dict) or sharing.get("profile") != PROFILE:
            raise ValueError("Unsupported sharing profile")
        enabled = sharing.get("share_query_text")
        if type(enabled) is not bool:
            raise ValueError("Invalid sharing settings")
        payload = _payload(copied, enabled, projected=True)
        if sharing.get("payload_sha256") != _digest(payload):
            raise ValueError("Sharing projection digest mismatch")
        source = sharing.get("source_sha256")
        if (
            not isinstance(source, str)
            or len(source) != 64
            or any(c not in "0123456789abcdef" for c in source)
        ):
            raise ValueError("Invalid source artifact digest")
        if set(sharing) != {"profile", "share_query_text", "source_sha256", "payload_sha256"}:
            raise ValueError("Unknown sharing metadata")
        if {**payload, "sharing": sharing} != copied:
            raise ValueError("Projection contains fields outside its sharing profile")
        return copied
    out = _payload(copied, share_query_text)
    out["sharing"] = {
        "profile": PROFILE,
        "share_query_text": share_query_text,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": _digest(out),
    }
    if len(_json(out)) > MAX_BYTES:
        raise ValueError("Projection exceeds the 64 MiB sharing limit")
    return out


def transfer_preview(
    record: Mapping[str, Any], *, share_query_text: bool = False
) -> dict[str, Any]:
    """Describe the exact projection without exposing its content or making a network call."""
    projected = project_run(record, share_query_text=share_query_text)
    return {
        **projected["sharing"],
        "run_id": projected["run_id"],
        "bytes": len(_json(projected)),
        "queries": len(projected["queries"]),
        "shares": [
            "query/document/family IDs",
            "tags",
            "run labels",
            "search parameters (text values as SHA-256 fingerprints)",
            "pipeline identity (adapter kind, name, configuration hash, versions)",
            "metrics",
            "timings",
            "safety assertions",
            "retry/attempt rankings",
            "corpus identity",
            "runner identity",
        ],
        "query_text_and_history": projected["sharing"]["share_query_text"],
        "excluded": [
            "request context/filters",
            "vectors",
            "other adapter configuration (endpoint hosts, module paths)",
            "raw evidence",
            "error messages",
            "arbitrary extensions",
            "dataset descriptions/provenance",
        ],
        "notice": "IDs, tags and labels can themselves contain sensitive information.",
    }
