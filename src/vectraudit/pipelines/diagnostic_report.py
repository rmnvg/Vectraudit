"""Bounded diagnostic reports with explicit evidence and safe, actionable failures."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from vectraudit.pipelines.errors import CALL_FAILURES
from vectraudit.pipelines.models import Contract, PipelineRequest, PipelineResult


class DiagnosticFinding(Contract):
    code: str
    status: Literal["ok", "warning", "error", "unknown", "skipped"]
    message: str
    hint: str = ""


class DiagnosticReport(Contract):
    format: Literal["vectraudit.connection_diagnostic"] = "vectraudit.connection_diagnostic"
    format_version: Literal[1] = 1
    executed: bool = False
    latency_ms: float | None = Field(default=None, ge=0)
    findings: list[DiagnosticFinding] = Field(default_factory=list, max_length=30)
    returned_ids: list[str] = Field(default_factory=list, max_length=100)
    missing_ids: list[str] = Field(default_factory=list, max_length=100)
    stage_count: int = Field(default=0, ge=0, le=64)
    reported_stage_count: int = Field(default=0, ge=0, le=64)
    corpus_version_known: bool = False


FAILURES = {
    **CALL_FAILURES,
    "missing_env": (
        "The registered adapter's credential environment variable is missing.",
        "Set the token_env variable named in your manifest before starting the workbench.",
    ),
    "dependency_missing": (
        "The adapter module or one of its dependencies could not be imported.",
        "Install its dependencies in the workbench Python environment and check module:function.",
    ),
    "adapter_configuration": (
        "The registered adapter could not be opened.",
        "Check its module:function target, callable, corpus path and startup configuration "
        "locally.",
    ),
    "timeout": (
        "The diagnostic exceeded its time limit.",
        "Check adapter startup and endpoint latency, or increase the diagnostic time limit.",
    ),
    "http_rate_limit": (
        "The endpoint rate-limited the sample request (HTTP 429).",
        "Wait before retrying and check the endpoint's quota. The diagnostic does not retry.",
    ),
    "adapter_exception": (
        "The sample call raised an adapter exception.",
        "Reproduce this input in the adapter locally. Exception text is omitted to protect "
        "secrets.",
    ),
    "cancelled": (
        "The pipeline test was cancelled before it finished.",
        "Run the test again when the adapter or endpoint is ready.",
    ),
    "worker_failed": (
        "The diagnostic process could not complete.",
        "Check the workbench Python installation, available memory and adapter startup locally.",
    ),
}


def failure_report(code: str, *, executed: bool = False) -> DiagnosticReport:
    """Only known categories reach the report; arbitrary exception messages never do."""
    code = code if code in FAILURES else "adapter_exception"
    message, hint = FAILURES[code]
    return DiagnosticReport(
        executed=executed,
        findings=[DiagnosticFinding(code=code, status="error", message=message, hint=hint)],
    )


def inspect_result(
    request: PipelineRequest,
    result: PipelineResult,
    documents: list[dict[str, Any]],
    *,
    latency_ms: float,
    corpus_version_known: bool,
    forbidden: list[str],
    no_answer: bool,
) -> DiagnosticReport:
    """Describe this sample's observations; never infer authorization from successful transport."""
    ids = [h.id for h in result.hits]
    report = DiagnosticReport(
        executed=True,
        latency_ms=latency_ms,
        returned_ids=ids,
        stage_count=len(result.stages),
        reported_stage_count=sum(s.source == "reported" for s in result.stages),
        corpus_version_known=corpus_version_known,
    )

    def add(code: str, status: Any, message: str, hint: str = "") -> None:
        report.findings.append(
            DiagnosticFinding(code=code, status=status, message=message, hint=hint)
        )

    add("response", "ok", "The sample call returned a valid response within the time limit.")
    add(
        "ranking",
        "ok" if ids else "warning",
        f"Returned {len(ids)} documents with unique IDs.",
        "An empty ranking is valid but cannot exercise document lookup or filter matching."
        if not ids
        else "This checks ID structure, not stability across runs or retrieval quality.",
    )
    by_id = {d["doc_id"]: d for d in documents}
    if not documents:
        add(
            "document_lookup",
            "unknown",
            "No local passage corpus was supplied.",
            "Attach a local JSONL corpus with doc_id and text to check returned IDs.",
        )
    elif not ids:
        add("document_lookup", "skipped", "There were no returned document IDs to look up.")
    else:
        report.missing_ids = [doc for doc in ids if doc not in by_id]
        add(
            "document_lookup",
            "warning" if report.missing_ids else "ok",
            f"Mapped {len(ids) - len(report.missing_ids)} of {len(ids)} returned IDs "
            "to local passages.",
            "Check chunk/document ID mapping and add missing passages to the local corpus."
            if report.missing_ids
            else "",
        )
    add(
        "stage_evidence",
        "ok" if result.stages else "warning",
        f"Captured {len(result.stages)} stages; "
        f"{report.reported_stage_count} use adapter-reported timings.",
        "Return stage traces to locate document losses inside the pipeline."
        if not result.stages
        else "",
    )
    add(
        "corpus_version",
        "ok" if corpus_version_known else "unknown",
        "A corpus version is configured."
        if corpus_version_known
        else "The retrieval corpus version is unknown.",
        "The configured version is supplied evidence; this sample does not independently "
        "inventory a remote corpus.",
    )
    if request.filters:
        add(
            "filter_forwarding",
            "unknown",
            "Filters were supplied to the adapter; internal forwarding is unverified.",
            "Use a known allowed/forbidden case and inspect the adapter's forwarding "
            "implementation.",
        )
        known = [
            doc
            for doc in ids
            if doc in by_id and all(k in by_id[doc].get("metadata", {}) for k in request.filters)
        ]
        mismatched = [
            doc
            for doc in known
            if any(by_id[doc]["metadata"][k] != v for k, v in request.filters.items())
        ]
        add(
            "filter_results",
            "error" if mismatched else "ok" if ids and len(known) == len(ids) else "unknown",
            f"{len(mismatched)} returned documents conflict with filters in the supplied local "
            f"metadata; {len(known)} checked.",
            "This checks sample outputs against local metadata, not filter enforcement. "
            "Confirm the corpus is current.",
        )
    else:
        add("filter_forwarding", "skipped", "No filters were supplied for this sample.")
    add(
        "context_forwarding",
        "unknown" if request.context else "skipped",
        "Context was supplied to the adapter; downstream use and authorization are unverified."
        if request.context
        else "No user or tenant context was supplied.",
        "Use reviewed permission scenarios to test expected results under each identity."
        if request.context
        else "",
    )
    if forbidden:
        violations = set(ids) & set(forbidden)
        add(
            "forbidden_documents",
            "error" if violations else "ok",
            f"{len(violations)} supplied forbidden IDs appeared in the final ranking.",
            "This assertion covers only the supplied IDs and this one sample call.",
        )
    else:
        add("forbidden_documents", "skipped", "No forbidden-document assertion was supplied.")
    if no_answer:
        add(
            "no_answer",
            "error" if ids else "ok",
            "The sample returned documents despite the no-answer assertion."
            if ids
            else "The sample returned no documents as asserted.",
        )
    return report
