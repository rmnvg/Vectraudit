"""Typed adapter failures; diagnostics render allowlisted remedies, never exception text."""

from __future__ import annotations


class PipelineCallError(ValueError):
    """A known failure category, compatible with existing ValueError handlers."""

    def __init__(self, code: str, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        """Seconds the endpoint asked us to wait (``Retry-After``), when it said."""


# Transient categories that runs retry automatically within their retry budget. Retrieval
# calls are reads, so repeating one cannot change what the benchmark measures.
RETRYABLE = frozenset({"timeout", "connection_failed", "http_rate_limit", "http_unavailable"})


# Only these stable categories may be persisted; adapters can supply arbitrary codes.
CALL_FAILURES = {
    "incomplete_evidence": (
        "The adapter did not observe a complete final ranking.",
        "Record the complete delivered ranking; partial or unavailable observations cannot pass.",
    ),
    "text_required": (
        "This retriever requires a text query.",
        "Supply sample text for the framework retriever.",
    ),
    "context_unsupported": (
        "The framework adapter cannot forward these filters or context with its current setup.",
        "For LangChain configure invoke_kwargs; for LlamaIndex use a request-aware retriever "
        "factory.",
    ),
    "document_id_missing": (
        "A retrieved document has no usable identifier.",
        "Configure id_key or map retriever document IDs to the IDs used in your benchmark corpus.",
    ),
    "http_auth": (
        "The endpoint rejected authentication or authorization (HTTP 401/403).",
        "Check the configured credential and its access to the retrieval endpoint.",
    ),
    "http_not_found": (
        "The retrieval endpoint returned HTTP 404.",
        "Check the registered endpoint path and that the intended deployment is running.",
    ),
    "http_rate_limit": (
        "The endpoint rate-limited the request (HTTP 429).",
        "Runs retry within their retry budget and honour Retry-After; if it persists, check "
        "the endpoint's quota or lower the request rate.",
    ),
    "http_unavailable": (
        "The endpoint was temporarily unavailable (HTTP 502, 503 or 504).",
        "Runs retry within their retry budget; if it persists, check the deployment's health "
        "and load.",
    ),
    "http_error": (
        "The endpoint returned an HTTP error.",
        "Inspect the endpoint's local logs and verify it accepts the PipelineRequest contract.",
    ),
    "connection_failed": (
        "The configured endpoint could not be contacted successfully.",
        "Check the server, DNS, port, TLS certificate and network access from this machine.",
    ),
    "timeout": (
        "The pipeline call exceeded its time limit.",
        "Check adapter startup and endpoint latency, or increase the configured timeout.",
    ),
    "response_too_large": (
        "The endpoint response exceeds the 8 MiB limit.",
        "Return bounded IDs, scores and stage traces; omit document bodies from the response.",
    ),
    "invalid_response": (
        "The adapter returned an invalid response contract.",
        "Python adapters must return PipelineResult; HTTP endpoints must return its JSON schema. "
        "Check unique IDs, finite scores, trace parents and the final-stage ranking.",
    ),
    "too_many_hits": (
        "The adapter returned more than the requested k documents.",
        "Explicitly truncate the delivered results to request.k, including the final trace stage.",
    ),
}


def run_failure(exc: Exception) -> dict[str, str]:
    """Keep known call causes without persisting adapter messages or arbitrary codes."""
    code = exc.code if isinstance(exc, PipelineCallError) else "pipeline_error"
    if code in CALL_FAILURES:
        message, hint = CALL_FAILURES[code]
        return {"code": code, "message": message, "hint": hint}
    return {
        "code": "pipeline_error",
        "message": "Pipeline call or response validation failed.",
        "hint": "Validate the adapter locally against PipelineRequest/PipelineResult.",
    }
