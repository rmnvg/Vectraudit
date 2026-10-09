"""Bounded contracts for customer-controlled passage lookup during local review."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import Field, model_validator

from vectraudit.contracts import Context, Filters
from vectraudit.pipelines.diagnostics import _kill
from vectraudit.pipelines.models import Contract


class LookupConfig(Contract):
    target: str = Field(min_length=3, max_length=512)
    timeout_s: float = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def configured(self) -> LookupConfig:
        module, sep, name = self.target.partition(":")
        if (
            not sep
            or not name.isidentifier()
            or not all(p.isidentifier() for p in module.split("."))
        ):
            raise ValueError("document_lookup.target must be module:function")
        return self


DocumentId = Annotated[str, Field(min_length=1, max_length=512)]


class LookupRequest(Contract):
    query_id: str = Field(min_length=1, max_length=512)
    doc_ids: list[DocumentId] = Field(min_length=1, max_length=10)
    context: Context = Field(default_factory=dict)
    filters: Filters = Field(default_factory=dict)
    corpus_version: str | None = Field(default=None, max_length=512)
    run_id: str


class LookupDocument(Contract):
    doc_id: str = Field(min_length=1, max_length=512)
    status: Literal["found", "missing", "forbidden", "version_unavailable"]
    text: str | None = Field(default=None, min_length=1, max_length=100000)
    title: str | None = Field(default=None, max_length=512)
    version: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def content(self) -> LookupDocument:
        if self.status == "found":
            if self.text is None or self.version is None:
                raise ValueError("found documents require text and a content version")
        elif any(value is not None for value in (self.text, self.title, self.version)):
            raise ValueError("unavailable documents must not return content")
        return self


class LookupResult(Contract):
    corpus_version: str | None = Field(default=None, max_length=512)
    documents: list[LookupDocument] = Field(max_length=10)


def validate_lookup(request: LookupRequest, result: LookupResult) -> LookupResult:
    """Reject unexpected IDs and snapshots before exposing any passage text."""
    ids = [doc.doc_id for doc in result.documents]
    if len(ids) != len(set(ids)) or set(ids) != set(request.doc_ids):
        raise ValueError("resolver must return exactly one status for each requested document")
    if request.corpus_version is not None and result.corpus_version != request.corpus_version:
        raise ValueError("resolver did not return the requested corpus version")
    return result


class LookupRunner:
    """One bounded fresh-process lookup at a time; no adapter output enters logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def run(self, config: LookupConfig, root: Path, request: LookupRequest) -> LookupResult:
        """Run trusted customer code, suppressing exception text and enforcing a timeout."""
        if not self._lock.acquire(blocking=False):
            raise HTTPException(429, "A document lookup is running. Try again when it finishes.")
        try:
            payload = json.dumps(
                {"target": config.target, "root": str(root), "request": request.model_dump()}
            ).encode()
            if len(payload) > 1024 * 1024:
                raise HTTPException(413, "Document lookup request exceeds 1 MiB")
            process = subprocess.Popen(
                [sys.executable, "-m", "vectraudit.pipelines.lookup_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
            try:
                out, _ = process.communicate(payload, timeout=config.timeout_s)
            except subprocess.TimeoutExpired:
                _kill(process)
                process.communicate()
                raise HTTPException(
                    504, "Document lookup timed out. Check the configured source."
                ) from None
            if process.returncode != 0 or len(out) > 6 * 1024 * 1024:
                raise HTTPException(
                    502, "Document lookup failed. Check the registered resolver locally."
                )
            try:
                result = LookupResult.model_validate_json(out)
                if (
                    request.corpus_version is not None
                    and result.corpus_version != request.corpus_version
                ):
                    raise HTTPException(
                        409,
                        "Document source returned a different corpus version. "
                        "No passages were accepted.",
                    )
                return validate_lookup(request, result)
            except ValueError:
                raise HTTPException(
                    502, "Document lookup returned invalid IDs, content or corpus version."
                ) from None
        finally:
            self._lock.release()


def content_hash(text: str) -> str:
    """Fingerprint the actual text shown, independently of adapter-reported versions."""
    return hashlib.sha256(text.encode()).hexdigest()
