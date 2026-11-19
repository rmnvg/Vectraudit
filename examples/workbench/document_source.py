"""Example read-only SQLite document resolver; configure DOCUMENT_LOOKUP_DB locally.

Schema: passages(snapshot, tenant, doc_id, text, title, content_version, category).
Index (snapshot, tenant, doc_id) for bounded lookups in a large corpus. Adapt the
authorization and filter handling to your real source before using user data.
"""

import os
import sqlite3
from pathlib import Path

from vectraudit.pipelines.document_lookup import LookupDocument, LookupRequest, LookupResult


def lookup(request: LookupRequest) -> LookupResult:
    """Fetch only requested IDs, scoped to the supplied tenant and frozen snapshot."""
    if not request.corpus_version:
        raise ValueError("This example requires a known corpus snapshot")
    if set(request.filters) - {"category"}:
        raise ValueError("Unsupported filter: adapt the resolver instead of ignoring it")
    tenant = request.context.get("tenant")
    documents = []
    if not isinstance(tenant, str) or not tenant:
        return LookupResult(
            corpus_version=request.corpus_version,
            documents=[LookupDocument(doc_id=doc, status="forbidden") for doc in request.doc_ids],
        )
    uri = Path(os.environ["DOCUMENT_LOOKUP_DB"]).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        for doc in request.doc_ids:
            row = connection.execute(
                "SELECT text, title, content_version, category FROM passages "
                "WHERE snapshot = ? AND tenant = ? AND doc_id = ?",
                (request.corpus_version, tenant, doc),
            ).fetchone()
            if row is None or (
                "category" in request.filters and row[3] != request.filters["category"]
            ):
                documents.append(LookupDocument(doc_id=doc, status="missing"))
            else:
                documents.append(
                    LookupDocument(
                        doc_id=doc, status="found", text=row[0], title=row[1], version=row[2]
                    )
                )
    finally:
        connection.close()
    return LookupResult(corpus_version=request.corpus_version, documents=documents)
