"""Execute a dataset against a connector and produce a run record.

A run record is plain JSON (see :mod:`vectraudit.evaluation.runfile`). It stores the
ranked ids, latency, metrics and errors of every query, plus everything needed to decide
later whether two runs can be compared: the dataset hash, backend description, search
parameters, timing protocol, metric contract and provenance.
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit import __version__
from vectraudit.connectors.base import (
    Connector,
    ConnectorError,
    ErrorCode,
    ParamValue,
    SearchRequest,
)
from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.metrics import (
    SEMANTIC_METRICS,
    ann_recall,
    latency_summary,
    metric_contract,
    semantic_metrics,
)

RUN_FORMAT = "vectraudit.run"
RUN_FORMAT_VERSION = 1
FloatArray = npt.NDArray[np.float32]
TextEmbedder = Callable[[Sequence[str]], npt.ArrayLike]
Progress = Callable[[int, int], None]

# Per-query failures; anything else stops the run, since later queries would fail too.
QUERY_LEVEL_ERRORS = frozenset({ErrorCode.QUERY, ErrorCode.UNSUPPORTED, ErrorCode.TIMEOUT})


@dataclass(frozen=True)
class RunOptions:
    """How to execute a run. ``exact=None`` uses the exact reference when available."""

    k: int = 10
    params: Mapping[str, ParamValue] = field(default_factory=dict)
    warmup: int = 5
    repeats: int = 1
    exact: bool | None = None
    label: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    """Provenance labels such as ``embedding_model`` or ``chunking_version``."""

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("Require warmup >= 0 and repeats >= 1.")


def execute_run(
    dataset: Dataset,
    connector: Connector,
    options: RunOptions | None = None,
    *,
    connection: Mapping[str, Any] | None = None,
    embed: TextEmbedder | None = None,
    embedder_name: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run every query and return the run record. Never raises for backend failures.

    Connection-level failures end the run with status ``failed``; per-query failures are
    recorded on the query (status ``partial``); Ctrl-C ends it with status ``cancelled``.
    """
    options = options or RunOptions()
    started = time.perf_counter()
    record = _skeleton(dataset, options, connection, embedder_name)
    try:
        info = connector.describe()
        caps = connector.capabilities()
        record["backend"] = info.as_dict()
        record["capabilities"] = caps.as_dict()
        use_exact = caps.exact_reference if options.exact is None else options.exact
        if use_exact and not caps.exact_reference:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"{connector.name} has no exact reference, so ANN recall can't be measured.",
                "Drop --exact, or give the connection raw vectors.",
            )
        record["search"]["exact_reference"] = use_exact
        record["search"]["exact_reference_note"] = caps.exact_reference_note if use_exact else None
        vectors, embedding_s = _query_vectors(dataset, embed, info.dim)
        record["timing"]["embedding_s"] = embedding_s
        _warm_up(connector, dataset, vectors, options)
        for n, (q, vector) in enumerate(zip(dataset.queries, vectors, strict=True)):
            if progress is not None:
                progress(n, len(dataset))
            record["queries"].append(_run_query(connector, q, vector, options, use_exact))
        if progress is not None:
            progress(len(dataset), len(dataset))
    except ConnectorError as e:
        record["status"] = "failed"
        record["error"] = e.as_dict()
    except KeyboardInterrupt:
        record["status"] = "cancelled"
    except Exception as e:  # a connector bug must still produce a failed run
        record["status"] = "failed"
        record["error"] = {
            "code": "internal",
            "message": f"{type(e).__name__}: {e}",
            "hint": "This is a bug in the connector or runner; please report it.",
        }
    else:
        failed = any(q["error"] for q in record["queries"])
        record["status"] = "partial" if failed else "complete"
    record["timing"]["wall_s"] = time.perf_counter() - started
    record["summary"] = summarize(record["queries"], len(dataset))
    return record


def _skeleton(
    dataset: Dataset,
    options: RunOptions,
    connection: Mapping[str, Any] | None,
    embedder_name: str | None,
) -> dict[str, Any]:
    return {
        "format": RUN_FORMAT,
        "format_version": RUN_FORMAT_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": options.label,
        "status": "failed",
        "error": None,
        "dataset": dataset.summary(),
        "connection": dict(connection or {}),
        "backend": None,
        "capabilities": None,
        "search": {"k": options.k, "params": dict(options.params), "embedder": embedder_name},
        "timing": {
            "scope": "connector search call, measured by the runner (client side)",
            "warmup": options.warmup,
            "repeats": options.repeats,
            "statistic": "median of repeats per query",
            "concurrency": 1,
        },
        "metric_contract": metric_contract(options.k),
        "provenance": provenance(options.labels),
        "queries": [],
    }


def _query_vectors(
    dataset: Dataset, embed: TextEmbedder | None, dim: int
) -> tuple[list[FloatArray], float | None]:
    texts = [q.text for q in dataset.queries if q.vector is None and q.text is not None]
    embedded: list[FloatArray] = []
    seconds = None
    if texts:
        if embed is None:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"{len(texts)} queries have text but no vector.",
                "Pass --embedder with the model that embedded the corpus, or add vectors.",
            )
        t0 = time.perf_counter()
        out = np.asarray(embed(texts), dtype=np.float32)
        seconds = time.perf_counter() - t0
        if out.ndim != 2 or len(out) != len(texts):
            raise ConnectorError(ErrorCode.CONFIG, "The embedder returned the wrong shape.")
        embedded = list(out)
    it = iter(embedded)
    vectors = [q.vector if q.vector is not None else next(it) for q in dataset.queries]
    for q, v in zip(dataset.queries, vectors, strict=True):
        if len(v) != dim:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"Query {q.query_id!r} has dimension {len(v)}, the backend expects {dim}.",
                "Use query vectors from the same embedding model as the corpus.",
            )
    return vectors, seconds


def _warm_up(
    connector: Connector, dataset: Dataset, vectors: list[FloatArray], options: RunOptions
) -> None:
    for q, v in list(zip(dataset.queries, vectors, strict=True))[: options.warmup]:
        try:
            connector.search(_request(q, v, options, exact=False))
        except ConnectorError as e:
            if e.code not in QUERY_LEVEL_ERRORS:
                raise


def _request(q: DatasetQuery, v: FloatArray, options: RunOptions, *, exact: bool) -> SearchRequest:
    return SearchRequest(v, options.k, filter=q.filter, params=options.params, exact=exact)


def _run_query(
    connector: Connector, q: DatasetQuery, v: FloatArray, options: RunOptions, use_exact: bool
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "query_id": q.query_id,
        "tags": list(q.tags),
        "ids": None,
        "exact_ids": None,
        "latency_ms": None,
        "metrics": None,
        "ann_recall": None,
        "nondeterministic": False,
        "error": None,
    }
    try:
        results, latencies = [], []
        for _ in range(options.repeats):
            t0 = time.perf_counter()
            results.append(connector.search(_request(q, v, options, exact=False)))
            latencies.append((time.perf_counter() - t0) * 1000)
        ids = results[0]
        entry["ids"] = ids
        entry["latency_ms"] = statistics.median(latencies)
        entry["nondeterministic"] = any(r != ids for r in results[1:])
        if q.relevant:
            entry["metrics"] = semantic_metrics(ids, q.relevant, options.k)
        if use_exact:
            exact = connector.search(_request(q, v, options, exact=True))
            entry["exact_ids"] = exact
            entry["ann_recall"] = ann_recall(ids, exact)
    except ConnectorError as e:
        if e.code not in QUERY_LEVEL_ERRORS:
            raise
        entry["error"] = e.as_dict()
    return entry


def summarize(queries: Sequence[Mapping[str, Any]], n_queries: int) -> dict[str, Any]:
    """Aggregate metrics over the queries that succeeded."""
    ok = [q for q in queries if not q["error"] and q["ids"] is not None]
    labelled = [q for q in ok if q["metrics"] is not None]
    recalls = [q["ann_recall"] for q in ok if q["ann_recall"] is not None]
    n_errors = sum(bool(q["error"]) for q in queries)
    return {
        "n_queries": n_queries,
        "n_executed": len(queries),
        "n_ok": len(ok),
        "n_errors": n_errors,
        "error_rate": n_errors / len(queries) if queries else 0.0,
        "n_labelled": len(labelled),
        "metrics": (
            {m: statistics.fmean(q["metrics"][m] for q in labelled) for m in SEMANTIC_METRICS}
            if labelled
            else None
        ),
        "ann_recall": statistics.fmean(recalls) if recalls else None,
        "n_ann_recall": len(recalls),
        "latency_ms": latency_summary([q["latency_ms"] for q in ok]),
        "n_nondeterministic": sum(bool(q["nondeterministic"]) for q in ok),
    }


def provenance(labels: Mapping[str, str]) -> dict[str, Any]:
    """Commit, CI run and runner environment, plus user labels."""
    env = os.environ
    ci = None
    if env.get("GITHUB_ACTIONS") == "true":
        server = env.get("GITHUB_SERVER_URL", "https://github.com")
        ci = {
            "provider": "github",
            "repository": env.get("GITHUB_REPOSITORY"),
            "ref": env.get("GITHUB_REF"),
            "event": env.get("GITHUB_EVENT_NAME"),
            "run_url": (
                f"{server}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
                if env.get("GITHUB_REPOSITORY") and env.get("GITHUB_RUN_ID")
                else None
            ),
        }
    return {
        "commit_sha": env.get("GITHUB_SHA") or _git_head(),
        "ci": ci,
        "runner": runner_environment(),
        "labels": dict(labels),
    }


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def runner_environment() -> dict[str, Any]:
    """Where the runner executed: latency is only comparable on matching environments."""
    return {
        "vectraudit": __version__,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ci": bool(os.environ.get("CI")),
    }
