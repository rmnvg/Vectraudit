"""Offline installation and release-check readiness diagnostics."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit import __version__
from vectraudit.connectors.base import ConnectorError
from vectraudit.connectors.config import (
    ConnectionConfig,
    load_connection,
    resolve_path,
    validate_connection,
)
from vectraudit.evaluation.dataset import Dataset, load_dataset
from vectraudit.evaluation.dataset_tools import SMALL_LABELLED, dataset_stats
from vectraudit.evaluation.metrics import SEMANTIC_METRICS
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, load_policy

EXTRAS = {
    "faiss": ("faiss-cpu", "faiss-cpu"),
    "qdrant": ("qdrant-client", "qdrant"),
    "pgvector": ("psycopg", "pgvector"),
    "weaviate": ("weaviate-client", "weaviate"),
}


@dataclass(frozen=True)
class Finding:
    """A stable diagnostic code with a user-facing remedy; never includes secret values."""

    code: str
    status: str
    message: str
    hint: str = ""


class Doctor:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.dataset: dict[str, Any] | None = None
        self.backend: str | None = None

    def add(self, code: str, status: str, message: str, hint: str = "") -> None:
        self.findings.append(Finding(code, status, message, hint))

    def dependency(self, distribution: str, extra: str) -> None:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            # FAISS may come from conda rather than a pip distribution.
            if distribution == "faiss-cpu" and importlib.util.find_spec("faiss") is not None:
                self.add(
                    "dependency.faiss-cpu", "ok", "FAISS module available (version unverified)."
                )
                return
            self.add(
                f"dependency.{distribution}",
                "error",
                f"Missing optional dependency: {distribution}.",
                f"Install vectraudit with the [{extra}] extra.",
            )
        else:
            self.add(f"dependency.{distribution}", "ok", f"{distribution} {version} is installed.")

    def inspect_connection(self, path: Path) -> ConnectionConfig | None:
        try:
            cfg = load_connection(path)
            validate_connection(cfg)
        except (ValueError, OSError, ConnectorError):
            self.add(
                "connection.invalid",
                "error",
                "Connection file is missing or invalid.",
                "Check backend fields and types; put secrets in named environment variables.",
            )
            return None
        self.backend = cfg.backend
        self.add("connection.valid", "ok", f"Valid {cfg.backend} connection configuration.")
        dependency = EXTRAS.get(cfg.backend)
        if dependency:
            self.dependency(*dependency)
        for key, value in cfg.options.items():
            if key.endswith("_env"):
                self.add(
                    f"secret.{key}",
                    "ok" if os.environ.get(value) else "error",
                    f"Secret for {key} is {'set' if os.environ.get(value) else 'not set'}.",
                    "Set the environment variable named in the connection file.",
                )
        keys = ("index", "vectors", "ids") if cfg.backend == "faiss" else ()
        for key in keys:
            if key in cfg.options:
                present = resolve_path(cfg, cfg.options[key]).is_file()
                self.add(
                    f"file.{key}",
                    "ok" if present else "error",
                    f"Local {key} file {'exists' if present else 'is missing'}.",
                    "Paths resolve relative to the connection file.",
                )
        return cfg

    def inspect_dataset(self, path: Path, policy: Policy, embedder: str | None) -> Dataset | None:
        try:
            ds = load_dataset(path)
        except (ValueError, OSError):
            self.add(
                "dataset.invalid",
                "error",
                "Dataset is missing or invalid.",
                "Run vectraudit dataset validate locally for the line-specific error.",
            )
            return None
        stats = dataset_stats(ds)
        self.dataset = {
            "sha256": ds.sha256,
            "n_queries": len(ds),
            "n_families": stats["n_families"],
            "n_labelled": ds.n_labelled,
            "dim": ds.dim,
            "needs_embedder": ds.needs_embedder,
        }
        self.add(
            "dataset.valid",
            "ok",
            f"Dataset validates: {len(ds)} rows, {stats['n_families']} families, "
            f"{ds.n_labelled} labelled rows.",
        )
        if stats["n_families"] < policy.min_queries:
            self.add(
                "dataset.minimum",
                "error",
                "Suite is below the policy minimum independent-family count.",
                f"Add independent question families to reach {policy.min_queries}.",
            )
        semantic = [
            name
            for name, rule in policy.metrics.items()
            if name in SEMANTIC_METRICS and rule.required
        ]
        n_positive = stats["n_positive_families"]
        if semantic and n_positive < policy.min_queries:
            self.add(
                "labels.minimum",
                "error",
                "Too few labelled families for the required quality metrics.",
                f"Label at least {policy.min_queries} independent families "
                "with positive source-document relevance.",
            )
        elif n_positive < SMALL_LABELLED:
            self.add(
                "labels.small",
                "warning",
                f"Only {n_positive} independent families have positive labels.",
                "Use label pool/serve/export; aim for 30+ representative labelled queries.",
            )
        if ds.n_labelled < len(ds):
            self.add(
                "labels.coverage",
                "warning",
                "Some questions have no semantic quality labels.",
                "Unlabelled questions contribute only ANN recall and latency.",
            )
        if any("query texts appear" in w for w in stats["warnings"]):
            self.add(
                "dataset.duplicates",
                "warning",
                "Repeated question texts reduce evaluation diversity.",
                "Review duplicates and sample distinct real intents.",
            )
        small = [n for n in stats["tag_families"].values() if n < policy.cohorts.min_queries]
        if small and policy.cohorts.metrics:
            self.add(
                "cohorts.small",
                "warning",
                "Some cohorts are below the policy minimum.",
                "Grow the labelled cohort; small cohorts are reported without judging.",
            )
        if ds.needs_embedder:
            if embedder:
                self.dependency("sentence-transformers", "text")
                self.add(
                    "embedding.unverified",
                    "warning",
                    "Embedding model supplied; model loading and dimensions are unverified.",
                    "Use the corpus embedding model; doctor never downloads or loads it.",
                )
            else:
                self.add(
                    "embedding.missing",
                    "error",
                    "Text-only queries need an embedder for record.",
                    "Pass --embedder MODEL here and to record, or use dataset embed first.",
                )
        return ds


def diagnose(
    *,
    connection: Path | None = None,
    dataset: Path | None = None,
    policy: Path | None = None,
    embedder: str | None = None,
) -> dict[str, Any]:
    """Inspect local inputs; never open a connector or call an embedding model."""
    doctor = Doctor()
    doctor.add("installation.version", "ok", f"Vectraudit {__version__} is installed.")
    from vectraudit.server.app import STATIC_DIR

    ui = (STATIC_DIR / "index.html").is_file()
    doctor.add(
        "installation.ui",
        "ok" if ui else "warning",
        "FAISS dashboard assets are installed." if ui else "FAISS dashboard assets are missing.",
        "Install the release wheel, or build frontend/ with npm run build.",
    )
    rules = DEFAULT_POLICY
    if policy is not None:
        try:
            rules = load_policy(policy)
            doctor.add("policy.valid", "ok", "Release policy validates.")
        except (ValueError, OSError):
            doctor.add(
                "policy.invalid",
                "error",
                "Policy file is missing or invalid.",
                "Check metric names, nonnegative limits and minimum query counts.",
            )
    cfg = doctor.inspect_connection(connection) if connection is not None else None
    ds = doctor.inspect_dataset(dataset, rules, embedder) if dataset is not None else None
    if cfg is not None and ds is not None:
        if cfg.backend == "faiss" and any(q.filter for q in ds.queries):
            doctor.add(
                "filters.unsupported",
                "error",
                "FAISS connections do not support dataset filters.",
                "Use a backend that supports these filters, or a compatible dataset.",
            )
        if (
            cfg.backend == "pinecone"
            and "ann_recall" in rules.metrics
            and rules.metrics["ann_recall"].required
        ):
            doctor.add(
                "reference.unsupported",
                "error",
                "Pinecone cannot provide an exact ANN reference.",
                "Use relevance metrics in the policy; ANN recall does not measure answer quality.",
            )
        if cfg.backend == "faiss" and "vectors" in cfg.options:
            path = resolve_path(cfg, cfg.options["vectors"])
            if path.is_file():
                try:
                    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
                    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.dtype.kind != "f":
                        raise ValueError("Invalid matrix")
                    if ds.dim is not None and vectors.shape[1] != ds.dim:
                        doctor.add(
                            "vectors.dimension",
                            "error",
                            "Query and corpus vector dimensions differ.",
                            "Use the same embedding model and dimension for corpus and queries.",
                        )
                    else:
                        doctor.add(
                            "vectors.shape",
                            "ok",
                            "Corpus vector shape is compatible with the dataset.",
                        )
                except (ValueError, OSError):
                    doctor.add(
                        "vectors.invalid",
                        "error",
                        "Corpus vectors are not a readable floating-point matrix.",
                        "Supply a .npy matrix; pickle loading is disabled.",
                    )
    statuses = {f.status for f in doctor.findings}
    status = (
        "blocked"
        if "error" in statuses
        else "needs_attention"
        if "warning" in statuses
        else "ready"
    )
    return {
        "format": "vectraudit.doctor.v1",
        "status": status,
        "offline": True,
        "scope": "release_setup"
        if any(p is not None for p in (connection, dataset, policy))
        else "installation",
        "backend": doctor.backend,
        "dataset": doctor.dataset,
        "checks": [asdict(f) for f in doctor.findings],
        "limits": [
            "Backend connectivity, permissions, snapshots and model execution are not tested."
        ],
    }
