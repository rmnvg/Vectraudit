"""Phase 14 acceptance: the release-check workflow end to end on every MVP backend.

For each backend, through the real CLI: a baseline passes against itself; a candidate
corpus with noisy embeddings (a controlled quality regression) fails; a broken connection
is an error; a different dataset version is not comparable. A deterministic latency
regression fails via the library. pgvector runs when ``VECTRAUDIT_TEST_PG_DSN`` is set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.connectors import Connector, SearchRequest
from vectraudit.connectors.base import BackendInfo, Capabilities
from vectraudit.connectors.config import load_connection, open_connector
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runner import RunOptions, execute_run

faiss = pytest.importorskip("faiss")
runner = CliRunner()
N, D, N_QUERIES = 2_000, 32, 50
POLICY = {
    "metrics": {"ndcg": {"max_drop": 0.02}, "recall": {"max_drop": 0.02}},
    "latency": {"p95_max_increase_pct": 100},
    "min_queries": 20,
}
PG_DSN = os.environ.get("VECTRAUDIT_TEST_PG_DSN")


@pytest.fixture(scope="module")
def fixture_dir(make_synthetic, tmp_path_factory) -> Path:
    """Clean and noisy corpora, a labelled dataset, a second dataset version and a policy."""
    out = tmp_path_factory.mktemp("acceptance")
    x_all, _ = make_synthetic.make_blobs(N + N_QUERIES, D, 20, 0)
    clean, queries = x_all[:N], x_all[N:]
    rng = np.random.default_rng(1)
    noisy = (clean + rng.normal(0, 0.1 * clean.std(), clean.shape)).astype(np.float32)
    np.save(out / "clean.npy", clean)
    np.save(out / "noisy.npy", noisy)
    for name, vectors in (("clean", clean), ("noisy", noisy)):
        index = faiss.IndexFlatL2(D)
        index.add(vectors)
        faiss.write_index(index, str(out / f"{name}.index"))
    _, nn = faiss.knn(queries, clean, 3)
    rows = [
        {
            "query_id": f"q{i}",
            "vector": q.tolist(),
            "relevant": {str(d): 3 - r for r, d in enumerate(ids)},
            "tags": ["head" if i < 25 else "tail"],
        }
        for i, (q, ids) in enumerate(zip(queries, nn, strict=True))
    ]
    (out / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "dataset-v2.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-1]) + "\n")
    (out / "policy.json").write_text(json.dumps(POLICY))
    return out


def _write(path: Path, obj: dict[str, Any]) -> Path:
    path.write_text(json.dumps(obj))
    return path


def faiss_connections(d: Path) -> dict[str, Path]:
    return {
        "base": _write(d / "faiss-base.json", {"backend": "faiss", "index": "clean.index"}),
        "noisy": _write(d / "faiss-noisy.json", {"backend": "faiss", "index": "noisy.index"}),
        "broken": _write(d / "faiss-broken.json", {"backend": "faiss", "index": "gone.index"}),
    }


def qdrant_connections(d: Path) -> dict[str, Path]:
    qdrant_client = pytest.importorskip("qdrant_client")
    models = qdrant_client.models
    for name in ("clean", "noisy"):
        store = d / f"qdrant-{name}"
        if not store.exists():
            client = qdrant_client.QdrantClient(path=str(store))
            client.create_collection(
                "docs", vectors_config=models.VectorParams(size=D, distance=models.Distance.EUCLID)
            )
            vectors = np.load(d / f"{name}.npy")
            client.upload_points(
                "docs",
                [models.PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)],
            )
            client.close()
    return {
        "base": _write(
            d / "qdrant-base.json",
            {"backend": "qdrant", "path": "qdrant-clean", "collection": "docs"},
        ),
        "noisy": _write(
            d / "qdrant-noisy.json",
            {"backend": "qdrant", "path": "qdrant-noisy", "collection": "docs"},
        ),
        "broken": _write(
            d / "qdrant-broken.json",
            {
                "backend": "qdrant",
                "url": "http://127.0.0.1:9",
                "collection": "docs",
                "timeout_s": 1,
            },
        ),
    }


def pgvector_connections(d: Path) -> dict[str, Path]:
    if not PG_DSN:
        pytest.skip("Set VECTRAUDIT_TEST_PG_DSN to run pgvector acceptance tests.")
    import psycopg

    from vectraudit.connectors.pgvector import vector_literal

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP SCHEMA IF EXISTS va_accept CASCADE")
        conn.execute("CREATE SCHEMA va_accept")
        for name in ("clean", "noisy"):
            conn.execute(f"CREATE TABLE va_accept.{name} (id bigint PRIMARY KEY, v vector({D}))")
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO va_accept.{name} VALUES (%s, %s::vector)",
                    [(i, vector_literal(v)) for i, v in enumerate(np.load(d / f"{name}.npy"))],
                )
            conn.execute(f"CREATE INDEX ON va_accept.{name} USING hnsw (v vector_l2_ops)")
            conn.execute(f"ANALYZE va_accept.{name}")
    os.environ["VA_ACCEPT_DSN"] = PG_DSN
    os.environ["VA_ACCEPT_BROKEN_DSN"] = "postgresql://postgres:x@127.0.0.1:9/va?connect_timeout=2"
    common = {"backend": "pgvector", "vector_column": "v", "distance": "l2"}
    return {
        "base": _write(
            d / "pg-base.json",
            {**common, "dsn_env": "VA_ACCEPT_DSN", "table": "va_accept.clean"},
        ),
        "noisy": _write(
            d / "pg-noisy.json",
            {**common, "dsn_env": "VA_ACCEPT_DSN", "table": "va_accept.noisy"},
        ),
        "broken": _write(
            d / "pg-broken.json",
            {
                **common,
                "dsn_env": "VA_ACCEPT_BROKEN_DSN",
                "table": "va_accept.clean",
                "connect_timeout_s": 2,
            },
        ),
    }


BACKENDS = {
    "faiss": faiss_connections,
    "qdrant": qdrant_connections,
    "pgvector": pgvector_connections,
}


@pytest.fixture(scope="module", params=sorted(BACKENDS))
def backend(request, fixture_dir) -> tuple[str, dict[str, Path]]:
    return request.param, BACKENDS[request.param](fixture_dir)


def record(conn: Path, dataset: Path, out: Path, label: str) -> int:
    result = runner.invoke(
        app,
        [
            "record",
            "--connection",
            str(conn),
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--k",
            "5",
            "--label",
            label,
            "--warmup",
            "2",
        ],
    )
    return result.exit_code


def check(base: Path, cand: Path, fixture_dir: Path, report: Path | None = None):
    args = [
        "check",
        str(base),
        str(cand),
        "--policy",
        str(fixture_dir / "policy.json"),
        "--no-github-summary",
    ]
    if report is not None:
        args += ["--html", str(report)]
    return runner.invoke(app, args)


def test_release_check_workflow(backend, fixture_dir, tmp_path) -> None:
    name, conns = backend
    ds = fixture_dir / "dataset.jsonl"
    base, again, noisy, broken, other = (
        tmp_path / f"{n}.json" for n in ("base", "again", "noisy", "broken", "other")
    )
    assert record(conns["base"], ds, base, "main") == 0
    assert record(conns["base"], ds, again, "main-again") == 0
    assert record(conns["noisy"], ds, noisy, "pr-noisy-embeddings") == 0
    assert record(conns["broken"], ds, broken, "pr-broken") == 1
    assert record(conns["base"], fixture_dir / "dataset-v2.jsonl", other, "v2") == 0

    run = json.loads(base.read_text())
    assert run["backend"]["backend"] == name
    assert run["summary"]["metrics"]["recall"] > 0.95, name

    result = check(base, again, fixture_dir)
    assert result.exit_code == 0, (name, result.output)

    report = tmp_path / "report.html"
    result = check(base, noisy, fixture_dir, report)
    assert result.exit_code == 3, (name, result.output)
    assert "ndcg dropped" in result.output
    page = report.read_text()
    assert "Fail" in page
    assert "pr-noisy-embeddings" in page

    assert json.loads(broken.read_text())["status"] == "failed"
    result = check(base, broken, fixture_dir)
    assert result.exit_code == 1, (name, result.output)
    assert "candidate run is failed" in result.output

    result = check(base, other, fixture_dir)
    assert result.exit_code == 4, (name, result.output)
    assert "different dataset versions" in result.output


class Slow(Connector):
    """Adds a fixed delay to every search: a deterministic latency regression."""

    def __init__(self, inner: Connector, delay_s: float) -> None:
        self.inner, self.delay_s, self.name = inner, delay_s, inner.name

    def describe(self) -> BackendInfo:
        return self.inner.describe()

    def capabilities(self) -> Capabilities:
        return self.inner.capabilities()

    def _search(self, request: SearchRequest) -> list[str]:
        time.sleep(self.delay_s)
        return self.inner.search(request)

    def close(self) -> None:
        self.inner.close()


def test_latency_regression(backend, fixture_dir) -> None:
    name, conns = backend
    ds = load_dataset(fixture_dir / "dataset.jsonl")
    options = RunOptions(k=5, warmup=2)
    cfg = load_connection(conns["base"])
    with open_connector(cfg) as conn:
        base = execute_run(ds, conn, options)
    with Slow(open_connector(cfg), 0.02) as slow:
        cand = execute_run(ds, slow, options)
    result = compare_runs(base, cand, parse_policy(POLICY))
    assert result["verdict"] == "fail", name
    assert any("p95 latency rose" in r["message"] for r in result["reasons"])
    assert result["metrics"]["ndcg"]["delta"] == 0
