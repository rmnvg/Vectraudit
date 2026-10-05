"""vectraudit init: scaffolded files are valid, and a fresh project reaches a passing check."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.connectors.config import parse_connection
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import ColumnMap, import_rows, read_records, render_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.scaffold import BACKENDS, init_project, render_workflow, scaffold_files

runner = CliRunner()
ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_scaffold_files_are_valid(backend, tmp_path) -> None:
    files = scaffold_files(backend)
    assert set(files) == {
        "retrieval/connection.json",
        "retrieval/policy.json",
        "retrieval/queries.csv",
        "retrieval/README.md",
        ".github/workflows/retrieval-check.yml",
    }
    assert parse_connection(json.loads(files["retrieval/connection.json"])).backend == backend
    parse_policy(json.loads(files["retrieval/policy.json"]))
    (tmp_path / "q.csv").write_text(files["retrieval/queries.csv"])
    rows = import_rows(read_records(tmp_path / "q.csv"), ColumnMap(tags="tags"))
    assert parse_dataset(render_dataset(rows)).n_labelled == 2
    workflow = files[".github/workflows/retrieval-check.yml"]
    assert "__" not in workflow.replace("__init__", "")
    assert "pull_request_target" not in workflow.replace("never `pull_request_target`", "")
    readme = files["retrieval/README.md"]
    assert f"--backend {backend}" in readme
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(workflow)
    steps = parsed["jobs"]["retrieval-check"]["steps"]
    assert any("vectraudit check" in s.get("run", "") for s in steps)


def test_workflow_secrets_per_backend() -> None:
    assert "QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}" in render_workflow("qdrant")
    assert "RETRIEVAL_PG_DSN: ${{ secrets.RETRIEVAL_PG_DSN }}" in render_workflow("pgvector")
    faiss = render_workflow("faiss")
    assert "[faiss-cpu]" in faiss
    assert "secrets.QDRANT" not in faiss
    assert "secrets.RETRIEVAL_PG_DSN" not in faiss


def test_example_workflow_matches_template() -> None:
    example = (ROOT / "examples" / "github" / "retrieval-check.yml").read_text()
    assert example == render_workflow("qdrant")


def test_init_does_not_overwrite(tmp_path) -> None:
    written, skipped = init_project(tmp_path, "qdrant")
    assert len(written) == 5
    assert skipped == []
    (tmp_path / "retrieval" / "policy.json").write_text("{}")
    written, skipped = init_project(tmp_path, "qdrant")
    assert written == []
    assert len(skipped) == 5
    assert (tmp_path / "retrieval" / "policy.json").read_text() == "{}"
    written, _ = init_project(tmp_path, "qdrant", force=True)
    assert len(written) == 5
    assert (tmp_path / "retrieval" / "policy.json").read_text() != "{}"


def test_init_cli(tmp_path) -> None:
    result = runner.invoke(app, ["init", "--backend", "pgvector", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "created" in result.output
    assert (tmp_path / ".github" / "workflows" / "retrieval-check.yml").exists()
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--backend", "pgvector"])
    assert "kept" in result.output
    result = runner.invoke(app, ["init", "--backend", "milvus", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Unknown backend" in result.output


def test_fresh_project_reaches_a_passing_check(synthetic, tmp_path, monkeypatch) -> None:
    """The pilot path: init, write queries, import, record twice, check."""
    import faiss

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert runner.invoke(app, ["init", "--backend", "faiss", "--dir", str(tmp_path)]).exit_code == 0
    retrieval = tmp_path / "retrieval"
    data = synthetic["vectors"].parent
    conn = json.loads((retrieval / "connection.json").read_text())
    conn.update(index=str(data / "ivf_flat.index"), vectors=str(data / "vectors.npy"))
    (retrieval / "connection.json").write_text(json.dumps(conn))

    vectors = np.load(synthetic["vectors"])
    queries = np.load(synthetic["queries"])[:40]
    _, nn = faiss.knn(queries, vectors, 3)
    with (retrieval / "queries.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "text", "vector", "relevant", "tags"])
        for i, (q, ids) in enumerate(zip(queries, nn, strict=True)):
            relevant = "; ".join(f"{d}:{3 - r}" for r, d in enumerate(ids))
            writer.writerow(
                [
                    f"q{i}",
                    f"question {i}",
                    json.dumps(q.tolist()),
                    relevant,
                    "head" if i < 20 else "tail",
                ]
            )
    dataset = retrieval / "dataset.jsonl"
    steps = [
        [
            "dataset",
            "import",
            str(retrieval / "queries.csv"),
            "--out",
            str(dataset),
            "--vector-column",
            "vector",
            "--tags-column",
            "tags",
        ],
        [
            "record",
            "--connection",
            str(retrieval / "connection.json"),
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "baseline.json"),
            "--repeats",
            "3",
        ],
        [
            "record",
            "--connection",
            str(retrieval / "connection.json"),
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "candidate.json"),
            "--repeats",
            "3",
        ],
    ]
    for args in steps:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args[0], result.output)
    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_path / "baseline.json"),
            str(tmp_path / "candidate.json"),
            "--policy",
            str(retrieval / "policy.json"),
            "--html",
            str(tmp_path / "report.html"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_release_check_demo_synthetic(tmp_path) -> None:
    """The before/after demo runs offline and tells its story."""
    import subprocess
    import sys

    out = tmp_path / "demo"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "release_check_demo.py"),
            "--synthetic",
            "--queries",
            "60",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    verdicts = {
        name: json.loads((out / f"result-{name}.json").read_text())["verdict"]
        for name in ("lower-nprobe", "fix-nprobe", "pq-compression")
    }
    assert verdicts == {"lower-nprobe": "fail", "fix-nprobe": "pass", "pq-compression": "fail"}
    pq = json.loads((out / "result-pq-compression.json").read_text())
    assert any(f["id"] == "ivf-quantization" for f in pq["findings"])
    assert (out / "report-lower-nprobe.html").exists()
    assert "## lower-nprobe: fail" in (out / "SUMMARY.md").read_text()
