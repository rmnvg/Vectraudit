"""The release-check CLI: dataset validate, record, check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_eval_runner import make_dataset
from vectraudit.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def workspace(synthetic, tmp_path_factory) -> dict[str, Path]:
    """A dataset plus connection files for a good baseline and a degraded candidate."""
    out = tmp_path_factory.mktemp("release")
    ds = make_dataset(synthetic, n=40)
    lines = []
    for q in ds.queries:
        assert q.vector is not None
        lines.append(
            json.dumps(
                {
                    "query_id": q.query_id,
                    "vector": q.vector.tolist(),
                    "relevant": q.relevant,
                    "tags": list(q.tags),
                }
            )
        )
    (out / "dataset.jsonl").write_text("\n".join(lines) + "\n")
    data = synthetic["vectors"].parent
    for name, index, params in [
        ("base", "ivf_flat.index", {"nprobe": 16}),
        ("cand", "ivf_pq.index", {"nprobe": 1}),
    ]:
        conn = {"backend": "faiss", "index": str(data / index), "params": params}
        conn["vectors"] = str(data / "vectors.npy")
        (out / f"{name}.json").write_text(json.dumps(conn))
    (out / "broken.json").write_text(
        json.dumps({"backend": "faiss", "index": str(data / "flat_l2.index")})
    )
    (out / "policy.json").write_text(json.dumps({"metrics": {"ndcg": {"max_drop": 0.01}}}))
    return {
        "dir": out,
        "dataset": out / "dataset.jsonl",
        "base": out / "base.json",
        "cand": out / "cand.json",
        "broken": out / "broken.json",
        "policy": out / "policy.json",
    }


def _record(ws: dict[str, Path], conn: str, out: Path, *extra: str):
    args = ["record", "--connection", str(ws[conn]), "--dataset", str(ws["dataset"])]
    return runner.invoke(app, [*args, "--out", str(out), "--k", "5", *extra])


def test_dataset_validate(workspace, tmp_path) -> None:
    result = runner.invoke(app, ["dataset", "validate", str(workspace["dataset"])])
    assert result.exit_code == 0, result.output
    assert "Labelled" in result.output
    result = runner.invoke(app, ["dataset", "validate", str(workspace["dataset"]), "--json"])
    assert json.loads(result.output)["n_queries"] == 40
    (tmp_path / "bad.jsonl").write_text('{"query_id": "q"}\n')
    result = runner.invoke(app, ["dataset", "validate", str(tmp_path / "bad.jsonl")])
    assert result.exit_code == 1
    assert "Line 1" in result.output


def test_record_and_check_end_to_end(workspace, tmp_path, monkeypatch) -> None:
    base, again, cand = tmp_path / "base.json", tmp_path / "again.json", tmp_path / "cand.json"
    result = _record(workspace, "base", base, "--label", "main", "--tag", "embedding=minilm")
    assert result.exit_code == 0, result.output
    assert "complete" in result.output
    assert "ANN recall" in result.output
    run = json.loads(base.read_text())
    assert run["label"] == "main"
    assert run["provenance"]["labels"] == {"embedding": "minilm"}
    assert run["search"]["params"] == {"nprobe": 16}
    assert run["connection"]["backend"] == "faiss"
    assert _record(workspace, "base", again).exit_code == 0
    # --param overrides the connection file's params.
    result = _record(workspace, "cand", cand, "--param", "nprobe=2")
    assert result.exit_code == 0, result.output
    assert json.loads(cand.read_text())["search"]["params"] == {"nprobe": 2}

    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    check = ["check", str(base), str(again), "--policy", str(workspace["policy"])]
    result = runner.invoke(app, check)
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "✅" in summary.read_text()

    reports = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "check",
            str(base),
            str(cand),
            "--policy",
            str(workspace["policy"]),
            "--json",
            str(reports / "result.json"),
            "--html",
            str(reports / "report.html"),
            "--markdown",
            str(reports / "summary.md"),
            "--no-github-summary",
        ],
    )
    assert result.exit_code == 3, result.output
    assert "FAIL" in result.output
    assert json.loads((reports / "result.json").read_text())["verdict"] == "fail"
    assert "<!doctype html>" in (reports / "report.html").read_text()
    assert "Fail" in (reports / "summary.md").read_text()
    assert summary.read_text().count("Retrieval check") == 1


def test_failed_run_is_written_and_checks_as_error(workspace, tmp_path) -> None:
    base, broken = tmp_path / "base.json", tmp_path / "broken.json"
    assert _record(workspace, "base", base).exit_code == 0
    # A dataset whose dimension doesn't match the index fails the whole run.
    bad = tmp_path / "dim.jsonl"
    bad.write_text('{"query_id": "q0", "vector": [1.0, 2.0]}\n')
    result = runner.invoke(
        app,
        [
            "record",
            "--connection",
            str(workspace["broken"]),
            "--dataset",
            str(bad),
            "--out",
            str(broken),
        ],
    )
    assert result.exit_code == 1
    assert "dimension 2" in result.output
    assert json.loads(broken.read_text())["status"] == "failed"
    result = runner.invoke(app, ["check", str(base), str(broken), "--no-github-summary"])
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_check_not_comparable(workspace, tmp_path) -> None:
    base, other = tmp_path / "base.json", tmp_path / "other.json"
    assert _record(workspace, "base", base).exit_code == 0
    assert _record(workspace, "base", other, "--k", "3").exit_code == 0
    result = runner.invoke(app, ["check", str(base), str(other), "--no-github-summary"])
    assert result.exit_code == 4
    assert "different k" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--param", "nprobe"], "key=value"),
        (["--param", "nprobe=fast"], "must be a number"),
        (["--tag", "=x"], "key=value"),
    ],
)
def test_record_argument_errors(workspace, tmp_path, args, message) -> None:
    result = _record(workspace, "base", tmp_path / "x.json", *args)
    assert result.exit_code == 1
    assert message in result.output


def test_record_and_check_input_errors(workspace, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--connection",
            str(tmp_path / "nope.json"),
            "--dataset",
            str(workspace["dataset"]),
            "--out",
            str(tmp_path / "x.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Connection file not found" in result.output
    (tmp_path / "secret.json").write_text('{"backend": "faiss", "index": "x", "api_key": "s"}')
    result = runner.invoke(
        app,
        [
            "record",
            "--connection",
            str(tmp_path / "secret.json"),
            "--dataset",
            str(workspace["dataset"]),
            "--out",
            str(tmp_path / "x.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Secrets must not" in result.output
    result = runner.invoke(app, ["check", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert result.exit_code == 1
    assert "not found" in result.output
    (tmp_path / "p.json").write_text("{}")
    base = tmp_path / "base.json"
    assert _record(workspace, "base", base).exit_code == 0
    empty_policy = ["--policy", str(tmp_path / "p.json")]
    result = runner.invoke(app, ["check", str(base), str(base), *empty_policy])
    assert result.exit_code == 1
    assert "at least one metric" in result.output
