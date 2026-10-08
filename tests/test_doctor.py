"""Preflight catches onboarding errors without sending customer data."""

import json
import socket

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_stats
from vectraudit.evaluation.doctor import diagnose


@pytest.fixture
def setup(tmp_path):
    connection = tmp_path / "connection.json"
    connection.write_text(
        json.dumps({"backend": "faiss", "index": "test.index", "vectors": "vectors.npy"})
    )
    (tmp_path / "test.index").write_bytes(b"not loaded by doctor")
    np.save(tmp_path / "vectors.npy", np.ones((4, 8), dtype=np.float32))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": f"q{i}",
                    "vector": [1.0] * 8,
                    "text": "PRIVATE QUESTION",
                    "relevant": {"doc": 2},
                }
            )
            for i in range(40)
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"min_queries": 30, "metrics": {"ndcg": {"max_drop": 0.02}}}))
    return connection, dataset, policy


def test_doctor_offline_and_no_payload_in_report(setup, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access in offline doctor")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    report = diagnose(connection=setup[0], dataset=setup[1], policy=setup[2])
    assert report["offline"]
    assert report["status"] != "blocked"
    assert "PRIVATE QUESTION" not in json.dumps(report)
    assert report["dataset"]["n_labelled"] == 40
    assert any(c["code"] == "dataset.duplicates" for c in report["checks"])


def test_preflight_blocks_sparse_labels_and_mismatched_dimensions(setup):
    dataset = setup[1]
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    for row in rows[2:]:
        row.pop("relevant")
    rows[0]["vector"] = [1.0] * 7
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    assert diagnose(dataset=dataset)["status"] == "blocked"
    for row in rows:
        row["vector"] = [1.0] * 7
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    report = diagnose(connection=setup[0], dataset=dataset, policy=setup[2])
    codes = {c["code"] for c in report["checks"] if c["status"] == "error"}
    assert codes == {"labels.minimum", "vectors.dimension"}


def test_preflight_counts_families_not_scenario_rows(setup):
    rows = [json.loads(line) for line in setup[1].read_text().splitlines()]
    for row in rows:
        row["family_id"] = "one-question"
    setup[1].write_text("\n".join(map(json.dumps, rows)))
    report = diagnose(connection=setup[0], dataset=setup[1], policy=setup[2])
    assert report["dataset"]["n_families"] == 1
    codes = {c["code"] for c in report["checks"] if c["status"] == "error"}
    assert {"dataset.minimum", "labels.minimum"} <= codes


def test_secret_presence_without_value_or_env_name(tmp_path, monkeypatch):
    connection = tmp_path / "connection.json"
    connection.write_text(
        json.dumps(
            {"backend": "pinecone", "index": "my-index", "api_key_env": "PRIVATE_VARIABLE_NAME"}
        )
    )
    monkeypatch.setenv("PRIVATE_VARIABLE_NAME", "VERY_SECRET_TOKEN")
    report = diagnose(connection=connection)
    assert "VERY_SECRET_TOKEN" not in json.dumps(report)
    assert "PRIVATE_VARIABLE_NAME" not in json.dumps(report)
    assert report["status"] != "blocked"
    monkeypatch.delenv("PRIVATE_VARIABLE_NAME")
    assert diagnose(connection=connection)["status"] == "blocked"


def test_cli_json_and_strict_warning_exit(setup):
    args = ["doctor", "--dataset", str(setup[1]), "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "needs_attention"
    assert CliRunner().invoke(app, [*args, "--strict"]).exit_code == 1
    assert (
        CliRunner().invoke(app, ["doctor", "--dataset", "missing.jsonl", "--json"]).exit_code == 1
    )


def test_span_only_dataset_counts_as_labelled():
    ds = parse_dataset(
        '{"query_id":"q","text":"question","relevant_spans":'
        '[{"doc_id":"a","start":0,"end":5,"grade":2}]}'
    )
    stats = dataset_stats(ds)
    assert stats["n_labelled"] == 1
    assert stats["relevant_per_query"]["max"] == 1


def test_doctor_rejects_credentials_embedded_in_url(tmp_path):
    path = tmp_path / "connection.json"
    path.write_text(
        json.dumps(
            {
                "backend": "qdrant",
                "collection": "docs",
                "url": "http://private:SECRET@example.test:6333",
            }
        )
    )
    report = diagnose(connection=path)
    assert report["status"] == "blocked"
    assert "SECRET" not in json.dumps(report)
