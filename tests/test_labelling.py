"""Local labelling privacy, integrity and interoperability."""

import json
import socket

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.experiments.corpus import Document, make_corpus
from vectraudit.labelling.pooling import load_questions, pool
from vectraudit.labelling.task import load_task, save_task


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text("text\n" + "\n".join(f"question {i}" for i in range(40)))
    corpus = make_corpus([Document(str(i), f"question {i}") for i in range(15)])
    return load_questions(path), corpus


def test_pool_offline_and_integrity(inputs, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access in local mode")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    dataset, corpus = inputs
    task = pool(dataset, corpus, k=10, seed=1)
    other = pool(dataset, corpus, k=10, seed=2)
    assert len(task.queries) == 40
    assert task.queries[0].documents != other.queries[0].documents
    assert set(task.queries[0].documents) == set(other.queries[0].documents)
    assert all(len(set(q.documents)) == 10 for q in task.queries)
    path = tmp_path / "task.json"
    save_task(task, path)
    assert load_task(path) == task
    raw = json.loads(path.read_text())
    raw["name"] = "tampered"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_task(path)


def test_pool_cli(inputs, tmp_path):
    _, corpus = inputs
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps({"doc_id": d.doc_id, "text": d.text}) for d in corpus.documents)
    )
    args = [
        "label",
        "pool",
        str(tmp_path / "questions.csv"),
        str(path),
        "--out",
        str(tmp_path / "task.json"),
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    assert CliRunner().invoke(app, args).exit_code == 1


def test_pool_candidates_and_runs(inputs, tmp_path):
    from vectraudit.evaluation.runfile import save_run
    from vectraudit.experiments.chunking import ChunkingSpec
    from vectraudit.experiments.embedders import EmbedderSpec
    from vectraudit.experiments.manifest import CandidateSpec
    from vectraudit.experiments.runner import run_candidate

    dataset, corpus = inputs
    result = run_candidate(
        CandidateSpec("lexical", EmbedderSpec(type="hashing"), ChunkingSpec()),
        corpus,
        dataset,
        3,
        experiment="test",
    )
    path = save_run(result.run, tmp_path / "run.json")
    task = pool(dataset, corpus, runs=[path, path], lexical=False, k=3)
    assert all(len(q.documents) == 3 for q in task.queries)
    result.run["queries"][0]["ids"] = ["chunk-id"]
    save_run(result.run, path)
    with pytest.raises(ValueError, match="source documents"):
        pool(dataset, corpus, runs=[path])
