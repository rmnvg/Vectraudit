"""Local labelling privacy, integrity and interoperability."""

import json
import socket

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.experiments.corpus import Document, make_corpus
from vectraudit.labelling.operations import revision
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
    result.run["queries"][0]["repeat_ids"] = [["chunk-id"]]
    save_run(result.run, path)
    with pytest.raises(ValueError, match="source documents"):
        pool(dataset, corpus, runs=[path])


def test_local_ui_csrf_escape_save(inputs, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from vectraudit.labelling.web import create_app

    dataset, corpus = inputs
    task = pool(dataset, corpus, k=2)
    task.queries[0].text = '<script>alert("query")</script>'
    path = tmp_path / "task.json"
    save_task(task, path)
    client = TestClient(create_app(path), base_url="http://127.0.0.1")
    page = client.get("/?annotator=A")
    assert "&lt;script&gt;" in page.text
    assert "default-src 'none'" in page.headers["Content-Security-Policy"]
    assert "run:" not in page.text
    assert "hashing" not in page.text
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    form = {"q": "0", "doc": task.queries[0].documents[0], "annotator": "A", "grade": "3"}
    assert client.post("/judge", data=form).status_code == 403
    import re

    form["csrf"] = re.search(r'name="csrf" value="([^"]+)"', page.text)[1]
    form["version"] = revision(task, task.queries[0].query_id, form["doc"], annotator="A")
    assert client.post("/judge", data=form).status_code == 200
    assert load_task(path).judgments[0].grade == 3
    form["version"] = revision(
        load_task(path), task.queries[0].query_id, form["doc"], annotator="A"
    )
    form.update(start="0", end="99999")
    assert client.post("/judge", data=form).status_code == 400
    form.update(start="0", end="3")
    assert client.post("/judge", data=form).status_code == 200
    assert len(load_task(path).judgments) == 1
    assert load_task(path).judgments[0].end == 3
    form["version"] = revision(
        load_task(path), task.queries[0].query_id, form["doc"], annotator="A"
    )
    form.update(start="", end="", grade="skip")
    assert client.post("/judge", data=form).status_code == 200
    assert load_task(path).judgments[0].grade is None


def test_agreement_resolution_and_sample(inputs):
    from vectraudit.labelling.agreement import agreement, assign_sample, kappa, resolve
    from vectraudit.labelling.task import Judgment
    from vectraudit.labelling.web import judge

    assert kappa([(0, 0), (0, 1), (1, 0), (1, 1)]) == 0
    assert kappa([(0, 0), (1, 1), (2, 2), (3, 3)]) == 1
    assert kappa([(0, 0), (0, 1), (1, 1), (1, 1)]) == 0.5
    assert kappa([(1, 2), (0, 0)], binary=True) == 1
    assert kappa([(1, 1)]) is None
    assert kappa([]) is None
    dataset, corpus = inputs
    task = pool(dataset, corpus, k=1)
    q = task.queries[0]
    for name, grade in [("A", 1), ("B", 3)]:
        judge(
            task, Judgment(query_id=q.query_id, doc_id=q.documents[0], annotator=name, grade=grade)
        )
    assert resolve(task, q.query_id, q.documents[0]) is None
    report = agreement(task)
    assert report["pairs"][0]["n"] == 1
    assert not report["disagreements"][0]["resolved"]
    task.resolutions.append(
        Judgment(query_id=q.query_id, doc_id=q.documents[0], annotator="R", grade=2)
    )
    assert resolve(task, q.query_id, q.documents[0]).grade == 2
    judge(task, Judgment(query_id=q.query_id, doc_id=q.documents[0], annotator="C", grade=3))
    assert resolve(task, q.query_id, q.documents[0]).grade == 3
    assert not task.resolutions
    assert len(assign_sample(task, "B", 0.2)) == 8


def test_export_versioning_and_conflicts(inputs, tmp_path):
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.evaluation.dataset_tools import diff_datasets
    from vectraudit.labelling.export import export_dataset
    from vectraudit.labelling.task import Judgment
    from vectraudit.labelling.web import judge

    dataset, corpus = inputs
    task = pool(dataset, corpus, k=2)
    q = task.queries[0]
    with pytest.raises(ValueError, match="No positive"):
        export_dataset(task)
    judge(
        task,
        Judgment(
            query_id=q.query_id, doc_id=q.documents[0], annotator="A", grade=2, start=0, end=3
        ),
    )
    before = parse_dataset(export_dataset(task))
    assert before.n_labelled == 1
    assert q.documents[1] not in before.queries[0].relevant
    assert before.queries[0].relevant_spans[0].end == 3
    assert before.meta["provenance"]["annotators"] == ["A"]
    judge(task, Judgment(query_id=q.query_id, doc_id=q.documents[1], annotator="A", grade=1))
    after = parse_dataset(export_dataset(task))
    assert before.sha256 != after.sha256
    assert diff_datasets(before, after)
    judge(task, Judgment(query_id=q.query_id, doc_id=q.documents[1], annotator="B", grade=3))
    with pytest.raises(ValueError, match="Resolve disagreement"):
        export_dataset(task)


def test_suggestions_require_human_acceptance(inputs, tmp_path):
    import re

    from fastapi.testclient import TestClient

    from vectraudit.labelling.export import export_dataset
    from vectraudit.labelling.suggestions import suggest, suggestion_agreement
    from vectraudit.labelling.web import create_app

    task = pool(*inputs, k=1)
    suggest(task, lambda q, d: 2, name="fixture judge")
    assert not task.judgments
    with pytest.raises(ValueError, match="No positive"):
        export_dataset(task)
    path = tmp_path / "task.json"
    save_task(task, path)
    client = TestClient(create_app(path), base_url="http://127.0.0.1")
    page = client.get("/?annotator=A")
    assert "Machine suggestion: 2" in page.text
    form = {
        "q": "0",
        "doc": task.queries[0].documents[0],
        "annotator": "A",
        "grade": "accept",
        "version": revision(
            task, task.queries[0].query_id, task.queries[0].documents[0], annotator="A"
        ),
        "suggestion": "2",
        "judge": "fixture judge",
        "csrf": re.search(r'name="csrf" value="([^"]+)"', page.text)[1],
    }
    assert client.post("/judge", data=form).status_code == 200
    saved = load_task(path)
    assert saved.judgments[0].accepted_suggestion
    assert suggestion_agreement(saved) == {"n": 1, "accepted": 1, "grade_agreement": 1.0}
    assert "accepted_suggestions" in export_dataset(saved)
    suggest(saved, lambda q, d: 1, name="new judge")
    save_task(saved, path)
    assert client.post("/judge", data=form).status_code == 409
    assert suggestion_agreement(saved)["grade_agreement"] == 1
    with pytest.raises(ValueError, match="valid integer"):
        suggest(saved, lambda q, d: True, name="bad judge")


def test_llm_loads_only_local_safe_files(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from vectraudit.labelling.suggestions import LocalLLMJudge

    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return object()

    loader = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=loader, AutoTokenizer=loader),
    )
    LocalLLMJudge(tmp_path)
    assert all(call["local_files_only"] and not call["trust_remote_code"] for call in calls)
    assert calls[1]["use_safetensors"]
    with pytest.raises(ValueError, match="existing local"):
        LocalLLMJudge(tmp_path / "missing")


def test_forty_questions_to_experiment_and_check_offline(inputs, tmp_path, monkeypatch):
    import re

    from fastapi.testclient import TestClient

    from vectraudit.labelling.web import create_app

    def forbidden(*args, **kwargs):
        raise AssertionError("Outbound network access in local workflow")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    _, corpus = inputs
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps({"doc_id": d.doc_id, "text": d.text}) for d in corpus.documents)
    )
    manifest = {
        "name": "acceptance",
        "corpus": "corpus.jsonl",
        "queries": "dataset.jsonl",
        "k": 2,
        "candidates": [
            {
                "name": f"lexical-{size}",
                "embedder": {"type": "hashing"},
                "chunking": {"size": size, "overlap": 0},
            }
            for size in (20, 40, 80)
        ],
    }
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(manifest))
    path = tmp_path / "task.json"
    runner = CliRunner()

    def invoke(args):
        result = runner.invoke(app, list(map(str, args)))
        assert result.exit_code == 0, result.output or repr(result.exception)

    invoke(
        [
            "label",
            "pool",
            tmp_path / "questions.csv",
            corpus_path,
            "--out",
            path,
            "--manifest",
            manifest_path,
            "--no-lexical",
            "--k",
            "2",
        ]
    )
    task = load_task(path)
    assert len(task.sources) == 3
    client = TestClient(create_app(path), base_url="http://127.0.0.1")
    page = client.get("/?annotator=Human")
    csrf = re.search(r'name="csrf" value="([^"]+)"', page.text)[1]
    for i, q in enumerate(task.queries):
        # Simulated human input for interoperability, never a product labelling heuristic.
        assert (
            client.post(
                "/judge",
                data={
                    "csrf": csrf,
                    "annotator": "Human",
                    "q": i,
                    "doc": q.documents[0],
                    "grade": 2,
                    "version": revision(task, q.query_id, q.documents[0], annotator="Human"),
                },
            ).status_code
            == 200
        )
    invoke(["label", "export", path, "--out", tmp_path / "dataset.jsonl"])
    invoke(["dataset", "validate", tmp_path / "dataset.jsonl"])
    invoke(["experiment", "run", manifest_path, "--out", tmp_path / "results", "--no-cache"])
    run = tmp_path / "results/runs/lexical-20.json"
    invoke(["check", run, run, "--json", tmp_path / "check.json"])
    invoke(
        [
            "label",
            "pool",
            tmp_path / "questions.csv",
            corpus_path,
            "--out",
            tmp_path / "again.json",
            "--experiment",
            tmp_path / "results",
            "--no-lexical",
            "--k",
            "2",
        ]
    )
    assert len(load_task(tmp_path / "again.json").sources) == 3


def test_span_export_survives_embedding_and_diff():
    import numpy as np

    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.evaluation.dataset_tools import diff_datasets, embed_dataset

    text = (
        '{"query_id":"q","text":"answer","relevant":{"doc":2},'
        '"relevant_spans":[{"doc_id":"doc","start":0,"end":3,"grade":2}]}'
    )
    original = parse_dataset(text)
    embedded = parse_dataset(
        embed_dataset(
            original, lambda texts: np.ones((len(texts), 8)), model="fixture", normalized=False
        )
    )
    assert original.queries[0].relevant_spans == embedded.queries[0].relevant_spans
    changed = parse_dataset(text.replace('"end":3', '"end":4'))
    assert diff_datasets(original, changed)["changed"] == {"q": ["relevant_spans"]}


def test_existing_dataset_hashes_unchanged():
    import hashlib

    from vectraudit.evaluation.dataset import parse_dataset

    # Pre-Phase-20 canonical representation: provenance absent means identical bytes.
    row = {
        "query_id": "q",
        "vector": None,
        "text": "answer",
        "relevant": {"doc": 2.0},
        "filter": {},
        "tags": [],
    }
    for meta in ({}, {"name": "legacy"}):
        canonical = {"meta": meta, "queries": [row]} if meta else [row]
        expected = hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        text = (json.dumps({"dataset": meta}) + "\n" if meta else "") + json.dumps(row)
        assert parse_dataset(text).sha256 == expected


def test_second_pass_is_blind_and_progress_is_sampled(inputs, tmp_path):
    from fastapi.testclient import TestClient

    from vectraudit.labelling.agreement import assign_sample
    from vectraudit.labelling.task import Judgment
    from vectraudit.labelling.web import create_app, judge

    task = pool(*inputs, k=1)
    ids = assign_sample(task, "B", 0.2)
    for q in task.queries:
        judge(task, Judgment(query_id=q.query_id, doc_id=q.documents[0], annotator="A", grade=3))
    path = tmp_path / "task.json"
    save_task(task, path)
    client = TestClient(create_app(path), base_url="http://127.0.0.1")
    page = client.get("/?annotator=B")
    assert "0/8 judged" in page.text
    assert "Saved grade 3" not in page.text
    assert "Question 1/8" in page.text
    first = next(q for q in task.queries if q.query_id in ids)
    assert first.text in page.text
