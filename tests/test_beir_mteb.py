"""A BEIR folder as an experiment's inputs, and the MTEB-style export of its results."""

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.experiments import beir as B
from vectraudit.experiments import mteb_export as M
from vectraudit.experiments.corpus import CorpusError
from vectraudit.experiments.manifest import ExperimentError, parse_manifest
from vectraudit.experiments.runner import run_experiment

runner = CliRunner()
TOPICS = {
    "billing": "invoice payment receipt billing statement",
    "reset": "password reset account recovery login",
    "ship": "shipping delivery tracking parcel courier",
}


def beir_folder(tmp_path, *, splits=("test",), extra_docs=()):
    d = tmp_path / "nf"
    (d / "qrels").mkdir(parents=True)
    docs, queries, qrels = [], [], []
    for topic, words in TOPICS.items():
        for i in range(12):
            docs.append(
                {"_id": f"{topic}-{i}", "title": f"{topic} guide {i}", "text": f"{words} no {i}"}
            )
    for topic, words in TOPICS.items():
        for i in range(4):
            first, second, *_ = words.split()
            queries.append({"_id": f"q-{topic}-{i}", "text": f"how do I {first} {second} no {i}"})
            qrels.append(f"q-{topic}-{i}\t{topic}-{i}\t1")
    (d / "corpus.jsonl").write_text("\n".join(map(json.dumps, [*docs, *extra_docs])))
    (d / "queries.jsonl").write_text("\n".join(map(json.dumps, queries)))
    for split in splits:
        (d / "qrels" / f"{split}.tsv").write_text("query-id\tcorpus-id\tscore\n" + "\n".join(qrels))
    return d


def candidate(name, size):
    return {
        "name": name,
        "embedder": {"type": "hashing"},
        "chunking": {"method": "words", "size": size},
    }


def manifest(path, **extra):
    raw = {
        "name": "bakeoff",
        "beir": {"path": "nf", "split": "test"},
        "k": 5,
        "candidates": [
            candidate("hash-50", 50),
            candidate("hash-20", 20),
        ],
        **extra,
    }
    return parse_manifest(raw, path)


# --- the BEIR folder ---------------------------------------------------------------------


def test_documents_use_title_and_text_and_keep_the_title(tmp_path) -> None:
    d = beir_folder(
        tmp_path,
        extra_docs=[
            {"_id": "t-only", "title": "Only a title", "text": ""},
            {"_id": "x-only", "title": "", "text": "only text here"},
        ],
    )
    corpus = B.load_beir_corpus(d)
    by_id = {doc.doc_id: doc for doc in corpus.documents}
    assert (
        by_id["billing-0"].text == "billing guide 0\ninvoice payment receipt billing statement no 0"
    )
    assert by_id["billing-0"].meta == {"title": "billing guide 0"}
    assert by_id["t-only"].text == "Only a title"
    assert by_id["x-only"].text == "only text here"
    assert by_id["x-only"].meta == {}
    assert len(corpus.documents) == 38


def test_a_document_with_nothing_in_it_is_refused_not_dropped(tmp_path) -> None:
    d = beir_folder(tmp_path, extra_docs=[{"_id": "blank", "title": " ", "text": ""}])
    with pytest.raises(CorpusError, match="neither a title nor text") as e:
        B.load_beir_corpus(d)
    assert "dropping it changes the corpus" in str(e.value)


def test_corpus_problems_are_named(tmp_path) -> None:
    d = beir_folder(tmp_path)
    (d / "corpus.jsonl").write_text('{"_id": "a", "text": "x"}\n{"_id": "a", "text": "y"}')
    with pytest.raises(CorpusError, match="duplicate _id"):
        B.load_beir_corpus(d)
    (d / "corpus.jsonl").write_text('{"text": "no id"}')
    with pytest.raises(CorpusError, match="needs an _id"):
        B.load_beir_corpus(d)
    (d / "corpus.jsonl").write_text("not json")
    with pytest.raises(CorpusError, match="not JSON"):
        B.load_beir_corpus(d)
    (d / "corpus.jsonl").unlink()
    with pytest.raises(CorpusError, match=r"No corpus\.jsonl"):
        B.load_beir_corpus(d)


def test_the_queries_are_labelled_by_the_same_importer_as_everywhere_else(tmp_path) -> None:
    corpus, dataset = B.load_beir(beir_folder(tmp_path), "test")
    assert len(dataset.queries) == 12
    q = next(q for q in dataset.queries if q.query_id == "q-reset-2")
    assert q.relevant == {"reset-2": 1.0}
    assert dataset.meta["provenance"]["source_format"] == "beir"
    assert dataset.meta["provenance"]["split"] == "test"
    assert len(corpus.documents) == 36


def test_a_missing_split_names_the_available_ones(tmp_path) -> None:
    d = beir_folder(tmp_path, splits=("dev", "train"))
    with pytest.raises(CorpusError, match="Available splits: dev, train"):
        B.load_beir(d, "test")


# --- the manifest and a real experiment ------------------------------------------------------


def test_the_manifest_takes_a_beir_folder_instead_of_corpus_and_queries(tmp_path) -> None:
    m = manifest(tmp_path)
    assert m.beir_split == "test"
    assert m.corpus_path == m.queries_path == tmp_path / "nf"
    other = manifest(tmp_path, beir={"path": "nf"})
    assert other.beir_split == "test"
    with pytest.raises(ExperimentError, match="not both"):
        manifest(tmp_path, corpus="c.jsonl", queries="q.jsonl")
    with pytest.raises(ExperimentError, match="unknown"):
        manifest(tmp_path, beir={"path": "nf", "shard": 1})
    with pytest.raises(ExperimentError, match=r"beir\.path"):
        manifest(tmp_path, beir={"split": "test"})


def test_an_experiment_runs_on_a_beir_folder(tmp_path) -> None:
    beir_folder(tmp_path)
    summary, results = run_experiment(manifest(tmp_path))
    assert summary["dataset"]["meta"]["name"] == "beir:nf"
    assert summary["corpus"]["documents"] == 36
    assert [r.spec.name for r in results] == ["hash-50", "hash-20"]
    for r in results:
        assert r.run["summary"]["metrics"]["recall"] == 1.0  # a small, separable corpus


# --- the MTEB-style export -----------------------------------------------------------------


@pytest.fixture
def finished(tmp_path):
    beir_folder(tmp_path)
    summary, _ = run_experiment(manifest(tmp_path))
    return summary


def test_the_export_has_the_layout_of_a_published_mteb_result(finished, tmp_path) -> None:
    paths = M.export_experiment(finished, tmp_path / "out", task_name="OurDocs")
    assert sorted(p.parent.name for p in paths) == ["hash-20", "hash-50"]
    body = json.loads(paths[0].read_text())
    # The top-level keys of an MTEB result file (read from a published one), plus our own block.
    assert set(body) == {
        "dataset_revision", "evaluation_time", "kg_co2_emissions", "mteb_version",
        "scores", "task_name", "vectraudit",
    }  # fmt: skip
    assert body["task_name"] == "OurDocs"
    assert list(body["scores"]) == ["test"]
    (score,) = body["scores"]["test"]
    assert score["hf_subset"] == "default"
    assert score["languages"] == []
    assert {
        "main_score",
        "ndcg_at_5",
        "map_at_5",
        "mrr_at_5",
        "precision_at_5",
        "recall_at_5",
    } <= set(score)
    assert "hit_rate_at_5" not in score  # MTEB has no such key
    assert paths[0].name == "OurDocs.json"


def test_the_scores_are_the_experiments_scores(finished, tmp_path) -> None:
    paths = M.export_experiment(finished, tmp_path / "out", task_name="OurDocs", split="dev")
    for path, cand in zip(paths, finished["candidates"], strict=True):
        body = json.loads(path.read_text())
        (score,) = body["scores"]["dev"]
        metrics = cand["summary"]["metrics"]
        assert score["ndcg_at_5"] == metrics["ndcg"]
        assert score["main_score"] == metrics["ndcg"]
        assert score["recall_at_5"] == metrics["recall"]
        assert score["precision_at_5"] == metrics["precision"]
        assert score["mrr_at_5"] == metrics["mrr"]
        assert score["map_at_5"] == metrics["map"]
        assert body["dataset_revision"] == finished["dataset"]["sha256"]
        assert body["evaluation_time"] > 0


def test_the_export_says_it_is_not_mteb_and_not_like_for_like(finished, tmp_path) -> None:
    (path, _) = M.export_experiment(finished, tmp_path / "out", task_name="OurDocs")
    body = json.loads(path.read_text())
    assert body["mteb_version"] is None
    assert body["kg_co2_emissions"] is None
    meta = body["vectraudit"]
    assert "not a public MTEB task" in meta["not_like_for_like"]
    assert meta["main_score_metric"] == "ndcg_at_5"
    assert meta["k"] == 5
    assert meta["candidate"]["name"] == "hash-50"
    assert meta["candidate"]["embedder"]["type"] == "hashing"
    assert meta["n_queries"] == 12
    assert meta["generator"].startswith("vectraudit ")


def test_an_incomplete_or_unlabelled_candidate_is_not_exported(finished, tmp_path) -> None:
    broken = json.loads(json.dumps(finished))
    broken["candidates"][0]["summary"]["n_errors"] = 2
    with pytest.raises(M.MtebExportError, match="2 failed questions"):
        M.export_experiment(broken, tmp_path / "o1", task_name="T")
    unlabelled = json.loads(json.dumps(finished))
    unlabelled["candidates"][1]["summary"]["metrics"] = None
    with pytest.raises(M.MtebExportError, match="no label metrics"):
        M.export_experiment(unlabelled, tmp_path / "o2", task_name="T")
    with pytest.raises(M.MtebExportError, match="task name"):
        M.export_experiment(finished, tmp_path / "o3", task_name="../escape")
    with pytest.raises(M.MtebExportError, match="not a vectraudit experiment"):
        M.export_experiment({"format": "other"}, tmp_path / "o4", task_name="T")


def test_the_command_writes_the_files(finished, tmp_path) -> None:
    (tmp_path / "result.json").write_text(json.dumps(finished))
    done = runner.invoke(
        app,
        ["experiment", "export-mteb", str(tmp_path), "--out", str(tmp_path / "mteb"),
         "--task", "OurDocs"],
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    assert (tmp_path / "mteb" / "hash-50" / "OurDocs.json").is_file()
    assert "Not like for like" in " ".join(done.output.split())
    bad = runner.invoke(
        app,
        ["experiment", "export-mteb", str(tmp_path / "missing"), "--out", str(tmp_path / "x"),
         "--task", "T"],
    )  # fmt: skip
    assert bad.exit_code == 1
