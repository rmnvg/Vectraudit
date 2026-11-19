"""RAGChecker's input from a run: what the model was given, with text from the user."""

from __future__ import annotations

import json
import stat
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.document_lookup import LookupConfig
from vectraudit.pipelines.ragchecker_input import (
    CorpusSource,
    LookupSource,
    RagCheckerInputError,
    build_input,
    context_documents,
    read_texts,
    render,
)

TEXTS = {"a": "Alpha text.", "b": "Beta text.", "c": "Gamma text.", "m": "Context-only text."}


def dataset(**second):
    rows = [
        {"query_id": "q0", "text": "first question", "relevant": {"a": 1}},
        {"query_id": "q1", "text": "second question", "relevant": {"b": 1}, **second},
        {"query_id": "q2", "text": "held out", "relevant": {"a": 1}, "split": "heldout"},
    ]
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def run_for(ds, **query):
    run = fake_run([1.0, 1.0, 1.0], run_id="run-1")
    run["dataset"]["sha256"] = ds.sha256
    run["search"]["k"] = run["metric_contract"]["k"] = 3
    for q, ids in zip(run["queries"], (["a", "b"], ["b", "c"], ["a"]), strict=True):
        q["ids"] = ids
    run["queries"][1] |= query
    return run


class Corpus:
    def __init__(self, texts=TEXTS):
        self.texts_by_id = texts

    def texts(self, query, doc_ids, run):
        found = {d: self.texts_by_id[d] for d in doc_ids if d in self.texts_by_id}
        return found if len(found) == len(set(doc_ids)) else None


REFERENCES = {"q0": "ref 0", "q1": "ref 1", "q2": "ref 2"}
ANSWERS = {"q0": "ans 0", "q1": "ans 1", "q2": "ans 2"}


def build(ds=None, run=None, references=REFERENCES, answers=ANSWERS, source=None):
    ds = ds or dataset()
    return build_input(run or run_for(ds), ds, references, answers, source or Corpus())


def test_each_question_gets_its_text_reference_response_and_context_in_order():
    data, counts = build()
    first, second = data["results"]
    assert first == {
        "query_id": "q0",
        "query": "first question",
        "gt_answer": "ref 0",
        "response": "ans 0",
        "retrieved_context": [
            {"doc_id": "a", "text": "Alpha text."},
            {"doc_id": "b", "text": "Beta text."},
        ],
    }
    assert [c["doc_id"] for c in second["retrieved_context"]] == ["b", "c"]
    assert counts["exported"] == 2
    assert counts["context_from_delivered"] == 2


def test_held_out_questions_are_never_exported():
    data, counts = build()
    assert {r["query_id"] for r in data["results"]} == {"q0", "q1"}
    assert counts["skipped_heldout"] == 1


def test_the_observed_model_context_is_the_context_when_the_run_recorded_one():
    context = {
        "coverage": "observed",
        "items": [{"id": "m", "truncated": True}, {"id": "b", "truncated": False}, {"id": "m"}],
    }
    data, counts = build(run=run_for(dataset(), model_context=context))
    assert [c["doc_id"] for c in data["results"][1]["retrieved_context"]] == ["m", "b"]
    assert counts["context_from_model_context"] == 1
    assert counts["truncated_context_items"] == 1
    unobserved = {"coverage": "partial", "items": [{"id": "m"}]}
    data, _ = build(run=run_for(dataset(), model_context=unobserved))
    assert [c["doc_id"] for c in data["results"][1]["retrieved_context"]] == ["b", "c"]
    assert context_documents({"ids": ["x", "x", "y"]}) == (["x", "y"], "delivered")


@pytest.mark.parametrize(
    ("change", "counter"),
    [
        ({"references": {"q0": "r"}}, "skipped_no_reference"),
        ({"answers": {"q0": "a"}}, "skipped_no_response"),
        ({"source": Corpus({"a": "A", "b": "B"})}, "skipped_incomplete_context"),
    ],
)
def test_a_question_missing_anything_is_skipped_and_counted_never_exported_partly(change, counter):
    data, counts = build(**change)
    # q1 lacks the reference, the response or the text of context document "c".
    assert [r["query_id"] for r in data["results"]] == ["q0"]
    assert counts[counter] == 1


def test_failed_textless_and_contextless_questions_are_skipped():
    ds = dataset()
    errored = run_for(ds, error={"code": "x", "message": "boom"})
    assert build(ds, errored)[1]["skipped_errored"] == 1
    nothing = run_for(ds, ids=[])
    assert build(ds, nothing)[1]["skipped_no_context"] == 1
    vectors = parse_dataset(
        '{"query_id": "q0", "vector": [1, 0], "relevant": {"a": 1}}\n'
        '{"query_id": "q1", "text": "second question", "relevant": {"b": 1}}'
    )
    run = fake_run([1.0, 1.0])
    run["dataset"]["sha256"] = vectors.sha256
    assert build_input(run, vectors, REFERENCES, ANSWERS, Corpus())[1]["skipped_no_text"] == 1


def test_a_run_of_another_dataset_or_with_nothing_to_export_is_refused():
    ds = dataset()
    other = run_for(ds)
    other["dataset"]["sha256"] = "0" * 64
    with pytest.raises(RagCheckerInputError, match="different dataset"):
        build_input(other, ds, REFERENCES, ANSWERS, Corpus())
    with pytest.raises(RagCheckerInputError, match="No question could be exported"):
        build(references={}, answers={})
    stray = run_for(ds)
    stray["queries"][0]["query_id"] = "unknown"
    with pytest.raises(RagCheckerInputError, match="not in --dataset"):
        build_input(stray, ds, REFERENCES, ANSWERS, Corpus())


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_corpus_and_text_files_are_read_strictly(tmp_path):
    corpus = write(
        tmp_path / "c.jsonl", [{"doc_id": "a", "text": "x"}, {"doc_id": "b", "text": "y"}]
    )
    assert CorpusSource(corpus).documents == {"a": "x", "b": "y"}
    for rows, message in [
        ([{"doc_id": "a", "text": "x"}, {"doc_id": "a", "text": "y"}], "repeats"),
        ([{"doc_id": "a"}], "needs a doc_id and a text"),
        ([{"doc_id": "", "text": "x"}], "needs a doc_id and a text"),
    ]:
        with pytest.raises(RagCheckerInputError, match=message):
            CorpusSource(write(tmp_path / "bad.jsonl", rows))
    (tmp_path / "junk.jsonl").write_text("not json")
    with pytest.raises(RagCheckerInputError, match="not JSON"):
        CorpusSource(tmp_path / "junk.jsonl")
    with pytest.raises(RagCheckerInputError, match="Cannot read"):
        CorpusSource(tmp_path / "missing.jsonl")
    good = write(tmp_path / "r.jsonl", [{"query_id": "q0", "reference": "ok"}])
    assert read_texts(good, "reference") == {"q0": "ok"}
    for rows, field, message in [
        ([{"query_id": "q0", "answer": "x"}], "reference", "reference must be"),
        ([{"reference": "x"}], "reference", "needs a query_id"),
        ([{"query_id": "q", "reference": " "}], "reference", "reference must be"),
        ([{"query_id": "q", "reference": "x"}] * 2, "reference", "repeats"),
        ([], "reference", "contains no rows"),
    ]:
        with pytest.raises(RagCheckerInputError, match=message):
            read_texts(write(tmp_path / "t.jsonl", rows), field)


def test_the_summary_warns_that_the_file_holds_private_text():
    _, counts = build()
    text = render(counts, "ragchecker_input.json")
    assert "Wrote 2 questions" in text
    assert "contains question, answer and document text" in text
    assert "1 questions skipped: held-out (never exported)" in text
    assert "use the delivered documents as the context" in text


def files(tmp_path, ds, run):
    corpus = write(tmp_path / "corpus.jsonl", [{"doc_id": d, "text": t} for d, t in TEXTS.items()])
    refs = write(
        tmp_path / "refs.jsonl", [{"query_id": q, "reference": r} for q, r in REFERENCES.items()]
    )
    answers = write(
        tmp_path / "ans.jsonl", [{"query_id": q, "answer": a} for q, a in ANSWERS.items()]
    )
    dataset_path = write(
        tmp_path / "benchmark.jsonl",
        [
            {"query_id": "q0", "text": "first question", "relevant": {"a": 1}},
            {"query_id": "q1", "text": "second question", "relevant": {"b": 1}},
            {"query_id": "q2", "text": "held out", "relevant": {"a": 1}, "split": "heldout"},
        ],
    )
    run_path = tmp_path / "run.json"
    run["dataset"]["sha256"] = parse_dataset(dataset_path.read_text()).sha256
    run_path.write_text(json.dumps(run))
    return corpus, refs, answers, dataset_path, run_path


def test_the_cli_writes_ragchecker_input_privately_and_says_so(tmp_path):
    ds = dataset()
    corpus, refs, answers, dataset_path, run_path = files(tmp_path, ds, run_for(ds))
    out = tmp_path / "out" / "ragchecker_input.json"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "ragchecker-input", str(run_path), "--dataset", str(dataset_path),
            "--references", str(refs), "--answers", str(answers), "--corpus", str(corpus),
            "--out", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 2 questions" in result.output
    assert "Keep it on this machine" in result.output
    data = json.loads(out.read_text())
    assert [r["query_id"] for r in data["results"]] == ["q0", "q1"]
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_the_cli_needs_exactly_one_text_source_and_a_resolver_in_the_manifest(tmp_path):
    ds = dataset()
    corpus, refs, answers, dataset_path, run_path = files(tmp_path, ds, run_for(ds))
    base = [
        "workbench", "ragchecker-input", str(run_path), "--dataset", str(dataset_path),
        "--references", str(refs), "--answers", str(answers), "--out", str(tmp_path / "o.json"),
    ]  # fmt: skip
    neither = CliRunner().invoke(app, base)
    assert neither.exit_code == 2
    assert "--corpus or --manifest" in neither.output
    both = CliRunner().invoke(app, [*base, "--corpus", str(corpus), "--manifest", "m.json"])
    assert both.exit_code == 2
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "benchmark.jsonl",
                "candidates": [
                    {"name": "c", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
                ],
            }
        )
    )
    unregistered = CliRunner().invoke(app, [*base, "--manifest", str(manifest)])
    assert unregistered.exit_code == 2
    assert "registers no document_lookup resolver" in unregistered.output


def test_the_registered_resolver_supplies_text_scoped_to_each_question(tmp_path, monkeypatch):
    """The same resolver the passage review uses: tenant, filters and version are passed on."""
    (tmp_path / "docs_resolver.py").write_text(
        textwrap.dedent(
            """
            from vectraudit.pipelines.document_lookup import (
                LookupDocument, LookupRequest, LookupResult,
            )

            TEXT = {"a": "Alpha from the resolver.", "b": "Beta from the resolver."}

            def lookup(request: LookupRequest) -> LookupResult:
                tenant = request.context.get("tenant")
                docs = []
                for doc in request.doc_ids:
                    if tenant == "acme" and doc in TEXT:
                        docs.append(LookupDocument(
                            doc_id=doc, status="found", text=TEXT[doc], version="v1"))
                    else:
                        docs.append(LookupDocument(doc_id=doc, status="forbidden"))
                return LookupResult(corpus_version=request.corpus_version, documents=docs)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    ds = parse_dataset(
        '{"query_id": "q0", "text": "first question", "context": {"tenant": "acme"}, '
        '"relevant": {"a": 1}}\n'
        '{"query_id": "q1", "text": "second question", "context": {"tenant": "other"}, '
        '"relevant": {"b": 1}}'
    )
    run = fake_run([1.0, 1.0], run_id="run-1")
    run["dataset"]["sha256"] = ds.sha256
    for q in run["queries"]:
        q["ids"] = ["a", "b"]
    source = LookupSource(LookupConfig(target="docs_resolver:lookup", timeout_s=10), tmp_path)
    data, counts = build_input(run, ds, REFERENCES, ANSWERS, source)
    assert [r["query_id"] for r in data["results"]] == ["q0"]
    assert data["results"][0]["retrieved_context"][0]["text"] == "Alpha from the resolver."
    assert counts["skipped_incomplete_context"] == 1  # the other tenant may not read them
