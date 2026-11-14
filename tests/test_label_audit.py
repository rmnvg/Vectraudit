"""The label audit finds suspect labels for review and never changes one by itself."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectraudit.evaluation.dataset import load_dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.evaluation.label_audit import (
    CorpusVectors,
    LabelAuditError,
    apply_label_decisions,
    audit_labels,
    queue_items,
)
from vectraudit.labelling.task import Judgment, Question, Task

ROOT = Path(__file__).parents[1]


def dataset(rows: list[dict[str, Any]], meta: dict[str, Any] | None = None):
    return parse_dataset(render_dataset(rows, meta))


def run(ds, rankings: dict[str, list[str]], run_id: str, k: int = 5) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "label": run_id,
        "dataset": {"sha256": ds.sha256},
        "search": {"k": k},
        "timing": {"source": "measured"},
        "status": "complete",
        "queries": [{"query_id": q, "ids": ids, "error": None} for q, ids in rankings.items()],
    }


ROWS = [
    {"query_id": "q1", "text": "refund window", "relevant": {"a": 2, "b": 1, "z": 1}},
    {"query_id": "q2", "text": "late fee", "relevant": {"c": 1}},
    {"query_id": "h1", "text": "held out", "relevant": {"a": 1, "b": 1}, "split": "heldout"},
]
# Every run puts a and b on top and a high unjudged document m beside them; z (labelled
# relevant) is never returned; x and y are ordinary non-relevant results.
RANKINGS = {
    "q1": ["a", "m", "b", "x", "y"],
    "q2": ["c", "x"],
    "h1": ["a", "m", "b", "x", "y"],
}


def two_runs(ds):
    return [run(ds, RANKINGS, "r1"), run(ds, {**RANKINGS, "q1": ["m", "a", "b", "y", "x"]}, "r2")]


def test_rank_signal_flags_a_missing_label_and_a_label_no_run_supports():
    ds = dataset(ROWS)
    report = audit_labels(ds, two_runs(ds))
    assert (report["format"], report["format_version"]) == ("vectraudit.label_audit", 1)
    found = {(s["query_id"], s["doc_id"]): s for s in report["suspects"]}
    assert set(found) == {("q1", "m"), ("q1", "z")}
    assert found[("q1", "m")]["kind"] == "label_may_be_missing"
    assert found[("q1", "m")]["label"] == {"state": "unjudged", "grade": None}
    assert found[("q1", "z")]["kind"] == "label_may_be_wrong"
    assert found[("q1", "z")]["signals"][0]["returned_by"] == 0
    assert "every run misses it" in found[("q1", "z")]["reason"]
    assert all(s["evidence"] == "hypothesis" for s in report["suspects"])
    # q2 has one relevant label: no stable threshold, so it is skipped and says why.
    assert report["signals"]["rank"]["questions_skipped"] == {"fewer than two relevant labels": 1}
    assert report["signals"]["rank"]["questions_audited"] == 1


def test_heldout_questions_and_their_repeats_stay_out_unless_asked():
    rows = [*ROWS, {"query_id": "q3", "text": "Held  OUT", "relevant": {"a": 1, "m": 1}}]
    ds = dataset(rows)
    report = audit_labels(ds, two_runs(ds))
    assert not any(s["query_id"] in ("h1", "q3") for s in report["suspects"])
    assert report["excluded"] == {
        "heldout_questions": 1,
        "exploratory_repeats_of_heldout": 1,
        "already_reviewed": 0,
    }
    included = audit_labels(ds, two_runs(ds), include_heldout=True)
    assert any(s["query_id"] == "h1" for s in included["suspects"])
    # The queue never carries held-out questions, whatever the report holds.
    assert not any(i["query_id"] == "h1" for i in queue_items(included, ds))


def test_runs_of_another_version_are_ignored_and_one_run_measures_nothing():
    ds = dataset(ROWS)
    other = dataset([*ROWS[:2], {"query_id": "q9", "text": "new", "relevant": {"a": 1}}])
    report = audit_labels(ds, [*two_runs(ds), run(other, RANKINGS, "old")])
    assert report["runs_ignored"] == 1
    assert len(report["runs"]) == 2
    with pytest.raises(LabelAuditError, match="No signal could be measured"):
        audit_labels(ds, [run(ds, RANKINGS, "r1")])
    # The same run given twice counts once.
    with pytest.raises(LabelAuditError, match="1 usable runs"):
        audit_labels(ds, [run(ds, RANKINGS, "r1"), run(ds, RANKINGS, "r1")])
    with pytest.raises(LabelAuditError, match="Nothing to audit"):
        audit_labels(ds, [])
    with pytest.raises(LabelAuditError, match="at least two runs"):
        audit_labels(ds, two_runs(ds), min_runs=1)


def test_a_question_every_run_misses_is_not_blamed_on_its_labels():
    ds = dataset([{"query_id": "q1", "text": "x", "relevant": {"a": 1, "b": 1}}])
    runs = [run(ds, {"q1": ["x", "y"]}, "r1"), run(ds, {"q1": ["y", "x"]}, "r2")]
    report = audit_labels(ds, runs)
    assert report["suspects"] == []
    assert report["signals"]["rank"]["questions_skipped"] == {
        "no run returns a relevant document": 1
    }


def test_forbidden_documents_are_never_suspects():
    rows = [{"query_id": "q1", "text": "x", "relevant": {"a": 1, "b": 1}, "forbidden": ["m"]}]
    ds = dataset(rows)
    rankings = {"q1": ["m", "a", "b", "x"]}
    report = audit_labels(ds, [run(ds, rankings, "r1"), run(ds, rankings, "r2")])
    assert not any(s["doc_id"] == "m" for s in report["suspects"])


def test_near_duplicates_labelled_differently_are_listed():
    rows = [
        {"query_id": "q1", "text": "x", "relevant": {"a": 1, "b": 0, "c": 1}},
        {"query_id": "q2", "text": "y", "relevant": {"c": 1}},
    ]
    ds = dataset(rows)
    base = np.eye(4, dtype=np.float32)
    vectors = np.vstack([base[0], base[0] + 0.001, base[1], base[1] + 0.001, base[2]])
    corpus = CorpusVectors(["a", "b", "c", "c2", "e"], vectors)
    report = audit_labels(ds, corpus=corpus)
    found = {(s["query_id"], s["doc_id"]): s for s in report["suspects"]}
    # a (relevant) and b (judged 0) are duplicates: one label is wrong, both are listed.
    assert {("q1", "a"), ("q1", "b")} <= set(found)
    assert found[("q1", "b")]["signals"][0]["duplicate_of"] == "a"
    # c is relevant and its duplicate c2 is unjudged: c2 may be relevant too.
    assert found[("q1", "c2")]["kind"] == "label_may_be_missing"
    assert found[("q2", "c2")]["kind"] == "label_may_be_missing"
    assert report["signals"]["near_duplicate"]["pairs"] == 3
    assert report["signals"]["rank"]["status"] == "not_measured"
    assert report["summary"]["by_signal"]["near_duplicate"] == len(report["suspects"])


def test_annotator_splits_without_a_reviewer_are_listed():
    ds = dataset([{"query_id": "q1", "text": "x", "relevant": {"a": 1, "b": 1}}])
    task = Task(
        name="t",
        corpus_sha256="0" * 64,
        sources=["s"],
        documents={"a": "alpha", "b": "beta", "c": "gamma"},
        queries=[Question(query_id="q1", text="x", documents=["a", "b", "c"])],
        judgments=[
            Judgment(query_id="q1", doc_id="a", annotator="p", grade=2),
            Judgment(query_id="q1", doc_id="a", annotator="q", grade=0),
            Judgment(query_id="q1", doc_id="a", annotator="r", grade=1),
            Judgment(query_id="q1", doc_id="b", annotator="p", grade=1),
            Judgment(query_id="q1", doc_id="b", annotator="q", grade=2),
            Judgment(query_id="q1", doc_id="c", annotator="p", grade=0),
            Judgment(query_id="q1", doc_id="c", annotator="q", grade=1),
        ],
        resolutions=[Judgment(query_id="q1", doc_id="c", annotator="lead", grade=0)],
    )
    report = audit_labels(ds, task=task)
    assert [(s["doc_id"], s["signals"][0]["relevant_votes"]) for s in report["suspects"]] == [
        ("a", 2)
    ]
    assert report["signals"]["annotators"]["items_with_several_judgments"] == 3
    other = Task(
        name="t",
        corpus_sha256="0" * 64,
        sources=["s"],
        documents={"a": "alpha"},
        queries=[Question(query_id="elsewhere", text="x", documents=["a"])],
    )
    with pytest.raises(LabelAuditError, match="shares no question"):
        audit_labels(ds, task=other)


def test_the_report_holds_no_text():
    rows = [{**r, "text": f"secret question {i}"} for i, r in enumerate(ROWS)]
    ds = dataset(rows)
    blob = json.dumps(audit_labels(ds, two_runs(ds)))
    assert "secret" not in blob


def test_suspects_enter_the_benchmark_queue_for_this_version_only():
    from vectraudit.pipelines.benchmark import audit_benchmark

    ds = dataset(ROWS)
    runs = two_runs(ds)
    report = audit_labels(ds, runs)
    queue = audit_benchmark(ds, runs, label_audit=report)["judgment_queue"]
    by_key = {(i["query_id"], i["doc_id"]): i for i in queue}
    assert by_key[("q1", "z")]["source"] == "label_audit"
    # m was already queued as an unjudged pooled result; the audit raises its priority.
    assert by_key[("q1", "m")]["priority"] == report["suspects"][0]["priority"]
    assert queue[0]["source"] == "label_audit"
    other = dataset([*ROWS, {"query_id": "q9", "text": "n", "relevant": {"a": 1}}])
    with pytest.raises(LabelAuditError, match="another version"):
        queue_items(report, other)


def test_reviewer_decisions_make_a_new_version_and_are_remembered():
    ds = dataset(ROWS)
    report = audit_labels(ds, two_runs(ds))
    text, recorded = apply_label_decisions(
        ds,
        report,
        [
            {"query_id": "q1", "doc_id": "m", "decision": 2},
            {"query_id": "q1", "doc_id": "z", "decision": "keep"},
        ],
        reviewer="Dana",
    )
    new = parse_dataset(text)
    assert new.sha256 != ds.sha256
    q1 = next(q for q in new.queries if q.query_id == "q1")
    assert q1.relevant == {"a": 2.0, "b": 1.0, "z": 1.0, "m": 2.0}
    assert [(r["doc_id"], r["decision"], r["from"], r["to"]) for r in recorded] == [
        ("m", "changed", None, 2.0),
        ("z", "kept", 1.0, 1.0),
    ]
    assert new.meta["provenance"]["label_review"][1]["reviewer"] == "Dana"
    # A later audit does not raise decided pairs again, unless asked to recheck them.
    again = audit_labels(new, [run(new, RANKINGS, "r3"), run(new, RANKINGS, "r4")])
    assert ("q1", "z") not in {(s["query_id"], s["doc_id"]) for s in again["suspects"]}
    assert again["excluded"]["already_reviewed"] >= 1
    rechecked = audit_labels(
        new, [run(new, RANKINGS, "r3"), run(new, RANKINGS, "r4")], recheck_reviewed=True
    )
    assert ("q1", "z") in {(s["query_id"], s["doc_id"]) for s in rechecked["suspects"]}


def test_decisions_are_explicit_and_refuse_what_would_break_the_question():
    rows = [
        {
            "query_id": "q1",
            "text": "x",
            "relevant": {"a": 1, "b": 1, "z": 1},
            "relevant_spans": [{"doc_id": "z", "start": 0, "end": 4}],
            "required_evidence": [["b"]],
        }
    ]
    ds = dataset(rows)
    report = {
        "format": "vectraudit.label_audit",
        "dataset_sha256": ds.sha256,
        "suspects": [
            {"query_id": "q1", "doc_id": d, "signals": [{"signal": "rank"}]} for d in "abz"
        ],
    }
    text, recorded = apply_label_decisions(
        ds, report, [{"query_id": "q1", "doc_id": "z", "decision": 0}], reviewer="Dana"
    )
    q1 = parse_dataset(text).queries[0]
    assert q1.relevant["z"] == 0.0
    assert q1.relevant_spans == ()
    assert recorded[0]["spans_removed"] == 1
    cases = [
        ([{"query_id": "q1", "doc_id": "b", "decision": 0}], "required evidence"),
        ([{"query_id": "q1", "doc_id": "q", "decision": 0}], "not a suspect"),
        ([{"query_id": "q1", "doc_id": "a", "decision": 5}], "grade from 0 to 3"),
        ([{"query_id": "q1", "doc_id": "a", "decision": True}], "grade from 0 to 3"),
        ([{"query_id": "q1", "doc_id": "a", "decision": 0}] * 2, "decided twice"),
        ([], "nothing is changed automatically"),
    ]
    for decisions, message in cases:
        with pytest.raises(LabelAuditError, match=message):
            apply_label_decisions(ds, report, decisions, reviewer="Dana")
    with pytest.raises(LabelAuditError, match="Name the reviewer"):
        apply_label_decisions(ds, report, cases[0][0], reviewer=" ")
    with pytest.raises(LabelAuditError, match="another version"):
        apply_label_decisions(ds, {**report, "dataset_sha256": "0"}, cases[0][0], reviewer="D")
    last = dataset([{"query_id": "q1", "text": "x", "relevant": {"a": 1, "b": 0}}])
    report_last = {**report, "dataset_sha256": last.sha256}
    with pytest.raises(LabelAuditError, match="last relevant document"):
        apply_label_decisions(
            last, report_last, [{"query_id": "q1", "doc_id": "a", "decision": 0}], reviewer="D"
        )


# --- acceptance: the spike's rules on a corpus with a seed the spike never used -------------


def _spike():
    spec = importlib.util.spec_from_file_location(
        "label_audit_spike", ROOT / "scripts" / "label_audit_spike.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def test_the_spike_thresholds_hold_on_a_new_seed(tmp_path):
    spike = _spike()
    corpus = spike.make_corpus(4)
    rankings = spike.run_rankings(corpus, 4)
    rows = [
        {
            "query_id": f"q{q}",
            "text": f"question {q}",
            "relevant": {f"d{d}": 1 for d in sorted(corpus.labels[q])},
        }
        for q in range(spike.QUESTIONS)
    ]
    path = tmp_path / "benchmark.jsonl"
    path.write_text(render_dataset(rows))
    before = path.read_bytes()
    ds = load_dataset(path)
    runs = [
        run(
            ds,
            {f"q{q}": [f"d{d}" for d in ranking[q]] for q in ranking},
            f"r{i}",
            k=spike.K,
        )
        for i, ranking in enumerate(rankings)
    ]
    ids = [f"d{i}" for i in range(len(corpus.docs))]
    report = audit_labels(ds, runs, corpus=CorpusVectors(ids, corpus.docs))
    planted = {(f"q{q}", f"d{d}") for q, d in corpus.planted}

    def rates(signal: str) -> tuple[float, float]:
        flagged = {
            (s["query_id"], s["doc_id"])
            for s in report["suspects"]
            if any(x["signal"] == signal for x in s["signals"])
        }
        return len(flagged & planted) / len(flagged), len(flagged & planted) / len(planted)

    precision, recall = rates("rank")
    # The adoption bar (D8): at least 70% of planted errors found, at most 30% of flags false.
    assert recall >= 0.7, recall
    assert precision >= 0.7, precision
    assert path.read_bytes() == before  # the audit never touches the dataset


# --- the commands ------------------------------------------------------------------------------


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    from vectraudit.evaluation.trec import import_run

    path = tmp_path / "benchmark.jsonl"
    path.write_text(render_dataset(ROWS))
    ds = load_dataset(path)
    folder = tmp_path / "runs"
    folder.mkdir()
    for name, rankings in (("r1", RANKINGS), ("r2", {**RANKINGS, "q1": ["m", "a", "b", "y", "x"]})):
        trec = tmp_path / f"{name}.trec"
        trec.write_text(
            "".join(
                f"{q} Q0 {d} {i + 1} {10 - i} {name}\n"
                for q, ids in rankings.items()
                for i, d in enumerate(ids)
            )
        )
        record = import_run(trec, ds, k=5, label=name)
        (folder / f"{name}.run.json").write_text(json.dumps(record))
    return path, folder


def test_label_audit_and_review_commands(tmp_path):
    from typer.testing import CliRunner

    from vectraudit.cli import app

    cli = CliRunner()
    path, folder = _workspace(tmp_path)
    before = path.read_bytes()
    out = tmp_path / "label-audit.json"
    result = cli.invoke(
        app, ["workbench", "label-audit", str(path), "--runs", str(folder), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "2 labels to review on 1 questions" in result.output
    assert "near duplicate: not measured" in result.output
    report = json.loads(out.read_text())
    assert report["format"] == "vectraudit.label_audit"
    assert path.read_bytes() == before

    queued = tmp_path / "benchmark-audit.json"
    result = cli.invoke(
        app,
        [
            "workbench",
            "benchmark",
            str(path),
            "--runs",
            str(folder),
            "--label-audit",
            str(out),
            "--out",
            str(queued),
        ],
    )
    assert result.exit_code == 0, result.output
    queue = json.loads(queued.read_text())["judgment_queue"]
    assert {i["doc_id"] for i in queue if i.get("source") == "label_audit"} == {"m", "z"}

    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps({"query_id": "q1", "doc_id": "m", "decision": 1})
        + "\n\n"
        + json.dumps({"query_id": "q1", "doc_id": "z", "decision": "keep"})
        + "\n"
    )
    new = tmp_path / "benchmark.v2.jsonl"
    args = ["workbench", "review-labels", str(path), str(out), "--decisions", str(decisions)]
    result = cli.invoke(app, [*args, "--reviewer", "Dana", "--out", str(new)])
    assert result.exit_code == 0, result.output
    assert "1 labels changed and 1 kept by Dana" in result.output
    assert load_dataset(new).queries[0].relevant["m"] == 1.0
    assert path.read_bytes() == before
    same = cli.invoke(app, [*args, "--reviewer", "Dana", "--out", str(path)])
    assert same.exit_code == 2
    assert "--out must differ" in same.output

    lonely = tmp_path / "lonely"
    lonely.mkdir()
    failed = cli.invoke(
        app, ["workbench", "label-audit", str(path), "--runs", str(lonely), "--out", str(out)]
    )
    assert failed.exit_code == 2
    assert "Nothing to audit" in failed.output
