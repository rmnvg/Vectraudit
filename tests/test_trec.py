"""TREC interchange: export, import, and the round trip that must lose nothing."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation import trec as T
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.metrics import semantic_metrics
from vectraudit.evaluation.runfile import load_run, save_run

FIX = Path(__file__).parent / "fixtures"
RANX = json.loads((FIX / "ranx/metrics.json").read_text())
TREC_EVAL = json.loads((FIX / "trec_eval/metrics.json").read_text())
runner = CliRunner()


def fixture_dataset(n=60, heldout=()):
    rows = []
    for q in RANX["queries"][:n]:
        rows.append(
            {
                "query_id": q["id"],
                "text": f"question {q['id']}",
                "relevant": q["labels"],
                "split": "heldout" if q["id"] in heldout else "explore",
            }
        )
    return parse_dataset("\n".join(map(json.dumps, rows)))


def write_trec(path, n=60, *, tag="sysA", skip=(), scores=None):
    lines = []
    for q in RANX["queries"][:n]:
        if q["id"] in skip:
            continue
        ranking = q["ranking"]
        for i, doc in enumerate(ranking):
            score = scores(q["id"], i, len(ranking)) if scores else len(ranking) - i
            lines.append(f"{q['id']} Q0 {doc} {i + 1} {score} {tag}")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


# --- import ---------------------------------------------------------------------------------


def test_an_imported_run_scores_exactly_what_trec_eval_scores(tmp_path) -> None:
    ds = fixture_dataset()
    run = T.import_run(write_trec(tmp_path / "a.trec"), ds, k=10)
    assert run["status"] == "complete"
    for entry in run["queries"]:
        want = TREC_EVAL["queries"][entry["query_id"]]
        for name in ("recall", "precision", "mrr", "ndcg", "map", "hit_rate"):
            assert entry["metrics"][name] == pytest.approx(want[f"{name}@10"], abs=1e-9)


def test_an_imported_run_is_a_valid_run_marked_as_external(tmp_path) -> None:
    ds = fixture_dataset()
    run = T.import_run(write_trec(tmp_path / "a.trec"), ds, k=10)
    save_run(run, tmp_path / "a.run.json")
    again = load_run(tmp_path / "a.run.json")  # check_run: the same validation as a recorded run
    assert again["timing"]["source"] == "external:trec"
    assert again["import"]["source"] == "external:trec"
    assert again["import"]["file_sha256"] == run["import"]["file_sha256"]
    assert again["summary"]["latency_ms"] is None
    assert all(q["latency_ms"] is None for q in again["queries"])
    assert again["dataset"]["sha256"] == ds.sha256
    assert "not measured" in again["timing"]["scope"]
    assert all(q.get("evidence") is None for q in again["queries"])


def test_two_imported_runs_compare_on_quality_and_say_latency_was_not_judged(tmp_path) -> None:
    ds = fixture_dataset()
    a = T.import_run(write_trec(tmp_path / "a.trec"), ds, k=10, label="a")
    shallow = lambda qid, i, n: n - i  # noqa: E731
    b_path = write_trec(tmp_path / "b.trec", scores=shallow)
    b = T.import_run(b_path, ds, k=3, label="b")
    same_k = T.import_run(b_path, ds, k=10, label="b10")
    result = compare_runs(a, same_k)
    assert result["verdict"] == "pass"
    assert result["metrics"]["ndcg"]["baseline"] == result["metrics"]["ndcg"]["candidate"]
    assert result.get("latency") is None
    assert any("imported" in w.lower() and "latency" in w.lower() for w in result["warnings"])
    assert compare_runs(a, b)["verdict"] == "not_comparable"  # different k


def test_a_question_missing_from_the_file_is_an_error_unless_asked_otherwise(tmp_path) -> None:
    ds = fixture_dataset()
    path = write_trec(tmp_path / "a.trec", skip={"q3", "q4"})
    run = T.import_run(path, ds, k=10)
    assert run["status"] == "partial"
    missing = [q for q in run["queries"] if q["error"]]
    assert {q["query_id"] for q in missing} == {"q3", "q4"}
    assert "has no ranking" in missing[0]["error"]["message"]
    assert missing[0]["metrics"] is None
    assert any("recorded as errors" in n for n in T.import_notes(run))
    lenient = T.import_run(path, ds, k=10, missing_as_empty=True)
    assert lenient["status"] == "complete"
    empty = next(q for q in lenient["queries"] if q["query_id"] == "q3")
    assert empty["ids"] == []
    assert empty["metrics"]["recall"] == 0.0
    assert lenient["import"]["dataset_questions_not_in_file"] == 2
    save_run(run, tmp_path / "p.json")
    load_run(tmp_path / "p.json")


def test_questions_the_dataset_does_not_have_are_counted_and_ignored(tmp_path) -> None:
    ds = fixture_dataset(n=40)
    path = write_trec(tmp_path / "a.trec", n=60)
    run = T.import_run(path, ds, k=10)
    assert run["import"]["questions_in_file"] == 60
    assert run["import"]["questions_not_in_dataset"] == 20
    assert run["summary"]["n_queries"] == 40
    assert any("ignored" in n for n in T.import_notes(run))


def test_the_rank_column_orders_the_run_and_a_disagreeing_score_is_reported(tmp_path) -> None:
    ds = fixture_dataset(n=10)
    flipped = write_trec(
        tmp_path / "f.trec", n=10, scores=lambda q, i, n: i
    )  # score rises with rank
    run = T.import_run(flipped, ds, k=10)
    assert run["import"]["rank_disagrees_with_score"] > 0
    assert run["import"]["ordered_by"] == "the rank column"
    plain = T.import_run(write_trec(tmp_path / "p.trec", n=10), ds, k=10)
    assert [q["ids"] for q in run["queries"]] == [q["ids"] for q in plain["queries"]]
    assert plain["import"]["rank_disagrees_with_score"] == 0
    assert any("rank column was used" in n for n in T.import_notes(run))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("q1 Q0 d1 1 5.0", "has 5 fields"),
        ("q1 Q0 d1 one 5.0 t", "rank must be an integer"),
        ("q1 Q0 d1 1 nan t", "not finite"),
        ("q1 Q0 d1 1 5.0 t\nq1 Q0 d2 1 4.0 t", "same rank"),
        ("q1 Q0 d1 1 5.0 t\nq1 Q0 d1 2 4.0 t", "lists a document twice"),
        ("\n\n", "holds no run lines"),
    ],
)
def test_ambiguous_files_are_refused(tmp_path, text, message) -> None:
    path = tmp_path / "bad.trec"
    path.write_text(text)
    with pytest.raises(T.TrecError, match=message):
        T.parse_run(path)
    with pytest.raises(T.TrecError, match="Cannot read"):
        T.parse_run(tmp_path / "nope.trec")


# --- export ---------------------------------------------------------------------------------


def test_a_run_survives_export_and_import_with_identical_quality(tmp_path) -> None:
    """Phase acceptance: nothing about quality is lost on the way out and back."""
    ds = fixture_dataset()
    first = T.import_run(write_trec(tmp_path / "a.trec"), ds, k=10, label="first")
    exported = T.export_run(first, ds)
    (tmp_path / "out.trec").write_text("\n".join(exported.lines) + "\n")
    second = T.import_run(tmp_path / "out.trec", ds, k=10, label="second")
    for a, b in zip(first["queries"], second["queries"], strict=True):
        assert a["ids"] == b["ids"]
        assert a["metrics"] == b["metrics"]
        assert a["repeat_metrics"] == b["repeat_metrics"]
    assert first["summary"]["metrics"] == second["summary"]["metrics"]
    assert exported.queries == 60
    assert exported.lines[0].split()[1] == "Q0"


def test_the_exported_file_is_trec_run_format_with_ranks_as_scores(tmp_path) -> None:
    ds = fixture_dataset(n=5)
    run = T.import_run(write_trec(tmp_path / "a.trec", n=5), ds, k=10, label="My system!")
    lines = T.export_run(run, ds).lines
    first = RANX["queries"][0]["ranking"][:10]
    top = [line.split() for line in lines if line.startswith("q0 ")]
    assert [row[2] for row in top] == first
    assert [row[3] for row in top] == [str(i + 1) for i in range(len(first))]
    assert [float(row[4]) for row in top] == [float(len(first) - i) for i in range(len(first))]
    assert {row[5] for row in top} == {"My-system"}  # a tag cannot hold spaces
    assert T.export_run(run, ds, tag="other").lines[0].endswith(" other")


def test_heldout_and_errored_questions_are_not_exported(tmp_path) -> None:
    ds = fixture_dataset(n=20, heldout={"q1", "q2"})
    path = write_trec(tmp_path / "a.trec", n=20, skip={"q5"})
    run = T.import_run(path, ds, k=10)  # q5 errors: no ranking in the file
    done = T.export_run(run, ds)
    ids = {line.split()[0] for line in done.lines}
    assert not ids & {"q1", "q2", "q5"}
    assert (done.heldout_left_out, done.errored_left_out) == (2, 1)
    with_heldout = T.export_run(run, ds, include_heldout=True)
    assert {"q1", "q2"} <= {line.split()[0] for line in with_heldout.lines}


def test_export_refuses_ids_it_cannot_represent_and_another_datasets_run(tmp_path) -> None:
    ds = fixture_dataset(n=5)
    run = T.import_run(write_trec(tmp_path / "a.trec", n=5), ds, k=10)
    spaced = json.loads(json.dumps(run))
    spaced["queries"][0]["ids"][0] = "has space"
    with pytest.raises(T.TrecError, match="whitespace"):
        T.export_run(spaced, ds)
    with pytest.raises(T.TrecError, match="not recorded on this dataset"):
        T.export_run(run, fixture_dataset(n=6))


def test_qrels_keep_grades_zeros_and_spans_and_leave_out_heldout() -> None:
    rows = [
        {"query_id": "a", "text": "a", "relevant": {"d1": 3, "d2": 0, "d3": 1}},
        {"query_id": "b", "text": "b", "relevant": {"d4": 1}, "split": "heldout"},
        {"query_id": "c", "text": "c", "relevant_spans": [{"doc_id": "d9", "start": 0, "end": 5}]},
        {"query_id": "u", "text": "unlabelled"},
    ]
    ds = parse_dataset("\n".join(map(json.dumps, rows)))
    lines, held = T.export_qrels(ds)
    assert lines == ["a 0 d1 3", "a 0 d2 0", "a 0 d3 1", "c 0 d9 1"]
    assert held == 1
    with_held, _ = T.export_qrels(ds, include_heldout=True)
    assert "b 0 d4 1" in with_held


def test_qrels_refuse_fractional_grades() -> None:
    ds = parse_dataset(json.dumps({"query_id": "a", "text": "a", "relevant": {"d1": 0.5}}))
    with pytest.raises(T.TrecError, match="need integers"):
        T.export_qrels(ds)


def test_exported_qrels_reproduce_the_reference_metrics(tmp_path) -> None:
    """A qrels file written here, read back by hand, gives the labels trec_eval was given."""
    ds = fixture_dataset(n=30)
    lines, _ = T.export_qrels(ds)
    labels: dict[str, dict[str, int]] = {}
    for line in lines:
        qid, _, doc, grade = line.split()
        labels.setdefault(qid, {})[doc] = int(grade)
    for q in RANX["queries"][:30]:
        got = semantic_metrics(q["ranking"], labels.get(q["id"], {}), 10)
        assert got["ndcg"] == pytest.approx(TREC_EVAL["queries"][q["id"]]["ndcg@10"], abs=1e-9)


# --- the commands ------------------------------------------------------------------------------


def test_the_three_commands_round_trip(tmp_path) -> None:
    ds_rows = [
        {"query_id": q["id"], "text": f"question {q['id']}", "relevant": q["labels"]}
        for q in RANX["queries"][:30]
    ]
    (tmp_path / "ds.jsonl").write_text("\n".join(map(json.dumps, ds_rows)))
    write_trec(tmp_path / "other.trec", n=30)
    imported = runner.invoke(
        app,
        ["runs", "import-trec", str(tmp_path / "other.trec"),
         "--dataset", str(tmp_path / "ds.jsonl"),
         "--out", str(tmp_path / "other.run.json"), "--label", "other"],
    )  # fmt: skip
    assert imported.exit_code == 0, imported.output
    assert "30 of 30 questions ranked" in " ".join(imported.output.split())
    exported = runner.invoke(
        app,
        ["runs", "export-trec", str(tmp_path / "other.run.json"), "--dataset",
         str(tmp_path / "ds.jsonl"), "--out", str(tmp_path / "back.trec")],
    )  # fmt: skip
    assert exported.exit_code == 0, exported.output
    original = [line.split() for line in (tmp_path / "other.trec").read_text().splitlines()]
    back = [line.split() for line in (tmp_path / "back.trec").read_text().splitlines()]
    top10 = [row[:3] for row in original if int(row[3]) <= 10]  # the import cut to k = 10
    assert top10 == [row[:3] for row in back]
    qrels = runner.invoke(
        app,
        ["dataset", "export-qrels", str(tmp_path / "ds.jsonl"), "--out", str(tmp_path / "q.qrels")],
    )
    assert qrels.exit_code == 0, qrels.output
    assert (tmp_path / "q.qrels").read_text().startswith("q0 0 ")


def test_a_partial_import_exits_nonzero_and_a_bad_file_is_a_clean_error(tmp_path) -> None:
    ds_rows = [
        {"query_id": q["id"], "text": f"question {q['id']}", "relevant": q["labels"]}
        for q in RANX["queries"][:10]
    ]
    (tmp_path / "ds.jsonl").write_text("\n".join(map(json.dumps, ds_rows)))
    write_trec(tmp_path / "part.trec", n=10, skip={"q2"})
    partial = runner.invoke(
        app,
        ["runs", "import-trec", str(tmp_path / "part.trec"),
         "--dataset", str(tmp_path / "ds.jsonl"), "--out", str(tmp_path / "p.json")],
    )  # fmt: skip
    assert partial.exit_code == 1
    assert "partial" in partial.output
    assert "recorded as errors" in " ".join(partial.output.split())
    (tmp_path / "bad.trec").write_text("not a run")
    bad = runner.invoke(
        app,
        ["runs", "import-trec", str(tmp_path / "bad.trec"),
         "--dataset", str(tmp_path / "ds.jsonl"), "--out", str(tmp_path / "b.json")],
    )  # fmt: skip
    assert bad.exit_code == 1
    assert "fields" in " ".join(bad.output.split())
