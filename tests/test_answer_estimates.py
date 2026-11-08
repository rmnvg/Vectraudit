"""Judge-corrected rates in the answers summary, and every reason they are withheld."""

import json
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_review_sample import dataset, run_for
from vectraudit.cli import app
from vectraudit.evaluation import answer_estimates as E
from vectraudit.evaluation import review_sample as R
from vectraudit.evaluation.answers import (
    AnswerJudgmentError,
    Judgment,
    attach_answers,
    attach_file,
    render_summary,
)

runner = CliRunner()
FAMILIES = 400
EVALUATOR = "ragas.correctness"


def judgment(kind, evaluator, verdict) -> Judgment:
    return Judgment(evaluator, kind, verdict, None, "answered", None, None)


def world(seed=0, *, flip=0.08):
    """A run and dataset, truth per question, and a lenient, noisy model judge."""
    rng = np.random.default_rng(seed)
    ds = dataset(FAMILIES, per_family=2)
    run = run_for(ds)
    truth = {q.query_id: bool(rng.random() < 0.55) for q in ds.queries}
    model = {}
    for qid, ok in truth.items():
        wrong = rng.random() < flip
        lenient = (not ok) and rng.random() < 0.12
        model[qid] = (ok != wrong) or lenient
    return ds, run, truth, model


def rows(truth, model, sample, *, humans=None, drop_model=()):
    out = []
    reviewed = (
        humans if humans is not None else {q for ids in sample.questions.values() for q in ids}
    )
    for qid, ok in model.items():
        if qid not in drop_model:
            out.append(
                ("query_id", qid, judgment("model", EVALUATOR, "correct" if ok else "incorrect"))
            )
        if qid in reviewed:
            out.append(
                (
                    "query_id",
                    qid,
                    judgment("human", "review", "correct" if truth[qid] else "incorrect"),
                )
            )
    return out


def sample_for(ds, run, tmp_path, size=150, seed=0):
    path = R.write_sample(R.build_sample(run, ds, size, seed), tmp_path / "sample.json")
    return R.read_sample(path)


def true_rate(ds, truth):
    fams = {}
    for q in ds.queries:
        fams.setdefault(q.family_id, []).append(truth[q.query_id])
    return float(np.mean([np.mean(v) for v in fams.values()]))


# --- the estimate -----------------------------------------------------------------------


def test_the_corrected_rate_contains_the_truth_and_the_judge_alone_does_not(tmp_path) -> None:
    ds, run, truth, model = world(seed=3)
    sample = sample_for(ds, run, tmp_path)
    out = attach_answers(run, rows(truth, model, sample), sample=sample)
    row = out["answers"]["estimates"]["by_evaluator"][EVALUATOR]
    assert row["status"] == "ok", row["reason"]
    t = true_rate(ds, truth)
    corrected, human, judge = row["corrected"], row["human_only"], row["judge_only"]
    assert corrected["ci_low"] <= t <= corrected["ci_high"]
    assert abs(judge["estimate"] - t) > 0.02  # the lenient judge overstates the rate
    assert corrected["ci_high"] - corrected["ci_low"] < human["ci_high"] - human["ci_low"]
    assert (row["sample_families"], row["frame_families"]) == (150, FAMILIES)
    assert 0.8 < row["agreement_on_sample"] < 1
    assert out["answers"]["estimates"]["sample"]["sha256"] == sample.sha256
    assert out["answers"]["estimates"]["metric"] == "correct_rate"


def test_humans_outside_the_sample_do_not_change_the_estimate_but_still_decide_verdicts(
    tmp_path,
) -> None:
    ds, run, truth, model = world(seed=4)
    sample = sample_for(ds, run, tmp_path)
    base = attach_answers(run, rows(truth, model, sample), sample=sample)
    inside = {q for ids in sample.questions.values() for q in ids}
    outside = {q.query_id for q in ds.queries if q.query_id not in inside}
    extra = sorted(outside)[:40]  # a hand-picked review: not a random sample
    more = attach_answers(
        run, rows(truth, model, sample, humans=inside | set(extra)), sample=sample
    )
    pick = lambda out: out["answers"]["estimates"]["by_evaluator"][EVALUATOR]["corrected"]  # noqa: E731
    assert pick(more) == pick(base)
    decided = [q["answer"]["decided_by"] for q in more["queries"] if q["query_id"] in set(extra)]
    assert set(decided) == {"human"}  # the extra reviews still decide those questions' verdicts
    assert all(
        q["answer"]["verdict"] in ("correct", "incorrect")
        for q in more["queries"]
        if q["query_id"] in inside
    )


def test_it_never_changes_a_verdict_or_a_diagnosis(tmp_path) -> None:
    ds, run, truth, model = world(seed=5)
    sample = sample_for(ds, run, tmp_path)
    with_sample = attach_answers(run, rows(truth, model, sample), sample=sample)
    without = attach_answers(run, rows(truth, model, sample))
    for a, b in zip(with_sample["queries"], without["queries"], strict=True):
        assert a["answer"] == b["answer"]
    assert with_sample["answers"]["verdicts"] == without["answers"]["verdicts"]
    assert with_sample["answers"]["diagnoses"] == without["answers"]["diagnoses"]


# --- when it is withheld ------------------------------------------------------------------


def absent(out):
    row = out["answers"]["estimates"]["by_evaluator"][EVALUATOR]
    assert row["status"] == "absent"
    assert "corrected" not in row
    return row


def test_without_a_sample_there_is_no_corrected_estimate_only_the_judges_rate(tmp_path) -> None:
    ds, run, truth, model = world(seed=6)
    sample = sample_for(ds, run, tmp_path)
    row = absent(attach_answers(run, rows(truth, model, sample)))
    assert "workbench review-sample" in row["reason"]
    assert row["judge_only"]["families"] == FAMILIES
    assert "biased" in row["judge_only"]["note"]


def test_a_sample_for_another_dataset_or_run_is_refused(tmp_path) -> None:
    ds, run, _, _ = world(seed=7)
    sample = sample_for(ds, run, tmp_path)
    other = run_for(dataset(FAMILIES, per_family=3))
    out = attach_answers(
        {**other, "queries": other["queries"]},
        [
            ("query_id", q["query_id"], judgment("model", EVALUATOR, "correct"))
            for q in other["queries"]
        ],
        sample=sample,
    )
    assert "different dataset" in absent(out)["reason"]


def test_a_judge_that_skipped_questions_gets_no_corrected_estimate(tmp_path) -> None:
    ds, run, truth, model = world(seed=8)
    sample = sample_for(ds, run, tmp_path)
    skipped = {q.query_id for q in ds.queries[:20]}
    out = attach_answers(run, rows(truth, model, sample, drop_model=skipped), sample=sample)
    reason = absent(out)["reason"]
    assert "needs all of them" in reason
    assert "randomness" in reason


def test_incomplete_or_disagreeing_human_review_is_not_used(tmp_path) -> None:
    ds, run, truth, model = world(seed=9)
    sample = sample_for(ds, run, tmp_path)
    inside = sorted({q for ids in sample.questions.values() for q in ids})
    partial = absent(
        attach_answers(run, rows(truth, model, sample, humans=set(inside[:-5])), sample=sample)
    )
    assert "of 150 sampled families" in partial["reason"]
    conflicting = rows(truth, model, sample)
    qid = inside[0]
    conflicting.append(
        (
            "query_id",
            qid,
            judgment("human", "second-reviewer", "incorrect" if truth[qid] else "correct"),
        )
    )
    assert "disagreed" in absent(attach_answers(run, conflicting, sample=sample))["reason"]


def test_a_tiny_review_is_refused_when_the_sample_is_drawn_not_here(tmp_path) -> None:
    ds, run, _, _ = world(seed=10)
    with pytest.raises(R.ReviewSampleError, match="too small"):
        R.build_sample(run, ds, size=10)


def test_the_unit_is_a_family_and_a_half_reviewed_family_does_not_count(tmp_path) -> None:
    ds, run, truth, model = world(seed=11)
    sample = sample_for(ds, run, tmp_path)
    inside = {q for ids in sample.questions.values() for q in ids}
    one_family = sorted(sample.chosen)[0]
    drop = sample.questions[one_family][0]  # one question of one sampled family not reviewed
    out = attach_answers(run, rows(truth, model, sample, humans=inside - {drop}), sample=sample)
    assert "of 150 sampled families" in absent(out)["reason"]
    assert str(len(sample.chosen) - 1) in absent(out)["reason"]


# --- the file and the command -------------------------------------------------------------


def test_attach_file_reads_the_sample_and_refuses_an_edited_one(tmp_path) -> None:
    ds, run, truth, model = world(seed=12)
    sample = sample_for(ds, run, tmp_path)
    lines = [
        {"query_id": q, "evaluator": e, "kind": k, "verdict": v}
        for _, q, j in rows(truth, model, sample)
        for e, k, v in [(j.evaluator, j.kind, j.verdict)]
    ]
    (tmp_path / "j.jsonl").write_text("\n".join(map(json.dumps, lines)))
    out = attach_file(run, tmp_path / "j.jsonl", sample_path=tmp_path / "sample.json")
    assert out["answers"]["estimates"]["by_evaluator"][EVALUATOR]["status"] == "ok"
    data = json.loads((tmp_path / "sample.json").read_text())
    data["chosen"][0] = next(f for f in data["frame"] if f not in data["chosen"])
    (tmp_path / "bad.json").write_text(json.dumps(data))
    with pytest.raises(AnswerJudgmentError, match="seed draws"):
        attach_file(run, tmp_path / "j.jsonl", sample_path=tmp_path / "bad.json")


def test_the_summary_shows_the_three_numbers_or_the_reason(tmp_path) -> None:
    ds, run, truth, model = world(seed=13)
    sample = sample_for(ds, run, tmp_path)
    text = render_summary(attach_answers(run, rows(truth, model, sample), sample=sample))
    assert f"{EVALUATOR} correct rate: judge only" in text
    assert "biased by the judge's errors" in text
    assert "people only" in text
    assert "corrected" in text
    assert "150 of 400 families reviewed" in text
    none = render_summary(attach_answers(run, rows(truth, model, sample)))
    assert f"{EVALUATOR}: no corrected estimate. No review sample was given" in none


def test_the_estimate_block_holds_no_text(tmp_path) -> None:
    ds, run, truth, model = world(seed=14)
    sample = sample_for(ds, run, tmp_path)
    block = attach_answers(run, rows(truth, model, sample), sample=sample)["answers"]["estimates"]
    dumped = json.dumps(block)
    assert not re.search(r"question \d", dumped)  # no question text from the dataset
    assert E.METRIC in dumped


def test_the_two_commands_work_together(tmp_path) -> None:
    from tests.test_review_sample import _valid_run
    from vectraudit.evaluation.dataset import load_dataset

    ds_small = dataset(120, per_family=1)
    (tmp_path / "ds.jsonl").write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "text": q.text, "family_id": q.family_id})
            for q in ds_small.queries
        )
    )
    loaded = load_dataset(tmp_path / "ds.jsonl")
    (tmp_path / "run.json").write_text(json.dumps(_valid_run(loaded)))
    sampled = runner.invoke(
        app,
        ["workbench", "review-sample", str(tmp_path / "run.json"), "--dataset",
         str(tmp_path / "ds.jsonl"), "--size", "50", "--out", str(tmp_path / "s.json")],
    )  # fmt: skip
    assert sampled.exit_code == 0, sampled.output
    chosen = {
        q
        for ids in json.loads((tmp_path / "s.json").read_text())["questions"].values()
        for q in ids
    }
    rng = np.random.default_rng(0)
    lines = []
    for q in loaded.queries:
        ok = bool(rng.random() < 0.6)
        lines.append({"query_id": q.query_id, "evaluator": "j", "kind": "model",
                      "verdict": "correct" if rng.random() < 0.9 else "incorrect"})  # fmt: skip
        if q.query_id in chosen:
            lines.append({"query_id": q.query_id, "evaluator": "h", "kind": "human",
                          "verdict": "correct" if ok else "incorrect"})  # fmt: skip
    (tmp_path / "j.jsonl").write_text("\n".join(map(json.dumps, lines)))
    done = runner.invoke(
        app,
        ["workbench", "attach-answers", str(tmp_path / "run.json"), str(tmp_path / "j.jsonl"),
         "--out", str(tmp_path / "out.json"), "--sample", str(tmp_path / "s.json")],
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    assert "j correct rate: judge only" in done.output
    assert "50 of 120 families reviewed" in done.output
    bad = runner.invoke(
        app,
        ["workbench", "attach-answers", str(tmp_path / "run.json"), str(tmp_path / "j.jsonl"),
         "--out", str(tmp_path / "o2.json"), "--sample", str(tmp_path / "missing.json")],
    )  # fmt: skip
    assert bad.exit_code == 2


def test_the_frontend_fixture_matches_what_the_code_writes(tmp_path, monkeypatch) -> None:
    """frontend/src/lib/fixtures/answer_estimates.json is real output; regenerate if this fails."""
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parents[1] / "frontend/src/lib/fixtures/answer_estimates.json").read_text()
    )
    monkeypatch.setitem(globals(), "FAMILIES", 100)
    ds, run, truth, model = world(seed=21)
    sample = sample_for(ds, run, tmp_path, size=40)
    ok = attach_answers(run, rows(truth, model, sample), sample=sample)["answers"]["estimates"]
    none = attach_answers(run, rows(truth, model, sample))["answers"]["estimates"]
    assert set(fixture["ok"]) == set(ok)
    assert set(fixture["ok"]["by_evaluator"][EVALUATOR]) == set(ok["by_evaluator"][EVALUATOR])
    assert set(fixture["ok"]["by_evaluator"][EVALUATOR]["corrected"]) == set(
        ok["by_evaluator"][EVALUATOR]["corrected"]
    )
    assert set(fixture["absent"]["by_evaluator"][EVALUATOR]) == set(none["by_evaluator"][EVALUATOR])
