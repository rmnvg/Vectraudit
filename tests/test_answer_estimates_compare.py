"""The judge-corrected change in correct rate between two runs, shown as information only."""

import copy

import numpy as np

from tests.test_answer_estimates import (
    EVALUATOR,
    FAMILIES,
    rows,
    sample_for,
    true_rate,
    world,
)
from tests.test_review_sample import _valid_run
from vectraudit.evaluation import answer_estimates as E
from vectraudit.evaluation.answers import attach_answers
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.report import render_html, render_markdown


def judged(rng, truth, flip=0.08):
    model = {}
    for qid, ok in truth.items():
        wrong = rng.random() < flip
        lenient = (not ok) and rng.random() < 0.12
        model[qid] = (ok != wrong) or lenient
    return model


def pair(seed=0):
    """A baseline and a candidate (about 12 points better) judged by the same noisy judge."""
    ds, run, truth_b, model_b = world(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    truth_c = {q: ok or bool(rng.random() < 0.27) for q, ok in truth_b.items()}
    model_c = judged(rng, truth_c)
    sample = sample_for(ds, run, __import__("pathlib").Path(_tmp()))
    base_run, cand_run = _valid_run(ds), _valid_run(ds)
    base = attach_answers(base_run, rows(truth_b, model_b, sample), sample=sample)
    cand = attach_answers(cand_run, rows(truth_c, model_c, sample), sample=sample)
    return ds, base, cand, truth_b, truth_c, sample


def _tmp():
    import tempfile

    return tempfile.mkdtemp()


def test_the_corrected_change_contains_the_true_change() -> None:
    ds, base, cand, tb, tc, _ = pair(seed=2)
    block = E.compare_estimates(base, cand)
    row = block["by_evaluator"][EVALUATOR]
    assert row["status"] == "ok", row["reason"]
    true_change = true_rate(ds, tc) - true_rate(ds, tb)
    assert true_change > 0.05
    corrected, human = row["corrected"], row["human_only"]
    assert corrected["ci_low"] <= true_change <= corrected["ci_high"]
    assert corrected["ci_high"] - corrected["ci_low"] < human["ci_high"] - human["ci_low"]
    assert (row["sample_families"], row["frame_families"]) == (150, FAMILIES)
    assert row["judge_only"]["n"] == FAMILIES
    assert corrected["method"] == "ppi++"


def test_the_change_is_not_clipped_to_zero_one() -> None:
    _, base, cand, *_ = pair(seed=3)
    corrected = E.compare_estimates(cand, base)["by_evaluator"][EVALUATOR]["corrected"]
    assert corrected["estimate"] < 0  # worse than the baseline: a negative change is kept


def test_runs_corrected_with_different_samples_are_not_compared(tmp_path) -> None:
    ds, base, _, _, tc, _ = pair(seed=4)
    other = sample_for(ds, _valid_run(ds), tmp_path, seed=99)
    again = attach_answers(
        _valid_run(ds), rows(tc, judged(np.random.default_rng(5), tc), other), sample=other
    )
    block = E.compare_estimates(base, again)
    assert block["by_evaluator"] == {}
    assert "not corrected with the same review sample" in block["note"]


def test_a_run_without_estimates_gives_no_comparison() -> None:
    ds, base, cand, *_ = pair(seed=5)
    plain = copy.deepcopy(cand)
    del plain["answers"]["estimates"]
    assert E.compare_estimates(base, plain) is None
    assert E.compare_estimates(base, _valid_run(ds)) is None


def test_a_missing_estimate_on_one_side_is_named() -> None:
    _, base, cand, *_ = pair(seed=6)
    broken = copy.deepcopy(cand)
    broken["answers"]["estimates"]["by_evaluator"][EVALUATOR]["status"] = "absent"
    row = E.compare_estimates(base, broken)["by_evaluator"][EVALUATOR]
    assert row["status"] == "absent"
    assert "candidate" in row["reason"]


# --- in the check -----------------------------------------------------------------------


def test_the_check_reports_the_change_and_the_verdict_is_unchanged() -> None:
    _, base, cand, *_ = pair(seed=7)
    with_estimates = compare_runs(base, cand)
    assert with_estimates["judge_corrected"]["by_evaluator"][EVALUATOR]["status"] == "ok"
    stripped_base, stripped_cand = copy.deepcopy(base), copy.deepcopy(cand)
    for run in (stripped_base, stripped_cand):
        del run["answers"]["estimates"]
    without = compare_runs(stripped_base, stripped_cand)
    assert "judge_corrected" not in without
    assert with_estimates["verdict"] == without["verdict"]
    assert with_estimates["reasons"] == without["reasons"]
    assert with_estimates["warnings"] == without["warnings"]
    assert with_estimates["metrics"] == without["metrics"]


def test_the_reports_show_it_as_information_only() -> None:
    _, base, cand, *_ = pair(seed=8)
    result = compare_runs(base, cand)
    md = render_markdown(result)
    assert "Judge-corrected change in the correct-answer rate" in md
    assert "Corrected (95%)" in md
    assert "the verdict does not use it" in md
    assert EVALUATOR in md
    html = render_html(result)
    assert "Judge-corrected change in correct answers" in html
    assert "Information only" in html or "information only" in html.lower()
    result_with_note = {
        **result,
        "judge_corrected": {"by_evaluator": {}, "note": "different samples"},
    }
    assert "different samples" in render_markdown(result_with_note)


def test_a_hand_edited_sample_in_a_run_cannot_pair_it_with_another() -> None:
    _, base, cand, *_ = pair(seed=9)
    forged = copy.deepcopy(cand)
    forged["answers"]["estimates"]["sample"]["sha256"] = "0" * 64
    block = E.compare_estimates(base, forged)
    assert block["by_evaluator"] == {}
