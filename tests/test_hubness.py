"""Hubs: documents retrieved for many unrelated questions."""

from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.hubness import compare, gini, hubness, render, skewness
from vectraudit.evaluation.runfile import check_run

N = 40


def with_rankings(rankings, corpus=1000):
    run = fake_run([1.0] * len(rankings))
    for q, ids in zip(run["queries"], rankings, strict=True):
        q["ids"] = ids
    run["backend"]["count"] = corpus
    return check_run(json.loads(json.dumps(run)))


def spread():
    return with_rankings([[f"d{i}", f"e{i}"] for i in range(N)])


def hubbed():
    return with_rankings([["hub", f"d{i}"] for i in range(N)])


LABELS = {f"q{i}": {f"d{i}": 1} for i in range(N)}  # the hub is never relevant


def test_gini_follows_evidentlys_formula():
    assert gini([3, 3, 3, 3]) == 0.0
    # sorted counts 1, 1, 2, 6: sum((2i - n - 1) x) / ((n - 1) sum x) with n = 4
    expected = ((-3) * 1 + (-1) * 1 + 1 * 2 + 3 * 6) / (3 * 10)
    assert gini([6, 1, 2, 1]) == pytest.approx(expected)
    assert gini([5]) == 0.0
    assert gini([0, 0, 4, 4]) == 0.0  # zeros are documents never retrieved: not counted


def test_skewness_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    values = [0, 0, 0, 1, 1, 2, 9]
    assert skewness(values) == pytest.approx(float(stats.skew(values)))
    assert skewness([2, 2, 2]) is None
    assert skewness([1, 2]) is None


def test_a_spread_retriever_has_no_hubs():
    report = hubness(spread())
    assert report["hubs"] == []
    assert report["documents_retrieved"] == 2 * N
    assert report["coverage"] == pytest.approx(2 * N / 1000)
    assert report["gini"] == 0.0
    assert report["skewness_includes_unretrieved"] is True
    assert report["hub_threshold"] == {"share": 0.1, "questions": 4}


def test_a_planted_hub_is_found_with_how_rarely_it_is_relevant():
    report = hubness(hubbed(), labels=LABELS)
    (hub,) = report["hubs"]
    assert hub == {
        "doc_id": "hub", "questions": N, "share": 1.0, "relevant_for": 0, "judged_questions": N,
    }  # fmt: skip
    assert report["gini"] > 0.4
    text = render(report)
    assert "hub: 40 questions (100%), labelled relevant for 0 of 40 judged" in text
    assert "crowd out answers: hub" in text


def test_a_comparison_lists_hubs_new_to_the_candidate():
    comparison = compare(hubness(spread()), hubness(hubbed()))
    assert [h["doc_id"] for h in comparison["new_hubs"]] == ["hub"]
    assert comparison["gone_hubs"] == []
    assert comparison["gini"]["candidate"] > comparison["gini"]["baseline"]
    assert "New hubs: hub" in render(hubness(hubbed()), comparison)


def test_failed_questions_and_repeats_in_one_ranking_are_not_counted():
    run = hubbed()
    for q in run["queries"][:5]:
        q["ids"] = ["hub", "hub", "hub"]  # one question, one occurrence
    report = hubness(run)
    assert report["hubs"][0]["questions"] == N
    failing = copy.deepcopy(hubbed())
    for q in failing["queries"][:30]:
        q["error"] = {"code": "timeout"}
    assert hubness(failing)["questions"] == 10


def test_unknown_corpus_size_and_small_runs():
    run = hubbed()
    run["backend"]["count"] = None
    report = hubness(run)
    assert report["coverage"] is None
    assert report["skewness_includes_unretrieved"] is False
    assert "unknown (the run does not record the corpus size)" in render(report)
    tiny = with_rankings([["hub", "a"], ["hub", "b"]])
    assert hubness(tiny)["hubs"] == []  # a hub needs at least three questions
    with pytest.raises(ValueError, match="hub_share"):
        hubness(run, hub_share=0)
    for q in run["queries"]:
        q["error"] = {"code": "timeout"}
    with pytest.raises(ValueError, match="no successful questions"):
        hubness(run)


def test_cli_reports_compares_and_fails_only_on_new_hubs(tmp_path):
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    base.write_text(json.dumps(spread()))
    cand.write_text(json.dumps(hubbed()))
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(json.dumps({"query_id": q, "text": q, "relevant": r}) for q, r in LABELS.items())
    )
    out = tmp_path / "hubs.json"
    runner = CliRunner()
    args = ["workbench", "hubs", str(cand), "--baseline", str(base), "--dataset", str(data)]
    result = runner.invoke(app, [*args, "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "New hubs: hub" in result.output
    assert json.loads(out.read_text())["comparison"]["new_hubs"][0]["doc_id"] == "hub"
    assert runner.invoke(app, [*args, "--fail-on-new-hubs"]).exit_code == 3
    same = runner.invoke(app, ["workbench", "hubs", str(base), "--baseline", str(base),
                               "--fail-on-new-hubs"])  # fmt: skip
    assert same.exit_code == 0, same.output
    assert runner.invoke(app, ["workbench", "hubs", str(tmp_path / "none.json")]).exit_code == 2
