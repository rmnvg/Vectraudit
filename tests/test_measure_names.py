"""Metric names in ir-measures notation, and where they appear."""

import pytest

from tests.test_trec import fixture_dataset, write_trec
from vectraudit.evaluation import measure_names as M
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.trec import import_run


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("ndcg", "nDCG@10"),
        ("precision", "P@10"),
        ("recall", "R@10"),
        ("mrr", "RR@10"),
        ("map", "AP@10"),
        ("hit_rate", "Success@10"),
    ],
)
def test_the_notation_is_ir_measures(metric, expected) -> None:
    assert M.notation(metric, 10) == expected
    assert M.label(metric, 10) == f"{expected} ({metric})"


def test_a_relevance_threshold_is_written_as_a_parameter() -> None:
    assert M.notation("precision", 5, rel=2) == "P(rel=2)@5"
    assert M.notation("ndcg", 20, rel=1) == "nDCG(rel=1)@20"


def test_metrics_without_a_standard_name_are_left_alone() -> None:
    assert M.notation("ann_recall", 10) is None
    assert M.label("ann_recall", 10, "ANN recall@k") == "ANN recall@k"
    assert M.label("custom", 10) == "custom"
    assert M.notations({"ndcg": 1, "ann_recall": 2, "map": 3}, 5) == {
        "ndcg": "nDCG@5",
        "map": "AP@5",
    }


def pair(tmp_path):
    ds = fixture_dataset(n=40)
    base = import_run(write_trec(tmp_path / "a.trec", n=40), ds, k=10, label="a")
    cand = import_run(write_trec(tmp_path / "b.trec", n=40), ds, k=10, label="b")
    return base, cand


def test_reports_show_the_notation_beside_the_key_at_the_runs_k(tmp_path) -> None:
    base, cand = pair(tmp_path)
    result = compare_runs(base, cand)
    md = render_markdown(result)
    assert "nDCG@10 (ndcg)" in md
    assert "P@10 (precision)" in md
    assert "AP@10 (map)" in md
    assert "Success@10 (hit_rate)" in md
    page = render_html(result)
    assert "nDCG@10 (ndcg)" in page
    assert "Recall@k" not in md  # the old generic name is gone
    assert result["metric_notation"]["ndcg"] == "nDCG@10"
    assert result["metric_notation"]["hit_rate"] == "Success@10"


def test_the_policy_keys_and_the_verdict_do_not_change(tmp_path) -> None:
    base, cand = pair(tmp_path)
    result = compare_runs(base, cand)
    assert set(result["metrics"]) >= {"ndcg", "precision", "recall", "mrr", "map", "hit_rate"}
    assert result["verdict"] == "pass"
    assert all(" " not in key for key in result["metrics"])  # keys stay machine names
