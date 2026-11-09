"""HTML and Markdown check reports."""

from __future__ import annotations

import copy

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.report import render_html, render_markdown

EVIL = '<script>alert("x")</script>|`tick`'


def _failing_result() -> dict:
    tags = [["faq"]] * 10 + [[EVIL]] * 30
    base = fake_run([1.0] * 40, ann=[1.0] * 40, tags=tags, label="main")
    cand = fake_run([0.0] * 10 + [1.0] * 30, ann=[0.9] * 40, tags=tags, label="pr-12")
    cand["queries"][0]["query_id"] = EVIL
    base["queries"][0]["query_id"] = EVIL
    cand["search"]["params"] = {"nprobe": 2}
    policy = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.01}},
            "cohorts": {"min_queries": 5, "metrics": {"ndcg": {"max_drop": 0.1}}},
        }
    )
    return compare_runs(base, cand, policy)


def test_html_report_contents_and_escaping() -> None:
    result = _failing_result()
    page = render_html(result)
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    assert "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page
    assert '<span class="badge fail">Fail</span>' in page
    for section in ("Why", "Quality", "Latency", "Queries by nDCG@", "Cohorts", "What changed"):
        assert f">{section}" in page
    assert "search.params.nprobe" in page
    assert "pr-12" in page
    assert "prefers-color-scheme: dark" in page


def test_html_report_for_a_run_that_could_not_be_compared() -> None:
    run = fake_run([1.0] * 3)
    cand = copy.deepcopy(run)
    cand["status"] = "failed"
    cand["error"] = {"code": "auth", "message": "bad <key>", "hint": None}
    page = render_html(compare_runs(run, cand))
    assert '<span class="badge error">Error</span>' in page
    assert "bad &lt;key&gt;" in page
    assert ">Quality" not in page


def test_markdown_summary() -> None:
    result = _failing_result()
    md = render_markdown(result, report_url="https://ci.example/report.html")
    assert md.startswith("## ❌ Retrieval check: Fail")
    assert "| nDCG@" in md  # ir-measures notation at the run's k, beside the key
    assert " (ndcg) |" in md
    assert "<script>" not in md
    assert "&lt;script&gt;" in md
    assert "\\|\\`tick\\`" in md
    assert "regressed" in md
    assert "[Full report](https://ci.example/report.html)" in md
    run = fake_run([1.0] * 40)
    ok = render_markdown(compare_runs(run, copy.deepcopy(run)))
    assert ok.startswith("## ✅ Retrieval check: Pass")
    assert "Worst:" not in ok
