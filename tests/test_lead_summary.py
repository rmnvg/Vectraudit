"""Plain-language decisions and trends from ledger rows: honest wording, no injected Markdown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_eval_compare import fake_run
from vectraudit.cloud.lead import explain, trends
from vectraudit.cloud.ledger import summarize_check
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.exceptions import apply_exceptions, failing_rules
from vectraudit.evaluation.policy import parse_policy

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
POLICY = {"metrics": {"ndcg": {"max_drop": 0.02}}, "min_queries": 10}


def entry(result, **extra):
    return {
        "kind": "decision",
        "at": T0,
        "verdict": result["verdict"],
        "baseline_name": "main",
        "baseline_run_id": "base-run",
        "candidate_run_id": "cand-run-123456789",
        "candidate_label": "reranker-v2",
        "candidate_commit": "abcdef1234567",
        "policy_digest": "f" * 64,
        "check_id": "c1",
        "evidence_available": True,
        "summary": summarize_check(result),
        **extra,
    }


def check(base_values, cand_values, raw=POLICY, **cand):
    base = fake_run(base_values)
    candidate = fake_run(cand_values, run_id="cand", **cand)
    return compare_runs(base, candidate, parse_policy(raw))


def test_a_clear_regression_reads_blocked_with_its_reasons_and_numbers():
    result = check([1.0] * 40, [1.0] * 20 + [0.0] * 20)
    summary = explain(entry(result))
    assert summary["verdict"] == "fail"
    assert summary["headline"] == "reranker-v2 @ abcdef1234 is blocked: 1 release rule fails."
    assert summary["action"].startswith("Do not release")
    assert summary["changes"][0] == (
        "nDCG (ranking quality) worsened from 1.000 to 0.500 (-0.500) on 40 questions."
    )
    assert any("ndcg" in m.lower() for m in summary["matters"])
    assert not any("includes zero" in u for u in summary["uncertain"])
    assert "**Fail**" in summary["markdown"]
    assert "**What changed**" in summary["markdown"]


def test_a_pass_inside_the_noise_says_it_is_not_proof_of_safety():
    values = [0.6 + 0.3 * ((q % 3) - 1) for q in range(40)]
    shuffled = values[1:] + values[:1]
    result = check(values, shuffled)
    assert result["verdict"] == "pass"
    summary = explain(entry(result))
    assert summary["headline"] == "reranker-v2 @ abcdef1234 passes the release policy."
    # Individual questions moved both ways; that is reported, but nothing blocks.
    assert [m for m in summary["matters"] if "scored lower" not in m] == []
    noise = [u for u in summary["uncertain"] if "includes zero for nDCG" in u]
    assert noise
    assert all("not proof of safety" in u for u in noise)


def test_small_samples_expired_reports_and_warnings_are_stated():
    result = check([1.0] * 12, [1.0] * 12)
    summary = explain(entry(result, evidence_available=False))
    joined = " ".join(summary["uncertain"])
    assert "Only 12 independent question families" in joined
    assert "expired under retention" in joined


def test_accepted_failures_never_read_as_a_pass():
    raw = {**POLICY, "exceptions": {"allow": ["metric"], "approver": "engineer"}}
    rules = parse_policy(raw)
    base, cand = fake_run([1.0] * 40), fake_run([1.0] * 20 + [0.0] * 20, run_id="cand")
    result = compare_runs(base, cand, rules)
    accepted = apply_exceptions(
        result,
        [
            {
                "id": "e1",
                "rule": rule,
                "accepted_by": "lead@example.test",
                "reason": "Known trade-off",
                "expires_at": "2026-11-01T00:00:00+00:00",
            }
            for rule in failing_rules(result)
        ],
        rules,
    )
    summary = explain(entry(accepted))
    assert summary["verdict"] == "accepted"
    assert "every failure is accepted by exception" in summary["headline"]
    assert "pass" not in summary["headline"].lower()
    accepted_lines = [m for m in summary["matters"] if m.startswith("Accepted by exception")]
    assert accepted_lines == ["Accepted by exception: metric:ndcg.max_drop."]
    assert summary["markdown"].startswith("**Fail, accepted**")


@pytest.mark.parametrize(
    ("verdict", "change", "words"),
    [
        ("error", {"status": "partial"}, "could not be completed"),
        (
            "not_comparable",
            {"dataset": {"sha256": "e" * 64, "n_queries": 40}},
            "could not be compared",
        ),
    ],
)
def test_errors_and_incomparable_runs_ask_for_a_rerecording(verdict, change, words):
    base = fake_run([1.0] * 40)
    cand = {**fake_run([1.0] * 40, run_id="cand"), **change}
    result = compare_runs(base, cand, parse_policy(POLICY))
    summary = explain(entry(result))
    assert summary["verdict"] == verdict
    assert words in summary["headline"]
    assert "Re-record" in summary["action"]
    assert not any(c.startswith("No configuration change") for c in summary["changes"])


def test_markdown_escapes_uploaded_labels():
    result = check([1.0] * 40, [1.0] * 40)
    hostile = "[click](https://evil.example) <img src=x> `code` | table"
    md = explain(entry(result, candidate_label=hostile, candidate_commit=None))["markdown"]
    assert "[click](" not in md
    assert "<img" not in md.replace("\\<img", "")
    assert "\\[click\\]" in md


def row(at, dataset, verdict, ndcg, base, p95=None, policy="p1", check_id=None):
    return {
        "at": at,
        "verdict": verdict,
        "dataset_sha256": dataset,
        "policy_digest": policy,
        "check_id": check_id,
        "candidate_label": f"cand-{ndcg}",
        "candidate_run_id": "run",
        "candidate_commit": None,
        "summary": {
            "primary_metric": "ndcg",
            "metrics": {
                "ndcg": {"baseline": base, "candidate": ndcg, "ci_low": -0.1, "ci_high": 0.1},
                "recall": {"baseline": base, "candidate": ndcg},
            },
            "latency": {"baseline_p95": 10.0, "candidate_p95": p95, "compared": True}
            if p95 is not None
            else None,
        },
    }


def test_trends_split_by_benchmark_version_and_mark_policy_changes():
    day = timedelta(days=1)
    rows = [
        row(T0 + 3 * day, "new", "pass", 0.70, 0.65, p95=12.0, check_id="c4"),
        row(T0, "old", "pass", 0.80, 0.80, p95=10.0),
        row(T0 + day, "old", "fail", 0.60, 0.80, p95=11.0, policy="p2"),
        row(T0 + 2 * day, "old", "pass", 0.82, 0.80),
    ]
    series = trends(rows)
    assert [s["dataset_sha256"] for s in series] == ["new", "old"]  # most recent first
    old = series[1]
    assert old["decisions"] == 3
    assert old["verdicts"] == {"fail": 1, "pass": 2}
    assert list(old["metrics"]) == ["ndcg", "recall"]  # primary metric first
    points = old["metrics"]["ndcg"]["points"]
    assert [p["value"] for p in points] == [0.80, 0.60, 0.82]  # oldest first
    assert old["metrics"]["ndcg"]["change"] == pytest.approx(0.02)
    assert points[0]["ci_low"] == -0.1
    assert [p["value"] for p in old["latency_p95"]["points"]] == [10.0, 11.0]
    assert old["latency_p95"]["change"] == pytest.approx(1.0)
    assert old["policy_changes"] == [T0 + day, T0 + 2 * day]
    new = series[0]
    assert new["metrics"]["ndcg"]["points"][0]["check_id"] == "c4"
    assert new["policy_changes"] == []
    assert trends([]) == []


def test_a_failure_inside_the_noise_still_stands():
    summary = {
        "metrics": {
            "ndcg": {
                "baseline": 0.75,
                "candidate": 0.72,
                "delta": -0.03,
                "ci_low": -0.07,
                "ci_high": 0.02,
                "n": 40,
                "n_families": 40,
            }
        },
        "primary_metric": "ndcg",
        "reasons": ["ndcg dropped by 0.03; the policy allows 0.01."],
        "n_reasons": 1,
    }
    result = explain({"verdict": "fail", "summary": summary, "candidate_label": "x"})
    (note,) = result["uncertain"]
    assert "includes zero for nDCG (-0.070 to +0.020)" in note
    assert note.endswith("so the failure still stands.")
    assert result["headline"] == "x is blocked: 1 release rule fails."
