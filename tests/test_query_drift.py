"""Query drift between the benchmark and production questions (Evidently's PSI method)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_trace_formats import docs, langfuse_page
from vectraudit.cli import app
from vectraudit.pipelines.drift import (
    describe,
    production_questions,
    psi,
    query_drift,
    script,
)
from vectraudit.pipelines.traces import Settings

ENGLISH = [
    "How do I reset my password?",
    "Where is my invoice for March?",
    "Can I change the billing email?",
    "How do I cancel my plan?",
    "Why was my card declined?",
    "How do I add a team member?",
] * 10  # 60 benchmark questions


def by_hand(ref_shares, cur_shares):
    return sum((r - c) * math.log(r / c) for r, c in zip(ref_shares, cur_shares, strict=True))


def test_categorical_psi_is_the_textbook_sum():
    ref = ["a"] * 50 + ["b"] * 50
    cur = ["a"] * 90 + ["b"] * 10
    assert psi(ref, cur, numeric=False) == pytest.approx(by_hand([0.5, 0.5], [0.9, 0.1]))
    assert psi(ref, list(ref), numeric=False) == 0.0


def test_empty_bins_get_evidentlys_floor():
    ref = ["a"] * 100
    cur = ["a"] * 50 + ["b"] * 50
    # reference has no "b": its share is floored at 0.0001, as Evidently does
    assert psi(ref, cur, numeric=False) == pytest.approx(by_hand([1.0, 0.0001], [0.5, 0.5]))
    tiny = ["a"] * 99_999 + ["b"]  # smallest share 0.00001 <= 0.0001: floor is a millionth of it
    got = psi(tiny, ["a"] * 10 + ["c"] * 10, numeric=False)
    floor = (1 / 100_000) / 10**6
    assert got == pytest.approx(by_hand([99_999 / 100_000, 1 / 100_000, floor], [0.5, 0.0001, 0.5]))


def test_numbers_are_binned_only_with_enough_distinct_values():
    rng = np.random.default_rng(0)
    ref = list(rng.integers(1, 60, 200))
    cur = list(rng.integers(20, 90, 200))
    edges = np.histogram_bin_edges(np.concatenate([ref, cur]).astype(float), bins="sturges")
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r[r == 0], c[c == 0] = 0.0001, 0.0001
    assert psi(ref, cur, numeric=True) == pytest.approx(float(np.sum((r - c) * np.log(r / c))))
    few = [5] * 30 + [6] * 30  # two distinct values: compared as categories
    assert psi(few, [6] * 60, numeric=True) == psi(few, [6] * 60, numeric=False)
    with pytest.raises(ValueError, match="at least one"):
        psi([], [1], numeric=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Where is my invoice?", "latin"),
        ("Где мой счёт?", "cyrillic"),
        ("我的发票在哪里", "cjk"),
        ("パスワードをリセット", "japanese"),
        ("أين فاتورتي", "arabic"),
        ("12345 ???", "none"),
    ],
)
def test_script_is_the_majority_script_of_the_letters(text, expected):
    assert script(text) == expected


def test_descriptors_of_one_question():
    assert describe("Where is invoice 42?", rare=True) == {
        "words": 4,
        "script": "latin",
        "has_digits": "yes",
        "question_mark": "yes",
        "unseen_term": "yes",
    }
    assert describe("請問發票在哪裡？", rare=False)["question_mark"] == "yes"  # noqa: RUF001 (the fullwidth mark is the point)


def bench(questions, tenant="acme"):
    return [(q, {"tenant": tenant}) for q in questions]


def test_production_like_the_benchmark_shows_no_drift():
    report = query_drift(bench(ENGLISH), bench(ENGLISH), context_fields=["tenant"])
    assert report["status"] == "no_drift"
    assert report["drifted"] == []
    assert not any(d["drift"] for d in report["descriptors"])
    assert report["benchmark_coverage"] == {"questions_in_benchmark": 60, "share": 1.0}
    assert report["unseen_terms"] == []
    assert report["warnings"] == []


def test_a_new_language_tenant_and_vocabulary_are_drift_and_named():
    production = (
        bench(ENGLISH[:30])
        + [
            ("Где мой счёт за март?", {"tenant": "globex"}),
            ("Как сбросить пароль?", {"tenant": "globex"}),
        ]
        * 10
        + [("Export my audit log to SIEM please", {"tenant": "globex"})] * 10
    )
    report = query_drift(bench(ENGLISH), production, context_fields=["tenant"])
    by = {d["name"]: d for d in report["descriptors"]}
    assert report["status"] == "drift"
    assert {"script", "unseen_term", "context.tenant"} <= set(report["drifted"])
    assert by["script"]["new_values"] == [{"value": "cyrillic", "questions": 20}]
    assert by["context.tenant"]["new_values"] == [{"value": "globex", "questions": 30}]
    assert by["script"]["test"] == "z_test"  # two values (latin, cyrillic): Evidently's choice
    assert by["script"]["p_value"] < 0.05
    terms = {t["term"]: t["questions"] for t in report["unseen_terms"]}
    assert terms["siem"] == 10
    assert "password" not in terms
    assert report["benchmark_coverage"]["share"] == pytest.approx(0.5)


def test_longer_questions_drift_on_word_count():
    long = [q.rstrip("?") + " for the enterprise workspace we migrated last quarter?"
            for q in ENGLISH]  # fmt: skip
    report = query_drift(bench(ENGLISH), bench(long))
    words = next(d for d in report["descriptors"] if d["name"] == "words")
    assert words["drift"]
    assert words["current"]["mean"] > words["reference"]["mean"]


def test_unseen_terms_are_measured_leave_one_out_on_the_benchmark():
    reference = bench(["alpha beta", "alpha beta", "alpha gamma"])  # gamma: only one question
    report = query_drift(reference, bench(["alpha beta", "alpha beta", "alpha gamma"]))
    unseen = next(d for d in report["descriptors"] if d["name"] == "unseen_term")
    # the benchmark's own share (1 of 3, leave-one-out) and production's (0 of 3: gamma is
    # in the benchmark vocabulary) are measured the same way and both reported
    assert unseen["reference"] == {"no": 0.6667, "yes": 0.3333}
    assert unseen["current"] == {"no": 1.0}
    assert report["warnings"][0].startswith("Small samples (3 benchmark, 3 production")


def test_missing_context_is_its_own_value():
    report = query_drift(bench(ENGLISH), [(q, {}) for q in ENGLISH], context_fields=["tenant"])
    tenant = next(d for d in report["descriptors"] if d["name"] == "context.tenant")
    assert tenant["new_values"] == [{"value": "(missing)", "questions": 60}]


def test_no_text_on_either_side_is_refused():
    with pytest.raises(ValueError, match="benchmark has no question text"):
        query_drift([("", {})], bench(ENGLISH))
    with pytest.raises(ValueError, match="No production question text"):
        query_drift(bench(ENGLISH), [("  ", {})])


def write_traces(tmp_path, questions):
    rows = []
    for i, question in enumerate(questions):
        page = langfuse_page(question, docs("d1"), trace=f"t{i}")["data"]
        rows += page
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_production_questions_are_read_from_traces_with_context(tmp_path):
    path = write_traces(tmp_path, ["How do I reset my password?", "Где мой счёт?"])
    found, unreadable = production_questions(path, Settings(), ["tenant"])
    assert sorted(t for t, _ in found) == ["How do I reset my password?", "Где мой счёт?"]
    assert all(c == {"tenant": "acme"} for _, c in found)  # from the root's metadata
    assert unreadable == 0


def test_cli_reports_drift_writes_json_and_can_fail(tmp_path):
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": f"q{i}", "text": q, "relevant": {"d1": 1},
                        "context": {"tenant": "acme"}})
            for i, q in enumerate(ENGLISH)
        )
    )  # fmt: skip
    traces = write_traces(tmp_path, ["Где мой счёт за март?", "Как сбросить пароль?"] * 20)
    out = tmp_path / "drift.json"
    runner = CliRunner()
    args = ["workbench", "drift", str(traces), str(data), "--context-field", "tenant"]
    result = runner.invoke(app, [*args, "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "Query drift: DRIFT (40 production vs 60 benchmark questions" in result.output
    assert "new: cyrillic (40)" in result.output
    assert json.loads(out.read_text())["status"] == "drift"
    failing = runner.invoke(app, [*args, "--fail-on-drift"])
    assert failing.exit_code == 3
    steady = write_traces(tmp_path, ENGLISH)
    calm = runner.invoke(app, ["workbench", "drift", str(steady), str(data), "--fail-on-drift"])
    assert calm.exit_code == 0, calm.output
    assert "No descriptor moved past the threshold." in calm.output
    missing = runner.invoke(app, ["workbench", "drift", str(tmp_path / "none.jsonl"), str(data)])
    assert missing.exit_code == 2


# --- the tests Evidently selects by default ------------------------------------------------

scipy = pytest.importorskip  # resolved per test so the module runs without SciPy


def test_p_values_and_distances_match_scipy():
    stats = scipy("scipy.stats")
    distance = scipy("scipy.spatial.distance")
    from vectraudit.pipelines import drift

    rng = np.random.default_rng(3)
    for _ in range(50):
        n, m = (int(x) for x in rng.integers(40, 400, 2))
        a, b = rng.normal(0, 1, n), rng.normal(rng.uniform(-0.4, 0.4), 1, m)
        # Asymptotic with Stephens' correction; SciPy's default (what Evidently calls) is
        # exact for small samples. Measured to stay within 0.03 of it at these sizes.
        assert drift.ks_p_value(a, b) == pytest.approx(stats.ks_2samp(a, b).pvalue, abs=0.03)
        assert drift.wasserstein_normed(a, b) == pytest.approx(
            stats.wasserstein_distance(a, b) / np.std(a), rel=1e-9
        )
        ca = list(rng.choice(list("abcd"), n, p=[0.4, 0.3, 0.2, 0.1]))
        cb = list(rng.choice(list("abcd"), m, p=[0.35, 0.3, 0.2, 0.15]))
        keys = sorted(set(ca) | set(cb))
        expected = [ca.count(k) * m / n for k in keys]
        observed = [cb.count(k) for k in keys]
        assert drift.chi_square_p_value(ca, cb) == pytest.approx(
            stats.chisquare(observed, expected).pvalue, rel=1e-9, abs=1e-12
        )
        ref_share, cur_share = drift._binned(ca, cb, numeric=False)
        assert drift.jensen_shannon(ca, cb, numeric=False) == pytest.approx(
            distance.jensenshannon(ref_share, cur_share), rel=1e-9
        )
        ya, yb = ["yes"] * 30 + ["no"] * (n - 30), ["yes"] * 60 + ["no"] * (m - 60)
        p1, p2 = 30 / n, 60 / m
        pooled = (30 + 60) / (n + m)
        z = (p1 - p2) / np.sqrt(pooled * (1 - pooled) * (1 / n + 1 / m))
        assert drift.z_test_p_value(ya, yb) == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-9)


def test_a_value_the_reference_never_has_is_drift_under_chi_square():
    from vectraudit.pipelines.drift import chi_square_p_value

    assert chi_square_p_value(["a", "b", "c"] * 20, ["a", "b", "d"] * 20) == 0.0


@pytest.mark.parametrize(
    ("n", "numeric", "values", "expected"),
    [
        (200, True, "many", "ks"),
        (200, True, "few", "chi_square"),
        (200, False, "few", "chi_square"),
        (200, False, "two", "z_test"),
        (1500, True, "many", "wasserstein"),
        (1500, True, "few", "jensen_shannon"),
        (1500, False, "few", "jensen_shannon"),
    ],
)
def test_the_test_is_chosen_as_evidently_chooses_it(n, numeric, values, expected):
    from vectraudit.pipelines.drift import drift_test

    rng = np.random.default_rng(0)
    sample = {
        "many": lambda: rng.normal(0, 1, n).tolist(),
        "few": lambda: rng.integers(0, 4, n).tolist(),
        "two": lambda: rng.choice(["yes", "no"], n).tolist(),
    }[values]
    result = drift_test(sample(), sample(), numeric=numeric)
    assert result["test"] == expected
    assert ("p_value" in result) == (expected in ("ks", "chi_square", "z_test"))


def test_default_tests_keep_false_alarms_near_five_percent_where_psi_does_not():
    from vectraudit.pipelines.drift import drift_test

    rng = np.random.default_rng(11)
    trials, n = 200, 200
    auto = psi_alarms = 0
    for _ in range(trials):
        a, b = rng.poisson(8, n).tolist(), rng.poisson(8, n).tolist()  # same distribution
        auto += drift_test(a, b, numeric=True)["drift"]
        psi_alarms += drift_test(a, b, numeric=True, method="psi")["drift"]
    assert auto / trials < 0.1
    assert psi_alarms / trials > 0.5  # why PSI is not the default at these sizes


def test_psi_can_still_be_chosen_explicitly():
    report = query_drift(bench(ENGLISH), bench(ENGLISH), method="psi")
    assert {d["test"] for d in report["descriptors"]} == {"psi"}
    assert report["method"]["name"] == "psi"
    with pytest.raises(ValueError, match="method must be one of"):
        query_drift(bench(ENGLISH), bench(ENGLISH), method="magic")
