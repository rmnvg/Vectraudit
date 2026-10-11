"""Cohorts derived from declared context fields: a change that hurts one tenant shows up."""

from __future__ import annotations

import json

import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, dataset_stats, render_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline

HEADER = {"dataset": {"name": "Support", "cohort_fields": ["tenant", "enterprise"]}}


def rows(n_big=20, n_small=12):
    out = []
    for i in range(n_big + n_small):
        tenant = "big" if i < n_big else "small"
        out.append(
            {
                "query_id": f"q{i}",
                "text": f"question {i}",
                "relevant": {f"doc-{i}": 1},
                "context": {"tenant": tenant, "enterprise": tenant == "big", "groups": ["a"]},
                **({"tags": ["faq"]} if i < 3 else {}),  # too few to judge as a cohort
            }
        )
    return out


def jsonl(header, body):
    return "\n".join(json.dumps(r) for r in ([header] if header else []) + body)


def test_declared_fields_become_cohorts_without_rewriting_the_benchmark():
    ds = parse_dataset(jsonl(HEADER, rows()))
    first = ds.queries[0]
    assert first.tags == ("faq",)
    assert first.cohorts == ("enterprise=true", "tenant=big")
    assert first.cohort_tags == ("enterprise=true", "faq", "tenant=big")
    # Derived cohorts are not authored content: exports keep the benchmark as written.
    assert all("tenant=big" not in r.get("tags", []) for r in dataset_rows(ds))
    assert parse_dataset(render_dataset(dataset_rows(ds), ds.meta)).sha256 == ds.sha256
    # Declaring fields is a change of version; without them nothing is derived.
    plain = parse_dataset(jsonl(None, rows()))
    assert plain.sha256 != ds.sha256
    assert plain.queries[0].cohorts == ()
    families = dataset_stats(ds)["tag_families"]
    assert (families["tenant=small"], families["tenant=big"], families["faq"]) == (12, 20, 3)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ([], "cohort_fields"),
        (["tenant", "tenant"], "cohort_fields"),
        (["bad field"], "cohort_fields"),
        ("tenant", "cohort_fields"),
        (["groups"], "must hold one value, not a list"),
    ],
)
def test_bad_declarations_are_refused(fields, message):
    with pytest.raises(DatasetError, match=message):
        parse_dataset(jsonl({"dataset": {"cohort_fields": fields}}, rows(2, 1)))


def fake(break_small: bool):
    def pipeline(request):
        hurt = break_small and request.context.get("tenant") == "small"
        return PipelineResult(hits=[] if hurt else [Hit(id=f"doc-{request.query_id[1:]}")])

    return pipeline


POLICY = parse_policy(
    {
        "metrics": {"ndcg": {"max_drop": 0.5}},  # the suite as a whole tolerates the change
        "min_queries": 10,
        "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.05}}},
    }
)


def test_a_change_that_hurts_one_tenant_fails_its_cohort_everywhere():
    ds = parse_dataset(jsonl(HEADER, rows()))
    options = PipelineOptions(k=1, repeats=1, warmup=0)
    base = execute_pipeline(ds, fake(False), options)
    cand = execute_pipeline(ds, fake(True), options)
    assert "tenant=small" in base["queries"][-1]["tags"]
    result = compare_runs(base, cand, POLICY)
    assert result["verdict"] == "fail"
    failing = {r["rule"] for r in result["reasons"] if r["verdict"] == "fail"}
    assert failing == {"cohort:tenant=small:ndcg.max_drop", "cohort:enterprise=false:ndcg.max_drop"}
    # The hosted check judges the uploaded projection and reaches the same decision.
    hosted = compare_runs(project_run(base), project_run(cand), POLICY)
    assert hosted["verdict"] == "fail"
    assert {r["rule"] for r in hosted["reasons"] if r["verdict"] == "fail"} == failing
    # Without the declaration the same change passes: the loss hides in the average.
    plain = parse_dataset(jsonl(None, rows()))
    assert (
        compare_runs(
            execute_pipeline(plain, fake(False), options),
            execute_pipeline(plain, fake(True), options),
            POLICY,
        )["verdict"]
        == "pass"
    )
