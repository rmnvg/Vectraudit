"""Permission variants are correlated evidence, not extra independent questions."""

import copy
import hashlib
import json

import numpy as np
import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import ColumnMap, dataset_rows, import_rows, render_dataset
from vectraudit.evaluation.policy import Policy, parse_policy
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.evaluation.runner import summarize
from vectraudit.pipelines.models import PipelineResult
from vectraudit.pipelines.runner import execute_pipeline
from vectraudit.pipelines.workflows import PermissionIn, permission_rows


def grouped(values, families, **kwargs):
    run = fake_run(values, **kwargs)
    for q, family in zip(run["queries"], families, strict=True):
        q["family_id"] = family
    run["summary"] = summarize(run["queries"], len(values))
    return run


def test_dataset_family_roundtrip_fingerprint_and_split_boundary():
    raw = [{"query_id": "q", "text": "question"}]
    old = parse_dataset(render_dataset(raw))
    # Existing ungrouped dataset hashes retain their previous canonical representation.
    canonical = [
        {
            "query_id": "q",
            "text": "question",
            "vector": None,
            "relevant": {},
            "filter": {},
            "tags": [],
        }
    ]
    assert (
        old.sha256
        == hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
    )
    rows = [
        {**raw[0], "family_id": "q"},
        {"query_id": "variant", "text": "other wording", "family_id": "q"},
    ]
    ds = parse_dataset(render_dataset(rows))
    assert ds.summary()["n_families"] == 1
    assert ds.sha256 != old.sha256
    assert parse_dataset(render_dataset(dataset_rows(ds))).sha256 == ds.sha256
    assert import_rows(rows, ColumnMap())[1]["family_id"] == "q"
    rows[1]["split"] = "heldout"
    with pytest.raises(DatasetError, match="family cannot cross"):
        parse_dataset("\n".join(map(json.dumps, rows)))


@pytest.mark.parametrize("family", [None, "", " ", 42, [], "x" * 513])
def test_invalid_family_ids_rejected_in_datasets_and_runs(family):
    with pytest.raises(DatasetError, match="family_id"):
        parse_dataset(json.dumps({"query_id": "q", "text": "q", "family_id": family}))
    run = fake_run([1.0])
    run["queries"][0]["family_id"] = family
    with pytest.raises(RunFileError, match="family_id"):
        check_run(run)


def test_permission_variants_keep_original_family_including_nested_scenarios():
    original = {"query_id": "q", "text": "question", "family_id": "intent"}
    body = PermissionIn(
        dataset=json.dumps(original),
        query_id="q",
        scenarios=[
            {"name": "first", "context": {"tenant": "a"}, "allowed": ["doc"]},
            {"name": "second", "context": {"tenant": "b"}, "forbidden": ["doc"]},
        ],
    )
    rows = permission_rows(body)["rows"]
    assert {r["family_id"] for r in rows} == {"intent"}
    nested = permission_rows(
        PermissionIn(
            dataset=render_dataset(rows),
            query_id=rows[0]["query_id"],
            scenarios=body.scenarios,
        )
    )["rows"]
    assert {r["family_id"] for r in nested} == {"intent"}
    ds = parse_dataset(render_dataset([original, *rows, *nested]))
    result = execute_pipeline(ds, lambda _: PipelineResult())
    assert result["summary"]["n_families"] == 1
    assert all(q["family_id"] == "intent" for q in result["queries"])


def test_unequal_family_sizes_get_equal_weight_and_identical_bootstrap():
    small = grouped([0.0, 1.0], ["a", "b"])
    large = grouped([0.0] * 40 + [1.0], ["a"] * 40 + ["b"])
    assert small["summary"]["metrics"] == large["summary"]["metrics"]
    assert large["summary"]["metrics"]["ndcg"] == 0.5
    base_small = grouped([0.0, 0.0], ["a", "b"])
    base_large = grouped([0.0] * 41, ["a"] * 40 + ["b"])
    expected = compare_runs(base_small, small, seed=7)["metrics"]["ndcg"]
    actual = compare_runs(base_large, large, seed=7)["metrics"]["ndcg"]
    assert actual["n"] == 41
    assert actual["n_families"] == 2
    assert {k: v for k, v in actual.items() if k != "n"} == {
        k: v for k, v in expected.items() if k != "n"
    }
    assert actual["ci_low"] == 0
    assert actual["ci_high"] == 1


def test_family_mean_uses_paired_measured_rows_only():
    base = grouped([1.0, 0.0, 0.5], ["a", "a", "b"])
    cand = grouped([0.0, 1.0, 0.5], ["a", "a", "b"])
    cand["queries"][1]["metrics"] = None
    result = compare_runs(base, cand)
    metric = result["metrics"]["ndcg"]
    assert metric["n"] == metric["n_families"] == 2
    assert metric["baseline"] == 0.75
    assert metric["candidate"] == 0.25
    assert metric["delta"] == -0.5


def test_minimum_cannot_be_satisfied_by_clones_or_sparse_labels():
    run = grouped([1.0] * 40, ["a"] * 40)
    policy = parse_policy({"min_queries": 30, "metrics": {"ndcg": {"min": 0.5}}})
    result = compare_runs(run, run, policy)
    assert result["verdict"] == "not_comparable"
    assert result["n_queries"] == 40
    assert result["n_families"] == 1
    for i, q in enumerate(run["queries"][10:], 10):
        q["family_id"] = str(i)
        q["metrics"] = None
    result = compare_runs(run, run, policy)
    assert result["n_families"] == 31
    assert result["metrics"]["ndcg"]["n_families"] == 1
    assert result["verdict"] == "not_comparable"


def test_cohort_minimum_uses_measured_families():
    base = grouped([1.0] * 12, ["a"] * 10 + ["b", "c"], tags=[["tenant"]] * 10 + [[], []])
    cand = copy.deepcopy(base)
    for q in cand["queries"][:10]:
        q["metrics"]["ndcg"] = 0
    policy = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 1}},
            "cohorts": {"min_queries": 2, "metrics": {"ndcg": {"max_drop": 0}}},
        }
    )
    result = compare_runs(base, cand, policy)
    cohort = result["cohorts"]["tenant"]
    assert cohort["n"] == 10
    assert cohort["n_families"] == 1
    assert not cohort["judged"]
    assert result["verdict"] == "pass"


def test_family_assignment_change_is_not_comparable():
    base = grouped([1.0, 1.0], ["a", "a"])
    cand = copy.deepcopy(base)
    cand["queries"][1]["family_id"] = "b"
    assert compare_runs(base, cand)["verdict"] == "not_comparable"


def test_hard_failures_are_not_averaged_away_even_below_minimum():
    base = grouped([1.0] * 40, ["a"] * 40)
    for q in base["queries"]:
        q["permission_checked"] = True
        q["no_answer_correct"] = True
    cand = copy.deepcopy(base)
    cand["queries"][0]["forbidden_ids"] = ["private"]
    cand["queries"][1]["no_answer_correct"] = False
    strict = parse_policy({"min_queries": 30, "metrics": {"ndcg": {"min": 0}}})
    result = compare_runs(base, cand, strict)
    assert result["verdict"] == "not_comparable"
    messages = str(result["reasons"])
    assert "Forbidden documents" in messages
    assert "No-answer assertion failed" in messages
    assert compare_runs(base, cand)["verdict"] == "fail"


def test_safety_only_family_counts_and_report_counts():
    base = grouped([1.0] * 40, ["a"] * 40, labelled=False)
    for q in base["queries"]:
        q["permission_checked"] = True
    strict = Policy(min_queries=30, metrics={})
    assert compare_runs(base, base, strict)["verdict"] == "not_comparable"
    result = compare_runs(base, base)
    assert result["safety"]["n_families"] == 1
    assert "40 query rows / 1 families" in render_markdown(result)
    assert "40 paired rows / 1 families" in render_html(result)


def test_checkpoint_resume_retains_family_metadata(tmp_path):
    ds = parse_dataset(
        render_dataset(
            [
                {"query_id": str(i), "text": "question", "family_id": "root", "no_answer": True}
                for i in range(3)
            ]
        )
    )

    def pipeline(_):
        return PipelineResult()

    options = {
        "checkpoint": tmp_path / "checkpoint.sqlite",
        "config": {"pipeline_version": "v1"},
        "corpus_version": "v1",
    }
    first = execute_pipeline(ds, pipeline, **options)
    resumed = execute_pipeline(ds, pipeline, **options, resume=True)
    assert first["queries"] == resumed["queries"]
    assert resumed["summary"]["n_families"] == 1


def test_connector_runs_preserve_families(synthetic):
    from vectraudit.connectors.faiss_index import FaissConnector
    from vectraudit.evaluation.runner import RunOptions, execute_run

    rows = [
        {"query_id": str(i), "family_id": "root", "vector": v.tolist()}
        for i, v in enumerate(np.load(synthetic["vectors"])[:3])
    ]
    ds = parse_dataset(render_dataset(rows))
    result = execute_run(ds, FaissConnector(synthetic["flat_l2"]), RunOptions(k=2, warmup=0))
    assert result["status"] == "complete"
    assert result["summary"]["n_families"] == 1
    assert all(q["family_id"] == "root" for q in result["queries"])


def test_label_export_and_legacy_task_hash(tmp_path):
    from vectraudit.experiments.corpus import Document, make_corpus
    from vectraudit.labelling.export import export_dataset
    from vectraudit.labelling.pooling import pool
    from vectraudit.labelling.task import Judgment, load_task

    ds = parse_dataset(
        render_dataset(
            [{"query_id": str(i), "text": "question", "family_id": "root"} for i in range(2)]
        )
    )
    task = pool(ds, make_corpus([Document("doc", "question")]), k=1)
    task.judgments = [
        Judgment(query_id=str(i), doc_id="doc", annotator="reviewer", grade=2) for i in range(2)
    ]
    exported = parse_dataset(export_dataset(task))
    assert exported.summary()["n_families"] == 1
    assert all(q.family_id == "root" for q in exported.queries)
    # A task written by the previous release has no family field.
    raw = task.model_dump(exclude={"sha256"})
    for q in raw["queries"]:
        del q["family_id"]
    raw["sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    path = tmp_path / "old-task.json"
    path.write_text(json.dumps(raw))
    assert load_task(path).digest() == raw["sha256"]
