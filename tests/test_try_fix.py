"""Investigate a failed question, try a declared fix, validate it under the policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines.investigate import (
    available_fixes,
    investigate,
    try_fix,
    try_plan,
    validate_fix,
)
from vectraudit.pipelines.manifest import Candidate, Fix, candidate_config, open_adapter
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

# Each question's fixed ranking. "search" keeps the top `depth`; `rerank` moves "rel" first;
# "final" keeps k=2. With depth 2, "rel" (third) is never retrieved.
APP = """
from vectraudit.pipelines import PythonPipeline, Stage, Hit

ORDER = {
    "target": ["x", "y", "rel", "z"],
    "victim": ["y", "x", "rel", "z"],
    "held-target": ["x", "y", "rel", "z"],
    "held-victim": ["y", "x", "rel", "z"],
}

def search(q, parents):
    order = ORDER.get(q.query_id, ["x", "y", "w"])
    return [Hit(id=d) for d in order[: q.params.get("depth", 2)]]

def rerank(q, parents):
    hits = parents["search"]
    if q.params.get("rerank"):
        hits = sorted(hits, key=lambda h: h.id != "rel")
    return hits

pipeline = PythonPipeline([
    Stage("search", "vector", search),
    Stage("rerank", "rerank", rerank, ("search",)),
    Stage("final", "truncate", lambda q, p: p["rerank"][: q.k], ("rerank",)),
])
"""
ROWS = [
    {"query_id": "target", "text": "t", "relevant": {"rel": 1}},
    {"query_id": "victim", "text": "v", "relevant": {"x": 1}},
    *({"query_id": f"plain{i}", "text": f"p{i}", "relevant": {"x": 1}} for i in range(3)),
    {"query_id": "held-target", "text": "ht", "relevant": {"rel": 1}, "split": "heldout"},
    {"query_id": "held-plain", "text": "hp", "relevant": {"x": 1}, "split": "heldout"},
]
FIXES = [
    {"name": "deeper search", "addresses": ["not_observed"], "params": {"depth": 4}},
    {
        "name": "deeper search and rerank",
        "addresses": ["not_observed", "lost_downstream:truncate"],
        "params": {"depth": 4, "rerank": True},
    },
]


def setup(tmp_path: Path, rows=ROWS, module="fix_trial_app") -> Path:
    (tmp_path / f"{module}.py").write_text(APP)
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    manifest = {
        "dataset": "queries.jsonl",
        "k": 2,
        "repeats": 1,
        "candidates": [
            {
                "name": "current",
                "adapter": {"kind": "python", "target": f"{module}:pipeline"},
                "pipeline_version": "1",
                "fixes": FIXES,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def parts(tmp_path: Path, rows=ROWS):
    path = setup(tmp_path, rows)
    candidate = Candidate.model_validate(json.loads(path.read_text())["candidates"][0])
    pipeline = open_adapter(candidate.adapter, tmp_path)
    data = parse_dataset((tmp_path / "queries.jsonl").read_text())
    return candidate, pipeline, data


def fix(candidate: Candidate, name: str) -> Fix:
    return next(f for f in candidate.fixes if f.name == name)


def test_investigation_names_the_loss_causes_and_matching_fixes(tmp_path):
    candidate, pipeline, data = parts(tmp_path)
    run = execute_pipeline(data, pipeline, PipelineOptions(k=2, warmup=0, label="current"))
    result = investigate(run, "target", available_fixes(candidate))
    assert result["losses"] == [{"class": "not_observed", "doc_id": "rel"}]
    assert [f["name"] for f in result["fixes"]] == ["deeper search", "deeper search and rerank"]
    assert "corpus and index" in result["hypotheses"][0]
    assert any("embedding" in c for c in result["cautions"])
    assert result["hypotheses_note"].endswith("none is established.")


def test_a_fix_that_only_moves_the_loss_is_not_a_recovery(tmp_path):
    candidate, pipeline, data = parts(tmp_path)
    trial = try_fix(
        data, pipeline, candidate, fix(candidate, "deeper search"), ["target"], k=2, sample=10
    )
    target = next(r for r in trial["rows"] if r["query_id"] == "target")
    assert target["recovered"] == []
    [moved] = target["moved"]
    assert (moved["to"]["class"], moved["to"]["stage"]) == ("lost_downstream", "final")
    assert trial["recovered_focus"] == 0
    assert trial["regressed"] == []
    assert trial["status"] == "exploratory"
    assert trial["runs"]["current"]["timing"]["paired_with"] == trial["runs"]["fixed"]["run_id"]


def test_recovery_and_regressions_are_both_reported(tmp_path):
    candidate, pipeline, data = parts(tmp_path)
    trial = try_fix(
        data,
        pipeline,
        candidate,
        fix(candidate, "deeper search and rerank"),
        ["target"],
        k=2,
        sample=10,
    )
    target = next(r for r in trial["rows"] if r["query_id"] == "target")
    assert [loss["doc_id"] for loss in target["recovered"]] == ["rel"]
    assert trial["recovered_focus"] == 1
    assert trial["regressed"] == ["victim"]  # "rel" now pushes its relevant "x" out
    victim = next(r for r in trial["rows"] if r["query_id"] == "victim")
    assert victim["introduced"][0]["stage"] == "final"
    assert all(
        q.split == "explore"
        for q in data.queries
        if q.query_id in {r["query_id"] for r in trial["rows"]}
    )


def test_trial_plan_is_seeded_and_never_uses_held_out_questions(tmp_path):
    _, _, data = parts(tmp_path)
    assert try_plan(data, ["target"], 2, 7) == try_plan(data, ["target"], 2, 7)
    assert len(try_plan(data, ["target"], 2, 7)) == 3
    with pytest.raises(ValueError, match="reserved for validation"):
        try_plan(data, ["held-target"], 2, 0)
    with pytest.raises(ValueError, match="not in the benchmark"):
        try_plan(data, ["nope"], 2, 0)


def test_only_a_policy_pass_on_held_out_questions_approves(tmp_path):
    candidate, pipeline, data = parts(tmp_path)
    policy = parse_policy({"metrics": {"ndcg": {"max_drop": 0.0}}, "min_queries": 1})
    good = fix(candidate, "deeper search and rerank")
    approved = validate_fix(data, pipeline, candidate, good, policy, k=2)
    assert (approved["status"], approved["approved"]) == ("approved_under_policy", True)
    assert approved["questions"] == 2
    assert approved["promote"]["params"] == {"depth": 4, "rerank": True}
    unchecked = validate_fix(data, pipeline, candidate, good, None, k=2)
    assert (unchecked["status"], unchecked["approved"]) == ("exploratory_only", False)

    rows = [
        *ROWS,
        {"query_id": "held-victim", "text": "hv", "relevant": {"x": 1}, "split": "heldout"},
    ]
    other = tmp_path / "victim"
    other.mkdir()
    candidate, pipeline, data = parts(other, rows)
    strict = parse_policy(
        {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 1, "max_regressed_queries": 0}
    )
    refused = validate_fix(data, pipeline, candidate, fix(candidate, good.name), strict, k=2)
    assert refused["status"] == "not_approved"
    assert any("regressed" in r for r in refused["check"]["reasons"])


def test_validation_refuses_without_held_out_questions(tmp_path):
    candidate, pipeline, data = parts(tmp_path, [r for r in ROWS if "split" not in r])
    with pytest.raises(ValueError, match="no held-out questions"):
        validate_fix(data, pipeline, candidate, candidate.fixes[0], None, k=2)


def test_fixes_cannot_address_safety_failures_or_change_identity(tmp_path):
    base = {"name": "c", "adapter": {"kind": "python", "target": "m:f"}}
    with pytest.raises(ValidationError, match="addresses 'permission'"):
        Candidate.model_validate(
            {**base, "fixes": [{"name": "f", "addresses": ["permission"], "params": {"a": 1}}]}
        )
    with_fixes = Candidate.model_validate({**base, "fixes": FIXES})
    assert candidate_config(with_fixes, tmp_path) == candidate_config(
        Candidate.model_validate(base), tmp_path
    )
    run = {
        "queries": [
            {
                "query_id": "q",
                "error": None,
                "forbidden_ids": ["secret"],
                "evidence": {"stages": [], "failures": []},
            }
        ]
    }
    result = investigate(run, "q", with_fixes.fixes)
    assert result["fixes"] == []
    assert result["unaddressed"] == ["permission"]
    assert any("never a deployable repair" in c for c in result["cautions"])


def test_hybrid_baseline_offers_only_fixes_it_can_run(tmp_path):
    (tmp_path / "plain.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x"}))
    (tmp_path / "vectors.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "vector": [1]}))
    for corpus, expected in [
        ("plain.jsonl", {"raise candidate depth", "use the phrase reranker"}),
        (
            "vectors.jsonl",
            {"raise candidate depth", "use the phrase reranker", "add vector similarity to fusion"},
        ),
    ]:
        candidate = Candidate.model_validate(
            {"name": "h", "adapter": {"kind": "hybrid", "corpus": corpus}}
        )
        assert {f.name for f in available_fixes(candidate, tmp_path)} == expected


def test_cli_investigates_and_tries_a_fix(tmp_path):
    manifest = setup(tmp_path, module="fix_trial_cli_app")
    candidate = Candidate.model_validate(json.loads(manifest.read_text())["candidates"][0])
    data = parse_dataset((tmp_path / "queries.jsonl").read_text())
    run = execute_pipeline(
        data,
        open_adapter(candidate.adapter, tmp_path),
        PipelineOptions(k=2, warmup=0, label="current"),
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    shown = CliRunner().invoke(
        app, ["workbench", "investigate", str(manifest), str(tmp_path / "run.json"), "target"]
    )
    assert shown.exit_code == 0, shown.output
    assert "lost rel: not observed" in shown.output
    assert "deeper search and rerank" in shown.output
    (tmp_path / "policy.json").write_text(
        json.dumps({"metrics": {"ndcg": {"max_drop": 0.0}}, "min_queries": 1})
    )
    tried = CliRunner().invoke(
        app,
        [
            "workbench",
            "try-fix",
            str(manifest),
            "--fix",
            "deeper search and rerank",
            "--query",
            "target",
            "--sample",
            "0",
            "--validate",
            "--policy",
            str(tmp_path / "policy.json"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert tried.exit_code == 0, tried.output
    assert "1 of 1 failing question(s) fully recovered" in tried.output
    assert "Approved under the release policy" in tried.output
    assert json.loads((tmp_path / "out" / "validation.json").read_text())["approved"] is True
    limited = CliRunner().invoke(
        app,
        [
            "workbench",
            "try-fix",
            str(manifest),
            "--fix",
            "deeper search",
            "--query",
            "target",
            "--max-calls",
            "1",
        ],
    )
    assert limited.exit_code == 2
    assert "exceed --max-calls" in limited.output
