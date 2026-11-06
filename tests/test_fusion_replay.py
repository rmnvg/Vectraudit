"""Offline fusion replay: validity rules, exactness, bounds and the held-out guard."""

from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.core.fusion import fuse
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.fusion_replay import (
    REPLAY_FORMAT,
    Config,
    ReplayError,
    expand_configs,
    next_step,
    parse_grid,
    render_replay,
    replay,
)
from vectraudit.pipelines.manifest import Fix
from vectraudit.pipelines.models import Hit
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import PythonPipeline, Stage

N = 6


def _keyword(q, parents):
    i = q.query_id[1:]
    return [Hit(id=f"A{i}", score=10.0), Hit(id=f"R{i}", score=9.9), Hit(id=f"C{i}", score=1.0)]


def _vector(q, parents):
    i = q.query_id[1:]
    return [Hit(id=f"B{i}", score=0.9), Hit(id=f"A{i}", score=0.8), Hit(id=f"D{i}", score=0.1)]


def _fusion(keep=None):
    def run(q, parents):
        fused = fuse(
            {n: [(h.id, h.score) for h in parents[n]] for n in ("keyword", "vector")},
            q.params.get("fusion", "rrf"),
            norm=q.params.get("fusion_norm", "min_max"),
        )
        return [Hit(id=d, score=s) for d, s in fused][:keep]

    return run


def _pipeline(tail="truncate", keep=None):
    stages = [
        Stage("keyword", "keyword", _keyword),
        Stage("vector", "vector", _vector),
        Stage("fusion", "fusion", _fusion(keep), ("keyword", "vector")),
    ]
    last = "fusion"
    if tail == "filter":
        stages.append(
            Stage(
                "filter", "filter", lambda q, p: [h for h in p["fusion"] if h.id[0] != "A"], (last,)
            )
        )
        last = "filter"
    elif tail == "rerank":
        stages.append(Stage("rerank", "rerank", lambda q, p: p["fusion"][::-1], (last,)))
        last = "rerank"
    stages.append(Stage("final", "truncate", lambda q, p, last=last: p[last][: q.k], (last,)))
    return PythonPipeline(stages)


def _dataset(relevant="R", extra=None):
    rows = []
    for i in range(N):
        row = {"query_id": f"q{i}", "text": f"question {i}", "relevant": {f"{relevant}{i}": 1}}
        rows.append(json.dumps({**row, **(extra or {}).get(i, {})}))
    return parse_dataset("\n".join(rows))


def _run(ds=None, params=None, **kwargs):
    ds = ds or _dataset()
    run = execute_pipeline(
        ds, _pipeline(**kwargs), PipelineOptions(k=2, warmup=0, params=params or {})
    )
    return ds, run


def _exact(report, label):
    return next(c for c in report["configurations"] if c["id"] == label)["results"]["exact"]


def test_replaying_the_recorded_settings_reproduces_the_recorded_ranking():
    ds, run = _run()
    report = replay(run, ds, [Config("rrf")])
    assert report["format"] == REPLAY_FORMAT
    assert report["summary"]["downstream"] == {"none": N, "filter": 0, "unknown": 0}
    exact = _exact(report, "rrf(k=60)")
    assert exact["n"] == N
    assert all(m["delta"] == 0 for m in exact["metrics"].values())
    for question in report["questions"]:
        shown = question["configurations"]["rrf(k=60)"]["exact"]
        assert shown["recovered"] == shown["lost"] == []


def test_score_fusion_recovers_a_document_rank_fusion_lost_and_the_real_pipeline_agrees():
    ds, run = _run()
    assert all(f"R{i}" not in q["ids"] for i, q in enumerate(run["queries"]))
    report = replay(run, ds, [Config("rrf"), Config("wsum", norm="none")])
    wsum = _exact(report, "wsum(norm=none)")
    assert wsum["metrics"]["recall"]["recorded"] == 0
    assert wsum["metrics"]["recall"]["replay"] == 1
    assert wsum["metrics"]["ndcg"]["ci_low"] > 0
    for question in report["questions"]:
        assert question["configurations"]["wsum(norm=none)"]["exact"]["recovered"] == [
            "R" + question["query_id"][1:]
        ]
    assert report["best"]["configuration"] == "wsum(norm=none)"
    assert report["best"]["basis"] == "exact"
    # The hypothesis holds on the real pipeline, which is what try-fix measures.
    _, fixed = _run(ds, {"fusion": "wsum", "fusion_norm": "none"})
    assert all(f"R{i}" in q["ids"] for i, q in enumerate(fixed["queries"]))


def test_report_is_deterministic_and_carries_no_text():
    ds, run = _run()
    configs = [Config("rrf"), Config("wsum", norm="none")]
    a, b = replay(run, ds, configs, seed=3), replay(run, ds, configs, seed=3)
    assert a == b
    assert "question 1" not in json.dumps(a)


def test_shallow_inputs_are_flagged_per_question_and_counted():
    ds, run = _run()
    run = copy.deepcopy(run)
    stages = run["queries"][0]["evidence"]["stages"]
    next(s for s in stages if s["name"] == "vector")["hits"] = stages[1]["hits"][:1]
    report = replay(run, ds, [Config("rrf")])
    assert report["summary"]["truncated_inputs"] == 1
    flagged = {q["query_id"] for q in report["questions"] if q.get("truncated_inputs")}
    assert flagged == {"q0"}


def test_a_reranker_after_fusion_makes_the_replay_downstream_unknown():
    ds, run = _run(tail="rerank")
    report = replay(run, ds, [Config("rrf")])
    assert report["summary"]["downstream"] == {"none": 0, "filter": 0, "unknown": N}
    assert report["configurations"][0]["results"] == {}
    assert report["best"] is None
    shown = report["questions"][0]["configurations"]["rrf(k=60)"]["before_downstream"]
    assert shown["metrics"] is None


def test_filters_after_fusion_give_bounds_not_a_point_estimate():
    ds = _dataset(relevant="C")
    # The recorded fusion keeps three documents, so the filter never saw C or D.
    ds, run = _run(ds, tail="filter", keep=3)
    report = replay(run, ds, [Config("wsum", norm="none")])
    assert report["summary"]["downstream"]["filter"] == N
    views = report["configurations"][0]["results"]
    assert set(views) == {"optimistic", "pessimistic"}
    # Unseen documents are kept by the optimistic bound and dropped by the pessimistic one.
    assert views["optimistic"]["metrics"]["recall"]["replay"] == 1
    assert views["pessimistic"]["metrics"]["recall"]["replay"] == 0
    assert report["best"]["basis"] == "pessimistic filter bound"


def test_held_out_questions_are_never_replayed_and_reuse_under_a_new_id_is_refused():
    ds = _dataset(extra={0: {"split": "heldout"}})
    ds, run = _run(ds)
    report = replay(run, ds, [Config("rrf")])
    assert report["summary"]["excluded"] == {"heldout": 1}
    assert _exact(report, "rrf(k=60)")["n"] == N - 1
    assert next(q for q in report["questions"] if q["query_id"] == "q0") == {
        "query_id": "q0",
        "excluded": "heldout",
    }
    reused = _dataset(extra={0: {"split": "heldout", "text": "QUESTION  1"}})
    reused_run = {**run, "dataset": {**run["dataset"], "sha256": reused.sha256}}
    with pytest.raises(ReplayError, match="cross explore/heldout"):
        replay(reused_run, reused, [Config("rrf")])


def test_forbidden_documents_in_a_replayed_top_k_are_reported():
    ds = _dataset(relevant="B", extra={i: {"forbidden": [f"R{i}"]} for i in range(N)})
    ds, run = _run(ds)
    report = replay(run, ds, [Config("rrf"), Config("wsum", norm="none")])
    by_id = {c["id"]: c for c in report["configurations"]}
    assert by_id["rrf(k=60)"]["forbidden_questions"] == 0
    assert by_id["wsum(norm=none)"]["forbidden_questions"] == N
    assert report["questions"][0]["configurations"]["wsum(norm=none)"]["exact"]["forbidden"] == [
        "R0"
    ]


def test_questions_without_traces_or_labels_or_a_result_are_excluded_and_counted():
    ds = _dataset(extra={1: {"relevant": {}, "no_answer": True}})
    ds, run = _run(ds)
    run = copy.deepcopy(run)
    del run["queries"][2]["evidence"]["stages"]
    run["queries"][3]["error"] = "boom"
    report = replay(run, ds, [Config("rrf")])
    assert report["summary"]["excluded"] == {"errored": 1, "no_stage_trace": 1, "unlabelled": 1}
    assert report["summary"]["replayed"] == N - 3


def test_inputs_that_cannot_be_replayed_are_refused():
    ds, run = _run()
    with pytest.raises(ReplayError, match="inputs of 'fusion'"):
        replay(run, ds, [Config("rrf")], inputs=["keyword", "nope"])
    with pytest.raises(ReplayError, match="not a fusion stage"):
        replay(run, ds, [Config("rrf")], fusion_stage="keyword")
    with pytest.raises(ReplayError, match="complete run"):
        replay({**run, "status": "partial"}, ds, [Config("rrf")])
    other = _dataset(relevant="X")
    with pytest.raises(ReplayError, match="different dataset"):
        replay(run, other, [Config("rrf")])
    with pytest.raises(ReplayError, match="metric must be"):
        replay(run, ds, [Config("rrf")], metric="f1")
    plain = copy.deepcopy(run)
    for q in plain["queries"]:
        q["evidence"]["stages"] = [s for s in q["evidence"]["stages"] if s["kind"] != "fusion"]
    with pytest.raises(ReplayError, match="no stage of kind 'fusion'"):
        replay(plain, ds, [Config("rrf")])
    with pytest.raises(ReplayError, match="Weights name stages"):
        replay(run, ds, [Config("rrf", weights=(("dense", 0.5), ("keyword", 0.5)))])


def test_a_weight_replay_changes_the_ranking_like_the_baseline_weights():
    ds, run = _run()
    report = replay(run, ds, [Config("rrf", weights=(("keyword", 0.9), ("vector", 0.1)))])
    exact = report["configurations"][0]["results"]["exact"]
    assert exact["metrics"]["recall"]["replay"] == 1  # R now beats B for the second slot
    assert report["configurations"][0]["weights"] == {"keyword": 0.9, "vector": 0.1}


def test_grid_parsing_and_expansion():
    grid = parse_grid("vector_weight=0.3,0.7;fusion_k=20,60")
    assert grid == {"vector_weight": ["0.3", "0.7"], "fusion_k": ["20", "60"]}
    configs = expand_configs(["rrf", "wsum", "bordafuse"], grid, ["keyword", "vector"])
    labels = [c.label for c in configs]
    # k is only for rrf, and rank methods ignore the norm, so nothing is repeated.
    assert len(configs) == 4 + 2 + 2
    assert len(set(labels)) == len(labels)
    assert "rrf(vector_weight=0.3, keyword_weight=0.7, k=20)" in labels
    assert Config("rrf", weights=(("vector", 0.3), ("keyword", 0.7)), rrf_k=20) in configs
    assert expand_configs(["rrf"], {}, ["keyword", "vector"]) == [Config("rrf")]


@pytest.mark.parametrize(
    ("methods", "grid", "message"),
    [
        (["rrf", "condorcet"], {}, "Unknown fusion method"),
        ([], {}, "at least one"),
        (["rrf"], {"bogus": ["1"]}, "Unknown grid key"),
        (["rrf"], {"vector_weight": ["2"]}, "between 0 and 1"),
        (["rrf"], {"fusion_k": ["x"]}, "fusion_k"),
        (["rrf"], {"fusion_k": ["99999"]}, "from 0 to 10000"),
        (["wsum"], {"fusion_norm": ["l2"]}, "fusion_norm must be one of"),
        (["rrf"], {"dense_weight": ["0.5"]}, "does not name an input"),
        (["rrf"], {"vector_weight": ["0.5"], "keyword_weight": ["0.5"]}, "one weight key"),
    ],
)
def test_bad_grids_are_refused(methods, grid, message):
    with pytest.raises(ReplayError, match=message):
        expand_configs(methods, grid, ["keyword", "vector"])


def test_weight_grids_need_two_inputs_and_the_grid_has_a_size_limit():
    with pytest.raises(ReplayError, match="exactly two inputs"):
        expand_configs(["rrf"], {"vector_weight": ["0.5"]}, ["a", "b", "vector"])
    with pytest.raises(ReplayError, match="more than 3"):
        expand_configs(["rrf"], {"fusion_k": ["1", "2", "3", "4"]}, ["a", "b"], limit=3)
    with pytest.raises(ReplayError, match="name=v1,v2"):
        parse_grid("fusion_k")
    with pytest.raises(ReplayError, match="twice"):
        parse_grid("fusion_k=1;fusion_k=2")


def test_next_step_names_a_declared_fix_or_shows_the_fix_to_declare():
    ds, run = _run()
    configs = [Config("rrf"), Config("wsum", norm="none")]
    report = replay(run, ds, configs)
    shown = next_step(report, configs)
    assert shown["kind"] == "declare_fix"
    assert shown["fix"]["params"] == {"fusion": "wsum", "fusion_norm": "none"}
    declared = Fix(
        name="raw score fusion",
        addresses=["lost_downstream:fusion"],
        params={"fusion": "wsum", "fusion_norm": "none"},
    )
    step = next_step(report, configs, [declared], "manifest.json", "base")
    assert step["kind"] == "try_fix"
    assert '--fix "raw score fusion"' in step["command"]
    assert "--candidate base" in step["command"]
    unrelated = Fix(name="other", addresses=["not_observed"], params={"depth": 100})
    assert next_step(report, configs, [unrelated])["kind"] == "declare_fix"
    assert next_step({**report, "best": None}, configs) is None


def test_baseline_weight_params_use_the_input_name():
    ds, run = _run()
    config = Config("rrf", weights=(("vector", 0.3), ("keyword", 0.7)))
    report = replay(run, ds, [config])
    step = next_step(report, [config])
    assert step["fix"]["params"] == {"fusion": "rrf", "fusion_k": 60, "vector_weight": 0.3}


def test_render_summarises_each_configuration():
    ds, run = _run()
    configs = [Config("rrf"), Config("wsum", norm="none")]
    report = replay(run, ds, configs)
    report["next_step"] = next_step(report, configs)
    text = render_replay(report)
    assert "wsum(norm=none) [exact]: ndcg" in text
    assert "6 of 6 questions replayed" in text
    assert "Next: declare this fix" in text


def test_cli_replays_a_run_file_and_writes_a_report(tmp_path):
    ds = _dataset()
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps({"query_id": f"q{i}", "text": f"question {i}", "relevant": {f"R{i}": 1}})
            for i in range(N)
        )
    )
    _, run = _run(ds)
    run_path = tmp_path / "current.run.json"
    run_path.write_text(json.dumps(run))
    out = tmp_path / "replay.json"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "fusion-replay", str(run_path), "--dataset", str(dataset_path),
            "--methods", "rrf,wsum", "--grid", "fusion_norm=none;vector_weight=0.5",
            "--out", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["format"] == REPLAY_FORMAT
    assert report["evidence"] == "hypothesis"
    assert report["best"]["configuration"].startswith("wsum")
    assert report["next_step"]["kind"] == "declare_fix"
    assert "Next: declare this fix" in result.output

    bad = CliRunner().invoke(
        app, ["workbench", "fusion-replay", str(run_path), "--dataset", str(dataset_path),
              "--methods", "condorcet"],
    )  # fmt: skip
    assert bad.exit_code == 2
    assert "Unknown fusion method" in bad.output
    heldout = tmp_path / "other.jsonl"
    heldout.write_text('{"query_id": "q0", "text": "different", "relevant": {"R0": 1}}')
    mismatch = CliRunner().invoke(
        app, ["workbench", "fusion-replay", str(run_path), "--dataset", str(heldout)]
    )
    assert mismatch.exit_code == 2
    assert "different dataset" in mismatch.output
