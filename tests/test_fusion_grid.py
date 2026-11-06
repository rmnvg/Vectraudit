"""--fusion-grid and --shortlist-from: candidates from fusion settings, and the held-out rule."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines import optimize as optimize_module
from vectraudit.pipelines.fusion_grid import expand_fusion_grid, shortlist
from vectraudit.pipelines.fusion_replay import REPLAY_FORMAT, ReplayError
from vectraudit.pipelines.manifest import Candidate

BASE = Candidate.model_validate(
    {
        "name": "base",
        "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"},
        "params": {"vector_weight": 0.5, "depth": 10},
        "fixes": [{"name": "keep me out", "addresses": ["not_observed"], "params": {"depth": 20}}],
    }
)


def _expand(grid, taken=("base",), room=20, base=BASE):
    return expand_fusion_grid(base, grid, list(taken), room=room)


def test_grid_points_are_candidates_that_share_the_base_adapter():
    points = _expand("fusion=rrf,wsum;vector_weight=0.3,0.7")
    assert [c.name for c in points] == [
        "base+fusion=rrf,vector_weight=0.3",
        "base+fusion=rrf,vector_weight=0.7",
        "base+fusion=wsum,vector_weight=0.3",
        "base+fusion=wsum,vector_weight=0.7",
    ]
    assert all(c.adapter == BASE.adapter and c.fixes == [] for c in points)
    assert points[2].params == {"vector_weight": 0.3, "depth": 10, "fusion": "wsum"}
    assert BASE.params == {"vector_weight": 0.5, "depth": 10}


def test_settings_a_method_ignores_are_not_repeated_and_the_base_itself_is_dropped():
    points = _expand("fusion=rrf,wsum,bordafuse;fusion_k=20,60;fusion_norm=max,zmuv")
    names = [c.name for c in points]
    assert len(names) == len(set(names))
    # rrf: 2 k values; wsum: 2 norms; bordafuse: neither.
    assert len(points) == 2 + 2 + 1
    assert "base+fusion=bordafuse" in names
    same = BASE.model_copy(update={"params": {"fusion": "rrf", "fusion_k": 60}})
    assert [c.params["fusion_k"] for c in _expand("fusion_k=60,20", base=same)] == [20]
    with pytest.raises(ReplayError, match="base candidate's own"):
        _expand("fusion=rrf;fusion_k=60", base=same)


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ("", "empty"),
        ("depth=10", "Unknown fusion-grid key"),
        ("fusion=condorcet", "fusion must be one of"),
        ("fusion_k=-1", "0 to 10000"),
        ("fusion_norm=l2", "fusion_norm must be one of"),
        ("vector_weight=2", "between 0 and 1"),
        ("vector_weight=nan", "between 0 and 1"),
        ("fusion", "name=v1,v2"),
    ],
)
def test_bad_grids_are_refused(grid, message):
    with pytest.raises(ReplayError, match=message):
        _expand(grid)


def test_grid_must_fit_max_candidates_and_use_a_hybrid_base_and_unique_names():
    with pytest.raises(ReplayError, match=r"makes 4 candidates, but only 3 more fit"):
        _expand("fusion=rrf,wsum;vector_weight=0.3,0.7", room=3)
    custom = Candidate.model_validate(
        {"name": "base", "adapter": {"kind": "python", "target": "pkg:attr"}}
    )
    with pytest.raises(ReplayError, match="hybrid"):
        _expand("fusion=wsum", base=custom)
    with pytest.raises(ReplayError, match="already exists"):
        _expand("fusion=wsum", taken=("base", "base+fusion=wsum"))


def _report(sha, **overrides):
    def entry(method, weight, delta, norm="min_max", k=60):
        return {
            "id": f"{method}-{weight}",
            "method": method,
            "weights": {"vector": weight, "keyword": 1 - weight},
            "fusion_k": k if method == "rrf" else None,
            "fusion_norm": None if method == "rrf" else norm,
            "depth": None,
            "results": {"exact": {"metrics": {"ndcg": {"delta": delta}}}},
        }

    report = {
        "format": REPLAY_FORMAT,
        "format_version": 1,
        "metric": "ndcg",
        "inputs": ["keyword", "vector"],
        "run": {"run_id": "r1", "dataset_sha256": sha},
        "configurations": [
            entry("rrf", 0.3, 0.0),
            entry("rrf", 0.7, 0.02),
            entry("wsum", 0.3, 0.2),
            entry("wsum", 0.7, 0.1),
        ],
    }
    return {**report, **overrides}


def test_shortlist_keeps_the_grid_points_a_replay_ranked_highest():
    points = _expand("fusion=rrf,wsum;vector_weight=0.3,0.7")
    kept, record = shortlist(points, BASE, _report("sha"), "sha", top=2)
    assert [c.name for c in kept] == [
        "base+fusion=wsum,vector_weight=0.3",
        "base+fusion=wsum,vector_weight=0.7",
    ]
    assert record["replay_run_id"] == "r1"
    assert record["grid_points"] == 4
    assert record["evidence"] == "hypothesis"
    assert record["shortlisted"] == [c.name for c in kept]


def test_shortlist_refuses_the_wrong_replay():
    points = _expand("fusion=rrf,wsum;vector_weight=0.3,0.7")
    with pytest.raises(ReplayError, match="different dataset"):
        shortlist(points, BASE, _report("other"), "sha")
    with pytest.raises(ReplayError, match="not a fusion replay"):
        shortlist(points, BASE, {"format": "vectraudit.run"}, "sha")
    with pytest.raises(ReplayError, match="no configuration"):
        shortlist(points, BASE, _report("sha", configurations=[]), "sha")
    with pytest.raises(ReplayError, match="is a point of --fusion-grid"):
        shortlist(_expand("fusion=rrf;vector_weight=0.3"), BASE, _report("sha"), "sha", top=1)
    with pytest.raises(ReplayError, match="at least 1"):
        shortlist(points, BASE, _report("sha"), "sha", top=0)


CORPUS = [
    {"doc_id": "kw", "text": "refund policy refund", "vector": [0.0, 1.0]},
    {"doc_id": "vec", "text": "returns", "vector": [1.0, 0.0]},
    {"doc_id": "mix", "text": "refund", "vector": [0.6, 0.8]},
]


def _manifest(tmp_path, max_candidates=20):
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in CORPUS))
    rows = []
    for i in range(6):
        rows.append(
            {
                "query_id": f"e{i}",
                "text": f"refund policy {i}",
                "vector": [1.0, 0.0],
                "relevant": {"kw": 1},
            }
        )
    for i in range(3):
        rows.append(
            {
                "query_id": f"h{i}",
                "text": f"returns question {i}",
                "vector": [1.0, 0.0],
                "relevant": {"vec": 1},
                "split": "heldout",
            }
        )
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    manifest = {
        "dataset": "queries.jsonl",
        "k": 2,
        "repeats": 1,
        "max_candidates": max_candidates,
        "candidates": [
            {
                "name": "base",
                "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"},
                "params": {"vector_weight": 0.5, "depth": 10},
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _optimize(tmp_path, *args, max_candidates=20):
    manifest = _manifest(tmp_path, max_candidates)
    out = tmp_path / "optimization.json"
    result = CliRunner().invoke(
        app, ["workbench", "optimize", str(manifest), "--out", str(out), *args]
    )
    return result, out


def test_grid_candidates_run_on_exploratory_questions_and_one_held_out_execution(
    tmp_path, monkeypatch
):
    calls = []
    real = optimize_module.execute_pipeline

    def spy(dataset, pipeline, options, **kwargs):
        calls.append(sorted({q.split for q in dataset.queries}))
        return real(dataset, pipeline, options, **kwargs)

    monkeypatch.setattr(optimize_module, "execute_pipeline", spy)
    result, out = _optimize(tmp_path, "--fusion-grid", "fusion=rrf,wsum;vector_weight=0.3,0.7")
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert len(report["candidates"]) == 1 + 4
    assert report["fusion_grid"]["base"] == "base"
    assert report["fusion_grid"]["points"] == len(report["fusion_grid"]["candidates"]) == 4
    assert "shortlist" not in report["fusion_grid"]
    assert calls.count(["heldout"]) == 1
    assert calls.count(["explore"]) == 5
    held = {
        q.query_id for q in load_dataset(tmp_path / "queries.jsonl").queries if q.split != "explore"
    }
    for candidate in report["candidates"]:
        assert not held & {q["query_id"] for q in candidate["run"]["queries"]}
    assert "Fusion grid: 4 of 4 points added to base" in result.output


def test_a_replay_shortlist_chooses_which_grid_points_run(tmp_path):
    sha = load_dataset(_manifest(tmp_path).parent / "queries.jsonl").sha256
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_report(sha)))
    result, out = _optimize(
        tmp_path,
        "--fusion-grid", "fusion=rrf,wsum;vector_weight=0.3,0.7",
        "--shortlist-from", str(replay), "--top", "1",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    names = [c["name"] for c in report["candidates"]]
    assert names == ["base", "base+fusion=wsum,vector_weight=0.3"]
    record = report["fusion_grid"]
    assert record["points"] == 4
    assert record["shortlist"]["shortlisted"] == [names[1]]
    assert record["shortlist"]["evidence"] == "hypothesis"
    assert "(shortlisted by a replay)" in result.output


def test_cli_errors_are_clear(tmp_path):
    too_many, _ = _optimize(
        tmp_path, "--fusion-grid", "fusion=rrf,wsum;vector_weight=0.3,0.7", max_candidates=3
    )
    assert too_many.exit_code == 2
    assert "only 2 more fit" in too_many.output
    orphan, _ = _optimize(tmp_path, "--shortlist-from", str(tmp_path / "replay.json"))
    assert orphan.exit_code == 2
    assert "need --fusion-grid" in orphan.output
    unknown, _ = _optimize(tmp_path, "--fusion-grid", "fusion=wsum", "--grid-base", "nope")
    assert unknown.exit_code == 2
    assert "no candidate 'nope'" in unknown.output
    missing, _ = _optimize(
        tmp_path, "--fusion-grid", "fusion=wsum", "--shortlist-from", str(tmp_path / "gone.json")
    )
    assert missing.exit_code == 2
