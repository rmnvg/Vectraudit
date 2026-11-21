"""Attribute a difference between two configurations to the settings that changed (47.1)."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.attribution import (
    AttributionError,
    attribute,
    combination_params,
    plan_attribution,
)
from vectraudit.pipelines.manifest import Candidate, open_adapter
from vectraudit.pipelines.runner import PairedSide, PipelineOptions, execute_interleaved

# The spike's pipeline (scripts/attribution_spike.py), per question and deterministic: the new
# embedding model is weaker on a domain (40% of questions), the planted culprit; a larger chunk
# size and the new reranker are each a small gain alone and a loss together, the planted
# interaction; the prompt changes nothing retrieval returns.
APP = """
import zlib
import numpy as np
from vectraudit.pipelines import Hit, PipelineResult

POOL, RELEVANT, K = 40, 4, 10


def world(i):
    rng = np.random.default_rng(i)
    truth = np.zeros(POOL, dtype=bool)
    truth[rng.choice(POOL, RELEVANT, replace=False)] = True
    return truth, rng.normal(size=POOL), rng.normal(size=POOL), rng.random() < 0.4


def ranking(i, params):
    truth, common, own, domain = world(i)
    signal = 1.5
    if params.get("embedding") == "new":
        signal -= 0.7 * domain
    big, v2 = params.get("chunk_size") == 512, params.get("reranker") == "v2"
    signal += 0.12 * big + 0.12 * v2 - 0.55 * (big and v2)
    key = zlib.crc32(repr(sorted(params.items())).encode())
    jitter = np.random.default_rng([i, key]).normal(size=POOL)
    score = truth * signal + 0.8 * common + 0.6 * own + 0.1 * jitter
    return [f"d{j}" for j in np.argsort(-score)[:K]]


def pipeline(request):
    i = int(request.query_id.split("-")[1])
    return PipelineResult(hits=[Hit(id=d) for d in ranking(i, request.params)])
"""
BASE = {"embedding": "old", "chunk_size": 256, "reranker": "v1", "prompt": "a"}
CAND = {"embedding": "new", "chunk_size": 512, "reranker": "v2", "prompt": "b"}
QUESTIONS = 200


def rows(n: int = QUESTIONS, heldout: int = 0, offset: int = 0) -> list[dict]:
    namespace: dict = {}
    exec(APP, namespace)
    out = []
    for i in range(offset, offset + n + heldout):
        truth = namespace["world"](i)[0]
        row = {
            "query_id": f"q-{i}",
            "text": f"question {i}",
            "relevant": {f"d{j}": 1 for j in np.flatnonzero(truth)},
        }
        if i >= offset + n:
            row["split"] = "heldout"
        out.append(row)
    return out


def setup(tmp_path: Path, n: int = QUESTIONS, heldout: int = 0, cand=CAND, offset: int = 0) -> Path:
    (tmp_path / "attribution_app.py").write_text(APP)
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows(n, heldout, offset))
    )
    adapter = {"kind": "python", "target": "attribution_app:pipeline"}
    manifest = {
        "dataset": "queries.jsonl",
        "k": 10,
        "repeats": 1,
        "candidates": [
            {"name": "current", "adapter": adapter, "pipeline_version": "1", "params": BASE},
            {"name": "release", "adapter": adapter, "pipeline_version": "1", "params": cand},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def parts(tmp_path: Path, **kwargs):
    path = setup(tmp_path, **kwargs)
    spec = json.loads(path.read_text())
    base, cand = (Candidate.model_validate(c) for c in spec["candidates"])
    data = parse_dataset((tmp_path / "queries.jsonl").read_text())
    return base, cand, open_adapter(base.adapter, tmp_path), data


def candidate(params: dict, **extra) -> Candidate:
    return Candidate.model_validate(
        {"name": "c", "adapter": {"kind": "python", "target": "m:f"}, "params": params, **extra}
    )


def true_contributions(n: int = 3000) -> dict[str, float]:
    """Exact Shapley recall@10 contributions over many questions the test did not use."""
    namespace: dict = {}
    exec(APP, namespace)
    names = list(BASE)
    data = rows(n, offset=10_000)
    values = {}
    for r in range(len(names) + 1):
        for on in itertools.combinations(names, r):
            params = {k: (CAND[k] if k in on else BASE[k]) for k in names}
            hits = [
                len(set(namespace["ranking"](10_000 + i, params)) & set(row["relevant"])) / 4
                for i, row in enumerate(data)
            ]
            values[frozenset(on)] = float(np.mean(hits))
    out = {}
    for p in names:
        others = [x for x in names if x != p]
        out[p] = sum(
            math.factorial(r)
            * math.factorial(len(names) - r - 1)
            / math.factorial(len(names))
            * (values[frozenset(s) | {p}] - values[frozenset(s)])
            for r in range(len(names))
            for s in itertools.combinations(others, r)
        )
    return out


def test_plan_lists_the_changed_settings_and_keeps_to_the_budget():
    base, cand = candidate(BASE), candidate(CAND)
    plan = plan_attribution(base, cand)
    assert plan.design == "full"
    assert plan.runs == 16
    assert [s.name for s in plan.settings] == sorted(BASE)
    assert plan.settings[0].as_dict()["baseline"] == {"chunk_size": 256}

    five = candidate({**CAND, "top_k": 50})
    with pytest.raises(AttributionError, match="default budget of 16 runs"):
        plan_attribution(base, five)
    assert plan_attribution(base, five, max_runs=32).design == "full"
    screen = plan_attribution(base, five, max_runs=20)
    assert (screen.design, screen.runs) == ("screen", 12)
    with pytest.raises(AttributionError, match="allows neither"):
        plan_attribution(base, five, max_runs=11)

    grouped = plan_attribution(base, five, together=[["embedding", "top_k"]])
    assert grouped.runs == 16
    assert grouped.settings[0].name == "embedding+top_k"
    assert grouped.settings[0].as_dict()["confounded"] is True


def test_plan_refuses_what_it_cannot_mix():
    base = candidate(BASE)
    with pytest.raises(AttributionError, match="same parameters"):
        plan_attribution(base, candidate(BASE))
    with pytest.raises(AttributionError, match="pipeline_version"):
        plan_attribution(base, candidate(CAND, pipeline_version="2"))
    other = Candidate.model_validate(
        {"name": "c", "adapter": {"kind": "python", "target": "m:g"}, "params": CAND}
    )
    with pytest.raises(AttributionError, match="adapter"):
        plan_attribution(base, other)
    with pytest.raises(AttributionError, match="do not change"):
        plan_attribution(base, candidate(CAND), together=[["embedding", "nope"]])
    with pytest.raises(AttributionError, match="only one"):
        plan_attribution(
            base, candidate(CAND), together=[["embedding", "prompt"], ["prompt", "reranker"]]
        )


def test_a_removed_parameter_is_removed_in_the_combinations(tmp_path):
    base = candidate({"depth": 4, "rerank": True})
    cand = candidate({"depth": 8})
    plan = plan_attribution(base, cand)
    assert [s.name for s in plan.settings] == ["depth", "rerank"]
    assert combination_params(plan, base.params, frozenset({"rerank"})) == {"depth": 4}


def test_the_planted_culprit_and_interaction_are_found(tmp_path):
    """Phase acceptance: the culprit's interval holds its planted share; the pair is named."""
    base, cand, pipeline, data = parts(tmp_path)
    report = attribute(data, pipeline, base, plan_attribution(base, cand), k=10, metric="recall")
    rows_ = {r["name"]: r for r in report["contributions"]}
    assert report["runs"] == 16
    assert report["leader"] == {"setting": "embedding", "clear": True, "tied_with": []}
    assert report["contributions"][0]["name"] == "embedding"
    assert sum(r["contribution"] for r in report["contributions"]) == pytest.approx(
        report["difference"]
    )
    truth = true_contributions()
    lo, hi = rows_["embedding"]["interval"]
    assert lo <= truth["embedding"] <= hi
    material = [i["settings"] for i in report["interactions"] if i["material"]]
    assert material == [["chunk_size", "reranker"]]
    # Each is a gain alone; their share of the drop comes from the interaction.
    assert rows_["reranker"]["alone"] > 0
    assert rows_["reranker"]["contribution"] < 0
    assert "chunk_size and reranker interact" in report["message"]
    # Reverting either of the pair also removes their interaction, so it recovers more than
    # reverting the culprit: the best single revert is not the largest contribution.
    revert = report["suggested_revert"]
    assert revert["setting"] in ("chunk_size", "reranker")
    assert revert["recovers"] > -rows_["embedding"]["reverted"]
    assert revert["tied_with"] == [] or revert["tied_with"][0] in ("chunk_size", "reranker")
    assert abs(rows_["prompt"]["contribution"]) < 0.01
    assert report["status"] == "exploratory"


def test_settings_that_change_together_are_one_group(tmp_path):
    base, cand, pipeline, data = parts(tmp_path, n=120)
    plan = plan_attribution(base, cand, together=[["embedding", "prompt"]])
    report = attribute(data, pipeline, base, plan, k=10, metric="recall")
    names = [r["name"] for r in report["contributions"]]
    assert sorted(names) == ["chunk_size", "embedding+prompt", "reranker"]
    assert "embedding" not in names
    assert report["runs"] == 8
    assert "Changed together and attributed as one: embedding+prompt" in report["message"]


def test_the_screen_design_finds_the_same_answer_in_fewer_runs(tmp_path):
    base, cand, pipeline, data = parts(tmp_path)
    plan = plan_attribution(base, cand, max_runs=11)
    assert (plan.design, plan.runs) == ("screen", 10)
    report = attribute(data, pipeline, base, plan, k=10, metric="recall")
    assert report["runs"] == 11  # one more run for the flagged pair
    assert report["leader"]["setting"] == "embedding"
    assert [i["settings"] for i in report["interactions"] if i["material"]] == [
        ["chunk_size", "reranker"]
    ]
    assert any("three settings" in a for a in report["assumptions"])
    assert report["untested_pairs"] == []


def test_a_screen_without_budget_for_the_pair_says_so(tmp_path):
    base, cand, pipeline, data = parts(tmp_path)
    report = attribute(
        data, pipeline, base, plan_attribution(base, cand, max_runs=10), k=10, metric="recall"
    )
    assert report["untested_pairs"] == [["chunk_size", "reranker"]]
    assert report["interactions"] == []
    assert "did not identify: chunk_size, reranker" in report["message"]


def test_close_contributions_are_not_ranked(tmp_path):
    # Without the embedding change only chunk size and reranker remain, sharing one interaction
    # and equal in truth. (Picking the larger of two equal shares and testing it is wrong about
    # one time in twenty; these questions are not such a time.)
    base, cand, pipeline, data = parts(
        tmp_path, cand={**CAND, "embedding": "old"}, n=80, offset=1000
    )
    report = attribute(data, pipeline, base, plan_attribution(base, cand), k=10, metric="recall")
    assert report["leader"]["clear"] is False
    revert = report["suggested_revert"]
    assert {revert["setting"], *revert["tied_with"]} == {"chunk_size", "reranker"}
    assert "validate both" in revert["note"]
    assert "cannot be told apart" in report["message"]


def test_held_out_questions_are_never_run(tmp_path):
    base, cand, pipeline, data = parts(tmp_path, n=30, heldout=10)
    report = attribute(data, pipeline, base, plan_attribution(base, cand), k=10)
    assert report["questions"] == 30
    for run in report["run_records"].values():
        assert all(
            not q["query_id"].endswith(tuple(f"-{i}" for i in range(30, 40)))
            for q in run["queries"]
        )


def test_runs_are_interleaved_and_rotate():
    calls: list[str] = []

    def make(name):
        def pipeline(request):
            from vectraudit.pipelines import Hit, PipelineResult

            calls.append(name)
            return PipelineResult(hits=[Hit(id="a")])

        return pipeline

    data = parse_dataset(
        "\n".join(
            json.dumps({"query_id": f"q{i}", "text": "t", "relevant": {"a": 1}}) for i in range(3)
        )
    )
    sides = [PairedSide(make(n), PipelineOptions(k=1, warmup=0, label=n)) for n in ("a", "b", "c")]
    runs = execute_interleaved(data, sides)
    assert calls == ["a", "b", "c", "b", "c", "a", "c", "a", "b"]
    assert len(runs[0]["timing"]["interleaved_with"]) == 2
    assert "paired_with" not in runs[0]["timing"]


def test_cli_writes_the_report_and_runs(tmp_path):
    path = setup(tmp_path, n=60)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "attribute",
            str(path),
            "--baseline",
            "current",
            "--candidate",
            "release",
            "--metric",
            "recall",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Plan: 4 settings differ" in result.output
    assert "every combination, 16 runs" in result.output
    assert "embedding" in result.output
    report = json.loads((out / "attribution.json").read_text())
    assert report["format"] == "vectraudit.attribution"
    assert "run_records" not in report
    assert len(list(out.glob("*.run.json"))) == 16

    refused = CliRunner().invoke(
        app,
        [
            "workbench",
            "attribute",
            str(path),
            "--baseline",
            "current",
            "--candidate",
            "release",
            "--max-calls",
            "10",
        ],
    )
    assert refused.exit_code == 2
    assert "exceed --max-calls" in refused.output
