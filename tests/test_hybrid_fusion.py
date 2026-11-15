"""Fusion options of the hybrid baseline, the fusion_stage helper and the built-in fix."""

from __future__ import annotations

import json

import pytest

from vectraudit.pipelines.investigate import available_fixes
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.manifest import Candidate
from vectraudit.pipelines.models import Hit, PipelineRequest
from vectraudit.pipelines.runtime import PythonPipeline, Stage, fusion_stage

DOCS = [
    {"doc_id": "kw-strong", "text": "refund policy refund", "vector": [0.0, 1.0]},
    {"doc_id": "kw-weak", "text": "refund", "vector": [0.2, 0.98]},
    {"doc_id": "vec-strong", "text": "returns", "vector": [1.0, 0.0]},
    {"doc_id": "vec-weak", "text": "money back", "vector": [0.9, 0.44]},
]


def _run(params, k=4):
    request = PipelineRequest(
        query_id="q", text="refund policy", vector=[1.0, 0.0], k=k, params=params
    )
    return hybrid_pipeline(DOCS)(request)


def test_default_fusion_is_weighted_rrf():
    explicit = _run({"vector_weight": 0.5, "fusion": "rrf", "fusion_k": 60})
    assert [h.id for h in _run({"vector_weight": 0.5}).hits] == [h.id for h in explicit.hits]


def test_score_fusion_changes_the_ranking_from_rank_fusion():
    rrf = [h.id for h in _run({"vector_weight": 0.5}).hits]
    wsum = [h.id for h in _run({"vector_weight": 0.5, "fusion": "wsum"}).hits]
    assert set(rrf) == set(wsum)
    assert rrf != wsum
    fused = next(
        s for s in _run({"vector_weight": 0.5, "fusion": "wsum"}).stages if s.kind == "fusion"
    )
    assert all(h.score is not None for h in fused.hits)


def test_fusion_k_and_norm_reach_the_stage():
    a = _run({"vector_weight": 0.5, "fusion_k": 1})
    b = _run({"vector_weight": 0.5, "fusion_k": 1000})
    assert [h.score for h in a.stages[2].hits] != [h.score for h in b.stages[2].hits]
    z = _run({"vector_weight": 0.5, "fusion": "wsum", "fusion_norm": "zmuv"})
    assert any(h.score < 0 for h in z.stages[2].hits)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"fusion": "condorcet"}, "fusion must be one of"),
        ({"fusion_k": -1}, "fusion_k"),
        ({"fusion_k": 2.5}, "fusion_k"),
        ({"fusion_norm": "l2"}, "fusion_norm must be one of"),
    ],
)
def test_invalid_fusion_params_are_refused(params, message):
    with pytest.raises(ValueError, match=message):
        _run(params)


def test_fusion_stage_declares_a_measured_fusion_of_its_parents():
    def keyword(q, parents):
        return [Hit(id="a", score=3.0), Hit(id="b", score=1.0)]

    def vector(q, parents):
        return [Hit(id="b", score=0.9), Hit(id="c", score=0.1)]

    pipeline = PythonPipeline(
        [
            Stage("keyword", "keyword", keyword),
            Stage("vector", "vector", vector),
            fusion_stage("fused", ["keyword", "vector"], "combmnz"),
            Stage("final", "truncate", lambda q, p: p["fused"][: q.k], ("fused",)),
        ]
    )
    result = pipeline(PipelineRequest(query_id="q", k=2))
    fused = result.stages[2]
    assert (fused.kind, fused.parents, fused.source) == (
        "fusion",
        ["keyword", "vector"],
        "measured",
    )
    assert [h.id for h in result.hits] == ["b", "a"]


def test_fusion_stage_checks_its_options_when_built():
    with pytest.raises(ValueError, match="exactly"):
        fusion_stage("f", ["a", "b"], "rrf", weights={"a": 1.0})
    with pytest.raises(ValueError, match="Unknown fusion"):
        fusion_stage("f", ["a"], "condorcet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parent"):
        fusion_stage("f", [], "rrf")


def test_score_fusion_fix_is_offered_once_vectors_are_fused(tmp_path):
    (tmp_path / "vectors.jsonl").write_text(json.dumps({"doc_id": "a", "text": "x", "vector": [1]}))

    def names(params):
        candidate = Candidate.model_validate(
            {
                "name": "h",
                "adapter": {"kind": "hybrid", "corpus": "vectors.jsonl"},
                "params": params,
            }
        )
        return {f.name for f in available_fixes(candidate, tmp_path)}

    assert "switch to weighted score fusion" not in names({})
    assert "switch to weighted score fusion" in names({"vector_weight": 0.5})
    assert "switch to weighted score fusion" not in names({"vector_weight": 0.5, "fusion": "wsum"})
    fix = next(
        f
        for f in available_fixes(
            Candidate.model_validate(
                {
                    "name": "h",
                    "adapter": {"kind": "hybrid", "corpus": "vectors.jsonl"},
                    "params": {"vector_weight": 0.5},
                }
            ),
            tmp_path,
        )
        if f.name == "switch to weighted score fusion"
    )
    assert fix.params == {"fusion": "wsum"}
    assert fix.addresses == ["not_observed", "lost_downstream:fusion"]
