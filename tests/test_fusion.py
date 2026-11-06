"""core.fusion against ranx 0.3.21 outputs, plus properties the release workflow relies on."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from vectraudit.core.fusion import FUSION_METHODS, NORM_METHODS, fuse, normalize

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranx/fusion.json").read_text())
QUERIES = {q["id"]: q for q in FIXTURE["queries"]}


def _case_id(case):
    return f"{case['method']}-{case['norm']}-{case.get('rrf_k', '')}"


def test_fixture_covers_every_method_and_normalisation():
    methods = {c["method"] for c in FIXTURE["cases"]}
    assert methods == set(FUSION_METHODS)
    score_norms = {c["norm"] for c in FIXTURE["cases"] if c["method"] == "wsum"}
    assert score_norms == set(NORM_METHODS)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=_case_id)
def test_fuse_matches_ranx(case):
    for qid, expected in case["fused"].items():
        stages = {name: [tuple(h) for h in hits] for name, hits in QUERIES[qid]["stages"].items()}
        weights = {n: FIXTURE["weights"][n] for n in stages} if case["weighted"] else None
        ours = fuse(
            stages,
            case["method"],
            weights=weights,
            rrf_k=case.get("rrf_k", 60),
            norm=case["norm"],
        )
        # ranx's output order on equal scores is arbitrary, so compare scores per document.
        got = dict(ours)
        assert set(got) == set(expected), qid
        for doc, score in expected.items():
            if case["method"] == "combmax" and score == RANX_COMBMAX_ABSENT:
                # ranx scores a document a stage did not return as -1e9 and lets that win
                # when every real score is lower (max-normalised negatives). Vectraudit takes
                # the maximum over the stages that returned the document.
                assert got[doc] <= score, (qid, doc)
                continue
            assert got[doc] == pytest.approx(score, abs=1e-9, rel=1e-9), (qid, doc)
        assert ours == sorted(ours, key=lambda t: (-t[1], t[0]))


RANX_COMBMAX_ABSENT = -1e9


def test_combmax_ignores_stages_that_did_not_return_the_document():
    stages = {"a": [("x", -2.0), ("y", -4.0)], "b": [("y", -1.0)]}
    assert dict(fuse(stages, "combmax", norm="none")) == {"x": -2.0, "y": -1.0}


def _random_stages(rng):
    pool = [f"d{i}" for i in range(30)]
    return {
        name: [(d, rng.uniform(-1, 5)) for d in rng.sample(pool, rng.randint(1, 20))]
        for name in ("keyword", "vector", "sparse")
    }


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_stage_order_does_not_matter(method):
    rng = random.Random(1)
    for _ in range(25):
        stages = _random_stages(rng)
        reordered = dict(reversed(list(stages.items())))
        a, b = fuse(stages, method), fuse(reordered, method)
        assert [d for d, _ in a] == [d for d, _ in b]
        assert [s for _, s in a] == pytest.approx([s for _, s in b])


@pytest.mark.parametrize("method", ["rrf", "wsum"])
def test_scaling_every_weight_keeps_the_order(method):
    rng = random.Random(2)
    for _ in range(25):
        stages = _random_stages(rng)
        weights = {name: rng.uniform(0.1, 1) for name in stages}
        scaled = {name: 7.5 * w for name, w in weights.items()}
        assert [d for d, _ in fuse(stages, method, weights=weights)] == [
            d for d, _ in fuse(stages, method, weights=scaled)
        ]


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_one_ranking_alone_keeps_its_order(method):
    ranking = [("c", 9.0), ("a", 5.0), ("b", 1.0), ("d", -2.0)]
    assert [d for d, _ in fuse({"only": ranking}, method)] == ["c", "a", "b", "d"]


def test_rrf_reproduces_the_hybrid_baselines_weighted_sum():
    keyword = [("a", None), ("b", None), ("c", None)]
    vector = [("c", None), ("a", None)]
    fused = fuse(
        {"keyword": keyword, "vector": vector}, "rrf", weights={"keyword": 0.6, "vector": 0.4}
    )
    expected = {"a": 0.6 / 61 + 0.4 / 62, "b": 0.6 / 62, "c": 0.6 / 63 + 0.4 / 61}
    assert dict(fused) == expected


def test_zero_weight_stage_contributes_no_candidates():
    fused = fuse(
        {"keyword": [("a", 1.0)], "vector": [("z", 1.0)]},
        "rrf",
        weights={"keyword": 1, "vector": 0},
    )
    assert [d for d, _ in fused] == ["a"]


def test_ties_break_by_id():
    fused = fuse({"x": [("b", None)], "y": [("a", None)]}, "rrf")
    assert fused == [("a", 1 / 61), ("b", 1 / 61)]


def test_score_methods_need_scores():
    with pytest.raises(ValueError, match="use 'rrf' or 'bordafuse'"):
        fuse({"keyword": [("a", None)], "vector": [("b", 1.0)]}, "wsum")
    with pytest.raises(ValueError, match="finite score"):
        fuse({"keyword": [("a", float("nan"))]}, "combsum")
    assert fuse({"keyword": [("a", None)]}, "bordafuse") == [("a", 1.0)]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"a": -1.0, "b": 1.0}, "non-negative"),
        ({"a": float("inf"), "b": 1.0}, "finite"),
        ({"a": 0.0, "b": 0.0}, "positive"),
        ({"a": 1.0}, "exactly"),
        ({"a": True, "b": 1.0}, "numbers"),
    ],
)
def test_weights_are_validated(weights, message):
    with pytest.raises(ValueError, match=message):
        fuse({"a": [("x", 1.0)], "b": [("y", 1.0)]}, "rrf", weights=weights)


def test_invalid_options_are_refused():
    stages = {"a": [("x", 1.0)]}
    with pytest.raises(ValueError, match="Unknown fusion"):
        fuse(stages, "condorcet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown normalisation"):
        fuse(stages, "wsum", norm="l2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rrf_k"):
        fuse(stages, "rrf", rrf_k=-1)
    with pytest.raises(ValueError, match="depth"):
        fuse(stages, "rrf", depth=0)
    with pytest.raises(ValueError, match="at least one"):
        fuse({}, "rrf")


def test_depth_cuts_each_input_and_duplicates_keep_their_best_rank():
    stages = {"a": [("x", 3.0), ("x", 2.0), ("y", 1.0), ("z", 0.5)]}
    assert [d for d, _ in fuse(stages, "rrf", depth=2)] == ["x", "y"]
    assert dict(fuse(stages, "rrf"))["x"] == 1 / 61


def test_normalize_edge_cases():
    assert normalize([("a", 4.0)], "zmuv") == [("a", 0.0)]
    assert normalize([("a", 2.0), ("b", 2.0)], "min_max") == [("a", 0.0), ("b", 0.0)]
    assert normalize([], "min_max") == []
    assert normalize([], "borda") == []
    assert normalize([("a", None), ("b", None)], "rank") == [("a", 1.0), ("b", 0.5)]
    assert normalize([("a", 1.0)], "borda", candidates=["a", "b", "c"]) == [
        ("a", 1.0),
        ("b", 0.5),
        ("c", 0.5),
    ]
    with pytest.raises(ValueError, match="candidates"):
        normalize([("a", 1.0)], "borda", candidates=["b"])
    with pytest.raises(ValueError, match="finite score"):
        normalize([("a", None)], "max")
