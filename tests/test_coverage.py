"""Coverage regions: groups of documents no benchmark question is close to."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.core import atlas as A
from vectraudit.core import coverage as V
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.benchmark import CorpusVectors, audit_benchmark

runner = CliRunner()
D = 16
TOPICS = ["billing invoices", "password reset", "shipping delivery", "refund policy",
          "api tokens", "tax receipts"]  # fmt: skip
SIZES = [300, 300, 300, 200, 100, 40]  # the refund cluster holds 200/1240 = 16% of documents
NO_QUESTIONS = 3  # index of "refund policy"


def scene(seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((len(TOPICS), D)).astype("float32") * 12
    ids, vectors, titles, group = [], [], [], []
    for g, (topic, n) in enumerate(zip(TOPICS, SIZES, strict=True)):
        for i in range(n):
            ids.append(f"{g}-{i}")
            vectors.append(centres[g] + rng.standard_normal(D).astype("float32"))
            titles.append(f"{topic} guide {i}")
            group.append(g)
    questions = [
        centres[g] + rng.standard_normal(D).astype("float32")
        for g in range(len(TOPICS))
        for _ in range(4)
        if g != NO_QUESTIONS
    ]
    return ids, np.asarray(vectors), titles, np.asarray(group), np.asarray(questions), centres


def test_the_group_no_question_is_near_is_found_and_described() -> None:
    ids, x, titles, group, q, _ = scene()
    result = V.coverage(ids, x, q, frozenset(), titles, seed=0)
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    regions = [r for r in result["regions"] if r["cluster"] in finding["clusters"]]
    # The flagged group really is the refund topic, not a fragment that happens to be empty.
    members = np.concatenate([group[r["member_rows"]] for r in regions])
    assert (members == NO_QUESTIONS).mean() > 0.95
    assert len(members) == SIZES[NO_QUESTIONS]
    region = regions[0]
    assert finding["evidence"] == "hypothesis"
    assert finding["message"].startswith("About 16% of documents form a group no benchmark")
    assert {"refund", "policy"} <= set(finding["terms"])
    assert region["questions"] == 0
    assert "2D maps distort distance" in result["note"]


def test_groups_with_questions_and_small_groups_are_not_findings() -> None:
    ids, x, titles, _, q, _ = scene()
    result = V.coverage(ids, x, q, frozenset(), titles)
    flagged = {c for f in result["findings"] for c in f["clusters"]}
    # A region with a question nearest to it is never part of a finding.
    assert all(r["questions"] == 0 for r in result["regions"] if r["cluster"] in flagged)
    assert any(r["questions"] > 0 for r in result["regions"])
    # Regions of covered topics sit close to their questions; the flagged ones are far.
    ratios = {r["cluster"]: r["distance_ratio"] for r in result["regions"]}
    assert min(ratios[c] for c in flagged) > result["far_factor"]
    # A stricter threshold removes the finding; a laxer one cannot add a covered group.
    assert V.coverage(ids, x, q, frozenset(), titles, min_share=0.2)["findings"] == []


def test_counts_per_region_add_up() -> None:
    ids, x, titles, _, q, _ = scene()
    relevant = frozenset(ids[:50])
    result = V.coverage(ids, x, q, relevant, titles)
    assert sum(r["documents"] for r in result["regions"]) == len(ids)
    assert sum(r["questions"] for r in result["regions"]) == len(q)
    assert sum(r["relevant_documents"] for r in result["regions"]) == 50
    assert result["k"] == V.default_k(len(ids))


def test_it_is_deterministic_and_seeded() -> None:
    ids, x, titles, _, q, _ = scene()
    a = V.public(V.coverage(ids, x, q, frozenset(), titles, seed=1))
    b = V.public(V.coverage(ids, x, q, frozenset(), titles, seed=1))
    assert a == b
    assert json.dumps(a, allow_nan=False)


def test_with_no_questions_there_are_no_findings() -> None:
    ids, x, titles, _, _, _ = scene()
    result = V.coverage(ids, x, np.empty((0, D)), frozenset(), titles)
    assert result["findings"] == []  # nothing to be far from


def test_kmeans_validates_and_handles_tiny_corpora() -> None:
    x = np.random.default_rng(0).standard_normal((5, 3)).astype("float32")
    with pytest.raises(ValueError, match="k must be"):
        V.kmeans(x, 0)
    with pytest.raises(ValueError, match="k must be"):
        V.kmeans(x, 6)
    centroids, labels = V.kmeans(x, 5)
    assert len(centroids) == 5
    assert sorted(labels.tolist()) == [0, 1, 2, 3, 4]
    assert V.default_k(1) == 1
    assert V.default_k(10_000) == V.MAX_CLUSTERS
    with pytest.raises(ValueError, match="min_share"):
        V.coverage(["a"], x[:1], x[:1], frozenset(), min_share=0)


def test_large_corpora_are_fitted_on_a_sample_and_still_assigned_in_full(monkeypatch) -> None:
    monkeypatch.setattr(V, "FIT_ROWS", 200)
    ids, x, titles, _, q, _ = scene()
    result = V.coverage(ids, x, q, frozenset(), titles)
    assert sum(r["documents"] for r in result["regions"]) == len(ids)
    assert len(result["findings"]) == 1


def test_top_terms_prefer_what_distinguishes_a_group() -> None:
    texts = {0: ["invoice export invoice"] * 3, 1: ["password reset invoice"] * 3}
    terms = V.top_terms(texts, n=2)
    assert terms[0][0] == "invoice" or "export" in terms[0]
    assert "password" in terms[1]
    assert "the" not in V.top_terms({0: ["the and for"]})[0]


# --- in the benchmark audit and the atlas -----------------------------------


def dataset_for(q, heldout_near=None):
    rows = [
        {"query_id": f"q{i}", "vector": v.tolist(), "relevant": {"0-0": 1}} for i, v in enumerate(q)
    ]
    if heldout_near is not None:
        rows.append(
            {"query_id": "h", "vector": heldout_near.tolist(), "relevant": {"3-0": 1},
             "split": "heldout"}
        )  # fmt: skip
    return parse_dataset("\n".join(map(json.dumps, rows)))


def test_audit_benchmark_reports_the_uncovered_group() -> None:
    ids, x, titles, _, q, _ = scene()
    audit = audit_benchmark(dataset_for(q), corpus=CorpusVectors(ids, x, titles))
    cov = audit["embedding_coverage"]
    assert len(cov["findings"]) == 1
    assert "member_rows" not in cov["regions"][0]
    assert audit["coverage"] == {}  # the tag coverage key is untouched
    assert audit_benchmark(dataset_for(q))["embedding_coverage"] is None


def test_a_heldout_question_near_the_group_does_not_cover_it() -> None:
    ids, x, titles, _, q, centres = scene()
    near = centres[NO_QUESTIONS]
    audit = audit_benchmark(dataset_for(q, heldout_near=near), corpus=CorpusVectors(ids, x, titles))
    assert len(audit["embedding_coverage"]["findings"]) == 1


def test_a_benchmark_without_vectors_says_coverage_was_not_measured() -> None:
    ids, x, titles, _, _, _ = scene()
    ds = parse_dataset(json.dumps({"query_id": "a", "text": "hello"}))
    cov = audit_benchmark(ds, corpus=CorpusVectors(ids, x, titles))["embedding_coverage"]
    assert cov["findings"] == []
    assert "not measured" in cov["note"]


def test_the_atlas_regions_agree_with_the_findings() -> None:
    ids, x, titles, group, q, _ = scene()
    questions = [A.AtlasQuestion(f"q{i}", v) for i, v in enumerate(q)]
    atlas = A.build_atlas(ids, x, questions, doc_titles=titles, seed=0)
    findings = atlas.sidecar["findings"]
    assert len(findings) == 1
    flagged = [r for r in atlas.sidecar["regions"] if r["uncovered"]]
    assert sorted(r["cluster"] for r in flagged) == findings[0]["clusters"]
    assert sum(r["documents"] for r in flagged) == findings[0]["documents"] == SIZES[NO_QUESTIONS]
    assert {"refund", "policy"} <= set(findings[0]["terms"])
    # The region's centre is where its documents are drawn.
    members = np.flatnonzero(group == NO_QUESTIONS)
    proj = A.fit_pca(x, 2).transform(x)[members]
    centres = np.array([r["centre"] for r in flagged])
    assert (centres.min(axis=0) >= proj.min(axis=0) - 1.5).all()
    assert (centres.max(axis=0) <= proj.max(axis=0) + 1.5).all()
    assert "member_rows" not in flagged[0]
    # An uncovered region lists its most representative documents, for a person to judge.
    pool = [d["id"] for r in flagged for d in r["pool"]]
    assert 0 < len(pool) <= A.POOL_SIZE * len(flagged)
    assert all(i.startswith(f"{NO_QUESTIONS}-") for i in pool)
    assert all("pool" not in r for r in atlas.sidecar["regions"] if not r["uncovered"])
    json.dumps(atlas.sidecar, allow_nan=False)


def test_a_heldout_question_does_not_cover_a_region_on_the_atlas() -> None:
    ids, x, titles, _, q, centres = scene()
    questions = [A.AtlasQuestion(f"q{i}", v) for i, v in enumerate(q)]
    questions.append(A.AtlasQuestion("h", centres[NO_QUESTIONS], heldout=True))
    for show in (False, True):
        atlas = A.build_atlas(ids, x, questions, doc_titles=titles, show_heldout=show)
        assert len(atlas.sidecar["findings"]) == 1


def test_the_benchmark_command_prints_the_hypothesis(tmp_path) -> None:
    ids, x, titles, _, q, _ = scene()
    rows = [
        {"doc_id": d, "text": t, "vector": v.tolist()}
        for d, t, v in zip(ids, titles, x, strict=True)
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(map(json.dumps, rows)))
    qrows = [
        {"query_id": f"q{i}", "vector": v.tolist(), "relevant": {"0-0": 1}} for i, v in enumerate(q)
    ]
    (tmp_path / "ds.jsonl").write_text("\n".join(map(json.dumps, qrows)))
    out = runner.invoke(
        app,
        ["workbench", "benchmark", str(tmp_path / "ds.jsonl"), "--out", str(tmp_path / "b.json"),
         "--corpus", str(tmp_path / "corpus.jsonl")],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "Hypothesis: About 16% of documents form a group no benchmark question" in out.output
    report = json.loads((tmp_path / "b.json").read_text())
    assert report["embedding_coverage"]["findings"][0]["evidence"] == "hypothesis"
