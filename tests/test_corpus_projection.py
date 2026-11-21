"""Project recall from corpus samples to the full corpus (Phase 48)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.experiments.manifest import ExperimentError, parse_manifest
from vectraudit.experiments.project import _pairs, plan_sizes, project_experiment
from vectraudit.experiments.scale import Thinned, fit_mixing, measured, project, rate_grid

DOCS = 6000
QUESTIONS = 150
DIM = 24


class Fixed:
    """A custom embedder whose vectors are fixed per text: documents and noisy queries."""

    def __init__(self, noise: float, seed: int) -> None:
        rng = np.random.default_rng(1)
        centers = rng.normal(size=(60, DIM))
        docs = centers[rng.integers(0, 60, DOCS)] + 0.8 * rng.normal(size=(DOCS, DIM))
        self.docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
        self.relevant = np.random.default_rng(2).choice(DOCS, QUESTIONS, replace=False)
        jitter = np.random.default_rng(seed).normal(size=(QUESTIONS, DIM))
        q = self.docs[self.relevant] + noise * jitter / np.sqrt(DIM) * 4
        self.queries = q / np.linalg.norm(q, axis=1, keepdims=True)

    def embed_documents(self, texts):
        return np.stack([self.docs[int(t.split()[1])] for t in texts]).astype(np.float32)

    def embed_queries(self, texts):
        return np.stack([self.queries[int(t.split()[1])] for t in texts]).astype(np.float32)

    def truth(self, k: int = 10) -> float:
        s = self.queries @ self.docs.T
        mine = s[np.arange(QUESTIONS), self.relevant][:, None]
        return float(((s > mine).sum(1) < k).mean())


EMBEDDERS = {"easy": Fixed(0.25, 3), "hard": Fixed(0.4, 4)}


def write_inputs(tmp_path: Path, extra_relevant: bool = False) -> Path:
    with (tmp_path / "corpus.jsonl").open("w") as f:
        for i in range(DOCS):
            f.write(json.dumps({"doc_id": f"d{i}", "text": f"doc {i}"}) + "\n")
    relevant = EMBEDDERS["easy"].relevant
    with (tmp_path / "queries.jsonl").open("w") as f:
        for n, r in enumerate(relevant):
            labels = {f"d{r}": 1}
            if extra_relevant and n == 0:
                labels[f"d{relevant[1]}"] = 1
            f.write(json.dumps({"query_id": f"q{n}", "text": f"query {n}", "relevant": labels}))
            f.write("\n")
    manifest = {
        "name": "projection",
        "corpus": {"path": "corpus.jsonl"},
        "queries": "queries.jsonl",
        "k": 10,
        "candidates": [
            {
                "name": name,
                "embedder": {"type": "custom", "model": name},
                "chunking": {"method": "document"},
            }
            for name in EMBEDDERS
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def manifest(tmp_path: Path, **kwargs):
    path = write_inputs(tmp_path, **kwargs)
    return parse_manifest(json.loads(path.read_text()), tmp_path)


def test_sizes_are_nested_and_an_extrapolation_beyond_the_multiple_is_refused():
    assert plan_sizes(10_000, 100, [0.1, 0.05], 10_000, 10) == [500, 1000]
    with pytest.raises(ExperimentError, match="measure at least 2000 documents"):
        plan_sizes(10_000, 100, [0.1], 20_000, 10)
    assert plan_sizes(10_000, 100, [0.1], 20_000, 20) == [1000]
    with pytest.raises(ExperimentError, match="no more than the 600 labelled"):
        plan_sizes(10_000, 600, [0.05, 0.1], 10_000, 10)
    with pytest.raises(ExperimentError, match="smaller than the corpus"):
        plan_sizes(10_000, 100, [0.1], 5_000, 10)
    with pytest.raises(ExperimentError, match="between 0 and 1"):
        plan_sizes(10_000, 100, [0.0], 10_000, 10)


def test_the_projection_covers_the_full_corpus_recall_and_ranks_the_candidates(tmp_path):
    """Phase acceptance on a small case: the interval holds the full-corpus truth."""
    report = project_experiment(
        manifest(tmp_path), fractions=[0.05, 0.075, 0.1], embedders=EMBEDDERS, draws=600
    )
    assert report["measured_sizes"] == [300, 450, 600]
    assert report["labelled_documents"] == QUESTIONS
    by_name = {c["name"]: c for c in report["candidates"]}
    for name, embedder in EMBEDDERS.items():
        c = by_name[name]
        lo, hi = c["projected"]["interval"]
        assert lo <= embedder.truth() <= hi, (name, embedder.truth(), c["projected"])
        assert c["sample_recall"] > embedder.truth()  # the sample flatters
        assert c["inflation"] > 0
        recalls = [m["recall"] for m in c["measured"]]
        assert recalls == sorted(recalls, reverse=True)
        cost = c["cost"]
        assert cost["sample"]["embedded_documents"] == 600
        # One chunk per document, 24 float32 dimensions each.
        assert cost["chunks"] == pytest.approx(DOCS)
        assert cost["chunks_interval"] == pytest.approx([DOCS, DOCS])
        assert cost["vector_bytes"] == pytest.approx(DOCS * DIM * 4)
        assert cost["embed_seconds"] == pytest.approx(cost["sample"]["embed_seconds"] * DOCS / 600)
    assert report["ranking"] == ["easy", "hard"]
    assert report["sample_leader"] == "easy"
    assert report["p_leader_stays"] > 0.9
    assert "is still the leader at full size" in report["message"]
    assert "to embed on this machine" in report["message"]
    assert "576 kB of vectors" in report["message"]
    assert any("approximate index" in a for a in report["assumptions"])


def test_the_full_size_may_exceed_the_corpus_file_within_the_multiple(tmp_path):
    m = manifest(tmp_path)
    with pytest.raises(ExperimentError, match="beyond the allowed 10-fold"):
        project_experiment(m, fractions=[0.05, 0.1], full_size=12_000, embedders=EMBEDDERS)
    report = project_experiment(
        m,
        fractions=[0.05, 0.1],
        full_size=12_000,
        max_multiple=20,
        embedders=EMBEDDERS,
        draws=200,
    )
    assert report["extrapolation"] == pytest.approx(20)
    # Twice the documents compete, so the projection is lower than for the file alone.
    alone = project_experiment(m, fractions=[0.05, 0.1], embedders=EMBEDDERS, draws=200)
    assert (
        report["candidates"][0]["projected"]["recall"]
        < alone["candidates"][0]["projected"]["recall"]
    )


def test_pairs_weigh_questions_equally(tmp_path):
    m = manifest(tmp_path, extra_relevant=True)
    from vectraudit.experiments.runner import _load_inputs

    _, dataset = _load_inputs(m)
    rows, units, weights = _pairs(dataset)
    assert len(rows) == QUESTIONS + 1
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(weights[1]) == pytest.approx(0.5 / QUESTIONS)
    assert units[0] == units[1]


def test_thinned_counts_are_checked_and_candidates_must_share_pairs():
    one = np.ones(3, dtype=np.int64)
    w = np.full(3, 1 / 3)
    with pytest.raises(ValueError, match="one entry per pair"):
        Thinned(one, one[:2], 0.1, one, w)
    with pytest.raises(ValueError, match="g must be"):
        Thinned(one, one, 0.0, one, w)
    t = Thinned(np.array([0, 9, 12]), np.array([0, 0, 0]), 0.1, np.arange(3), w)
    assert measured(t, 10) == pytest.approx(2 / 3)
    other = Thinned(t.a, t.u, 0.1, np.array([0, 0, 1]), w)
    with pytest.raises(ValueError, match="same pairs"):
        project([t, other], 10, 1000)


def test_with_everything_sampled_the_projection_is_the_measurement():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 12, 200)
    u = rng.integers(0, 6, 200)
    t = Thinned(a, u, 1.0, np.arange(200), np.full(200, 1 / 200))
    result = project([t], 10, 1000, draws=100)[0]
    assert result["recall"] == pytest.approx(measured(t, 10))
    assert result["interval"][0] == pytest.approx(result["interval"][1])


def test_the_mixing_distribution_recovers_two_groups():
    rng = np.random.default_rng(0)
    lam = np.where(rng.random(4000) < 0.5, 2.0, 200.0)
    u = rng.poisson(0.1 * lam)
    grid = rate_grid(1000)
    w = fit_mixing(u, 0.1, grid)
    assert w[grid < 10].sum() == pytest.approx(0.5, abs=0.05)
    assert w[grid > 50].sum() == pytest.approx(0.5, abs=0.05)


def test_cli_refuses_and_writes_the_report(tmp_path, monkeypatch):
    from vectraudit.cli import app
    from vectraudit.experiments import project as module

    path = write_inputs(tmp_path)
    original = module.project_experiment

    def with_embedders(*args, **kwargs):
        kwargs["draws"] = 200
        return original(*args, embedders=EMBEDDERS, **kwargs)

    monkeypatch.setattr(module, "project_experiment", with_embedders)
    out = tmp_path / "p.json"
    ok = CliRunner().invoke(
        app, ["experiment", "project", str(path), "--out", str(out), "--no-cache"]
    )
    assert ok.exit_code == 0, ok.output
    assert "Wrote" in ok.output
    assert json.loads(out.read_text())["format"] == "vectraudit.scale_projection"
    refused = CliRunner().invoke(
        app,
        ["experiment", "project", str(path), "--full-size", "100000", "--no-cache"],
    )
    assert refused.exit_code == 2
    assert "Projection refused" in refused.output


def test_costs_scale_chunks_per_document_with_an_interval():
    from vectraudit.experiments.project import project_cost

    stats = {
        "chunks": 30,
        "dimension": 8,
        "embedded_documents": 15,
        "embed_seconds": 3.0,
        "vector_bytes": 30 * 32,
        "labelled_chunks": 10,
        "unlabelled_chunks_per_document": np.array([1, 1, 2, 2, 4, 1, 1, 2, 2, 4]),
    }
    cost = project_cost(stats, full_size=1005, labelled=5)
    assert cost["chunks"] == pytest.approx(10 + 1000 * 2.0)
    lo, hi = cost["chunks_interval"]
    assert lo < cost["chunks"] < hi
    assert cost["embed_seconds"] == pytest.approx(cost["chunks"] * 0.1)
    assert cost["vector_bytes"] == pytest.approx(cost["chunks"] * 32)
    assert "unlabelled_chunks_per_document" not in cost["sample"]

    cached = project_cost({**stats, "embed_seconds": None}, full_size=1005, labelled=5)
    assert cached["embed_seconds"] is None
    assert "--no-cache" in cached["embed_note"]
