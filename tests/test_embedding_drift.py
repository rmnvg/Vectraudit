"""Embedding drift follows Evidently's methods and keeps false alarms near their level."""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_query_drift import write_traces
from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.pipelines import embedding_drift as E
from vectraudit.pipelines.embedding_drift import (
    EmbeddingDriftError,
    benchmark_drift,
    corpus_drift,
    embedding_drift,
    question_model,
    roc_auc,
    wants_normalized,
    window_drift,
)

DIM = 32


def centres(rng, k=9, dim=DIM):
    c = rng.normal(size=(k, dim))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def draw(rng, topics, n, noise=0.6):
    pick = rng.integers(0, len(topics), n)
    x = topics[pick] + noise * rng.normal(size=(n, topics.shape[1])) / math.sqrt(DIM) * 4
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def windows(seed, n_ref=300, n_cur=300, new_share=0.0):
    rng = np.random.default_rng(seed)
    topics = centres(rng)
    old, new = topics[:8], topics[8:]
    k = round(n_cur * new_share)
    current = (
        np.vstack([draw(rng, old, n_cur - k), draw(rng, new, k)]) if k else draw(rng, old, n_cur)
    )
    return draw(rng, old, n_ref), current


def binomial_upper(n, p, tail=0.005):
    """The smallest count whose upper tail under Binomial(n, p) is below ``tail``."""
    total = 0.0
    for k in range(n, -1, -1):
        total += math.comb(n, k) * p**k * (1 - p) ** (n - k)
        if total > tail:
            return k + 1
    return 0


# --- the statistics --------------------------------------------------------------------------


def test_roc_auc_is_the_mann_whitney_statistic_with_ties_counted_half():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert roc_auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert roc_auc([3, 2, 1], [1, 1, 0]) == 1.0
    with pytest.raises(EmbeddingDriftError, match="both sets"):
        roc_auc([1, 2], [1, 1])


@pytest.mark.parametrize("method", ["model", "mmd"])
def test_false_alarms_stay_within_binomial_tolerance_of_five_percent(method):
    repeats = 100 if method == "model" else 60
    n = 300 if method == "model" else 150
    alarms = sum(
        embedding_drift(*windows(s, n, n), method=method, seed=s)["drift"] for s in range(repeats)
    )
    assert alarms <= binomial_upper(repeats, 0.05), alarms


@pytest.mark.parametrize("method", ["model", "mmd"])
def test_a_new_topic_in_a_fifth_of_production_is_detected(method):
    # Production is larger than the benchmark, as it usually is.
    found = [
        embedding_drift(*windows(s, 300, 3000, 0.2), method=method, seed=s)["drift"]
        for s in range(10)
    ]
    assert sum(found) >= 9


def test_the_default_follows_evidently():
    reference, current = windows(1, 300, 300, 0.5)
    small = embedding_drift(reference, current)
    assert small["method"] == "model"
    assert small["evidently_default"] is True
    assert "random scores" in small["rule"]  # under 1,000 reference vectors
    assert 0.5 < small["threshold"] < 0.6
    assert small["drift"] is True
    large_ref, large_cur = windows(2, 1200, 600, 0.5)
    large = embedding_drift(large_ref, large_cur)
    assert large["threshold"] == 0.55
    assert large["rule"] == "drift when the ROC AUC is above 0.55"
    assert large["drift"] is True
    forced = embedding_drift(reference, current, bootstrap=False)
    assert forced["threshold"] == 0.55


def test_ratio_and_distance_say_when_their_defaults_mislead():
    reference, current = windows(3, 300, 300)
    ratio = embedding_drift(reference, current, method="ratio")
    assert ratio["components"] == DIM
    assert "noisy" in ratio["warnings"][0]
    big = embedding_drift(*windows(3, 1500, 1500), method="ratio")
    assert big["warnings"] == []
    assert big["drift"] is False
    distance = embedding_drift(reference, current, method="distance")
    assert distance["threshold"] == 0.2
    assert "scale" in distance["warnings"][0]
    shifted = embedding_drift(*windows(3, 300, 300, 0.5), method="distance", bootstrap=True)
    assert shifted["warnings"] == []
    assert "resamples of the reference" in shifted["rule"]
    assert shifted["drift"] is True


def test_mmd_reports_a_permutation_p_value():
    calm = embedding_drift(*windows(4, 150, 150), method="mmd")
    moved = embedding_drift(*windows(4, 150, 150, 0.5), method="mmd")
    assert calm["p_value"] > 0.05
    assert moved["p_value"] < 0.01
    assert moved["value"] > calm["value"]
    assert moved["rule"].startswith("drift when the permutation p-value (200 permutations)")


def test_principal_components_and_sampling_are_reported(monkeypatch):
    reference, current = windows(5, 300, 300, 0.5)
    projected = embedding_drift(reference, current, pca_components=8, method="ratio")
    assert projected["components"] == 8
    assert projected["dimensions"] == DIM
    monkeypatch.setattr(E, "MAX_ROWS", 100)
    sampled = embedding_drift(reference, current)
    assert sampled["sampled"] == {"reference": 100, "current": 100}
    assert (sampled["n_reference"], sampled["n_current"]) == (300, 300)


def test_bad_inputs_are_refused_with_reasons():
    reference, current = windows(6, 50, 50)
    cases = [
        (lambda: embedding_drift(reference, current[:, :10]), "different dimensions"),
        (lambda: embedding_drift(reference[:5], current), "at least 10"),
        (lambda: embedding_drift(reference, current, method="psi"), "method must be one of"),
        (lambda: embedding_drift(reference, current, pca_components=99), "pca_components"),
    ]
    broken = current.copy()
    broken[0, 0] = np.nan
    cases.append((lambda: embedding_drift(reference, broken), "NaN"))
    for call, message in cases:
        with pytest.raises(EmbeddingDriftError, match=message):
            call()


# --- inputs ----------------------------------------------------------------------------------


def hashed(text: str) -> np.ndarray:
    """A bag of hashed words, unit length: different topics land on different coordinates."""
    v = np.zeros(DIM)
    for word in re.findall(r"\w+", text.lower()):
        v[int(hashlib.sha1(word.encode()).hexdigest(), 16) % DIM] += 1
    return v / max(np.linalg.norm(v), 1e-9)


BILLING = [
    f"how do I {verb} my {thing} {detail}"
    for verb in ("pay", "change", "download", "dispute", "split")
    for thing in ("invoice", "bill", "statement", "receipt")
    for detail in ("today", "online", "this month")
]
CRYPTO = [
    f"{verb} {coin} wallet {detail} seed phrase"
    for verb in ("restore", "connect", "stake", "bridge")
    for coin in ("bitcoin", "ethereum", "solana", "token")
    for detail in ("ledger", "metamask", "phantom", "exchange", "hardware")
]


def bench_rows(vectors=False, **extra):
    rows = []
    for i, text in enumerate(BILLING):
        row = {"query_id": f"q{i}", "text": text, "relevant": {"d1": 1}, **extra}
        if vectors:
            row["vector"] = hashed(text).tolist()
        rows.append(row)
    return rows


def test_the_model_is_the_benchmarks_own_unless_it_has_none():
    meta = {"embedding_model": "mini", "normalized": True}
    with_vectors = parse_dataset(render_dataset(bench_rows(vectors=True), meta))
    assert question_model(with_vectors, None) == "mini"
    assert question_model(with_vectors, "mini") == "mini"
    assert wants_normalized(with_vectors) is True
    with pytest.raises(EmbeddingDriftError, match="would compare two models"):
        question_model(with_vectors, "other")
    plain = parse_dataset(render_dataset(bench_rows()))
    with pytest.raises(EmbeddingDriftError, match="--embedder"):
        question_model(plain, None)
    assert question_model(plain, "other") == "other"
    assert wants_normalized(plain) is False


def test_benchmark_drift_uses_exploratory_questions_and_holds_no_text():
    rows = bench_rows()
    rows += [
        {"query_id": "h1", "text": "secret held out", "relevant": {"d1": 1}, "split": "heldout"}
    ]
    ds = parse_dataset(render_dataset(rows))
    production = [(f"t{i}", text) for i, text in enumerate(CRYPTO[:40] + BILLING[:20])]
    seen: list[str] = []

    def embed(text):
        seen.append(text)
        return hashed(text)

    report = benchmark_drift(ds, production, embed, model="hash")
    assert "secret held out" not in seen
    assert report["format"] == "vectraudit.embedding_drift"
    assert report["status"] == "drift"
    assert report["evidence"] == "hypothesis"
    assert report["reference"]["count"] == len(BILLING)
    assert report["reference"]["embedded_here"] == len(BILLING)
    blob = json.dumps(report)
    assert "seed phrase" not in blob
    assert "invoice" not in blob
    calm = benchmark_drift(ds, [(f"t{i}", t) for i, t in enumerate(BILLING)], embed, model="hash")
    assert calm["status"] == "no_drift"


def test_windows_and_corpus_exports_compare_too():
    earlier = [(f"a{i}", t) for i, t in enumerate(BILLING)]
    later = [(f"b{i}", t) for i, t in enumerate(CRYPTO)]
    report = window_drift(earlier, later, hashed, model="hash")
    assert (report["reference"]["label"], report["current"]["label"]) == ("earlier", "later")
    assert report["status"] == "drift"
    with pytest.raises(EmbeddingDriftError, match="Both windows"):
        window_drift([], later, hashed, model="hash")
    old_ids = [f"d{i}" for i in range(len(BILLING))]
    new_ids = old_ids[10:] + [f"n{i}" for i in range(len(CRYPTO))]
    vectors = [hashed(t) for t in BILLING[10:] + CRYPTO]
    moved = corpus_drift((old_ids, [hashed(t) for t in BILLING]), (new_ids, vectors))
    assert moved["kind"] == "corpus"
    assert moved["current"]["documents_added"] == len(CRYPTO)
    assert moved["current"]["documents_removed"] == 10
    assert moved["status"] == "drift"


# --- the commands ------------------------------------------------------------------------------


@pytest.fixture
def fake_embedder(monkeypatch):
    import vectraudit.core.embed as embed_module

    calls = []

    def make(model_name, *, normalize):
        calls.append((model_name, normalize))
        return hashed

    monkeypatch.setattr(embed_module, "sentence_transformer_embedder", make)
    return calls


def test_drift_command_compares_embeddings_and_can_fail(tmp_path, fake_embedder):
    data = tmp_path / "questions.jsonl"
    data.write_text(render_dataset(bench_rows(), {"embedding_model": "mini"}))
    traces = write_traces(tmp_path, CRYPTO[:40] + BILLING[:20])
    out = tmp_path / "embedding-drift.json"
    runner = CliRunner()
    args = ["workbench", "drift", str(traces), str(data), "--embeddings", str(out)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Embedding drift: DRIFT (60 production vs 60 benchmark questions, method model)" in (
        result.output
    )
    assert fake_embedder == [("mini", False)]
    report = json.loads(out.read_text())
    assert report["status"] == "drift"
    assert report["current"]["source"] == "traces.jsonl"
    failing = runner.invoke(app, [*args, "--fail-on-drift"])
    assert failing.exit_code == 3
    plain = tmp_path / "plain.jsonl"
    plain.write_text(render_dataset(bench_rows()))
    refused = runner.invoke(
        app, ["workbench", "drift", str(traces), str(plain), "--embeddings", str(out)]
    )
    assert refused.exit_code == 2
    assert "--embedder" in refused.output


def test_embedding_drift_command_for_windows_and_corpus(tmp_path, fake_embedder):
    runner = CliRunner()
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    earlier = write_traces(first, BILLING)
    later = write_traces(second, CRYPTO)
    out = tmp_path / "windows.json"
    args = ["workbench", "embedding-drift", str(earlier), str(later), "--out", str(out)]
    missing = runner.invoke(app, args)
    assert missing.exit_code == 2
    assert "--embedder" in missing.output
    result = runner.invoke(app, [*args, "--embedder", "mini", "--fail-on-drift"])
    assert result.exit_code == 3, result.output
    assert "80 later vs 60 earlier questions" in result.output
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    old.write_text(
        "".join(
            json.dumps({"doc_id": f"d{i}", "vector": hashed(t).tolist()}) + "\n"
            for i, t in enumerate(BILLING)
        )
    )
    new.write_text(
        "".join(
            json.dumps({"doc_id": f"d{i}", "vector": hashed(t).tolist()}) + "\n"
            for i, t in enumerate(BILLING[::-1])
        )
    )
    corpus = tmp_path / "corpus.json"
    calm = runner.invoke(
        app, ["workbench", "embedding-drift", str(old), str(new), "--corpus", "--out", str(corpus)]
    )
    assert calm.exit_code == 0, calm.output
    assert json.loads(corpus.read_text())["status"] == "no_drift"
    assert "documents, method model" in calm.output
