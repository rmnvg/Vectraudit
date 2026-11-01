"""Retrieval score drift between two windows of production traces."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_query_drift import ENGLISH
from tests.test_trace_formats import langfuse_page
from vectraudit.cli import app
from vectraudit.pipelines.drift import retrieval_samples, score_drift
from vectraudit.pipelines.traces import Settings

SETTINGS = Settings(id_field="doc_id")


def window(tmp_path, name, questions, *, scale=1.0, shift=0.0, k=5, seed=0, scored=True,
           second=None):  # fmt: skip
    """One trace per question; the retriever returns k documents with descending scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, question in enumerate(questions):
        top = rng.uniform(0.6, 0.9)
        scores = sorted((top - rng.uniform(0, 0.3, k)).tolist(), reverse=True)
        retrieved = [
            {
                "page_content": f"passage {j}",
                "metadata": {"doc_id": f"d{j}"},
                **({"score": s * scale + shift} if scored else {}),
            }
            for j, s in enumerate(scores)
        ]
        page = langfuse_page(question, retrieved, trace=f"{name}{i}")["data"]
        if second is not None:  # a second retriever span with its own scores
            extra = dict(page[1], id=f"{name}{i}-ret2", name="rerank_kb")
            extra["output"] = json.dumps([dict(d, score=second) for d in retrieved])
            page.append(extra)
        rows += page
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_the_same_retriever_on_the_same_questions_does_not_drift(tmp_path):
    a = window(tmp_path, "a", ENGLISH, seed=1)
    b = window(tmp_path, "b", ENGLISH, seed=2)
    report = score_drift(a, b, SETTINGS)
    assert report["status"] == "no_drift", report["descriptors"]
    assert report["questions"]["status"] == "no_drift"
    assert report["reference_traces"] == report["current_traces"] == 60


def test_rescaled_scores_with_steady_questions_point_at_the_retriever(tmp_path):
    a = window(tmp_path, "a", ENGLISH, seed=1)
    b = window(tmp_path, "b", ENGLISH, seed=2, scale=40.0)  # e.g. cosine to dot product
    report = score_drift(a, b, SETTINGS)
    by = {d["name"]: d for d in report["descriptors"]}
    assert report["status"] == "retriever_shift_likely"
    assert by["top_score"]["drift"]
    assert by["mean_score"]["drift"]
    assert by["top_score"]["current"]["mean"] > 20 * by["top_score"]["reference"]["mean"]
    assert not by["returned"]["drift"]
    assert "retriever likely changed" in report["reading"]


def test_a_smaller_result_set_is_drift_in_returned(tmp_path):
    report = score_drift(
        window(tmp_path, "a", ENGLISH, seed=1),
        window(tmp_path, "b", ENGLISH, seed=1, k=2),
        SETTINGS,
    )
    returned = next(d for d in report["descriptors"] if d["name"] == "returned")
    assert returned["drift"]
    assert (returned["reference"]["mean"], returned["current"]["mean"]) == (5.0, 2.0)


def test_when_questions_moved_too_the_shift_is_not_blamed_on_the_retriever(tmp_path):
    russian = ["Где мой счёт за март?", "Как сбросить пароль?", "Как отменить план?"] * 20
    report = score_drift(
        window(tmp_path, "a", ENGLISH, seed=1),
        window(tmp_path, "b", russian, seed=2, shift=-0.3),
        SETTINGS,
    )
    assert report["status"] == "drift_with_question_drift"
    assert "script" in report["questions"]["drifted"]
    assert "may explain the shift" in report["reading"]


def test_traces_without_scores_are_counted_not_guessed(tmp_path):
    scored = window(tmp_path, "a", ENGLISH, seed=1)
    bare = window(tmp_path, "b", ENGLISH[:6], scored=False)
    samples, questions, skipped = retrieval_samples(bare, SETTINGS)
    assert samples == []
    assert len(questions) == 6
    assert skipped == {"scores_unrecorded": 6}
    with pytest.raises(ValueError, match="No retrieval scores in the later traces"):
        score_drift(scored, bare, SETTINGS)


def test_a_named_stage_is_used_instead_of_the_first_retriever(tmp_path):
    a = window(tmp_path, "a", ENGLISH, seed=1, second=0.5)
    b = window(tmp_path, "b", ENGLISH, seed=1, second=0.9)
    first = score_drift(a, b, SETTINGS)
    named = score_drift(a, b, SETTINGS, stage="rerank_kb")
    assert first["status"] == "no_drift"  # the first retriever is unchanged
    assert named["status"] == "retriever_shift_likely"  # the second one moved
    assert named["stage"] == "rerank_kb"
    with pytest.raises(ValueError, match="no_stage"):
        score_drift(a, b, SETTINGS, stage="missing_span")


def test_small_windows_are_flagged(tmp_path):
    report = score_drift(
        window(tmp_path, "a", ENGLISH[:10], seed=1),
        window(tmp_path, "b", ENGLISH[:10], seed=2),
        SETTINGS,
    )
    assert report["warnings"][0].startswith("Small samples (10 earlier, 10 later")


def test_cli_reports_writes_json_and_fails_only_on_a_retriever_shift(tmp_path):
    a = window(tmp_path, "a", ENGLISH, seed=1)
    shifted = window(tmp_path, "b", ENGLISH, seed=2, scale=40.0)
    steady = window(tmp_path, "c", ENGLISH, seed=3)
    out = tmp_path / "drift.json"
    runner = CliRunner()
    base = ["workbench", "score-drift", "--id-field", "doc_id"]
    result = runner.invoke(app, [*base, str(a), str(shifted), "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "Retrieval score drift: retriever shift likely (60 earlier vs 60 later" in result.output
    assert json.loads(out.read_text())["status"] == "retriever_shift_likely"
    assert runner.invoke(app, [*base, str(a), str(shifted), "--fail-on-drift"]).exit_code == 3
    calm = runner.invoke(app, [*base, str(a), str(steady), "--fail-on-drift"])
    assert calm.exit_code == 0, calm.output
    missing = runner.invoke(app, [*base, str(a), str(tmp_path / "none.jsonl")])
    assert missing.exit_code == 2
