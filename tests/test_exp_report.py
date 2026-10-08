"""Experiment ranking, ties, cohorts, reports and the CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_exp_runner import LEXICAL, SEMANTIC, SynonymEmbedder, write_fixture
from vectraudit.cli import app
from vectraudit.experiments.manifest import load_manifest
from vectraudit.experiments.report import rank, render_html, render_markdown, verdict_sentence
from vectraudit.experiments.runner import run_experiment

PARAGRAPHS = {
    **SEMANTIC,
    "name": "semantic-paragraphs",
    "chunking": {"method": "paragraphs", "size": 40},
}


@pytest.fixture
def ranked(tmp_path):
    manifest = load_manifest(write_fixture(tmp_path, [LEXICAL, SEMANTIC, PARAGRAPHS]))
    summary, results = run_experiment(manifest, embedders={"synonyms": SynonymEmbedder()})
    return rank(summary, results)


def test_ranking_ties_and_cohorts(ranked) -> None:
    order = [r["name"] for r in ranked["ranking"]]
    assert order[-1] == "lexical"
    assert ranked["leader"] in ("semantic", "semantic-paragraphs")
    by = {r["name"]: r for r in ranked["ranking"]}
    # Two candidates with the same quality can't be separated: reported as tied.
    assert set(ranked["tied_with_leader"]) == {"semantic", "semantic-paragraphs"}
    lexical = by["lexical"]
    assert not lexical["tied_with_leader"]
    assert lexical["ci_vs_leader"][1] < 0
    assert lexical["findings_vs_leader"]
    assert set(ranked["cohorts"]) == {"refund", "password", "invoice", "shipping", "warranty"}
    assert all(c["winner"] != "lexical" for c in ranked["cohorts"].values())
    assert "statistically tied" in verdict_sentence(ranked)
    json.dumps(ranked, allow_nan=False, default=str)


def test_reports(ranked) -> None:
    page = render_html(ranked)
    assert page.startswith("<!doctype html>")
    assert "<script" not in page
    for section in ("Quality", "Cost", "Best per cohort", "Compared with the leader"):
        assert f">{section}</h2>" in page
    assert '<span class="tag best">leader</span>' in page
    md = render_markdown(ranked)
    assert md.startswith("## Experiment fixture:")
    assert "(tied)" in md
    assert "| 3 | lexical |" in md


def test_experiments_keep_family_weights_and_counts(tmp_path):
    path = write_fixture(tmp_path, [LEXICAL, SEMANTIC])
    questions = tmp_path / "queries.jsonl"
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    for row in rows:
        row["family_id"] = row["tags"][0]
    questions.write_text("\n".join(map(json.dumps, rows)))
    summary, results = run_experiment(
        load_manifest(path), embedders={"synonyms": SynonymEmbedder()}
    )
    ranked = rank(summary, results)
    assert all(r.run["summary"]["n_families"] == 5 for r in results)
    assert all(r["n"] == 20 and r["n_families"] == 5 for r in ranked["ranking"])
    assert all(c["n"] == 4 and c["n_families"] == 1 for c in ranked["cohorts"].values())


def test_rank_errors(ranked, tmp_path) -> None:
    manifest = load_manifest(write_fixture(tmp_path, [LEXICAL]))
    summary, results = run_experiment(manifest)
    with pytest.raises(ValueError, match="primary metric"):
        rank(summary, results, primary="bleu")
    for q in results[0].run["queries"]:
        q["metrics"] = None
    results[0].run["summary"]["metrics"] = None
    with pytest.raises(ValueError, match="No candidate has labelled metrics"):
        rank(summary, results)


def test_cli_and_winner_runs_are_checkable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECTRAUDIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    tiny = {**LEXICAL, "name": "lexical-16", "embedder": {"type": "hashing", "dim": 16}}
    path = write_fixture(tmp_path, [LEXICAL, tiny])
    out = tmp_path / "results"
    runner = CliRunner()
    result = runner.invoke(app, ["experiment", "run", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Report:" in result.output
    for name in (
        "result.json",
        "report.html",
        "summary.md",
        "runs/lexical.json",
        "runs/lexical-16.json",
    ):
        assert (out / name).exists(), name
    leader = json.loads((out / "result.json").read_text())["leader"]
    winner = out / "runs" / f"{leader}.json"
    check = runner.invoke(app, ["check", str(winner), str(winner), "--no-github-summary"])
    assert check.exit_code == 0, check.output
    bad = runner.invoke(app, ["experiment", "run", str(tmp_path / "missing.json")])
    assert bad.exit_code == 1
    assert "not found" in bad.output


def test_experiment_demo_synthetic(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parents[1] / "examples" / "experiment_demo.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--queries",
            "30",
            "--out",
            str(tmp_path / "demo"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**__import__("os").environ, "VECTRAUDIT_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads((tmp_path / "demo" / "result.json").read_text())
    assert result["ranking"][-1]["name"] == "lexical-64"  # tiny hashing space collides
    assert (tmp_path / "demo" / "runs" / "lexical-2048.json").exists()
