"""Hosted policies, pinned baselines and server-side checks."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_runs import a_run, client_for, via
from vectraudit.cli import app as cli
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import save_run
from vectraudit.integrations import cloud as cloud_client

P = "/api/v1/workspaces/acme/projects/search"
POLICY = {"metrics": {"ndcg": {"max_drop": 0.01}}}


@pytest.fixture
def uploaded(world):
    """Baseline, an equal candidate, a regressed candidate and a partial run, uploaded."""
    client = client_for(world, "acme_runner")
    runs = {
        "base": a_run("base"),
        "same": a_run("same"),
        "worse": a_run("worse"),
        "partial": a_run("partial", status="partial"),
    }
    for q in runs["worse"]["queries"][:8]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    from vectraudit.evaluation.runner import summarize

    runs["worse"]["summary"] = summarize(runs["worse"]["queries"], 20)
    for run in runs.values():
        client.upload_run(run)
    return runs


def pin(world, who: str, run_id: str, name: str = "main", note: str = ""):
    return world.client.put(
        f"{P}/baselines/{name}", json={"run_id": run_id, "note": note}, headers=world.as_(who)
    )


def test_pinning_rules(world, uploaded) -> None:
    assert pin(world, "acme_viewer", "base").status_code == 403
    assert pin(world, "acme_runner", "base").status_code == 403
    assert pin(world, "globex_owner", "base").status_code == 404
    r = pin(world, "acme_engineer", "partial")
    assert r.status_code == 409
    assert r.json()["error_code"] == "run_not_complete"
    assert pin(world, "acme_engineer", "nope").status_code == 404
    assert pin(world, "acme_engineer", "base", name="bad name!").status_code == 422
    first = pin(world, "acme_engineer", "base", note="first good run")
    assert first.status_code == 200
    assert first.json()["run_id"] == "base"
    assert first.json()["pinned_by"].startswith("user:")
    moved = pin(world, "acme_owner", "same", note="re-baseline after reindex")
    assert moved.json()["run_id"] == "same"
    listed = world.client.get(f"{P}/baselines", headers=world.as_("acme_runner")).json()
    assert [(b["name"], b["run_id"]) for b in listed] == [("main", "same")]
    audit = world.client.get("/api/v1/workspaces/acme/audit", headers=world.as_("acme_owner"))
    pins = [e for e in audit.json() if e["action"] == "baseline.pin"]
    assert pins[0]["details"]["previous_run_id"] == "base"
    assert pins[0]["details"]["note"] == "re-baseline after reindex"
    assert (
        world.client.get(f"{P}/baselines/missing", headers=world.as_("acme_viewer")).status_code
        == 404
    )


def test_policy(world) -> None:
    got = world.client.get(f"{P}/policy", headers=world.as_("acme_runner")).json()
    assert got["is_default"]
    assert got["policy"] is None
    assert (
        world.client.put(f"{P}/policy", json=POLICY, headers=world.as_("acme_viewer")).status_code
        == 403
    )
    bad = world.client.put(
        f"{P}/policy", json={"metrics": {"bleu": {"min": 1}}}, headers=world.as_("acme_engineer")
    )
    assert bad.status_code == 422
    assert bad.json()["error_code"] == "invalid_policy"
    ok = world.client.put(f"{P}/policy", json=POLICY, headers=world.as_("acme_engineer"))
    assert ok.status_code == 200
    assert not ok.json()["is_default"]
    assert world.client.get(f"{P}/policy", headers=world.as_("globex_owner")).status_code == 404


def test_hosted_check_matches_the_cli(world, uploaded) -> None:
    world.client.put(f"{P}/policy", json=POLICY, headers=world.as_("acme_engineer"))
    pin(world, "acme_engineer", "base")
    client = client_for(world, "acme_runner")
    worse = client.create_check("worse")
    local = compare_runs(uploaded["base"], uploaded["worse"], parse_policy(POLICY))
    assert worse["verdict"] == local["verdict"] == "fail"
    assert worse["result"]["reasons"] == local["reasons"]
    assert worse["baseline_run_id"] == "base"
    assert client.create_check("same")["verdict"] == "pass"
    assert client.create_check("partial")["verdict"] == "error"
    # Idempotent per (baseline, candidate, policy): a retry returns the same check.
    assert client.create_check("worse")["id"] == worse["id"]
    world.client.put(
        f"{P}/policy",
        json={"metrics": {"ndcg": {"max_drop": 0.5}}},
        headers=world.as_("acme_engineer"),
    )
    relaxed = client.create_check("worse")
    assert relaxed["id"] != worse["id"]
    assert relaxed["verdict"] == "pass"
    got = world.client.get(f"{P}/checks/{worse['id']}", headers=world.as_("acme_viewer"))
    assert got.json()["verdict"] == "fail"
    assert (
        world.client.get(f"{P}/checks/{worse['id']}", headers=world.as_("globex_owner")).status_code
        == 404
    )


def test_check_errors(world, uploaded) -> None:
    h = world.as_("acme_runner")
    no_baseline = world.client.post(f"{P}/checks", json={"candidate_run_id": "worse"}, headers=h)
    assert no_baseline.status_code == 404
    pin(world, "acme_engineer", "base")
    missing = world.client.post(f"{P}/checks", json={"candidate_run_id": "nope"}, headers=h)
    assert missing.status_code == 404
    viewer = world.client.post(
        f"{P}/checks", json={"candidate_run_id": "worse"}, headers=world.as_("acme_viewer")
    )
    assert viewer.status_code == 403


def test_cli_cloud_baseline_workflow(world, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    runner = CliRunner()
    base = save_run(a_run("cli-base"), tmp_path / "base.json")
    worse_run = a_run("cli-worse")
    for q in worse_run["queries"]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    worse = save_run(worse_run, tmp_path / "worse.json")

    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    assert runner.invoke(cli, ["cloud", "push", str(base)]).exit_code == 0
    denied = runner.invoke(cli, ["cloud", "pin", "cli-base"])
    assert denied.exit_code == 1
    assert "Runner tokens can't pin baseline" in " ".join(denied.output.split())

    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_engineer"])
    result = runner.invoke(cli, ["cloud", "pin", "cli-base", "--note", "first"])
    assert result.exit_code == 0, result.output
    assert "Baseline main → run cli-base" in result.output
    assert "cli-base" in runner.invoke(cli, ["cloud", "baselines"]).output

    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    result = runner.invoke(cli, ["check", "cloud:main", str(base), "--no-github-summary"])
    assert result.exit_code == 0, result.output
    md = tmp_path / "summary.md"
    result = runner.invoke(
        cli,
        [
            "check",
            "cloud:main",
            str(worse),
            "--upload",
            "--no-github-summary",
            "--markdown",
            str(md),
        ],
    )
    assert result.exit_code == 3, result.output
    assert "/checks/" in md.read_text()
    assert "/report)" in md.read_text()  # the summary links to the hosted report
    assert "Uploaded run cli-worse" in result.output
    flat = "".join(result.output.split())
    # Links use the service's configured public URL.
    assert (
        "Hostedcheck:http://localhost:8800/api/v1/workspaces/acme/projects/search/checks/" in flat
    )
    no_cloud = runner.invoke(cli, ["check", str(base), str(worse), "--upload"])
    assert no_cloud.exit_code == 1
    assert "pass cloud:<name>" in no_cloud.output
    missing = runner.invoke(cli, ["check", "cloud:release", str(base)])
    assert missing.exit_code == 1
    assert "Baseline 'release' not found" in missing.output
    assert json.loads(worse.read_text())["run_id"] == "cli-worse"
