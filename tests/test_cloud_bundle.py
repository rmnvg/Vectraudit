"""Evidence bundles for hosted checks, and named in pull-request summaries.

A reviewer downloads a stored check's bundle (API, dashboard or `vectraudit cloud bundle`)
and `vectraudit verify` checks it offline; CI names the bundle and its manifest SHA-256 in
the summary and comment it writes.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_checks import POLICY, P, pin
from tests.test_cloud_runs import a_run, client_for, via
from vectraudit.cli import app as cli
from vectraudit.evaluation.bundle import verify_bundle
from vectraudit.evaluation.report import render_markdown
from vectraudit.integrations import cloud as cloud_client
from vectraudit.templates import retrieval_check


@pytest.fixture
def uploaded(world):
    """A baseline and a candidate whose first eight questions lose everything, uploaded."""
    from vectraudit.evaluation.runner import summarize

    client = client_for(world, "acme_runner")
    runs = {"base": a_run("base"), "worse": a_run("worse")}
    for q in runs["worse"]["queries"][:8]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    runs["worse"]["summary"] = summarize(runs["worse"]["queries"], 20)
    for run in runs.values():
        client.upload_run(run)
    return runs


def failing_check(world) -> dict:
    world.client.put(f"{P}/policy", json=POLICY, headers=world.as_("acme_engineer"))
    pin(world, "acme_engineer", "base")
    return client_for(world, "acme_runner").create_check("worse")


def download(world, check_id: str, who: str = "acme_viewer"):
    return world.client.get(f"{P}/checks/{check_id}/bundle", headers=world.as_(who))


def manifest_sha(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return hashlib.sha256(archive.read("manifest.json")).hexdigest()


def test_a_hosted_check_bundle_verifies_offline(world, uploaded, tmp_path) -> None:
    check = failing_check(world)
    r = download(world, check["id"])
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["x-vectraudit-bundle-runs"] == "true"
    assert r.headers["x-vectraudit-manifest-sha256"] == manifest_sha(r.content)
    assert f"vectraudit-check-{check['id']}.zip" in r.headers["content-disposition"]
    path = tmp_path / "evidence.zip"
    path.write_bytes(r.content)
    report = verify_bundle(path)
    assert report["status"] == "verified"
    assert report["verdict"] == "fail"


def test_bundle_still_verifies_after_the_project_policy_changes(world, uploaded, tmp_path) -> None:
    check = failing_check(world)
    world.client.put(
        f"{P}/policy",
        json={"metrics": {"ndcg": {"max_drop": 0.5}}},
        headers=world.as_("acme_engineer"),
    )
    r = download(world, check["id"])
    assert r.status_code == 200
    path = tmp_path / "evidence.zip"
    path.write_bytes(r.content)
    assert verify_bundle(path)["status"] == "verified"  # decided under the original policy


def test_bundles_follow_tenant_access(world, uploaded) -> None:
    check = failing_check(world)
    assert download(world, check["id"], "acme_runner").status_code == 200
    assert download(world, check["id"], "globex_owner").status_code == 404
    assert download(world, "0" * 32).status_code == 404


def test_runs_with_shared_query_text_stay_out_of_the_bundle(world, tmp_path) -> None:
    client = client_for(world, "acme_runner")
    base, cand = a_run("text-base"), a_run("text-cand")
    for run in (base, cand):
        client.upload_run(run, share_query_text=True)
    pin(world, "acme_engineer", "text-base")
    check = client.create_check("text-cand")
    r = download(world, check["id"])
    assert r.status_code == 200
    assert r.headers["x-vectraudit-bundle-runs"] == "false"
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        assert not any(
            name.startswith("runs/") and name.endswith("e.json") for name in archive.namelist()
        )
    path = tmp_path / "evidence.zip"
    path.write_bytes(r.content)
    assert verify_bundle(path)["status"] == "hashes_consistent"  # never "verified" without runs


def test_cli_downloads_a_hosted_bundle_and_never_overwrites(
    world,
    uploaded,
    tmp_path,
    monkeypatch,
) -> None:
    check = failing_check(world)
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_viewer"])
    out = tmp_path / "evidence.zip"
    runner = CliRunner()
    result = runner.invoke(cli, ["cloud", "bundle", check["id"], "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert manifest_sha(out.read_bytes()) in " ".join(result.output.split())
    assert runner.invoke(cli, ["verify", str(out)]).exit_code == 0
    again = runner.invoke(cli, ["cloud", "bundle", check["id"], "--out", str(out)])
    assert again.exit_code == 1
    assert "already exists" in again.output


def test_dashboard_check_page_links_the_bundle(cloud_app, world, uploaded) -> None:
    from tests.test_cloud_web import sign_in

    check = failing_check(world)
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = viewer.get(f"/app/acme/search/checks/{check['id']}").text
    link = f"{P}/checks/{check['id']}/bundle"
    assert link in page
    assert "vectraudit verify" in page
    download = viewer.get(link)  # the browser session alone may read it
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"


# --- named in the pull-request summary and comment ----------------------------------------


def test_summary_names_the_bundle_and_its_manifest(world, uploaded) -> None:
    result = failing_check(world)["result"]
    evidence = {
        "file": "evidence.zip",
        "manifest_sha256": "a" * 64,
        "includes_runs": True,
        "url": "https://github.com/acme/search/actions/runs/42",
    }
    md = render_markdown(result, evidence=evidence)
    assert "[`evidence.zip`](https://github.com/acme/search/actions/runs/42)" in md
    assert f"Manifest SHA-256 `{'a' * 64}`" in md
    assert "`vectraudit verify evidence.zip`" in md
    assert "Evidence bundle" not in render_markdown(result)


def test_check_names_the_bundle_in_its_summary(tmp_path, world, uploaded) -> None:
    from vectraudit.evaluation.runfile import save_run

    base = save_run(uploaded["base"], tmp_path / "base.json")
    worse = save_run(uploaded["worse"], tmp_path / "worse.json")
    md, bundle = tmp_path / "summary.md", tmp_path / "evidence.zip"
    runner = CliRunner()
    args = ["check", str(base), str(worse), "--no-github-summary", "--markdown", str(md)]
    bad = runner.invoke(cli, [*args, "--bundle", str(bundle), "--bundle-url", "http://x"])
    assert bad.exit_code == 1  # only https links
    result = runner.invoke(
        cli,
        [*args, "--bundle", str(bundle), "--bundle-runs", "--bundle-url", "https://ci/run/1"],
    )
    assert result.exit_code in (0, 3), result.output
    text = md.read_text()
    assert manifest_sha(bundle.read_bytes()) in text
    assert "[`evidence.zip`](https://ci/run/1)" in text


def test_action_writes_the_bundle_and_links_the_run(tmp_path) -> None:
    env = {
        "VA_BASE_REF": "main",
        "VA_ARTIFACT_NAME": "retrieval-check",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "acme/search",
        "GITHUB_RUN_ID": "42",
    }
    inputs = retrieval_check.read_inputs(env)
    assert inputs.bundle
    flags = retrieval_check._bundle_flags(inputs, tmp_path)
    assert flags == [
        "--bundle",
        str(tmp_path / "evidence.zip"),
        "--bundle-runs",
        "--bundle-url",
        "https://github.com/acme/search/actions/runs/42",
    ]
    # No artifact upload: nothing to link. A malformed repository is not linked either.
    assert retrieval_check.read_inputs({**env, "VA_ARTIFACT_NAME": ""}).bundle_url == ""
    assert retrieval_check.read_inputs({**env, "GITHUB_REPOSITORY": "a/b c"}).bundle_url == ""
    off = retrieval_check.read_inputs({**env, "VA_BUNDLE": "false"})
    assert retrieval_check._bundle_flags(off, tmp_path) == []
    with pytest.raises(retrieval_check.InputError):
        retrieval_check.read_inputs({**env, "VA_BUNDLE": "maybe"})


def test_an_older_check_whose_policy_changed_is_refused(cloud_app, world, uploaded) -> None:
    """Checks made before the policy source was kept can only use the current policy."""
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.models import Check

    check = failing_check(world)
    with transaction(cloud_app.state.sessions) as s:
        row = s.get(Check, check["id"])
        row.result = {k: v for k, v in row.result.items() if k != "policy_source"}
    assert download(world, check["id"]).status_code == 200  # current policy still matches
    world.client.put(
        f"{P}/policy",
        json={"metrics": {"ndcg": {"max_drop": 0.5}}},
        headers=world.as_("acme_engineer"),
    )
    r = download(world, check["id"])
    assert r.status_code == 409
    assert r.json()["error_code"] == "POLICY_CHANGED"
