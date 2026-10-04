"""Run upload to the hosted service: validation, idempotency, isolation, client and CLI."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app as cli
from vectraudit.evaluation.runfile import save_run
from vectraudit.integrations import cloud as cloud_client
from vectraudit.integrations.cloud import (
    CloudClient,
    CloudClientError,
    CloudTarget,
)

RUNS = "/api/v1/workspaces/acme/projects/search/runs"


def via(test_client):
    """A CloudClient transport that calls the in-process app."""

    def send(method, url, headers, body):
        r = test_client.request(
            method, url.removeprefix("http://testserver"), content=body, headers=dict(headers)
        )
        return r.status_code, r.content

    return send


def client_for(world, who: str, project: str = "acme/search") -> CloudClient:
    target = CloudTarget.from_env(
        project,
        {"VECTRAUDIT_CLOUD_URL": "http://testserver", "VECTRAUDIT_CLOUD_TOKEN": world.tokens[who]},
    )
    return CloudClient(target, via(world.client))


def a_run(run_id: str = "run-1", label: str = "main", **overrides):
    run = fake_run([1.0, 0.5] * 10, run_id=run_id, label=label, **overrides)
    run["provenance"]["commit_sha"] = "abc123"
    return run


def test_upload_is_idempotent(world) -> None:
    client = client_for(world, "acme_runner")
    run = a_run()
    stored = client.upload_run(run)
    assert stored["run_id"] == "run-1"
    assert stored["status"] == "complete"
    assert stored["commit_sha"] == "abc123"
    assert stored["n_queries"] == 20
    assert stored["uploaded_by"].startswith("token:va_run_")
    # Re-sending the same run (e.g. a CI retry, keys reordered) returns the stored one.
    again = client.upload_run(json.loads(json.dumps(run, sort_keys=True)))
    assert again["id"] == stored["id"]
    altered = copy.deepcopy(run)
    altered["label"] = "tampered"
    with pytest.raises(CloudClientError) as e:
        client.upload_run(altered)
    assert e.value.status == 409
    assert e.value.code == "run_conflict"
    assert client.get_run_file("run-1") == run


def test_status_codes(world) -> None:
    h = world.as_("acme_runner")
    body = json.dumps(a_run("run-2")).encode()
    assert world.client.post(RUNS, content=body, headers=h).status_code == 201
    assert world.client.post(RUNS, content=body, headers=h).status_code == 200


def test_invalid_uploads(world) -> None:
    h = world.as_("acme_runner")
    good = json.dumps(a_run("run-3")).encode()
    wrong = {**h, "X-Vectraudit-Digest": "sha256=" + "0" * 64}
    r = world.client.post(RUNS, content=good, headers=wrong)
    assert r.status_code == 400
    assert r.json()["error_code"] == "digest_mismatch"
    right = {**h, "X-Vectraudit-Digest": "sha256=" + hashlib.sha256(good).hexdigest()}
    assert world.client.post(RUNS, content=good, headers=right).status_code == 201
    for body in (b"{not json", json.dumps({"format": "other"}).encode()):
        r = world.client.post(RUNS, content=body, headers=h)
        assert r.status_code == 422
        assert r.json()["error_code"] == "invalid_run"
    huge = a_run("run-4")
    huge["dataset"]["n_queries"] = 50_000
    r = world.client.post(RUNS, content=json.dumps(huge).encode(), headers=h)
    assert r.status_code == 413
    assert r.json()["error_code"] == "too_many_queries"


def test_failed_runs_are_stored(world) -> None:
    run = a_run(
        "run-failed", status="failed", error={"code": "auth", "message": "bad key", "hint": None}
    )
    stored = client_for(world, "acme_runner").upload_run(run)
    assert stored["status"] == "failed"


def test_upload_permissions_and_isolation(world) -> None:
    body = json.dumps(a_run("run-5")).encode()
    for who, status in [
        ("acme_viewer", 403),
        ("acme_engineer", 201),
        ("globex_runner", 404),
        ("globex_owner", 404),
    ]:
        r = world.client.post(RUNS, content=body, headers=world.as_(who))
        assert r.status_code == status, who
    # The runner of acme/search can't write to acme/chat.
    chat = "/api/v1/workspaces/acme/projects/chat/runs"
    r = world.client.post(
        chat, content=json.dumps(a_run("x")).encode(), headers=world.as_("acme_runner")
    )
    assert r.status_code == 404
    # Reading: viewers yes, other tenants no.
    assert world.client.get(f"{RUNS}/run-5", headers=world.as_("acme_viewer")).status_code == 200
    for who in ("globex_owner", "globex_runner"):
        assert world.client.get(f"{RUNS}/run-5/file", headers=world.as_(who)).status_code == 404
    assert world.client.get(f"{RUNS}/nope", headers=world.as_("acme_viewer")).status_code == 404


def test_same_run_id_in_two_projects_is_fine(world) -> None:
    run = a_run("shared-id")
    client_for(world, "acme_runner").upload_run(run)
    client_for(world, "globex_runner", "globex/search").upload_run(run)


def test_target_validation() -> None:
    env = {
        "VECTRAUDIT_CLOUD_URL": "https://cloud.vectraudit.test",
        "VECTRAUDIT_CLOUD_TOKEN": "va_run_x",
    }
    assert CloudTarget.from_env("acme/search", env).project == "search"
    with pytest.raises(CloudClientError, match="Refusing to send a token over http"):
        CloudTarget.from_env("a/b", {**env, "VECTRAUDIT_CLOUD_URL": "http://cloud.example"})
    for spec in ("acme", "acme/", "/search", "a/b/c"):
        with pytest.raises(CloudClientError, match="workspace/project"):
            CloudTarget.from_env(spec, env)
    with pytest.raises(CloudClientError, match="URL is not set"):
        CloudTarget.from_env("a/b", {})
    with pytest.raises(CloudClientError, match="TOKEN is not set"):
        CloudTarget.from_env("a/b", {"VECTRAUDIT_CLOUD_URL": "https://x"})


def test_cli_push_and_record_upload(world, synthetic, tmp_path, monkeypatch) -> None:
    from tests.test_eval_runner import make_dataset

    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    runner = CliRunner()
    path = save_run(a_run("pushed"), tmp_path / "run.json")
    result = runner.invoke(cli, ["cloud", "push", str(path)])
    assert result.exit_code == 0, result.output
    assert "Uploaded run pushed (complete)" in result.output
    assert runner.invoke(cli, ["cloud", "push", str(path)]).exit_code == 0  # retry is safe
    (tmp_path / "bad.json").write_text("{}")
    assert runner.invoke(cli, ["cloud", "push", str(tmp_path / "bad.json")]).exit_code == 1

    ds = make_dataset(synthetic, n=5)
    lines = [
        json.dumps({"query_id": q.query_id, "vector": q.vector.tolist(), "relevant": q.relevant})
        for q in ds.queries
    ]
    (tmp_path / "ds.jsonl").write_text("\n".join(lines))
    conn = {"backend": "faiss", "index": str(synthetic["flat_l2"])}
    (tmp_path / "conn.json").write_text(json.dumps(conn))
    result = runner.invoke(
        cli,
        [
            "record",
            "--connection",
            str(tmp_path / "conn.json"),
            "--dataset",
            str(tmp_path / "ds.jsonl"),
            "--out",
            str(tmp_path / "rec.json"),
            "--upload",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Uploaded run" in result.output
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_viewer"])
    result = runner.invoke(
        cli,
        [
            "record",
            "--connection",
            str(tmp_path / "conn.json"),
            "--dataset",
            str(tmp_path / "ds.jsonl"),
            "--out",
            str(tmp_path / "rec2.json"),
            "--upload",
        ],
    )
    assert result.exit_code == 1
    assert "Upload failed" in result.output
    assert (tmp_path / "rec2.json").exists()
