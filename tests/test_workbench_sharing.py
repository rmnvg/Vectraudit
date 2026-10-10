"""Sharing from the workbench: preview what leaves the machine, then upload only on confirm."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from tests.test_shared_records import PRIVATE, TEXT, case
from vectraudit.pipelines.web import create_workbench

CASE_ID = "a" * 32


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    for name in ("VECTRAUDIT_CLOUD_URL", "VECTRAUDIT_CLOUD_TOKEN", "VECTRAUDIT_CLOUD_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    folder = tmp_path / ".workbench" / "cases"
    folder.mkdir(parents=True)
    (folder / f"{CASE_ID}.json").write_text(json.dumps(case()))
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        yield client, tmp_path


def verified_join(monkeypatch):
    """Stand in for the local draft join (tested with real drafts in the display tests)."""
    from vectraudit.pipelines.local_display import CaseQuestionDisplay

    monkeypatch.setattr(
        CaseQuestionDisplay,
        "decorate",
        lambda self, record: {**record, "question_display": {"text": TEXT}},
    )


def body(**extra):
    return {"kind": "case", "id": CASE_ID, **extra}


def test_preview_shows_what_would_leave_and_sends_nothing(workbench, monkeypatch) -> None:
    client, root = workbench
    from vectraudit.integrations.cloud import CloudClient

    monkeypatch.setattr(
        CloudClient, "upload_record", lambda *a, **k: pytest.fail("preview must not upload")
    )
    before = (root / ".workbench" / "cases" / f"{CASE_ID}.json").read_text()
    shown = client.post("/api/guided/share/preview", json=body())
    assert shown.status_code == 200
    preview = shown.json()
    assert preview["destination"]["configured"] is False
    assert "VECTRAUDIT_CLOUD_URL" in preview["destination"]["message"]
    assert preview["transfer"]["query_text"] is False
    summary = preview["summary"]
    assert summary["included"][0] == "Question ID q1 and its status (open)"
    assert "investigation notes" in summary["withheld"]
    assert summary["question_text"] == []
    assert PRIVATE not in shown.text
    # Opting in sends text only from the verified local draft join; this workspace has no
    # draft, so no text is sent, whatever the stored case once displayed.
    unjoined = client.post("/api/guided/share/preview", json=body(share_query_text=True)).json()
    assert unjoined["summary"]["question_text"] == []
    verified_join(monkeypatch)
    with_text = client.post("/api/guided/share/preview", json=body(share_query_text=True)).json()
    assert with_text["summary"]["question_text"] == [TEXT]
    assert with_text["transfer"]["query_text"] is True
    assert (root / ".workbench" / "cases" / f"{CASE_ID}.json").read_text() == before


def test_upload_needs_confirmation_and_configuration(workbench) -> None:
    client, _ = workbench
    unconfirmed = client.post("/api/guided/share", json=body())
    assert unconfirmed.status_code == 422
    assert "confirm" in unconfirmed.json()["detail"]
    unconfigured = client.post("/api/guided/share", json=body(confirm=True))
    assert unconfigured.status_code == 422
    assert "VECTRAUDIT_CLOUD_URL" in unconfigured.json()["detail"]
    assert client.post("/api/guided/share/preview", json=body(id="../x")).status_code == 422
    missing = client.post("/api/guided/share/preview", json=body(id="f" * 32))
    assert missing.status_code == 404
    client.headers.pop("X-Workbench-Token")
    assert client.post("/api/guided/share/preview", json=body()).status_code in (401, 403)


def test_broken_records_are_refused_without_echoing_their_content(workbench) -> None:
    client, root = workbench
    broken = copy.deepcopy(case())
    broken["status"] = PRIVATE  # not an allowed status: validation would echo it
    (root / ".workbench" / "cases" / f"{CASE_ID}.json").write_text(json.dumps(broken))
    refused = client.post("/api/guided/share/preview", json=body())
    assert refused.status_code == 422
    assert PRIVATE not in refused.text


def test_confirmed_upload_reaches_the_project_and_retries_are_idempotent(
    workbench, world, monkeypatch
) -> None:
    pytest.importorskip("sqlalchemy")
    from tests.test_cloud_checks import via
    from vectraudit.integrations import cloud as cloud_client

    client, root = workbench
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    preview = client.post("/api/guided/share/preview", json=body()).json()
    assert preview["destination"] == {
        "configured": True,
        "url": "http://testserver",
        "project": "acme/search",
    }
    assert world.tokens["acme_runner"] not in json.dumps(preview)
    verified_join(monkeypatch)
    shared = client.post("/api/guided/share", json=body(confirm=True, share_query_text=True))
    assert shared.status_code == 200, shared.text
    result = shared.json()
    record_id = result["uploaded"]["id"]
    assert result["url"] == f"http://testserver/app/acme/search/records/{record_id}"
    hosted = world.client.get(
        f"/api/v1/workspaces/acme/projects/search/records/{record_id}/file",
        headers=world.as_("acme_viewer"),
    )
    assert hosted.json()["data"]["question_text"] == TEXT
    assert PRIVATE not in hosted.text
    again = client.post("/api/guided/share", json=body(confirm=True, share_query_text=True))
    assert again.json()["uploaded"]["id"] == record_id
    changed = copy.deepcopy(case())
    changed["status"] = "investigating"  # same local revision, different content
    (root / ".workbench" / "cases" / f"{CASE_ID}.json").write_text(json.dumps(changed))
    conflict = client.post("/api/guided/share", json=body(confirm=True, share_query_text=True))
    assert conflict.status_code == 409
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_viewer"])
    denied = client.post("/api/guided/share", json=body(confirm=True))
    assert denied.status_code == 403
