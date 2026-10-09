"""Project isolation, durable history, migration and hosted reading of shared records."""

import copy
import json
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from tests.test_shared_records import PRIVATE, benchmark, case, policy
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import Run, SharedRecord, utcnow
from vectraudit.integrations.cloud import CloudClientError
from vectraudit.integrations.shared_records import project_record

P = "/api/v1/workspaces/acme/projects/search"


def test_upload_retries_conflicts_and_distinct_revisions(world):
    client = client_for(world, "acme_runner")
    source = case()
    record = project_record("case", source)
    first = client.upload_record(record)
    assert client.upload_record(record)["id"] == first["id"]
    source["status"] = "investigating"
    with pytest.raises(CloudClientError) as err:
        client.upload_record(project_record("case", source))
    assert err.value.status == 409
    source["revision"] = "f" * 32
    second = client.upload_record(project_record("case", source))
    assert second["id"] != first["id"]
    assert (
        world.client.get(
            f"{P}/records/{first['id']}/file", headers=world.as_("acme_viewer")
        ).json()["data"]["status"]
        == "open"
    )
    page = world.client.get(f"{P}/records?limit=1", headers=world.as_("acme_viewer")).json()
    assert page["records"][0]["id"] == second["id"]
    rest = world.client.get(
        f"{P}/records",
        params={"cursor": page["next_cursor"], "limit": 1},
        headers=world.as_("acme_viewer"),
    ).json()
    assert rest["records"][0]["id"] == first["id"]
    assert rest["next_cursor"] is None


def test_scope_roles_digest_and_server_allowlist(world):
    record = project_record("case", case())
    stored = client_for(world, "acme_runner").upload_record(record)
    for who in ("globex_owner", "globex_runner"):
        assert world.client.get(f"{P}/records", headers=world.as_(who)).status_code == 404
        assert (
            world.client.get(f"{P}/records/{stored['id']}/file", headers=world.as_(who)).status_code
            == 404
        )
    assert (
        world.client.post(f"{P}/records", json=record, headers=world.as_("acme_viewer")).status_code
        == 403
    )
    headers = {**world.as_("acme_runner"), "X-Vectraudit-Digest": "sha256=wrong"}
    assert world.client.post(f"{P}/records", json=record, headers=headers).status_code == 400
    injected = copy.deepcopy(record)
    injected["data"]["context"] = PRIVATE
    rejected = world.client.post(f"{P}/records", json=injected, headers=world.as_("acme_runner"))
    assert rejected.status_code == 422
    assert PRIVATE not in rejected.text
    assert (
        world.client.post(f"{P}/records", json=case(), headers=world.as_("acme_runner")).status_code
        == 422
    )


def test_archive_survives_retention_is_exported_and_deleted_with_project(world, cloud_app):
    client = client_for(world, "acme_runner")
    for kind, record in [("case", case()), ("benchmark", benchmark()), ("policy", policy())]:
        client.upload_record(project_record(kind, record))
    client.upload_run(a_run("expired"))
    with transaction(cloud_app.state.sessions) as s:
        run = s.query(Run).filter(Run.run_id == "expired").one()
        run.uploaded_at = utcnow() - timedelta(days=40)
    with transaction(cloud_app.state.sessions) as s:
        assert quotas.purge_expired(s) == {"acme": 1}
        assert s.query(SharedRecord).count() == 3
    exported = world.client.get(
        "/api/v1/workspaces/acme/export", headers=world.as_("acme_owner")
    ).json()
    search = next(p for p in exported["projects"] if p["slug"] == "search")
    assert len(search["shared_records"]) == 3
    assert PRIVATE not in json.dumps(search["shared_records"])
    deleted = world.client.request(
        "DELETE", P, json={"confirm": "search"}, headers=world.as_("acme_owner")
    )
    assert deleted.status_code == 200
    assert deleted.json()["shared_records"] == 3
    with transaction(cloud_app.state.sessions) as s:
        assert s.query(SharedRecord).count() == 0


def test_second_engineer_can_read_cases_benchmarks_and_policies(world, cloud_app):
    client = client_for(world, "acme_runner")
    records = [
        client.upload_record(project_record(kind, raw, share_query_text=True))
        for kind, raw in [("case", case()), ("benchmark", benchmark()), ("policy", policy())]
    ]
    browser = sign_in(cloud_app, world.tokens["acme_engineer"])
    listing = browser.get("/app/acme/search/records")
    assert listing.status_code == 200
    assert "not hosted release approval" in listing.text
    assert "Shared cases" in browser.get("/app/acme/search").text
    for row in records:
        page = browser.get(f"/app/acme/search/records/{row['id']}")
        assert page.status_code == 200
        assert PRIVATE not in page.text
        assert "<script>" not in page.text
        assert "Download projected record" in page.text
        if row["kind"] != "policy":
            assert "Question explicitly shared" in page.text
        assert browser.get(f"{P}/records/{row['id']}/file").status_code == 200
    outsider = sign_in(cloud_app, world.tokens["globex_owner"])
    assert outsider.get(f"/app/acme/search/records/{records[0]['id']}").status_code == 404
