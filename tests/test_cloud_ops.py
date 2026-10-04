"""Operational behaviour: request ids, safe access logs, security headers."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from vectraudit.cloud.app import create_app
from vectraudit.cloud.logs import JsonFormatter


def test_request_ids(cloud_app) -> None:
    client = TestClient(cloud_app)
    generated = client.get("/healthz").headers["x-request-id"]
    assert len(generated) == 32
    assert (
        client.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"]
        == "abc-123"
    )
    weird = client.get("/healthz", headers={"X-Request-ID": "<script>"}).headers["x-request-id"]
    assert weird != "<script>"


def test_access_log_has_no_secrets(world, caplog) -> None:
    caplog.set_level(logging.INFO, logger="vectraudit.cloud.access")
    world.client.get("/r/abc?e=99999999999&s=SIGNATURE-SECRET")
    world.client.get("/api/v1/me", headers=world.as_("acme_viewer"))
    records = [r for r in caplog.records if r.name == "vectraudit.cloud.access"]
    assert [r.fields["path"] for r in records] == ["/r/abc", "/api/v1/me"]
    text = " ".join(json.dumps(r.fields) for r in records)
    assert "SIGNATURE-SECRET" not in text
    assert world.tokens["acme_viewer"] not in text
    assert records[1].fields["status"] == 200


def test_security_headers(cloud_app) -> None:
    plain = TestClient(cloud_app).get("/healthz")
    assert plain.headers["x-content-type-options"] == "nosniff"
    assert plain.headers["x-frame-options"] == "DENY"
    assert "strict-transport-security" not in plain.headers
    https = dataclasses.replace(cloud_app.state.settings, public_url="https://cloud.example")
    secure = TestClient(create_app(https, engine=cloud_app.state.engine)).get("/healthz")
    assert "max-age=31536000" in secure.headers["strict-transport-security"]


def test_json_formatter() -> None:
    record = logging.LogRecord(
        "vectraudit.cloud.access", logging.INFO, __file__, 1, "request", None, None
    )
    record.fields = {"path": "/x", "status": 200}
    line = json.loads(JsonFormatter().format(record))
    assert line["msg"] == "request"
    assert line["path"] == "/x"
    assert line["level"] == "info"
