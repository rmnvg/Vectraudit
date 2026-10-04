"""Hosted control plane: settings, migrations, health and error handling."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from vectraudit.cloud.app import create_app
from vectraudit.cloud.db import make_engine
from vectraudit.cloud.migrate import current_revision, head_revision, upgrade
from vectraudit.cloud.models import Base
from vectraudit.cloud.settings import Settings, SettingsError, normalize_database_url

SECRET = "s" * 40


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "VECTRAUDIT_CLOUD_DATABASE_URL": "postgres://u:p@db:5432/va",
            "VECTRAUDIT_CLOUD_SECRET_KEY": SECRET,
            "VECTRAUDIT_CLOUD_PUBLIC_URL": "https://cloud.example/",
            "VECTRAUDIT_CLOUD_SHARE_LINKS": "true",
            "VECTRAUDIT_CLOUD_MAX_UPLOAD_BYTES": "1000",
        }
    )
    assert s.database_url == "postgresql+psycopg://u:p@db:5432/va"
    assert s.public_url == "https://cloud.example"
    assert s.share_links
    assert s.max_upload_bytes == 1000
    defaults = Settings.from_env(
        {"VECTRAUDIT_CLOUD_DATABASE_URL": "sqlite://", "VECTRAUDIT_CLOUD_SECRET_KEY": SECRET}
    )
    assert not defaults.share_links
    assert normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "DATABASE_URL is not set"),
        ({"VECTRAUDIT_CLOUD_DATABASE_URL": "sqlite://"}, "at least 32 characters"),
        (
            {
                "VECTRAUDIT_CLOUD_DATABASE_URL": "sqlite://",
                "VECTRAUDIT_CLOUD_SECRET_KEY": SECRET,
                "VECTRAUDIT_CLOUD_MAX_UPLOAD_BYTES": "lots",
            },
            "must be an integer",
        ),
    ],
)
def test_settings_errors(env, message) -> None:
    with pytest.raises(SettingsError, match=message):
        Settings.from_env(env)


def test_migrations_match_the_models(cloud_url) -> None:
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    upgrade(cloud_url)
    engine = make_engine(cloud_url)
    assert current_revision(engine) == head_revision()
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    # JSON vs JSONB variants and UTCDateTime vs DateTime render identically in the DB.
    assert [d for d in diff if d[0] not in ("modify_type",)] == []


def test_readiness_tracks_migrations(cloud_settings) -> None:
    app = create_app(cloud_settings)
    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "ok"
    pending = client.get("/readyz")
    assert pending.status_code == 503
    assert pending.json()["error_code"] == "migrations_pending"
    assert "vectraudit-cloud migrate" in pending.json()["hint"]
    upgrade(cloud_settings.database_url)
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "schema": head_revision()}
    app.state.engine.dispose()


def test_unreachable_database_is_not_ready(tmp_path) -> None:
    settings = Settings(database_url="postgresql+psycopg://u:p@127.0.0.1:9/x", secret_key=SECRET)
    client = TestClient(create_app(settings))
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["error_code"] == "database_unavailable"


def test_errors_are_structured(cloud_app) -> None:
    client = TestClient(cloud_app)
    missing = client.get("/api/v1/nothing-here")
    assert missing.status_code == 404
    big = client.post("/api/v1/anything", content=b"x" * 300_000)
    assert big.status_code == 413
    assert big.json()["error_code"] == "payload_too_large"
