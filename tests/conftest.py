"""Shared fixtures: small synthetic indexes of every type, built once per test session."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

SMALL = {"n": 2_000, "d": 32, "n_centers": 20, "n_queries": 50, "nlist": 16, "hnsw_m": 16}


def _load_make_synthetic() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_synthetic"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def make_synthetic() -> ModuleType:
    """The ``examples/make_synthetic.py`` module."""
    return _load_make_synthetic()


@pytest.fixture(scope="session")
def synthetic(
    tmp_path_factory: pytest.TempPathFactory, make_synthetic: ModuleType
) -> dict[str, Path]:
    """Paths to small synthetic indexes and side files (see ``make_synthetic.build_dataset``)."""
    out = tmp_path_factory.mktemp("synthetic")
    return make_synthetic.build_dataset(out, **SMALL)


@pytest.fixture(scope="session")
def binary_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A binary index file (unsupported by vectraudit)."""
    path = tmp_path_factory.mktemp("binary") / "binary_flat.index"
    faiss.write_index_binary(faiss.IndexBinaryFlat(64), str(path))
    return path


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Never touch the real ~/.cache during tests.

    Session-scoped so it is active before any module-scoped fixture creates a Session.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VECTRAUDIT_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        yield


@pytest.fixture(scope="session")
def uniform_ivf(tmp_path_factory) -> tuple[Path, Path, Any]:
    """IVF-Flat over unclustered data, where a low nprobe really misses neighbours."""
    from vectraudit.evaluation.dataset import parse_dataset

    out = tmp_path_factory.mktemp("uniform")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2_000, 16)).astype(np.float32)
    queries = rng.normal(size=(20, 16)).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(16), 16, 16)
    index.train(x)
    index.add(x)
    faiss.write_index(index, str(out / "ivf.index"))
    np.save(out / "x.npy", x)
    rows = [{"query_id": f"q{i}", "vector": q.tolist()} for i, q in enumerate(queries)]
    return out / "ivf.index", out / "x.npy", parse_dataset("\n".join(map(json.dumps, rows)))


SECRET = "test-secret-key-0123456789-abcdefghijklmnop"


def _cloud_databases() -> list[str]:
    dbs = ["sqlite"]
    if os.environ.get("VECTRAUDIT_TEST_PG_DSN"):
        dbs.append("postgresql")
    return dbs


@pytest.fixture(params=_cloud_databases())
def cloud_url(request, tmp_path) -> Iterator[str]:
    """An empty database (not migrated): SQLite file, or a fresh PostgreSQL database."""
    pytest.importorskip("sqlalchemy")
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'cloud.db'}"
        return
    import uuid

    import psycopg
    from psycopg import sql

    admin = os.environ["VECTRAUDIT_TEST_PG_DSN"]
    name = f"va_cloud_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = psycopg.conninfo.make_conninfo(admin, dbname=name)
    info = psycopg.conninfo.conninfo_to_dict(url)
    yield (
        f"postgresql+psycopg://{info['user']}:{info.get('password', '')}@"
        f"{info.get('host', 'localhost')}:{info.get('port', 5432)}/{name}"
    )
    with psycopg.connect(admin, autocommit=True) as conn:
        drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)")
        conn.execute(drop.format(sql.Identifier(name)))


@pytest.fixture
def cloud_settings(cloud_url):
    from vectraudit.cloud.settings import Settings

    return Settings(database_url=cloud_url, secret_key=SECRET, max_upload_bytes=200_000)


@pytest.fixture
def cloud_app(cloud_settings):
    """A migrated database and its app."""
    from vectraudit.cloud.app import create_app
    from vectraudit.cloud.migrate import upgrade

    upgrade(cloud_settings.database_url)
    app = create_app(cloud_settings)
    yield app
    app.state.engine.dispose()


@dataclass
class World:
    client: Any
    tokens: dict[str, str]
    """Name -> secret: admin, acme_owner, acme_engineer, acme_viewer, globex_owner,
    acme_runner (project acme/search), globex_runner (project globex/search)."""

    def as_(self, who: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens[who]}"}


@pytest.fixture
def world(cloud_app) -> World:
    """Two tenants (acme, globex) with members, projects and runner tokens."""
    from fastapi.testclient import TestClient

    from vectraudit.cloud import repo
    from vectraudit.cloud.auth import Principal
    from vectraudit.cloud.db import transaction

    tokens: dict[str, str] = {}
    with transaction(cloud_app.state.sessions) as s:
        admin = repo.get_or_create_user(s, "ops@vectraudit.test")
        admin.is_admin = True
        tokens["admin"] = repo.create_personal_token(s, admin, "admin", "test")[1].token
        for slug in ("acme", "globex"):
            _, owner = repo.create_workspace(s, "test", slug, slug.title(), f"owner@{slug}.test")
            tokens[f"{slug}_owner"] = repo.create_personal_token(s, owner, "o", "test")[1].token
            owner_p = Principal("user", None, None, user_id=owner.id)
            access = repo.authorize(s, owner_p, slug, "manage_project")
            repo.create_project(s, access, "search", "Search")
            if slug == "acme":
                repo.create_project(s, access, "chat", "Chat")
                for role in ("engineer", "viewer"):
                    user, _ = repo.add_member(s, access, f"{role}@acme.test", role)
                    tokens[f"acme_{role}"] = repo.create_personal_token(s, user, role, "t")[1].token
            proj = repo.authorize(s, owner_p, slug, "manage_runner_tokens", "search")
            tokens[f"{slug}_runner"] = repo.create_runner_token(s, proj, "ci")[1].token
    return World(TestClient(cloud_app), tokens)


LIVE_SKIP = "Set VECTRAUDIT_TEST_"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """With VECTRAUDIT_REQUIRE_LIVE=1, a live test skipped for a missing server fails.

    The live-backends workflow sets it, so a server that never started (or a renamed
    variable) cannot turn the whole job green by skipping every test that needed it.
    """
    outcome = yield
    report = outcome.get_result()
    if not (report.skipped and os.environ.get("VECTRAUDIT_REQUIRE_LIVE") == "1"):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if LIVE_SKIP in reason:
        report.outcome = "failed"
        report.longrepr = f"Live backend test skipped under VECTRAUDIT_REQUIRE_LIVE=1: {reason}"
