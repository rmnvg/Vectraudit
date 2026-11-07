"""pgvector connector.

Database tests need ``VECTRAUDIT_TEST_PG_DSN`` (a superuser DSN for a disposable database
with the pgvector extension available, e.g. the ``pgvector/pgvector`` image); they are
skipped without it. CI runs them against a service container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_eval_runner import make_dataset
from vectraudit.connectors import ConnectorError, Distance, ErrorCode, SearchRequest
from vectraudit.connectors.config import open_connector, parse_connection
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.connectors.pgvector import (
    PgvectorConnector,
    _pg_error,
    check_identifier,
    vector_literal,
)
from vectraudit.evaluation.runner import RunOptions, execute_run

psycopg = pytest.importorskip("psycopg")
ADMIN_DSN = os.environ.get("VECTRAUDIT_TEST_PG_DSN")
N = 300
READER_PASSWORD = "va-reader-pw"


# --- no database needed ------------------------------------------------------------------


def test_identifiers_and_literals() -> None:
    assert check_identifier("chunks_2024", "table") == "chunks_2024"
    for bad in ("chunks; drop table x", 'a"b', "1abc", "", "a" * 64, "naïve"):
        with pytest.raises(ConnectorError) as e:
            check_identifier(bad, "table")
        assert e.value.code is ErrorCode.CONFIG
    assert vector_literal(np.array([0.5, -1.0], dtype=np.float32)) == "[0.5,-1.0]"


@pytest.mark.parametrize(
    ("sqlstate", "code"),
    [
        ("28P01", ErrorCode.AUTH),
        ("42501", ErrorCode.AUTH),
        ("57014", ErrorCode.TIMEOUT),
        ("42P01", ErrorCode.CONFIG),
        ("42703", ErrorCode.CONFIG),
        ("08006", ErrorCode.CONNECTION),
        ("22P02", ErrorCode.QUERY),
    ],
)
def test_sqlstates_are_classified(sqlstate, code) -> None:
    # _pg_error reads only diag.sqlstate, which psycopg sets from the server's error.
    error = SimpleNamespace(diag=SimpleNamespace(sqlstate=sqlstate))
    assert _pg_error(error, "t").code is code


def test_unreachable_server_is_a_connection_error() -> None:
    with pytest.raises(ConnectorError) as e:
        PgvectorConnector("postgresql://u:p@127.0.0.1:9/db", table="t", connect_timeout_s=2)
    assert e.value.code is ErrorCode.CONNECTION


def test_connection_config(tmp_path, monkeypatch) -> None:
    with pytest.raises(ConnectorError, match="Secrets must not"):
        parse_connection({"backend": "pgvector", "dsn": "postgresql://u:p@h/db", "table": "t"})
    with pytest.raises(ConnectorError, match='Missing "dsn_env"'):
        open_connector(parse_connection({"backend": "pgvector", "table": "t"}))
    monkeypatch.setenv("VA_PG", "postgresql://u:p@127.0.0.1:9/db")
    for extra, message in [
        ({"filter_columns": "lang"}, "must be a list"),
        ({"filter_columns": [1]}, "list of column names"),
        ({"distance": "manhattan"}, "not supported"),
        ({"table": "a.b.c"}, "schema.table"),
        ({"statement_timeout_ms": 0}, ">= 1"),
        ({"vector": "x"}, "Unknown field"),
    ]:
        raw = {"backend": "pgvector", "dsn_env": "VA_PG", "table": "t", **extra}
        with pytest.raises(ConnectorError, match=message):
            open_connector(parse_connection(raw, tmp_path))


# --- against PostgreSQL ------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg(synthetic) -> dict[str, str]:
    """Load a test schema and return DSNs: admin and a read-only reader role."""
    if not ADMIN_DSN:
        pytest.skip("Set VECTRAUDIT_TEST_PG_DSN to run pgvector integration tests.")
    vectors = np.load(synthetic["vectors"])[:N]
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP SCHEMA IF EXISTS va_test CASCADE")
        conn.execute("CREATE SCHEMA va_test")
        conn.execute(
            "CREATE TABLE va_test.chunks (id bigint PRIMARY KEY, embedding vector(32), "
            "lang text, public boolean, score double precision, body text)"
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO va_test.chunks VALUES (%s, %s::vector, %s, %s, %s, %s)",
                [
                    (
                        i,
                        vector_literal(v),
                        "en" if i % 2 == 0 else "de",
                        i < 100,
                        float(i % 3),
                        f"chunk {i}",
                    )
                    for i, v in enumerate(vectors)
                ],
            )
        conn.execute(
            "CREATE INDEX chunks_hnsw ON va_test.chunks USING hnsw (embedding vector_l2_ops)"
        )
        conn.execute("CREATE TABLE va_test.secret (id bigint, embedding vector(32))")
        conn.execute("CREATE TABLE va_test.loose (id bigint, embedding vector)")
        conn.execute("INSERT INTO va_test.loose VALUES (1, '[1,2,3]')")
        conn.execute("ANALYZE va_test.chunks")
        conn.execute("DROP ROLE IF EXISTS va_reader")
        conn.execute(f"CREATE ROLE va_reader LOGIN PASSWORD '{READER_PASSWORD}'")
        conn.execute("GRANT USAGE ON SCHEMA va_test TO va_reader")
        conn.execute("GRANT SELECT ON va_test.chunks, va_test.loose TO va_reader")
        # Tiny tables favour sequential scans; make the planner use the HNSW index when it
        # can, as it would on a production-sized table. Exact search still scans every row.
        conn.execute("ALTER ROLE va_reader SET enable_seqscan = off")
    info = psycopg.conninfo.conninfo_to_dict(ADMIN_DSN)
    reader = psycopg.conninfo.make_conninfo(ADMIN_DSN, user="va_reader", password=READER_PASSWORD)
    return {"admin": ADMIN_DSN, "reader": reader, "host": str(info.get("host"))}


def connect(dsn: str, **kwargs) -> PgvectorConnector:
    kwargs.setdefault("table", "va_test.chunks")
    kwargs.setdefault("distance", "l2")
    return PgvectorConnector(dsn, **kwargs)


@pytest.fixture(scope="module")
def query(synthetic) -> np.ndarray:
    return np.load(synthetic["queries"])[0]


def test_describe(pg) -> None:
    with connect(pg["reader"]) as c:
        info = c.describe()
        caps = c.capabilities()
    assert info.backend == "pgvector"
    assert info.distance is Distance.L2
    assert info.dim == 32
    assert info.count == N
    assert info.target == {
        "table": "va_test.chunks",
        "id_column": "id",
        "vector_column": "embedding",
    }
    assert "pgvector" in info.backend_version
    assert info.config["indexes"][0]["name"] == "chunks_hnsw"
    assert info.config["indexes"][0]["method"] == "hnsw"
    assert info.config["matching_index"]
    assert info.snapshot is None
    assert caps.exact_reference
    assert caps.query_plans
    assert not caps.filters
    assert not caps.filter_count
    assert caps.tuning_params == ("hnsw.ef_search", "ivfflat.probes")
    with connect(pg["reader"], distance="cosine") as c:
        assert not c.describe().config["matching_index"]


def test_search_matches_exact_faiss(pg, synthetic, tmp_path, query) -> None:
    import faiss

    flat = faiss.IndexFlatL2(32)
    flat.add(np.load(synthetic["vectors"])[:N])
    faiss.write_index(flat, str(tmp_path / "flat.index"))
    expected = FaissConnector(tmp_path / "flat.index").search(SearchRequest(query, 10))
    with connect(pg["reader"]) as c:
        assert c.search(SearchRequest(query, 10, exact=True)) == expected
        approx = c.search(SearchRequest(query, 10, params={"hnsw.ef_search": 100}))
        assert len(set(approx) & set(expected)) >= 8


def test_query_plans(pg, query) -> None:
    with connect(pg["reader"]) as c:
        indexed = c.explain(SearchRequest(query, 10, params={"hnsw.ef_search": 40}))
        exact = c.explain(SearchRequest(query, 10, exact=True))
    assert indexed["indexes"] == ["chunks_hnsw"]
    assert "Index Scan" in indexed["nodes"]
    assert exact["indexes"] == []
    assert "Seq Scan" in exact["nodes"]


def test_filters(pg, query) -> None:
    with connect(pg["reader"], filter_columns=["lang", "public", "score"]) as c:
        assert c.capabilities().filters
        en = c.search(SearchRequest(query, 20, filter={"lang": "en"}, exact=True))
        both = c.search(SearchRequest(query, 20, filter={"lang": "en", "public": True}))
        floats = c.search(SearchRequest(query, 20, filter={"score": 2.0}, exact=True))
        with pytest.raises(ConnectorError) as e:
            c.search(SearchRequest(query, 5, filter={"body": "x"}))
        assert e.value.code is ErrorCode.UNSUPPORTED
        with pytest.raises(ConnectorError) as e:
            c.search(SearchRequest(query, 5, filter={"lang": 3}))
        assert e.value.code is ErrorCode.QUERY
        # An exact count with the search's own condition, not the planner's estimate.
        assert c.capabilities().filter_count
        assert c.count({}) == N
        assert c.count({"lang": "en"}) == N // 2
        assert c.count({"lang": "en", "public": True}) == 50
        with pytest.raises(ConnectorError) as e:
            c.count({"body": "x"})
        assert e.value.code is ErrorCode.UNSUPPORTED
    assert len(en) == 20
    assert all(int(i) % 2 == 0 for i in en)
    assert all(int(i) % 2 == 0 and int(i) < 100 for i in both)
    assert all(int(i) % 3 == 2 for i in floats)


def test_sessions_are_read_only(pg) -> None:
    with connect(pg["admin"]) as c:
        assert c._fetch("SHOW transaction_read_only") == [("on",)]
        with pytest.raises(ConnectorError, match="read-only transaction"):
            c._fetch("DELETE FROM va_test.chunks")
    with psycopg.connect(pg["admin"]) as conn:
        assert conn.execute("SELECT count(*) FROM va_test.chunks").fetchone() == (N,)


def test_statement_timeout(pg) -> None:
    with connect(pg["reader"], statement_timeout_ms=50) as c:
        with pytest.raises(ConnectorError) as e:
            c._fetch("SELECT pg_sleep(1)")
        assert e.value.code is ErrorCode.TIMEOUT
        # The connection stays usable after a cancelled statement.
        assert c._fetch("SELECT 1") == [(1,)]


def test_errors(pg, query) -> None:
    bad_password = psycopg.conninfo.make_conninfo(pg["reader"], password="wrong")
    cases = [
        (lambda: connect(bad_password), ErrorCode.AUTH),
        (lambda: connect(pg["reader"], table="va_test.missing"), ErrorCode.CONFIG),
        (lambda: connect(pg["reader"], vector_column="nope"), ErrorCode.CONFIG),
        (lambda: connect(pg["reader"], vector_column="lang"), ErrorCode.UNSUPPORTED),
        (lambda: connect(pg["reader"], table="va_test.secret"), ErrorCode.AUTH),
    ]
    for make, code in cases:
        with pytest.raises(ConnectorError) as e:
            make()
        assert e.value.code is code, e.value.message
    with connect(pg["reader"]) as c:
        for request, code in [
            (SearchRequest(np.zeros(3, dtype=np.float32), 5), ErrorCode.QUERY),
            (SearchRequest(query, 5, params={"hnsw.ef_search": 0}), ErrorCode.CONFIG),
            (SearchRequest(query, 5, params={"work_mem": 64}), ErrorCode.CONFIG),
        ]:
            with pytest.raises(ConnectorError) as e:
                c.search(request)
            assert e.value.code is code


def test_unconstrained_vector_column(pg) -> None:
    with connect(pg["reader"], table="va_test.loose") as c:
        assert c.describe().dim == 3
        assert c.search(SearchRequest(np.ones(3, dtype=np.float32), 2)) == ["1"]


def test_run_via_connection_file(pg, synthetic, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VA_PG_DSN", pg["reader"])
    conn = {
        "backend": "pgvector",
        "dsn_env": "VA_PG_DSN",
        "table": "va_test.chunks",
        "distance": "l2",
        "filter_columns": ["lang"],
        "params": {"hnsw.ef_search": 64},
    }
    path = tmp_path / "conn.json"
    path.write_text(json.dumps(conn))
    cfg = parse_connection(json.loads(path.read_text()), Path(tmp_path))
    ds = make_dataset(synthetic, n=10, filter={"lang": "en"})
    with open_connector(cfg) as c:
        run = execute_run(ds, c, RunOptions(k=5, params=cfg.params), connection=cfg.public())
    assert run["status"] == "complete"
    assert run["summary"]["ann_recall"] > 0.8
    evidence = run["queries"][0]["evidence"]
    assert evidence["kind"] == "pg_plan"
    assert evidence["source"].startswith("planner estimate")
    assert evidence["indexes"] == ["chunks_hnsw"]
    text = json.dumps(run)
    assert READER_PASSWORD not in text
    assert "VA_PG_DSN" in text
