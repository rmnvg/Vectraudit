"""Connector contract, connection files and the FAISS connector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vectraudit.connectors import (
    ConnectorError,
    Distance,
    ErrorCode,
    SearchRequest,
    load_connection,
    open_connector,
)
from vectraudit.connectors.config import parse_connection, secret
from vectraudit.connectors.faiss_index import FaissConnector


def _write(path: Path, obj: object) -> Path:
    path.write_text(json.dumps(obj))
    return path


@pytest.fixture(scope="module")
def query(synthetic: dict[str, Path]) -> np.ndarray:
    return np.load(synthetic["queries"])[0]


def test_ivf_connector_describes_and_searches(synthetic, query) -> None:
    with FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"]) as c:
        info = c.describe()
        caps = c.capabilities()
        assert info.backend == "faiss"
        assert info.distance is Distance.L2
        assert info.dim == 32
        assert info.count == 2_000
        assert info.config["kind"] == "IVF_FLAT"
        assert info.snapshot is not None
        assert info.snapshot.startswith("sha256:")
        assert info.target == {"index": "ivf_flat.index"}
        assert caps.tuning_params == ("nprobe",)
        assert caps.exact_reference
        assert caps.exact_reference_note is None
        assert not caps.filters

        low = c.search(SearchRequest(query, 10, params={"nprobe": 1}))
        exact = c.search(SearchRequest(query, 10, exact=True))
        full = c.search(SearchRequest(query, 10, params={"nprobe": 16}))
        assert len(low) == 10
        assert all(isinstance(i, str) for i in low)
        # Probing every list of an IVF-Flat index is exact.
        assert full == exact


def test_idmap_ids_are_user_facing(synthetic, query) -> None:
    ids = np.load(synthetic["ids_idmap"])
    c = FaissConnector(synthetic["idmap_flat"])
    found = c.search(SearchRequest(query, 5, exact=True))
    assert set(found) <= {str(i) for i in ids}
    # A flat index is its own exact reference.
    assert c.capabilities().exact_reference


def test_hnsw_tuning_and_ip_distance(synthetic, query) -> None:
    hnsw = FaissConnector(synthetic["hnsw_flat"])
    assert hnsw.capabilities().tuning_params == ("ef_search",)
    assert not hnsw.capabilities().exact_reference
    assert len(hnsw.search(SearchRequest(query, 3, params={"ef_search": 64}))) == 3
    assert FaissConnector(synthetic["ivf_flat_ip"]).describe().distance is Distance.IP


def test_reconstructed_reference_is_marked_weaker(synthetic, query) -> None:
    c = FaissConnector(synthetic["ivf_pq"], reconstructed_reference=True)
    caps = c.capabilities()
    assert caps.exact_reference
    assert caps.exact_reference_note is not None
    assert "compression" in caps.exact_reference_note
    assert len(c.search(SearchRequest(query, 5, exact=True))) == 5


@pytest.mark.parametrize(
    ("request_kwargs", "code"),
    [
        ({"filter": {"lang": "en"}}, ErrorCode.UNSUPPORTED),
        ({"exact": True}, ErrorCode.UNSUPPORTED),
        ({"params": {"ef_search": 10}}, ErrorCode.CONFIG),
        ({"params": {"nprobe": 0}}, ErrorCode.CONFIG),
        ({"k": 0}, ErrorCode.QUERY),
    ],
)
def test_unsupported_requests_are_errors_not_empty_results(
    synthetic, query, request_kwargs, code
) -> None:
    c = FaissConnector(synthetic["ivf_pq"])
    kwargs = {"k": 5, **request_kwargs}
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(query, **kwargs))
    assert e.value.code is code


def test_wrong_dimension_is_a_query_error(synthetic) -> None:
    c = FaissConnector(synthetic["flat_l2"])
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(np.zeros(3, dtype=np.float32), 5))
    assert e.value.code is ErrorCode.QUERY


def test_faiss_connector_config_errors(synthetic, tmp_path, binary_index_path) -> None:
    with pytest.raises(ConnectorError) as e:
        FaissConnector(tmp_path / "missing.index")
    assert e.value.code is ErrorCode.CONFIG
    with pytest.raises(ConnectorError) as e:
        FaissConnector(binary_index_path)
    assert e.value.code is ErrorCode.UNSUPPORTED
    with pytest.raises(ConnectorError, match="Vectors file not found"):
        FaissConnector(synthetic["flat_l2"], vectors=tmp_path / "nope.npy")
    with pytest.raises(ConnectorError, match="ids needs vectors"):
        FaissConnector(synthetic["flat_l2"], ids=synthetic["ids_idmap"])
    np.save(tmp_path / "short.npy", np.zeros((3, 32), dtype=np.float32))
    with pytest.raises(ConnectorError) as e:
        FaissConnector(synthetic["flat_l2"], vectors=tmp_path / "short.npy")
    assert e.value.code is ErrorCode.CONFIG
    assert e.value.hint


def test_connection_file_resolves_relative_paths(synthetic, tmp_path, query) -> None:
    data = synthetic["ivf_flat"].parent
    path = _write(
        data / "conn.json",
        {
            "backend": "faiss",
            "name": "prod-ivf",
            "index": "ivf_flat.index",
            "vectors": "vectors.npy",
            "params": {"nprobe": 4},
        },
    )
    cfg = load_connection(path)
    assert cfg.name == "prod-ivf"
    assert cfg.params == {"nprobe": 4}
    assert cfg.public() == {
        "backend": "faiss",
        "name": "prod-ivf",
        "index": "ivf_flat.index",
        "vectors": "vectors.npy",
        "params": {"nprobe": 4},
    }
    with open_connector(cfg) as c:
        assert c.capabilities().exact_reference
        assert len(c.search(SearchRequest(query, 3))) == 3


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "JSON object"),
        ({"backend": "redis"}, "Unknown backend"),
        ({"backend": "faiss", "index": "x", "api_key": "s3cret"}, "Secrets must not"),
        ({"backend": "faiss", "index": "x", "dsn": "postgres://u:p@h/db"}, "Secrets must not"),
        ({"backend": "faiss", "index": "x", "params": {"nprobe": "8"}}, "params"),
        ({"backend": "faiss", "index": "x", "params": {"nprobe": True}}, "params"),
        ({"backend": "faiss", "index": "x", "name": 3}, "name"),
    ],
)
def test_invalid_connections(raw, message) -> None:
    with pytest.raises(ConnectorError, match=message) as e:
        parse_connection(raw)
    assert e.value.code is ErrorCode.CONFIG


def test_faiss_options_are_validated(tmp_path) -> None:
    for raw, message in [
        ({"backend": "faiss"}, 'needs "index"'),
        ({"backend": "faiss", "index": 3}, "must be a str"),
        ({"backend": "faiss", "index": "x", "colour": "red"}, "Unknown field"),
    ]:
        with pytest.raises(ConnectorError, match=message):
            open_connector(parse_connection(raw, tmp_path))


def test_connection_file_errors(tmp_path) -> None:
    with pytest.raises(ConnectorError, match="not found"):
        load_connection(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ConnectorError, match="not valid JSON"):
        load_connection(tmp_path / "bad.json")


def test_secret_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VA_TEST_KEY", "abc")
    assert secret({"api_key_env": "VA_TEST_KEY"}, "api_key") == "abc"
    assert secret({}, "api_key", required=False) is None
    with pytest.raises(ConnectorError, match='Missing "api_key_env"'):
        secret({}, "api_key")
    monkeypatch.delenv("VA_TEST_KEY")
    with pytest.raises(ConnectorError, match="VA_TEST_KEY is not set"):
        secret({"api_key_env": "VA_TEST_KEY"}, "api_key")
    with pytest.raises(ConnectorError, match="must name an environment variable"):
        secret({"api_key_env": 5}, "api_key")


def test_results_are_deduplicated_and_truncated() -> None:
    from vectraudit.connectors.base import BackendInfo, Capabilities, Connector

    class Echo(Connector):
        def describe(self) -> BackendInfo:
            return BackendInfo("echo", Distance.L2, 1, None)

        def capabilities(self) -> Capabilities:
            return Capabilities()

        def _search(self, request: SearchRequest) -> list[str]:
            return ["a", "b", "a", "c", "d"]

    c = Echo()
    assert c.search(SearchRequest(np.zeros(1, dtype=np.float32), 3)) == ["a", "b", "c"]
    c.close()
    c.close()
