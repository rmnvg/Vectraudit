"""Filtered queries in release runs: counted pass rates, selectivity cohorts, unknown rates."""

import json
from pathlib import Path

import numpy as np
import pytest

from vectraudit.connectors import Capabilities, ConnectorError, ErrorCode, SearchRequest
from vectraudit.connectors.base import BackendInfo, Connector, Distance
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.core.metadata import Metadata
from vectraudit.evaluation import pass_rates as P
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.runner import RunOptions, execute_run

N = 2000


def _dataset(synthetic, buckets=None, *, filter_=None, n=6) -> str:
    queries = np.load(synthetic["queries"])
    header = {"dataset": {"name": "filtered"}}
    if buckets is not None:
        header["dataset"]["selectivity_buckets"] = buckets
    rows = [json.dumps(header)]
    for i in range(n):
        row = {"query_id": f"q{i}", "vector": queries[i].tolist(), "relevant": {str(i): 1}}
        # Alternate filtered and unfiltered queries.
        if i % 2 == 0:
            row["filter"] = filter_ if filter_ is not None else {"source": "blob_00.txt"}
        rows.append(json.dumps(row))
    return "\n".join(rows)


def _connector(synthetic, index="flat_l2", **extra) -> FaissConnector:
    return FaissConnector(
        synthetic[index], vectors=synthetic["vectors"], metadata=synthetic["chunks"], **extra
    )


# --- the FAISS connector's filters ------------------------------------------


def test_faiss_connector_filters_and_counts_from_its_metadata(synthetic) -> None:
    with _connector(synthetic) as c:
        caps = c.capabilities()
        assert caps.filters
        assert caps.filter_count
        total = c.count({})
        some = c.count({"source": "blob_00.txt"})
        assert total == N
        assert 0 < some < total
        assert c.count({"source": "nowhere.txt"}) == 0
        q = np.load(synthetic["queries"])[0]
        found = c.search(SearchRequest(q, 10, filter={"source": "blob_00.txt"}))
        assert len(found) == 10
        # Every id returned really has that source.
        lines = Path(synthetic["chunks"]).read_text().splitlines()
        meta = {row["id"]: row for row in map(json.loads, lines)}
        assert all(meta[int(i)]["source"] == "blob_00.txt" for i in found)


def test_faiss_exact_reference_applies_the_same_filter(synthetic) -> None:
    with _connector(synthetic, "ivf_flat") as c:
        q = np.load(synthetic["queries"])[1]
        flt = {"source": "blob_01.txt"}
        exact = c.search(SearchRequest(q, 10, filter=flt, exact=True))
        flat = _connector(synthetic).search(SearchRequest(q, 10, filter=flt))
        assert exact == flat  # brute force over the allowed ids == a flat index's own search
        full = c.search(SearchRequest(q, 10, filter=flt, params={"nprobe": 16}))
        assert full == exact  # every cell probed: exact


def test_faiss_connector_without_metadata_cannot_filter_or_count(synthetic) -> None:
    c = FaissConnector(synthetic["flat_l2"], vectors=synthetic["vectors"])
    assert not c.capabilities().filters
    assert not c.capabilities().filter_count
    q = np.load(synthetic["queries"])[0]
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(q, 5, filter={"source": "x"}))
    assert e.value.code is ErrorCode.UNSUPPORTED
    with pytest.raises(ConnectorError) as e:
        c.count({"source": "x"})
    assert e.value.code is ErrorCode.UNSUPPORTED


def test_faiss_connector_names_an_unknown_filter_field(synthetic) -> None:
    c = _connector(synthetic)
    q = np.load(synthetic["queries"])[0]
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(q, 5, filter={"colour": "red"}))
    assert e.value.code is ErrorCode.QUERY
    assert "colour" in e.value.message
    assert "source" in (e.value.hint or "")


def test_faiss_connector_filters_with_an_idmap_index(synthetic) -> None:
    ids = np.load(synthetic["ids_idmap"])
    meta = [{"id": int(i), "grp": "a" if n % 4 == 0 else "b"} for n, i in enumerate(ids)]
    c = FaissConnector(
        synthetic["idmap_flat"],
        vectors=synthetic["vectors"],
        ids=synthetic["ids_idmap"],
        metadata=Metadata(meta),
    )
    q = np.load(synthetic["queries"])[0]
    got = c.search(SearchRequest(q, 10, filter={"grp": "a"}))
    assert len(got) == 10
    assert {int(i) for i in got} <= {m["id"] for m in meta if m["grp"] == "a"}
    assert c.count({"grp": "a"}) == N // 4


# --- the run ----------------------------------------------------------------


def test_run_records_counted_pass_rates_and_selectivity_cohorts(synthetic) -> None:
    ds = parse_dataset(_dataset(synthetic, [0.01, 0.1]))
    with _connector(synthetic, "ivf_flat") as c:
        run = execute_run(ds, c, RunOptions(k=10, params={"nprobe": 4}))
    assert run["status"] == "complete"
    filtered = [q for q in run["queries"] if q["filtered"]]
    plain = [q for q in run["queries"] if not q["filtered"]]
    assert len(filtered) == 3
    rate = filtered[0]["pass_rate"]
    assert 0.01 < rate < 0.1  # 20 sources in the synthetic corpus: about 5%
    assert all(q["tags"] == ["pass_rate<10%"] for q in filtered)
    assert all("pass_rate" not in q and q["tags"] == [] for q in plain)
    assert all(q["ann_recall"] is not None for q in run["queries"])  # filtered exact reference
    assert run["search"]["pass_rates"] == {
        "supported": True,
        "filtered_queries": 3,
        "counted": 3,
        "buckets": [0.01, 0.1],
        "meaning": "fraction of the corpus that satisfies the filter, counted by the backend",
    }


def test_buckets_are_exclusive_and_a_permissive_filter_is_in_none(synthetic) -> None:
    # Pass rate ~5%: below 0.1 only; with a bucket at 0.5 it is the 0.5 bucket, not both.
    ds = parse_dataset(_dataset(synthetic, [0.5, 0.1]))
    with _connector(synthetic) as c:
        run = execute_run(ds, c, RunOptions(k=5))
    assert {tuple(q["tags"]) for q in run["queries"] if q["filtered"]} == {("pass_rate<10%",)}
    ds = parse_dataset(_dataset(synthetic, [0.01]))
    with _connector(synthetic) as c:
        run = execute_run(ds, c, RunOptions(k=5))
    filtered = [q for q in run["queries"] if q["filtered"]]
    assert all(q["pass_rate"] > 0.01 and q["tags"] == [] for q in filtered)


def test_bucket_helpers() -> None:
    assert P.bucket_label(0.001) == "pass_rate<0.1%"
    assert P.bucket_label(0.1) == "pass_rate<10%"
    buckets = [0.1, 0.001, 0.01]
    assert P.bucket_for(0.0005, buckets) == "pass_rate<0.1%"
    assert P.bucket_for(0.001, buckets) == "pass_rate<1%"  # the bound itself is the next bucket
    assert P.bucket_for(0.05, buckets) == "pass_rate<10%"
    assert P.bucket_for(0.1, buckets) is None
    assert P.bucket_for(None, buckets) is None


class _NoCount(Connector):
    """Filters, but no way to count them."""

    name = "nocount"

    def describe(self) -> BackendInfo:
        return BackendInfo("nocount", Distance.L2, 32, None)

    def capabilities(self) -> Capabilities:
        return Capabilities(filters=True)

    def _search(self, request: SearchRequest) -> list[str]:
        return ["1", "2"]


def test_an_uncountable_pass_rate_is_unknown_not_low(synthetic) -> None:
    ds = parse_dataset(_dataset(synthetic, [0.01, 0.1]))
    run = execute_run(ds, _NoCount(), RunOptions(k=2, warmup=0))
    assert run["status"] == "complete"
    filtered = [q for q in run["queries"] if q["filtered"]]
    assert filtered
    assert all(q["pass_rate"] is None and q["tags"] == [] for q in filtered)
    summary = run["search"]["pass_rates"]
    assert (summary["supported"], summary["counted"], summary["filtered_queries"]) == (False, 0, 3)
    with pytest.raises(ConnectorError) as e:
        _NoCount().count({"a": 1})
    assert e.value.code is ErrorCode.UNSUPPORTED


def test_a_run_without_filters_has_no_pass_rate_fields(synthetic) -> None:
    rows = [json.dumps({"query_id": "q", "vector": np.load(synthetic["queries"])[0].tolist()})]
    with _connector(synthetic) as c:
        run = execute_run(parse_dataset("\n".join(rows)), c, RunOptions(k=5))
    assert "pass_rates" not in run["search"]
    assert "pass_rate" not in run["queries"][0]


def test_each_distinct_filter_is_counted_once(synthetic) -> None:
    class Counting(FaissConnector):
        calls = 0

        def _count(self, filter):
            Counting.calls += 1
            return super()._count(filter)

    ds = parse_dataset(_dataset(synthetic, [0.1], n=8))
    c = Counting(synthetic["flat_l2"], vectors=synthetic["vectors"], metadata=synthetic["chunks"])
    execute_run(ds, c, RunOptions(k=5))
    assert Counting.calls == 2  # the corpus total, and the one filter used by four queries


# --- the dataset header -----------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [0],
        [1],
        [1.5],
        [0.1, 0.1],
        "0.1",
        [0.1, "x"],
        [True],
        [0.1] * 7,
        [0.01 * i for i in range(1, 8)],
    ],
)
def test_header_rejects_bad_buckets(synthetic, bad) -> None:
    with pytest.raises(DatasetError, match="selectivity_buckets"):
        parse_dataset(_dataset(synthetic, bad))


def test_buckets_are_sorted_in_the_dataset_identity(synthetic) -> None:
    a = parse_dataset(_dataset(synthetic, [0.1, 0.01]))
    b = parse_dataset(_dataset(synthetic, [0.01, 0.1]))
    assert a.meta["selectivity_buckets"] == [0.01, 0.1]
    assert a.sha256 == b.sha256
    assert a.sha256 != parse_dataset(_dataset(synthetic)).sha256
