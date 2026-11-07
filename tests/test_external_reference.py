"""ANN recall against an external exact reference, and when two such runs can be compared."""

import json

import faiss
import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_vdbbench_import import D, K, N, make
from vectraudit.cli import app
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation import external_reference as X
from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runner import RunOptions, execute_run
from vectraudit.evaluation.vdbbench_import import import_vdbbench

runner = CliRunner()


@pytest.fixture
def world(tmp_path):
    src, data = make(tmp_path)
    out = tmp_path / "out"
    import_vdbbench(src, out, distance="l2")
    flat = faiss.IndexFlatL2(D)
    flat.add(data["x"])
    faiss.write_index(flat, str(out / "flat.index"))
    hnsw = faiss.IndexHNSWFlat(D, 4)
    hnsw.hnsw.efSearch = 1
    hnsw.add(data["x"])
    faiss.write_index(hnsw, str(out / "hnsw.index"))
    return out, data


def connector(out, index="flat.index", *, metadata=False):
    return FaissConnector(
        out / index,
        vectors=out / "vectors.npy",
        ids=out / "ids.npy",
        metadata=out / "metadata.jsonl" if metadata else None,
    )


def record(out, index="flat.index", *, variant="", metadata=False, **options):
    ds = load_dataset(out / f"dataset{variant}.jsonl")
    ref = X.load_reference(out / f"reference{variant}.json")
    opts = RunOptions(k=K, warmup=0, reference=ref, **options)
    with connector(out, index, metadata=metadata) as c:
        return execute_run(ds, c, opts)


def test_ann_recall_is_measured_against_the_published_neighbours(world) -> None:
    out, _ = world
    run = record(out)
    assert run["status"] == "complete", run["error"]
    assert run["search"]["exact_reference"] is True
    assert run["search"]["reference"]["source"] == "external:vdbbench"
    assert "not relevance labelling" in run["search"]["exact_reference_note"]
    assert all(q["ann_recall"] == 1.0 for q in run["queries"])  # a flat index is exact
    ref = X.load_reference(out / "reference.json")
    assert run["queries"][2]["exact_ids"] == ref.neighbours("q2", K)


def test_the_backend_is_not_asked_for_its_own_exact_search(world) -> None:
    out, _ = world

    class Counting(FaissConnector):
        exact_calls = 0

        def _search(self, request):
            Counting.exact_calls += request.exact
            return super()._search(request)

    ds = load_dataset(out / "dataset.jsonl")
    ref = X.load_reference(out / "reference.json")
    c = Counting(out / "flat.index", vectors=out / "vectors.npy", ids=out / "ids.npy")
    execute_run(ds, c, RunOptions(k=K, warmup=0, reference=ref))
    assert Counting.exact_calls == 0


def test_an_approximate_index_scores_below_one(world) -> None:
    out, _ = world
    run = record(out, "hnsw.index")
    recalls = [q["ann_recall"] for q in run["queries"]]
    assert all(r is not None for r in recalls)
    assert 0.0 < np.mean(recalls) < 1.0


def test_a_filtered_variant_is_checked_with_its_filter(world) -> None:
    out, _ = world
    run = record(out, variant="-int-90p", metadata=True)
    assert run["status"] == "complete", run["error"]
    assert all(q["filtered"] for q in run["queries"])
    assert all(q["ann_recall"] == 1.0 for q in run["queries"])
    assert run["search"]["reference"]["filter"] == {"vdb_int_90p": True}
    assert run["search"]["pass_rates"]["counted"] == len(run["queries"])
    # 90% removed is a 10% pass rate.
    assert all(q["pass_rate"] == pytest.approx(30 / N) for q in run["queries"])


def _failed(run, text):
    assert run["status"] == "failed"
    assert text in run["error"]["message"]
    return run["error"]


def test_a_reference_for_another_filter_is_refused(world) -> None:
    out, _ = world
    ds = load_dataset(out / "dataset.jsonl")  # unfiltered queries
    ref = X.load_reference(out / "reference-int-90p.json")
    with connector(out, metadata=True) as c:
        run = execute_run(ds, c, RunOptions(k=K, warmup=0, reference=ref))
    _failed(run, "assumes the filter")


def test_a_reference_computed_under_another_distance_is_refused(world) -> None:
    out, data = world
    ip = faiss.IndexFlatIP(D)
    ip.add(data["x"])
    faiss.write_index(ip, str(out / "ip.index"))
    run = record(out, "ip.index")
    error = _failed(run, "computed under l2 distance")
    assert "another" in error["hint"] or "distance" in error["hint"]


def test_k_beyond_the_reference_and_competing_references_are_refused(world) -> None:
    out, _ = world
    _failed(_with_k(out, K + 1), "larger than")
    _failed(record(out, exact=True), "two different exact references")


def _with_k(out, k):
    ds = load_dataset(out / "dataset.jsonl")
    ref = X.load_reference(out / "reference.json")
    with connector(out) as c:
        return execute_run(ds, c, RunOptions(k=k, warmup=0, reference=ref))


def test_queries_missing_from_the_reference_are_refused(world, tmp_path) -> None:
    out, _ = world
    lines = (out / "dataset.jsonl").read_text().splitlines()
    extra = json.loads(lines[1])
    extra["query_id"] = "not-in-reference"
    (out / "other.jsonl").write_text("\n".join([*lines, json.dumps(extra)]))
    ref = X.load_reference(out / "reference.json")
    with connector(out) as c:
        run = execute_run(
            load_dataset(out / "other.jsonl"), c, RunOptions(k=K, warmup=0, reference=ref)
        )
    _failed(run, "not in the reference")


# --- comparison ----------------------------------------------------------------


POLICY = parse_policy({"min_queries": 1, "metrics": {"ann_recall": {"max_drop": 0.5}}})


def native(out):
    ds = load_dataset(out / "dataset.jsonl")
    with connector(out) as c:
        return execute_run(ds, c, RunOptions(k=K, warmup=0, exact=True))


def test_runs_against_the_same_reference_compare_on_ann_recall(world) -> None:
    out, _ = world
    verdict = compare_runs(record(out), record(out, "hnsw.index"), POLICY)
    assert verdict["verdict"] in (Verdict.PASS.value, Verdict.FAIL.value)
    assert "ann_recall" in verdict["metrics"]
    assert not any("different exact references" in w for w in verdict["warnings"])


def test_runs_against_different_references_are_not_compared_on_ann_recall(world) -> None:
    out, _ = world
    verdict = compare_runs(native(out), record(out), POLICY)
    assert verdict["verdict"] == Verdict.NOT_COMPARABLE.value
    assert "ann_recall" not in verdict["metrics"]
    assert any("different exact references" in w for w in verdict["warnings"])
    text = " ".join(r["message"] for r in verdict["reasons"])
    assert "the backend's own exact search vs external:vdbbench" in text
    # The same holds for two different external references (here: an unfiltered and a filtered).
    other = record(out, variant="-int-90p", metadata=True)
    assert X.reference_identity(other) != X.reference_identity(record(out))


def test_a_policy_that_does_not_need_ann_recall_is_not_blocked(world) -> None:
    out, _ = world
    verdict = compare_runs(native(out), record(out))  # default policy: no metric rules
    assert any("different exact references" in w for w in verdict["warnings"])


# --- the command ---------------------------------------------------------------


def test_record_accepts_a_reference_and_records_its_source(world, tmp_path) -> None:
    out, _ = world
    conn = tmp_path / "conn.json"
    conn.write_text(
        json.dumps(
            {
                "backend": "faiss",
                "index": str(out / "flat.index"),
                "vectors": str(out / "vectors.npy"),
                "ids": str(out / "ids.npy"),
            }
        )
    )
    target = tmp_path / "run.json"
    args = [
        "--connection",
        str(conn),
        "--dataset",
        str(out / "dataset.jsonl"),
        "--out",
        str(target),
    ]
    done = runner.invoke(app, ["record", *args, "--reference", str(out / "reference.json")])
    assert done.exit_code == 0, done.output
    assert json.loads(target.read_text())["search"]["reference"]["source"] == "external:vdbbench"
    bad = runner.invoke(app, ["record", *args, "--reference", str(out / "missing.json")])
    assert bad.exit_code == 1
    assert "Cannot read" in " ".join(bad.output.split())


# --- the sweep -----------------------------------------------------------------


def sweep(out, *extra, index="hnsw.index"):
    return runner.invoke(
        app,
        [
            "sweep",
            str(out / index),
            "--queries",
            str(out / "queries.npy"),
            "--values",
            "1,64",
            "--n-queries",
            "12",
            "--repeats",
            "1",
            "--json",
            *extra,
        ],
    )


def test_a_sweep_against_the_published_neighbours_equals_the_one_computed_here(world) -> None:
    out, _ = world
    external = sweep(out, "--reference", str(out / "reference.json"))
    assert external.exit_code in (0, 2), external.output
    local = sweep(out, "--vectors", str(out / "vectors.npy"), "--ids", str(out / "ids.npy"))
    assert local.exit_code in (0, 2), local.output
    a, b = json.loads(external.stdout), json.loads(local.stdout)
    assert a["truth_source"] == "external"
    assert a["reference"]["source"] == "external:vdbbench"
    assert b["truth_source"] == "raw"
    # The published neighbours ARE the exact ones here, so the measured recall is identical.
    assert [p["recall"] for p in a["points"]] == [p["recall"] for p in b["points"]]


def test_a_sweep_reference_needs_its_own_queries(world, tmp_path) -> None:
    out, data = world
    ref = str(out / "reference.json")
    base = ["sweep", str(out / "hnsw.index"), "--values", "1", "--reference", ref]
    missing = runner.invoke(app, base)
    assert missing.exit_code == 1
    assert "needs the queries" in " ".join(missing.output.split())
    np.save(tmp_path / "other.npy", data["q"] + 1.0)
    wrong = runner.invoke(app, [*base, "--queries", str(tmp_path / "other.npy")])
    assert wrong.exit_code == 1
    assert "not the ones the reference was built for" in " ".join(wrong.output.split())


def test_a_sweep_refuses_a_filtered_reference_and_filter_options(world) -> None:
    out, _ = world
    filtered = sweep(out, "--reference", str(out / "reference-int-90p.json"))
    assert filtered.exit_code == 1
    assert "assumes a metadata filter" in " ".join(filtered.output.split())
    both = sweep(out, "--reference", str(out / "reference.json"), "--pass-rates", "0.5")
    assert both.exit_code == 1
    assert "cannot be combined" in " ".join(both.output.split())


def test_a_sweep_refuses_a_reference_under_another_distance(world) -> None:
    out, data = world
    ip = faiss.IndexHNSWFlat(D, 4, faiss.METRIC_INNER_PRODUCT)
    ip.add(data["x"])
    faiss.write_index(ip, str(out / "hnsw_ip.index"))
    result = sweep(out, "--reference", str(out / "reference.json"), index="hnsw_ip.index")
    assert result.exit_code == 1
    assert "computed under l2" in " ".join(result.output.split())


def test_sweep_runs_with_different_ground_truth_are_not_compared(world, tmp_path) -> None:
    out, _ = world
    a, b = tmp_path / "local.json", tmp_path / "external.json"
    local = ["--vectors", str(out / "vectors.npy"), "--ids", str(out / "ids.npy")]
    assert sweep(out, *local, "--save", str(a)).exit_code in (0, 2)
    assert sweep(out, "--reference", str(out / "reference.json"), "--save", str(b)).exit_code in (
        0,
        2,
    )
    diff = runner.invoke(app, ["runs", "diff", str(a), str(b)])
    assert diff.exit_code == 4  # nothing could be judged: not a pass
    assert "different kinds of ground truth" in " ".join(diff.output.split())
