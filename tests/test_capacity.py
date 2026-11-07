"""The capacity curve: bounded read-only load, refused unless confirmed, honest about limits."""

import json
import threading
from itertools import pairwise

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.connectors import Capabilities, ConnectorError, ErrorCode, SearchRequest
from vectraudit.connectors.base import BackendInfo, Connector, Distance
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation import capacity as C
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.ratelimit import RateLimiter

runner = CliRunner()


def _dataset(synthetic, n=20) -> str:
    queries = np.load(synthetic["queries"])
    return "\n".join(
        json.dumps({"query_id": f"q{i}", "vector": queries[i % len(queries)].tolist()})
        for i in range(n)
    )


def _plan(**kw) -> C.CapacityPlan:
    base = {"concurrency": (1, 2), "duration_s": 1.0, "max_rps": 200.0, "serial_queries": 5}
    return C.CapacityPlan(**{**base, **kw})


def _info(backend="qdrant", **target) -> BackendInfo:
    return BackendInfo(backend, Distance.L2, 4, 10, target or {"collection": "staging-docs"})


# --- limits and safety -------------------------------------------------------


def test_the_rate_limiter_spaces_calls_across_threads() -> None:
    limiter = RateLimiter(100.0)
    waits = [limiter.reserve() for _ in range(5)]
    assert waits[0] == 0.0
    assert all(b > a for a, b in pairwise(waits))
    assert waits[4] == pytest.approx(0.04, abs=0.01)
    assert RateLimiter(None).reserve() == 0.0


@pytest.mark.parametrize(
    "kw",
    [
        {"concurrency": ()},
        {"concurrency": (0,)},
        {"concurrency": (33,)},
        {"concurrency": (1, 1)},
        {"concurrency": tuple(range(1, 10))},
        {"duration_s": 0.5},
        {"duration_s": 601},
        {"max_rps": 0},
        {"max_rps": 1001},
        {"k": 0},
    ],
)
def test_plan_limits(kw) -> None:
    with pytest.raises(C.CapacityRefused):
        C.CapacityPlan(**kw)


def test_defaults_are_the_owner_decided_limits() -> None:
    plan = C.CapacityPlan()
    assert (plan.duration_s, plan.max_rps) == (30.0, 50.0)
    assert plan.level_request_ceiling == 1501


def test_a_run_is_refused_without_the_matching_target() -> None:
    info = _info()
    ok = C.safety_problems(info, None, confirm_target="staging-docs", allow_production=False)
    assert ok == []
    assert (
        "--confirm-target staging-docs"
        in C.safety_problems(info, None, confirm_target=None, allow_production=False)[0]
    )
    wrong = C.safety_problems(info, None, confirm_target="other", allow_production=False)
    assert "does not match" in wrong[0]
    no_target = C.safety_problems(
        BackendInfo("qdrant", Distance.L2, 4, 1, {}),
        None,
        confirm_target="x",
        allow_production=False,
    )
    assert "does not name a target" in no_target[0]


def test_production_needs_an_explicit_override_even_when_confirmed() -> None:
    prod = _info(collection="docs-PROD")
    problems = C.safety_problems(prod, None, confirm_target="docs-PROD", allow_production=False)
    assert len(problems) == 1
    assert "production" in problems[0]
    assert C.safety_problems(prod, None, confirm_target="docs-PROD", allow_production=True) == []
    named = C.safety_problems(
        _info(), "prod-eu", confirm_target="staging-docs", allow_production=False
    )
    assert "'prod-eu' looks like production" in named[0]


def test_each_backend_has_a_target_name() -> None:
    for backend, key in C.TARGET_KEYS.items():
        assert C.target_name(BackendInfo(backend, Distance.L2, 4, 1, {key: "t"})) == "t"
    assert C.target_name(_info("unknown", collection="x")) is None


# --- a run -------------------------------------------------------------------


def test_a_local_run_measures_each_level_within_its_ceiling(synthetic) -> None:
    ds = parse_dataset(_dataset(synthetic))
    plan = _plan(max_rps=100.0)
    report = C.run_capacity(
        ds,
        lambda: FaissConnector(synthetic["flat_l2"], vectors=synthetic["vectors"]),
        plan,
        connection={"backend": "faiss"},
    )
    assert report["status"] == "complete", report["error"]
    assert (report["format"], report["format_version"]) == ("vectraudit.capacity", 1)
    assert [lv["concurrency"] for lv in report["levels"]] == [1, 2]
    sent = report["serial"]["requests"] + sum(lv["requests"] for lv in report["levels"])
    assert sent <= plan.request_ceiling(True)
    for level in report["levels"]:
        assert level["requests"] <= plan.level_request_ceiling
        assert level["achieved_qps"] <= plan.max_rps * 1.05
        assert level["p50_ms"] <= level["p95_ms"] <= level["p99_ms"]
        assert level["error_rate"] == 0.0
        assert level["at_ceiling"]  # a flat index on 20 queries is far faster than 100/s
        assert level["recall"]["mean"] == pytest.approx(1.0)  # flat: its own exact reference
        assert level["recall"]["sample"] > 0
    json.dumps(report, allow_nan=False)
    assert any("at_ceiling" in n for n in report["notes"])
    assert report["runner"]["python"]


def test_one_connector_per_worker_unless_it_is_thread_safe(synthetic) -> None:
    class Slow(FaissConnector):
        thread_safe = False

    opened: list[object] = []

    def open_worker():
        opened.append(1)
        return Slow(synthetic["flat_l2"], vectors=synthetic["vectors"])

    ds = parse_dataset(_dataset(synthetic))
    C.run_capacity(ds, open_worker, _plan(concurrency=(3,), max_rps=50.0))
    assert len(opened) == 3
    shared: list[object] = []

    def open_shared():
        shared.append(1)
        return FaissConnector(synthetic["flat_l2"], vectors=synthetic["vectors"])

    C.run_capacity(ds, open_shared, _plan(concurrency=(3,), max_rps=50.0))
    assert len(shared) == 1


class _Stub(Connector):
    """A connector whose search outcome the test decides."""

    name = "stub"
    thread_safe = True

    def __init__(self, behave, *, exact=False):
        self.behave, self.exact, self.calls = behave, exact, 0
        self.lock = threading.Lock()

    def describe(self) -> BackendInfo:
        return BackendInfo("qdrant", Distance.L2, 32, 100, {"collection": "staging"})

    def capabilities(self) -> Capabilities:
        return Capabilities(filters=True, exact_reference=self.exact)

    def _search(self, request: SearchRequest) -> list[str]:
        with self.lock:
            self.calls += 1
            n = self.calls
        return self.behave(n, request)


def test_errors_are_counted_and_do_not_stop_the_run(synthetic) -> None:
    def behave(n, request):
        if n % 3 == 0:
            raise ConnectorError(ErrorCode.TIMEOUT, "slow")
        return ["1", "2"]

    stub = _Stub(behave)
    report = C.run_capacity(
        parse_dataset(_dataset(synthetic)), lambda: stub, _plan(concurrency=(2,), max_rps=60.0)
    )
    assert report["status"] == "complete"
    level = report["levels"][0]
    assert 0.2 < level["error_rate"] < 0.5
    assert level["errors"] == {"timeout": sum(level["errors"].values())}
    assert level["recall"]["mean"] is None  # no exact reference: unknown, not zero
    assert "no exact reference" in level["recall"]["basis"]


def test_an_error_that_will_repeat_ends_the_run_at_once(synthetic) -> None:
    stub = _Stub(lambda n, r: (_ for _ in ()).throw(ConnectorError(ErrorCode.AUTH, "denied")))
    report = C.run_capacity(
        parse_dataset(_dataset(synthetic)), lambda: stub, _plan(concurrency=(4,))
    )
    assert report["status"] == "failed"
    assert report["error"]["code"] == "auth"
    assert stub.calls <= C.WARMUP + 1  # stopped on the first rejected request, not thousands
    assert report["levels"] == []


def test_all_requests_failing_gives_null_latencies_not_nan(synthetic) -> None:
    stub = _Stub(lambda n, r: (_ for _ in ()).throw(ConnectorError(ErrorCode.QUERY, "bad")))
    report = C.run_capacity(
        parse_dataset(_dataset(synthetic)), lambda: stub, _plan(concurrency=(1,), max_rps=20.0)
    )
    level = report["levels"][0]
    assert level["error_rate"] == 1.0
    assert level["p95_ms"] is None
    assert level["achieved_qps"] == 0.0
    json.dumps(report, allow_nan=False)


def test_ctrl_c_keeps_what_was_measured(synthetic) -> None:
    def behave(n, request):
        if n > 30:
            raise KeyboardInterrupt
        return ["1"]

    report = C.run_capacity(
        parse_dataset(_dataset(synthetic)),
        lambda: _Stub(behave),
        _plan(concurrency=(1, 2), max_rps=500.0),
    )
    assert report["status"] == "cancelled"
    assert any("Cancelled" in n for n in report["notes"])


def test_filters_in_the_dataset_are_sent_as_they_are(synthetic) -> None:
    seen: list[object] = []

    def behave(n, request):
        seen.append(dict(request.filter))
        return ["1"]

    rows = [
        json.dumps({"query_id": "q0", "vector": [0.0] * 32, "filter": {"tenant": "a"}}),
    ]
    C.run_capacity(parse_dataset("\n".join(rows)), lambda: _Stub(behave), _plan(concurrency=(1,)))
    assert seen
    assert all(f == {"tenant": "a"} for f in seen)


def test_list_filters_and_contexts_are_refused() -> None:
    rows = [json.dumps({"query_id": "q", "vector": [0.0] * 32, "filter": {"t": ["a", "b"]}})]
    report = C.run_capacity(parse_dataset("\n".join(rows)), lambda: _Stub(lambda n, r: []), _plan())
    assert report["status"] == "failed"
    assert report["error"]["code"] == "unsupported"


# --- comparison --------------------------------------------------------------


def _report(**over):
    base = {
        "format": C.FORMAT,
        "format_version": 1,
        "status": "complete",
        "runner": {"system": "Linux", "machine": "x86_64", "cpu_count": 4, "python": "3.12.1"},
        "dataset": {"sha256": "abc"},
        "backend": {"backend": "qdrant"},
        "plan": {"k": 10, "params": {}, "max_rps": 50.0},
        "levels": [
            {
                "concurrency": 1,
                "achieved_qps": 40.0,
                "p50_ms": 5.0,
                "p95_ms": 9.0,
                "p99_ms": 12.0,
                "error_rate": 0.0,
                "recall": {"mean": 0.95},
            }
        ],
    }
    return {**base, **over}


def test_reports_compare_only_on_matching_environments() -> None:
    base = _report()
    faster = _report()
    faster["levels"] = [{**base["levels"][0], "p95_ms": 7.0, "achieved_qps": 48.0}]
    same = C.compare_capacity(base, faster)
    assert same["comparable"]
    assert same["levels"][0]["p95_ms"] == [9.0, 7.0]
    other_machine = _report(
        runner={"system": "Darwin", "machine": "arm64", "cpu_count": 8, "python": "3.12.1"}
    )
    out = C.compare_capacity(base, other_machine)
    assert not out["comparable"]
    assert out["levels"] == []
    assert any("system" in r for r in out["reasons"])


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"status": "cancelled"}, "did not complete"),
        ({"dataset": {"sha256": "other"}}, "datasets differ"),
        ({"backend": {"backend": "pgvector"}}, "backends differ"),
        ({"plan": {"k": 10, "params": {}, "max_rps": 100.0}}, "max_rps"),
        ({"plan": {"k": 5, "params": {}, "max_rps": 50.0}}, "in k"),
        ({"format": "vectraudit.run"}, "not a vectraudit.capacity"),
    ],
)
def test_unlike_reports_are_not_compared(change, reason) -> None:
    out = C.compare_capacity(_report(), _report(**change))
    assert not out["comparable"]
    assert any(reason in r for r in out["reasons"])


def test_no_shared_level_is_not_comparable() -> None:
    other = _report()
    other["levels"] = [{**other["levels"][0], "concurrency": 8}]
    assert not C.compare_capacity(_report(), other)["comparable"]


# --- the command -------------------------------------------------------------


def _files(tmp_path, synthetic, *, index="flat_l2", name=None):
    (tmp_path / "ds.jsonl").write_text(_dataset(synthetic))
    conn = {
        "backend": "faiss",
        "index": str(synthetic[index]),
        "vectors": str(synthetic["vectors"]),
    }
    if name:
        conn["name"] = name
    (tmp_path / "conn.json").write_text(json.dumps(conn))


def _cmd(tmp_path, *extra):
    return runner.invoke(
        app,
        [
            "capacity",
            "--connection",
            str(tmp_path / "conn.json"),
            "--dataset",
            str(tmp_path / "ds.jsonl"),
            "--out",
            str(tmp_path / "cap.json"),
            "--concurrency",
            "1,2",
            "--duration",
            "1",
            "--serial-queries",
            "5",
            *extra,
        ],
    )


def test_the_command_prints_what_it_will_send_and_refuses_without_confirmation(
    tmp_path, synthetic
) -> None:
    _files(tmp_path, synthetic)
    out = _cmd(tmp_path)
    assert out.exit_code == 1
    assert "This will send" in out.output
    assert "The most this can send" in out.output
    assert "Not started" in out.output
    assert "--confirm-target flat_l2.index" in " ".join(out.output.split())
    assert not (tmp_path / "cap.json").exists()
    wrong = _cmd(tmp_path, "--confirm-target", "elsewhere")
    assert wrong.exit_code == 1
    assert not (tmp_path / "cap.json").exists()


def test_the_command_refuses_production_without_the_override(tmp_path, synthetic) -> None:
    _files(tmp_path, synthetic, name="prod-search")
    out = _cmd(tmp_path, "--confirm-target", "flat_l2.index")
    assert out.exit_code == 1
    assert "looks like production" in " ".join(out.output.split())
    assert not (tmp_path / "cap.json").exists()
    allowed = _cmd(tmp_path, "--confirm-target", "flat_l2.index", "--allow-production")
    assert allowed.exit_code == 0, allowed.output


def test_the_command_writes_a_report_and_compares_with_a_baseline(tmp_path, synthetic) -> None:
    _files(tmp_path, synthetic)
    first = _cmd(tmp_path, "--confirm-target", "flat_l2.index", "--max-rps", "100")
    assert first.exit_code == 0, first.output
    report = json.loads((tmp_path / "cap.json").read_text())
    assert report["format"] == "vectraudit.capacity"
    assert report["safety"] == {
        "confirmed_target": "flat_l2.index",
        "target": "flat_l2.index",
        "production_override": False,
    }
    (tmp_path / "base.json").write_text(json.dumps(report))
    again = _cmd(
        tmp_path,
        "--confirm-target",
        "flat_l2.index",
        "--max-rps",
        "100",
        "--baseline",
        str(tmp_path / "base.json"),
    )
    assert again.exit_code == 0
    comparison = json.loads((tmp_path / "cap.json").read_text())["comparison"]
    assert comparison["comparable"]
    assert [lv["concurrency"] for lv in comparison["levels"]] == [1, 2]


def test_bad_command_options_are_refused_cleanly(tmp_path, synthetic) -> None:
    _files(tmp_path, synthetic)
    assert _cmd(tmp_path, "--max-rps", "5000").exit_code != 0
    assert _cmd(tmp_path, "--duration", "9999").exit_code != 0
    bad = runner.invoke(
        app,
        [
            "capacity",
            "--connection",
            str(tmp_path / "conn.json"),
            "--dataset",
            str(tmp_path / "ds.jsonl"),
            "--out",
            str(tmp_path / "cap.json"),
            "--concurrency",
            "1,50",
        ],
    )
    assert bad.exit_code == 1
    assert "1 to 32" in " ".join(bad.output.split())
