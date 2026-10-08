"""Pipeline contracts, measured evidence, safety gates and dataset compatibility."""

import json

import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.policy import Policy
from vectraudit.evaluation.runfile import check_run
from vectraudit.evaluation.runner import RunOptions
from vectraudit.pipelines import (
    Hit,
    HTTPPipeline,
    PipelineRequest,
    PipelineResult,
    PythonPipeline,
    Stage,
    StageTrace,
    execute_pipeline,
)
from vectraudit.pipelines.errors import PipelineCallError


def dataset(**fields):
    return parse_dataset(json.dumps({"query_id": "q1", "text": "invoice", **fields}))


def test_branch_trace_and_release_compatibility():
    pipeline = PythonPipeline(
        [
            Stage("keyword", "keyword", lambda q, p: [Hit(id="a")]),
            Stage("vector", "vector", lambda q, p: [Hit(id="b")]),
            Stage(
                "fusion", "fusion", lambda q, p: p["keyword"] + p["vector"], ("keyword", "vector")
            ),
            Stage("final", "truncate", lambda q, p: p["fusion"][: q.k], ("fusion",)),
        ]
    )
    run = execute_pipeline(dataset(relevant={"b": 1}), pipeline, RunOptions(k=1, warmup=0))
    assert check_run(run)["status"] == "complete"
    query = run["queries"][0]
    assert query["metrics"]["recall"] == 0
    assert query["evidence"]["failures"] == [
        {
            "doc_id": "b",
            "stage": "final",
            "kind": "truncate",
            "parents": ["fusion"],
            "evidence": "measured",
            "reason": "present in parent output; absent from this stage",
        }
    ]
    assert all(s["latency_ms"] >= 0 for s in query["evidence"]["stages"])


def test_capabilities_separate_supported_from_captured_traces():
    traced = PythonPipeline([Stage("keyword", "keyword", lambda q, p: [Hit(id="a")])])
    untraced = execute_pipeline(dataset(), lambda q: PipelineResult(hits=[Hit(id="a")]))
    captured = execute_pipeline(dataset(), traced)
    assert untraced["capabilities"]["stage_traces_supported"]
    assert not untraced["capabilities"]["stage_traces"]
    assert captured["capabilities"]["stage_traces"]
    assert check_run(untraced)["status"] == "complete"


def test_no_answer_is_not_missing_annotation():
    def empty(q):
        return PipelineResult()

    missing = execute_pipeline(dataset(), empty)
    labelled = execute_pipeline(dataset(no_answer=True), empty)
    assert missing["summary"]["no_answer"]["accuracy"] is None
    assert labelled["summary"]["no_answer"] == {"n": 1, "accuracy": 1.0}
    assert labelled["queries"][0]["metrics"] is None
    assert missing["dataset"]["sha256"] != labelled["dataset"]["sha256"]


def test_context_and_forbidden_are_evaluated_and_roundtrip():
    ds = dataset(context={"tenant": "a"}, forbidden=["secret"], no_answer=True, split="heldout")
    assert parse_dataset(render_dataset(dataset_rows(ds))).sha256 == ds.sha256

    def retrieve(q):
        assert q.context == {"tenant": "a"}
        return PipelineResult(hits=[Hit(id="secret")])

    run = execute_pipeline(ds, retrieve)
    assert run["summary"]["permissions"]["violations"] == 1
    result = compare_runs(run, run, Policy(min_queries=1, metrics={}))
    assert result["verdict"] == "fail"


@pytest.mark.parametrize(
    "fields",
    [
        {"no_answer": True, "relevant": {"a": 1}},
        {"no_answer": "true"},
        {"split": "train"},
        {"forbidden": ["a"], "relevant": {"a": 1}},
    ],
)
def test_conflicting_annotations_rejected(fields):
    with pytest.raises(DatasetError):
        dataset(**fields)


def test_errors_are_redacted_and_never_pass():
    def broken(q):
        raise RuntimeError("secret=123")

    run = execute_pipeline(dataset(), broken)
    assert run["status"] == "partial"
    assert "secret=123" not in json.dumps(run)
    assert compare_runs(run, run)["verdict"] == "error"


@pytest.mark.parametrize(
    "code",
    [
        "http_auth",
        "http_not_found",
        "http_rate_limit",
        "http_error",
        "timeout",
        "connection_failed",
        "response_too_large",
        "invalid_response",
        "too_many_hits",
        "text_required",
        "context_unsupported",
        "document_id_missing",
    ],
)
def test_typed_errors_keep_causes_without_exception_text(code):
    def broken(q):
        raise PipelineCallError(code, "SECRET-token and private query")

    run = check_run(execute_pipeline(dataset(), broken))
    assert run["queries"][0]["error"]["code"] == code
    assert run["queries"][0]["error"]["hint"]
    assert run["summary"]["errors_by_code"] == {code: 1}
    assert run["status"] == "partial"
    assert "SECRET" not in json.dumps(run)
    assert compare_runs(run, run)["verdict"] == "error"


def test_unknown_typed_error_codes_are_redacted():
    def broken(q):
        raise PipelineCallError("SECRET-in-code", "SECRET-in-message")

    run = execute_pipeline(dataset(), broken)
    assert run["queries"][0]["error"]["code"] == "pipeline_error"
    assert run["summary"]["errors_by_code"] == {"pipeline_error": 1}
    assert "SECRET" not in json.dumps(run)


def test_failure_summary_counts_failed_queries_by_cause():
    ds = parse_dataset(
        "\n".join(json.dumps({"query_id": str(i), "text": "invoice"}) for i in range(4))
    )

    def mixed(q):
        if q.query_id != "3":
            code = "timeout" if q.query_id in {"0", "1"} else "http_auth"
            raise PipelineCallError(code, "private")
        return PipelineResult(hits=[Hit(id="a")])

    run = execute_pipeline(ds, mixed, RunOptions(warmup=0, repeats=2))
    assert run["summary"]["errors_by_code"] == {"http_auth": 1, "timeout": 2}
    assert run["summary"]["n_errors"] == 3
    assert run["summary"]["n_ok"] == 1


def test_invalid_traces_and_overflow():
    with pytest.raises(ValueError, match="earlier parents"):
        PipelineResult(
            stages=[StageTrace(name="a", kind="filter", parents=["b"], hits=[], latency_ms=0)]
        )
    with pytest.raises(ValueError, match="abstaining"):
        PipelineResult(abstained=True, hits=[Hit(id="a")])
    run = execute_pipeline(
        dataset(), lambda q: PipelineResult(hits=[Hit(id=str(i)) for i in range(11)])
    )
    assert run["status"] == "partial"


def test_http_rejects_credential_urls_and_insecure_auth():
    with pytest.raises(ValueError, match="credentials"):
        HTTPPipeline("https://user:password@example.com/search")
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPPipeline("http://example.com/search", token="secret")
    assert PipelineRequest(query_id="a").k == 10


def test_http_pipeline_transports_context_and_downgrades_evidence():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "hits": [{"id": "allowed"}],
                        "stages": [
                            {
                                "name": "remote",
                                "kind": "custom",
                                "hits": [{"id": "allowed"}],
                                "parents": [],
                                "latency_ms": 1,
                                "source": "measured",
                            }
                        ],
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = HTTPPipeline(f"http://127.0.0.1:{server.server_port}/search")
        result = adapter(PipelineRequest(query_id="q", text="test", context={"tenant": "acme"}))
        assert received[0]["context"] == {"tenant": "acme"}
        assert result.stages[0].source == "reported"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
