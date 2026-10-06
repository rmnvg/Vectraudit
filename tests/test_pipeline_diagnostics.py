"""Live connection checks run real adapters in a fresh process and report bounded evidence."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from vectraudit.integrations.retrievers import _require_text
from vectraudit.pipelines.diagnostic_report import failure_report, inspect_result
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.models import Hit, PipelineRequest, PipelineResult, StageTrace
from vectraudit.pipelines.web import create_workbench

CORPUS = "\n".join(
    json.dumps(d)
    for d in [
        {"doc_id": "a", "text": "invoice", "metadata": {"tenant": "acme"}},
        {"doc_id": "private", "text": "invoice", "metadata": {"tenant": "other"}},
    ]
)

ADAPTER = """
import os, sys, time
from vectraudit.pipelines.models import Hit, PipelineResult

def unmapped(request):
    return PipelineResult(hits=[Hit(id="a"), Hit(id="ghost")])

def leaky(request):
    print("SECRET-stdout")
    print("SECRET-stderr", file=sys.stderr)
    raise RuntimeError("SECRET-message")

def slow(request):
    time.sleep(60)

def wrong_type(request):
    return {"hits": []}

def too_many(request):
    return PipelineResult(hits=[Hit(id=str(i)) for i in range(request.k + 1)])
"""


def finding(report, code):
    return next(f for f in report["findings"] if f["code"] == code)


def codes(report):
    return [f["code"] for f in report["findings"]]


def workbench(tmp_path, candidates):
    (tmp_path / "probe_adapter.py").write_text(ADAPTER)
    manifest = tmp_path / "pipelines.json"
    manifest.write_text(json.dumps({"dataset": "unused.jsonl", "candidates": candidates}))
    client = TestClient(create_workbench(tmp_path, manifest))
    client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
    return client


def python(name):
    return {"name": name, "adapter": {"kind": "python", "target": f"probe_adapter:{name}"}}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        yield client


def diagnose(client, **body):
    response = client.post("/api/guided/diagnostics", json={"text": "invoice", **body})
    assert response.status_code == 200, response.text
    return response.json()["report"]


def test_builtin_reports_lookup_stages_and_forbidden_ids(client, tmp_path):
    report = diagnose(client, corpus=CORPUS, k=2, forbidden=["private"])
    assert report["executed"]
    assert report["latency_ms"] >= 0
    assert sorted(report["returned_ids"]) == ["a", "private"]
    assert finding(report, "response")["status"] == "ok"
    assert finding(report, "document_lookup")["status"] == "ok"
    assert finding(report, "stage_evidence")["status"] == "ok"
    assert report["stage_count"] > 0
    assert finding(report, "forbidden_documents")["status"] == "error"
    assert finding(report, "filter_forwarding")["status"] == "skipped"
    assert finding(report, "context_forwarding")["status"] == "skipped"
    # A connection check writes no run artifacts into the workspace.
    assert not list(tmp_path.glob("*.run.json"))


def test_context_is_supplied_but_never_reported_as_verified(client):
    report = diagnose(
        client,
        corpus=CORPUS,
        k=2,
        context={"tenant": "acme"},
        filters={"tenant": "acme"},
        forbidden=["private"],
    )
    assert report["returned_ids"] == ["a"]
    assert finding(report, "context_forwarding")["status"] == "unknown"
    assert finding(report, "filter_forwarding")["status"] == "unknown"
    assert finding(report, "filter_results")["status"] == "ok"
    assert finding(report, "forbidden_documents")["status"] == "ok"


def test_no_answer_assertion(client):
    assert finding(diagnose(client, corpus=CORPUS, no_answer=True), "no_answer")["status"] == (
        "error"
    )
    report = diagnose(client, corpus=CORPUS, text="volcano", no_answer=True)
    assert finding(report, "no_answer")["status"] == "ok"
    assert finding(report, "document_lookup")["status"] == "skipped"


def test_input_boundaries(client):
    url = "/api/guided/diagnostics"
    assert client.post(url, json={"text": "invoice"}).status_code == 422
    assert client.post(url, json={"text": "x", "pipeline": "configured-9"}).status_code == 422
    duplicate = CORPUS + "\n" + CORPUS.splitlines()[0]
    assert client.post(url, json={"text": "x", "corpus": duplicate}).status_code == 422
    del client.headers["X-Workbench-Token"]
    assert client.post(url, json={"text": "x", "corpus": CORPUS}).status_code == 403


def test_registered_python_adapters_report_specific_failures(tmp_path):
    names = ["unmapped", "leaky", "wrong_type", "too_many"]
    client = workbench(tmp_path, [python(n) for n in names])
    with client:
        report = diagnose(client, pipeline="configured-0", corpus=CORPUS)
        assert report["missing_ids"] == ["ghost"]
        assert finding(report, "document_lookup")["status"] == "warning"
        assert finding(report, "stage_evidence")["status"] == "warning"
        assert finding(report, "corpus_version")["status"] == "unknown"

        response = client.post(
            "/api/guided/diagnostics", json={"text": "invoice", "pipeline": "configured-1"}
        )
        assert "SECRET" not in response.text
        assert codes(response.json()["report"]) == ["adapter_exception"]
        assert response.json()["report"]["executed"]

        assert codes(diagnose(client, pipeline="configured-2")) == ["invalid_response"]
        assert codes(diagnose(client, pipeline="configured-3", k=2)) == ["too_many_hits"]


def test_unimportable_adapter_is_a_dependency_failure(tmp_path):
    bad = {"name": "Bad", "adapter": {"kind": "python", "target": "no_such_module:run"}}
    with workbench(tmp_path, [bad]) as client:
        report = diagnose(client, pipeline="configured-0")
    assert codes(report) == ["dependency_missing"]
    assert not report["executed"]


def test_timeout_kills_a_hung_adapter(tmp_path):
    with workbench(tmp_path, [python("slow")]) as client:
        started = time.perf_counter()
        report = diagnose(client, pipeline="configured-0", timeout_s=3)
    assert codes(report) == ["timeout"]
    assert time.perf_counter() - started < 20


def test_cancel_and_single_probe_capacity(tmp_path):
    with workbench(tmp_path, [python("slow")]) as client:
        results = {}

        def call():
            results["report"] = diagnose(client, pipeline="configured-0", timeout_s=60)

        thread = threading.Thread(target=call)
        thread.start()
        deadline = time.monotonic() + 10
        while client.post("/api/guided/diagnostics/cancel").json() == {"cancel_requested": False}:
            # Until the probe starts, a second request may win the slot instead.
            assert time.monotonic() < deadline
            time.sleep(0.05)
        thread.join(10)
        assert codes(results["report"]) == ["cancelled"]


def test_capacity_rejects_concurrent_probe(tmp_path):
    with workbench(tmp_path, [python("slow")]) as client:
        runner_thread = threading.Thread(
            target=lambda: client.post(
                "/api/guided/diagnostics",
                json={"text": "x", "pipeline": "configured-0", "timeout_s": 5},
            )
        )
        runner_thread.start()
        time.sleep(1)
        second = client.post(
            "/api/guided/diagnostics", json={"text": "x", "pipeline": "configured-0"}
        )
        assert second.status_code == 429
        client.post("/api/guided/diagnostics/cancel")
        runner_thread.join(10)


class _Handler(BaseHTTPRequestHandler):
    status = 200
    body = b"{}"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def http(url, **adapter):
    return {"name": "Remote", "adapter": {"kind": "http", "target": url, **adapter}}


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (401, b"{}", "http_auth"),
        (404, b"{}", "http_not_found"),
        (429, b"{}", "http_rate_limit"),
        (500, b"SECRET", "http_error"),
        (200, b"not json SECRET", "invalid_response"),
        (200, b'{"hits": [{"id": "a"}, {"id": "a"}]}', "invalid_response"),
    ],
)
def test_http_failures_are_categorised(tmp_path, endpoint, monkeypatch, status, body, code):
    monkeypatch.setattr(_Handler, "status", status)
    monkeypatch.setattr(_Handler, "body", body)
    url = f"http://127.0.0.1:{endpoint.server_port}/retrieve"
    with workbench(tmp_path, [http(url)]) as client:
        response = client.post(
            "/api/guided/diagnostics",
            json={"text": "invoice", "pipeline": "configured-0", "confirm_remote": True},
        )
    assert codes(response.json()["report"]) == [code]
    assert "SECRET" not in response.text


def test_http_success_requires_consent(tmp_path, endpoint, monkeypatch):
    monkeypatch.setattr(_Handler, "body", b'{"hits": [{"id": "a", "score": 1.0}]}')
    url = f"http://127.0.0.1:{endpoint.server_port}/retrieve"
    with workbench(tmp_path, [http(url)]) as client:
        body = {"text": "invoice", "pipeline": "configured-0"}
        assert client.post("/api/guided/diagnostics", json=body).status_code == 422
        report = diagnose(client, pipeline="configured-0", confirm_remote=True, corpus=CORPUS)
    assert report["returned_ids"] == ["a"]
    assert finding(report, "document_lookup")["status"] == "ok"


def test_http_missing_credential_and_unreachable_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("VECTRAUDIT_TEST_TOKEN", raising=False)
    candidates = [
        http("http://127.0.0.1:9/retrieve", token_env="VECTRAUDIT_TEST_TOKEN"),
        http("http://127.0.0.1:9/retrieve", timeout=2),
    ]
    with workbench(tmp_path, candidates) as client:
        missing = diagnose(client, pipeline="configured-0", confirm_remote=True)
        unreachable = diagnose(client, pipeline="configured-1", confirm_remote=True)
    assert codes(missing) == ["missing_env"]
    assert codes(unreachable) == ["connection_failed"]


def test_failure_report_only_renders_known_categories():
    report = failure_report("arbitrary text SECRET")
    assert codes(report.model_dump()) == ["adapter_exception"]
    assert "SECRET" not in report.model_dump_json()


def test_inspect_result_flags_filter_conflicts_in_local_metadata():
    request = PipelineRequest(query_id="q", text="x", k=2, filters={"tenant": "acme"})
    hits = [Hit(id="a"), Hit(id="private")]
    result = PipelineResult(
        hits=hits, stages=[StageTrace(name="s", kind="custom", hits=hits, latency_ms=1)]
    )
    docs = [json.loads(line) for line in CORPUS.splitlines()]
    report = inspect_result(
        request,
        result,
        docs,
        latency_ms=1,
        corpus_version_known=False,
        forbidden=[],
        no_answer=False,
    ).model_dump()
    assert finding(report, "filter_results")["status"] == "error"
    assert report["reported_stage_count"] == 1


def test_framework_errors_carry_codes_and_stay_value_errors():
    with pytest.raises(ValueError, match="no text") as raised:
        _require_text(PipelineRequest(query_id="q"))
    assert isinstance(raised.value, PipelineCallError)
    assert raised.value.code == "text_required"
