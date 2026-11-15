"""Pulling traces from OpenObserve, whose storage flattens attribute keys."""

from __future__ import annotations

import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from tests import test_openllmetry_traces as olm
from tests.test_trace_import import SECRET, dataset
from vectraudit.cli import app
from vectraudit.pipelines import openobserve_api
from vectraudit.pipelines.drift import production_questions
from vectraudit.pipelines.openobserve_api import (
    OpenObserveAPIError,
    OpenObserveSource,
    fetch,
    restore_key,
    stored_key,
    to_otlp,
)
from vectraudit.pipelines.traces import Settings, _otlp_attributes, import_traces

USER, PASSWORD = "dev@example.com", "secret"
WINDOW = {"start": "2023-11-14T22:00:00Z", "end": "2023-11-15T00:00:00Z"}
PASSAGE = f"{SECRET} To reset your password, open Settings and choose Security."


def store(span: dict) -> dict:
    """One OTLP span as OpenObserve stores it: flattened, rewritten keys, events as JSON."""
    row = {
        "_timestamp": int(span["startTimeUnixNano"]) // 1000,
        "trace_id": span["traceId"],
        "span_id": span["spanId"],
        "reference_parent_span_id": span.get("parentSpanId") or "",
        "operation_name": span["name"],
        "start_time": int(span["startTimeUnixNano"]),
        "end_time": int(span["endTimeUnixNano"]),
        "duration": 5,
        "service_name": "rag-service",
        "span_kind": "2",
        "span_status": "UNSET",
    }
    for key, value in _otlp_attributes(span["attributes"]).items():
        column = stored_key(key)
        row["attr_" + column if column in row else column] = value
    row["events"] = json.dumps(
        [
            {"name": e["name"], "_timestamp": 1, **_otlp_attributes(e["attributes"])}
            for e in span.get("events", [])
        ]
    )
    return row


def openinference_trace(trace: str, question: str, start: int = 0) -> list[dict]:
    def span(span_id, name, attributes, parent=None, offset=0):
        return olm.span(span_id, name, attributes, parent=parent, start=start + offset) | {
            "traceId": trace
        }

    documents = {}
    for i, (doc, score) in enumerate((("d1", 0.9), ("d4", 0.4))):
        base = f"retrieval.documents.{i}.document"
        documents |= {
            f"{base}.id": f"chunk-{doc}",
            f"{base}.score": score,
            f"{base}.content": PASSAGE if doc == "d1" else "Unrelated billing text.",
            f"{base}.metadata": json.dumps({"doc_id": doc}),
        }
    return [
        span(f"{trace[:8]}r", "rag", {"openinference.span.kind": "CHAIN", "input.value": question}),
        span(f"{trace[:8]}v", "search_kb", {"openinference.span.kind": "RETRIEVER", **documents},
             parent=f"{trace[:8]}r", offset=2),
        span(f"{trace[:8]}l", "llm", {"openinference.span.kind": "LLM",
             "llm.input_messages.0.message.role": "user",
             "llm.input_messages.0.message.content": f"Use: {PASSAGE}",
             "prompt.id": "support-answer", "llm.prompt_template.version": "4"},
             parent=f"{trace[:8]}r", offset=10),
    ]  # fmt: skip


class FakeOpenObserve(BaseHTTPRequestHandler):
    traces: ClassVar[list[str]] = []
    rows: ClassVar[list[dict]] = []
    requests: ClassVar[list[tuple[str, str, dict, dict | None]]] = []
    page: ClassVar[int] = 500

    def _send(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorised(self):
        expected = "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
        return self.headers.get("Authorization") == expected

    def do_GET(self):
        url = urlsplit(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        type(self).requests.append(("GET", url.path, params, None))
        if not self._authorised():
            return self._send(401, {"message": "Unauthorized"})
        if url.path != "/api/acme/default/traces/latest":
            return self._send(404, {"message": "not found"})
        start, size = int(params["from"]), min(int(params["size"]), self.page)
        hits = [{"trace_id": t} for t in self.traces[start : start + size]]
        return self._send(200, {"hits": hits, "total": len(self.traces)})

    def do_POST(self):
        url = urlsplit(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(("POST", url.path, params, body))
        if not self._authorised():
            return self._send(401, {"message": "Unauthorized"})
        query = body["query"]
        wanted = set(re.findall(r"'([0-9a-f]{32})'", query["sql"]))
        rows = [
            r for r in self.rows
            if r["trace_id"] in wanted
            and query["start_time"] <= r["_timestamp"] < query["end_time"]
        ]  # fmt: skip
        start, size = query["from"], min(query["size"], self.page)
        return self._send(200, {"hits": rows[start : start + size], "total": len(rows)})

    def log_message(self, *args):
        pass


T1, T2 = "1" * 32, "2" * 32
BASE_NS = 1_700_000_000_000_000_000  # 2023-11-14T22:13:20Z


@pytest.fixture
def openobserve():
    spans = [
        *openinference_trace(T1, "How do I reset my password?"),
        *openinference_trace(T2, "Where is the invoice?", start=50),
    ]
    FakeOpenObserve.traces = [T1, T2]
    FakeOpenObserve.rows = [store(s) for s in spans]
    FakeOpenObserve.requests = []
    FakeOpenObserve.page = 500
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenObserve)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def source(url, **kw):
    return OpenObserveSource(url=url, org="acme", user=USER, password=PASSWORD, **kw)


def test_every_key_the_importer_reads_survives_storage_and_restoration():
    keys = [
        *openobserve_api._KNOWN_KEYS,
        "retrieval.documents.12.document.metadata",
        "reranker.output_documents.3.document.score",
        "llm.input_messages.2.message.content",
        "llm.input_messages.0.message.contents.1.message_content.text",
        "gen_ai.prompt.5.role",
        "traceloop.association.properties.session_id",
    ]
    for key in keys:
        assert restore_key(stored_key(key)) == key
    assert restore_key("attr_duration") == "duration"  # a user attribute that collided
    assert restore_key("http_status_code") == "http_status_code"  # unknown keys stay as stored


def test_an_openinference_trace_round_trips_through_openobserve(openobserve, tmp_path):
    env = {"OPENOBSERVE_URL": openobserve, "OPENOBSERVE_ORG": "acme",
           "OPENOBSERVE_USER": USER, "OPENOBSERVE_PASSWORD": PASSWORD}  # fmt: skip
    out = tmp_path / "otlp.json"
    result = CliRunner().invoke(
        app,
        ["workbench", "openobserve-pull", "--since", WINDOW["start"], "--until", WINDOW["end"],
         "--traces", str(out)],
        env=env,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 2 traces and 6 spans" in result.output
    run, report = import_traces(dataset(), out, Settings(id_field="doc_id"))
    by = {q["query_id"]: q for q in run["queries"]}
    assert by["q1"]["ids"] == ["d1", "d4"]
    assert [i["id"] for i in by["q1"]["model_context"]["items"]] == ["d1"]
    assert by["q1"]["prompts"] == [{"name": "support-answer", "version": "4"}]
    assert report["source"]["shape"] == "otlp-json"


def test_an_openllmetry_trace_round_trips_with_events_and_context(openobserve, tmp_path):
    trace = [olm.workflow(), olm.pinecone_query(), olm.llm("Answer briefly.")]
    FakeOpenObserve.traces = ["a" * 32]
    FakeOpenObserve.rows = [store(s) for s in trace]
    _, rows = fetch(source(openobserve), **WINDOW)
    path = tmp_path / "otlp.json"
    path.write_text(json.dumps(to_otlp(rows)))
    run, _ = import_traces(dataset(), path, Settings(id_field="doc_id"))
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["ids"] == ["d1", "d4"]  # from the db.query.result events' repr metadata
    assert q1["prompts"] == [{"name": "support-answer", "version": "7"}]
    found, _ = production_questions(path, Settings(), ["tenant"])
    assert found == [("How do I reset my password?", {"tenant": "acme"})]


def test_traces_are_taken_by_start_and_read_whole_with_the_grace(openobserve):
    late = store(openinference_trace(T1, "x", start=0)[2] | {"spanId": "late0001"})
    late["_timestamp"] = (BASE_NS // 1000) + 3 * 3600 * 1_000_000  # after the window
    FakeOpenObserve.rows.append(late)
    _, without = fetch(source(openobserve), **WINDOW, grace_minutes=0)
    _, within = fetch(source(openobserve), **WINDOW, grace_minutes=24 * 60)
    assert "late0001" not in {r["span_id"] for r in without}
    assert "late0001" in {r["span_id"] for r in within}
    (_, path, params, _), (_, spath, sparams, body) = FakeOpenObserve.requests[:2]
    assert path == "/api/acme/default/traces/latest"
    assert params["start_time"] == str(1_699_999_200_000_000)  # microseconds
    assert (spath, sparams["type"]) == ("/api/acme/_search", "traces")
    assert body["query"]["sql"] == (
        f"SELECT * FROM \"default\" WHERE trace_id IN ('{T1}', '{T2}') "
        "ORDER BY start_time ASC, span_id ASC"
    )


def test_pages_are_followed_and_trace_ids_are_validated_before_any_query(openobserve, monkeypatch):
    monkeypatch.setattr(openobserve_api, "PAGE", 2)
    FakeOpenObserve.page = 2
    FakeOpenObserve.traces = [T1, "x' OR '1'='1", T2, T1]
    trace_ids, rows = fetch(source(openobserve), **WINDOW)
    assert trace_ids == [T1, T2]  # the malformed ID never reaches SQL; repeats kept once
    assert len(rows) == 6
    sqls = [b["query"]["sql"] for m, _, _, b in FakeOpenObserve.requests if m == "POST"]
    assert all("'1'='1" not in sql for sql in sqls)
    assert len(sqls) == 3  # six spans in pages of two


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (401, b"", "refused the credentials"),
        (404, b"", "check the organisation and stream"),
        (429, b"", "rate limit"),
        (500, b'{"message": "PLANTED-SERVER-DETAIL"}', "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b'{"results": []}', "Unexpected OpenObserve response"),
        (200, b'{"hits": [1], "total": 1}', "Unexpected record"),
    ],
)
def test_bad_responses_are_refused_plainly(status, body, match):
    def send(method, url, headers, data, timeout):
        return status, body

    with pytest.raises(OpenObserveAPIError, match=match) as info:
        fetch(source("https://o2.example", send=send), **WINDOW)
    assert "PLANTED" not in str(info.value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"url": "http://o2.example"}, "Refusing to send OpenObserve credentials over http"),
        ({"url": "https://u:p@o2.example"}, "must not contain credentials"),
        ({"url": "ftp://o2.example"}, "Invalid OpenObserve URL"),
        ({"url": "https://o2.example", "org": "acme; drop"}, "Invalid organisation"),
        ({"url": "https://o2.example", "stream": 'default" --'}, "Invalid stream"),
        ({"url": "https://o2.example", "password": None}, "credentials are missing"),
        ({"url": "https://o2.example", "token": "abc"}, "not both"),
    ],
)
def test_unsafe_or_incomplete_settings_are_refused(kwargs, match):
    settings = {"org": "acme", "user": USER, "password": PASSWORD, **kwargs}
    with pytest.raises(OpenObserveAPIError, match=match):
        OpenObserveSource(**settings)


def test_a_token_is_sent_as_given():
    seen = {}

    def send(method, url, headers, data, timeout):
        seen.update(headers)
        return 200, b'{"hits": [], "total": 0}'

    fetch(OpenObserveSource(url="https://o2.example", org="acme", token="dG9rZW4=", send=send),
          **WINDOW)  # fmt: skip
    assert seen["Authorization"] == "Basic dG9rZW4="


def test_cli_needs_url_and_org(tmp_path):
    window = ["--since", WINDOW["start"], "--until", WINDOW["end"]]
    result = CliRunner().invoke(
        app,
        ["workbench", "openobserve-pull", *window, "--traces", str(tmp_path / "o.json")],
        env={"OPENOBSERVE_URL": "", "OPENOBSERVE_ORG": ""},
    )
    assert result.exit_code == 2
    assert "set OPENOBSERVE_URL" in result.output
    assert not (tmp_path / "o.json").exists()
