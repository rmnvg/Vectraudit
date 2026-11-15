"""Pulling observations and scores from the Langfuse public API."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from tests.test_trace_formats import docs, langfuse_page
from tests.test_trace_import import dataset
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_file
from vectraudit.pipelines.langfuse_api import (
    LangfuseAPIError,
    LangfuseSource,
    fetch_observations,
    fetch_scores,
    scores_to_judgments,
)
from vectraudit.pipelines.traces import Settings, import_traces

KEYS = ("pk-lf-test", "sk-lf-test")
COMMENT = "PLANTED-REVIEWER-COMMENT"


def observations() -> list[dict]:
    first = langfuse_page("How do I reset my password?", docs("d1", "d4"), trace="tf1")["data"]
    second = langfuse_page("Where is the invoice?", docs("d9"), trace="tf2")["data"]
    return [*first, *second]


def scores() -> list[dict]:
    def score(i, name, data_type, value, source="EVAL", subject=None):
        return {
            "id": f"s{i}",
            "name": name,
            "dataType": data_type,
            "value": value,
            "source": source,
            "timestamp": "2026-10-20T10:05:00Z",
            "comment": COMMENT,  # a real API omits it unless asked; never copied either way
            "subject": subject or {"kind": "trace", "id": "tf1"},
        }

    return [
        score(1, "review", "BOOLEAN", True, source="ANNOTATION"),
        score(2, "correctness", "NUMERIC", 0.42),
        score(3, "grade", "CATEGORICAL", "Incorrect", subject={"kind": "trace", "id": "tf2"}),
        score(4, "tone", "CATEGORICAL", "friendly"),
        score(5, "note", "TEXT", "free text"),
        score(6, "csat", "NUMERIC", 4, subject={"kind": "session", "id": "s-1"}),
        score(7, "obs", "BOOLEAN", False,
              subject={"kind": "observation", "id": "x", "traceId": "tf2"}),
    ]  # fmt: skip


class FakeLangfuse(BaseHTTPRequestHandler):
    """The two read endpoints, paged by cursor, checking authentication."""

    pages: ClassVar[dict[str, list[list[dict]]]] = {}
    requests: ClassVar[list[tuple[str, dict]]] = []

    @staticmethod
    def expected() -> str:
        return "Basic " + base64.b64encode(":".join(KEYS).encode()).decode()

    def do_GET(self):
        url = urlsplit(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        type(self).requests.append((url.path, params))
        if self.headers.get("Authorization") != self.expected():
            return self._send(401, {"message": "Invalid credentials"})
        chunks = self.pages.get(url.path)
        if chunks is None:
            return self._send(404, {"message": "not found"})
        index = int(params.get("cursor", "0"))
        meta = {"cursor": str(index + 1)} if index + 1 < len(chunks) else {}
        return self._send(200, {"data": chunks[index], "meta": meta})

    def _send(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def langfuse():
    obs = observations()
    FakeLangfuse.pages = {
        "/api/public/v2/observations": [obs[:2], obs[2:5], obs[4:]],  # obs[4] repeats
        "/api/public/v3/scores": [scores()[:3], scores()[3:]],
    }
    FakeLangfuse.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLangfuse)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def source(host: str, **kw) -> LangfuseSource:
    return LangfuseSource(host=host, public_key=KEYS[0], secret_key=KEYS[1], **kw)


def test_observations_are_paged_deduplicated_and_requested_with_their_content(langfuse):
    got = fetch_observations(
        source(langfuse), start="2026-10-20T00:00:00Z", end="2026-10-21T00:00:00+00:00"
    )
    assert [o["id"] for o in got] == [
        "tf1-root", "tf1-ret", "tf1-gen", "tf2-root", "tf2-ret", "tf2-gen",
    ]  # fmt: skip
    paths = [(p, q.get("cursor")) for p, q in FakeLangfuse.requests]
    assert paths == [("/api/public/v2/observations", c) for c in (None, "1", "2")]
    params = FakeLangfuse.requests[0][1]
    assert params["fields"] == "core,basic,io,metadata,prompt"
    assert (params["fromStartTime"], params["toStartTime"]) == (
        "2026-10-20T00:00:00Z",
        "2026-10-21T00:00:00Z",
    )


def test_pulled_observations_and_scores_flow_into_import_and_attach_answers(langfuse, tmp_path):
    env = {
        "LANGFUSE_HOST": langfuse,
        "LANGFUSE_PUBLIC_KEY": KEYS[0],
        "LANGFUSE_SECRET_KEY": KEYS[1],
    }
    traces, judged = tmp_path / "obs.jsonl", tmp_path / "judgments.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "langfuse-pull", "--since", "2026-10-20T00:00:00Z",
            "--until", "2026-10-21T00:00:00Z", "--traces", str(traces), "--judgments", str(judged),
        ],
        env=env,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 6 observations from 2 traces" in result.output
    assert "Wrote 4 of 7 scores as judgments" in result.output
    assert "1 category not a verdict, 1 free text, 1 session subject" in result.output
    assert "Numeric scores need --threshold" in result.output
    assert COMMENT not in judged.read_text()

    run, report = import_traces(dataset(), traces, Settings(id_field="doc_id"))
    assert report["source"]["shape"] == "langfuse"
    by = {q["query_id"]: q for q in run["queries"]}
    assert by["q1"]["ids"] == ["d1", "d4"]
    assert by["q2"]["ids"] == ["d9"]

    out = attach_file(run, judged, {"correctness": 0.7})
    answers = {q["query_id"]: q["answer"] for q in out["queries"] if "answer" in q}
    assert answers["q1"]["verdict"] == "correct"  # human annotation decides over the model
    assert answers["q1"]["decided_by"] == "human"
    assert answers["q2"]["verdict"] == "incorrect"  # categorical and observation scores agree
    assert out["answers"]["unmatched_judgments"] == 0


def test_scores_map_to_judgments_without_guessing():
    rows, skipped = scores_to_judgments(scores())
    assert rows == [
        {"trace_id": "tf1", "evaluator": "review", "kind": "human", "verdict": "correct"},
        {"trace_id": "tf1", "evaluator": "correctness", "kind": "model", "score": 0.42},
        {"trace_id": "tf2", "evaluator": "grade", "kind": "model", "verdict": "incorrect"},
        {"trace_id": "tf2", "evaluator": "obs", "kind": "model", "verdict": "incorrect"},
    ]
    assert skipped == {"category_not_a_verdict": 1, "free_text": 1, "session_subject": 1}
    as_human, _ = scores_to_judgments(scores(), human_sources=("ANNOTATION", "EVAL"))
    assert {r["kind"] for r in as_human} == {"human"}
    trace = {"kind": "trace", "id": "t"}
    bad, why = scores_to_judgments(
        [
            {"name": "x", "dataType": "NUMERIC", "value": "NaN", "subject": trace},
            {"name": "x", "dataType": "BOOLEAN", "value": 1, "subject": trace},
            {"name": "", "dataType": "BOOLEAN", "value": True, "subject": trace},
            {"name": "x", "dataType": "BOOLEAN", "value": True},
        ]
    )
    assert bad == []
    assert why == {"invalid_value": 2, "unnamed": 1, "unknown_subject": 1}


def test_scores_are_paged_with_their_subject_and_name_filter(langfuse):
    got = fetch_scores(
        source(langfuse), start="2026-10-20T00:00:00Z", end="2026-10-21T00:00:00Z", names=["a", "b"]
    )
    assert len(got) == 7
    params = FakeLangfuse.requests[0][1]
    assert (params["fields"], params["name"], params["limit"]) == ("subject", "a,b", "100")


def test_wrong_keys_name_the_problem_without_echoing_the_response(langfuse):
    wrong = LangfuseSource(host=langfuse, public_key="pk", secret_key="nope")
    with pytest.raises(LangfuseAPIError, match="refused the keys"):
        fetch_scores(wrong, start="2026-10-20T00:00:00Z", end="2026-10-21T00:00:00Z")


def stub(*responses):
    calls = []

    def send(url, headers, timeout):
        calls.append(url)
        return responses[min(len(calls), len(responses)) - 1]

    return send, calls


WINDOW = {"start": "2026-10-20T00:00:00Z", "end": "2026-10-21T00:00:00Z"}


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (429, b"", "rate limit"),
        (500, b'{"message": "PLANTED-SERVER-DETAIL"}', "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b'{"items": []}', "Unexpected Langfuse response"),
        (200, b'{"data": [1]}', "Unexpected record"),
    ],
)
def test_bad_responses_are_refused_plainly(status, body, match):
    send, _ = stub((status, body))
    with pytest.raises(LangfuseAPIError, match=match) as info:
        fetch_observations(source("https://lf.example", send=send), **WINDOW)
    assert "PLANTED" not in str(info.value)


def test_a_repeated_cursor_and_an_oversized_window_stop_the_pull():
    page = json.dumps({"data": [], "meta": {"cursor": "same"}}).encode()
    send, calls = stub((200, page))
    with pytest.raises(LangfuseAPIError, match="repeated a pagination cursor"):
        fetch_observations(source("https://lf.example", send=send), **WINDOW)
    assert len(calls) == 2
    many = json.dumps({"data": observations(), "meta": {}}).encode()
    send, _ = stub((200, many))
    with pytest.raises(LangfuseAPIError, match="More than 5 records"):
        fetch_observations(source("https://lf.example", send=send), **WINDOW, max_records=5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"host": "http://lf.example"}, "Refusing to send Langfuse keys over http"),
        ({"host": "https://user:pw@lf.example"}, "must not contain credentials"),
        ({"host": "ftp://lf.example"}, "Invalid Langfuse host"),
        ({"host": "https://lf.example", "secret_key": ""}, "keys are missing"),
    ],
)
def test_unsafe_or_incomplete_settings_are_refused(kwargs, match):
    settings = {"public_key": KEYS[0], "secret_key": KEYS[1], **kwargs}
    with pytest.raises(LangfuseAPIError, match=match):
        LangfuseSource(**settings)


def test_times_need_a_zone():
    with pytest.raises(LangfuseAPIError, match="needs a time zone"):
        fetch_scores(source("https://lf.example"), start="2026-10-20T00:00:00", end=WINDOW["end"])
    with pytest.raises(LangfuseAPIError, match="ISO 8601"):
        fetch_scores(source("https://lf.example"), start="yesterday", end=WINDOW["end"])


def test_cli_needs_an_output_and_reports_errors(tmp_path):
    runner = CliRunner()
    window = ["--since", WINDOW["start"], "--until", WINDOW["end"]]
    result = runner.invoke(app, ["workbench", "langfuse-pull", *window])
    assert result.exit_code == 2
    assert "give --traces, --judgments or both" in result.output
    result = runner.invoke(
        app,
        ["workbench", "langfuse-pull", *window, "--traces", str(tmp_path / "o.jsonl")],
        env={"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
    )
    assert result.exit_code == 2
    assert "keys are missing" in result.output
    assert not (tmp_path / "o.jsonl").exists()


@pytest.mark.parametrize(
    ("part", "code"),
    [(slice(0, 2), None), (slice(2, 3), "trace_missing"), (slice(0, 1), "trace_no_retrieval")],
)
def test_a_trace_cut_at_the_window_edge_is_a_gap_never_a_result(tmp_path, part, code):
    cut = observations()[part]  # tf1 only: root, retriever, generation
    path = tmp_path / "cut.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in cut))
    run, _ = import_traces(dataset(), path, Settings(id_field="doc_id"))
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert (q1.get("error") or {}).get("code") == code
    if code is None:  # retrieval was recorded; what reached the model is not claimed
        assert q1["ids"] == ["d1", "d4"]
        assert q1.get("model_context") is None
