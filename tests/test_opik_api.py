"""Pulling traces, spans and feedback scores from the Opik REST API."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from tests.test_trace_formats import docs
from tests.test_trace_import import SECRET, dataset
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_file
from vectraudit.pipelines.opik_api import (
    OpikAPIError,
    OpikSource,
    fetch,
    records,
    scores_to_judgments,
)
from vectraudit.pipelines.traces import Settings, TraceImportError, import_traces

KEY, WORKSPACE = "opik-test-key", "acme"
REASON = "PLANTED-SCORE-REASON"
WINDOW = {"start": "2026-10-20T00:00:00Z", "end": "2026-10-21T00:00:00Z"}
STAGES = Settings(id_field="doc_id", stage_kinds={"search_kb": "vector"})


def trace(n: int, question: str, retrieved: list[dict], scores=()) -> tuple[dict, list[dict]]:
    """An Opik trace (question on the trace) with a retriever and an LLM span."""
    tid = f"0190-trace-{n}"
    t = f"2026-10-20T10:00:0{n}"
    record = {
        "id": tid,
        "project_id": "p1",
        "name": "rag",
        "start_time": f"{t}.000Z",
        "end_time": f"{t}.900Z",
        "input": {"question": question},
        "output": {"answer": "an answer"},
        "metadata": {"tenant": "acme"},
        "feedback_scores": list(scores),
        "span_feedback_scores": [{"name": "avg", "value": 0.5, "source": "sdk"}],
    }
    spans = [
        {
            "id": f"0190-ret-{n}",
            "trace_id": tid,
            "parent_span_id": None,
            "name": "search_kb",
            "type": "general",
            "start_time": f"{t}.010Z",
            "end_time": f"{t}.030Z",
            "input": {"query": question},
            "output": {"documents": retrieved},
            "metadata": {},
            "model": None,
        },
        {
            "id": f"0190-llm-{n}",
            "trace_id": tid,
            "parent_span_id": None,
            "name": "answer",
            "type": "llm",
            "start_time": f"{t}.100Z",
            "end_time": f"{t}.800Z",
            "input": {"messages": [
                {"role": "user", "content": "Use: " + retrieved[0]["page_content"]},
            ]},
            "output": {"content": "an answer"},
            "metadata": {},
            "feedback_scores": [],
        },
    ]  # fmt: skip
    return record, spans


def score(name, source, value=None, category=None):
    out = {"name": name, "source": source, "value": value, "reason": REASON}
    if category is not None:
        out["category_name"] = category
    return out


def project():
    t1, s1 = trace(
        1,
        "How do I reset my password?",
        docs("d1", "d4"),
        [score("review", "ui", 1, "Correct"), score("correctness", "online_scoring", 0.4)],
    )
    t2, s2 = trace(
        2,
        "Where is the invoice?",
        docs("d9"),
        [score("tone", "ui", 1, "friendly")],
    )
    s2[1]["feedback_scores"] = [score("grade", "sdk", 0, "incorrect")]  # on the LLM span
    stray = {**s1[0], "id": "0190-other", "trace_id": "0190-trace-outside"}
    return [t1, t2], [*s1, *s2, stray]


class FakeOpik(BaseHTTPRequestHandler):
    """Traces and spans, page-numbered, behind Opik Cloud's headers."""

    traces: ClassVar[list[dict]] = []
    spans: ClassVar[list[dict]] = []
    requests: ClassVar[list[tuple[str, dict]]] = []
    page_size: ClassVar[int] = 100

    def do_GET(self):
        url = urlsplit(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        type(self).requests.append((url.path, params))
        if (self.headers.get("Authorization"), self.headers.get("Comet-Workspace")) != (
            KEY,
            WORKSPACE,
        ):
            return self._send(401, {"errors": ["unauthorized"]})
        rows = {"/api/v1/private/traces": self.traces, "/api/v1/private/spans": self.spans}
        if url.path not in rows or params.get("project_name") != "support-bot":
            return self._send(404, {"errors": ["not found"]})
        page, size = int(params["page"]), min(int(params["size"]), self.page_size)
        chunk = rows[url.path][(page - 1) * size : page * size]
        return self._send(
            200, {"page": page, "size": len(chunk), "total": len(rows[url.path]), "content": chunk}
        )

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
def opik():
    FakeOpik.traces, FakeOpik.spans = project()
    FakeOpik.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpik)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api"
    server.shutdown()


def source(url: str, **kw) -> OpikSource:
    return OpikSource(url=url, api_key=KEY, workspace=WORKSPACE, **kw)


def test_traces_and_their_spans_are_pulled_with_a_grace_for_late_spans(opik):
    found, spans = fetch(source(opik), "support-bot", **WINDOW, grace_minutes=30)
    assert [t["id"] for t in found] == ["0190-trace-1", "0190-trace-2"]
    assert "0190-other" not in {s["id"] for s in spans}  # a span of a trace not taken
    assert len(spans) == 4
    (_, traces_q), (_, spans_q) = FakeOpik.requests
    assert (traces_q["from_time"], traces_q["to_time"]) == (WINDOW["start"], WINDOW["end"])
    assert spans_q["to_time"] == "2026-10-21T00:30:00Z"


def test_pages_are_followed_until_the_total(opik):
    FakeOpik.page_size = 2
    try:
        FakeOpik.traces = FakeOpik.traces * 3  # six rows, three of them repeats
        found, _ = fetch(source(opik), "support-bot", **WINDOW)
    finally:
        FakeOpik.page_size = 100
    assert len(found) == 2
    pages = [q["page"] for p, q in FakeOpik.requests if p.endswith("/traces")]
    assert pages == ["1", "2", "3"]  # smaller pages than asked for are followed to the total


def test_without_a_total_a_short_page_ends_the_listing():
    calls = []

    def send(url, headers, timeout):
        calls.append(url)
        rows = [{"id": f"t{len(calls)}-{i}"} for i in range(100 if len(calls) == 1 else 3)]
        return 200, json.dumps({"content": rows}).encode()

    def spans_too(url, headers, timeout):  # traces listing, then an empty span listing
        if "/spans" in url:
            return 200, b'{"content": []}'
        return send(url, headers, timeout)

    found, _ = fetch(source("https://opik.example/api", send=spans_too), "p", **WINDOW)
    assert len(found) == 103
    assert len(calls) == 2


def test_pulled_records_import_and_judgments_attach(opik, tmp_path):
    env = {"OPIK_URL_OVERRIDE": opik, "OPIK_API_KEY": KEY, "OPIK_WORKSPACE": WORKSPACE}
    out, judged = tmp_path / "spans.jsonl", tmp_path / "judgments.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "opik-pull", "--project", "support-bot",
            "--since", WINDOW["start"], "--until", WINDOW["end"],
            "--traces", str(out), "--judgments", str(judged),
        ],
        env=env,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 2 traces and 4 spans" in result.output
    assert "Wrote 3 feedback scores as judgments" in result.output
    assert "1 category not a verdict" in result.output
    assert REASON not in judged.read_text()

    run, report = import_traces(dataset(), out, STAGES)
    assert report["source"]["shape"] == "opik"
    by = {q["query_id"]: q for q in run["queries"]}
    assert by["q1"]["ids"] == ["d1", "d4"]
    assert [i["id"] for i in by["q1"]["model_context"]["items"]] == ["d1"]
    assert by["q2"]["ids"] == ["d9"]

    answers = attach_file(run, judged, {"correctness": 0.7})
    got = {q["query_id"]: q["answer"] for q in answers["queries"] if "answer" in q}
    assert (got["q1"]["verdict"], got["q1"]["decided_by"]) == ("correct", "human")
    assert got["q2"]["verdict"] == "incorrect"  # a span's score counts for its trace


def test_a_retriever_must_be_named_because_opik_has_no_retriever_type(opik, tmp_path):
    found, spans = fetch(source(opik), "support-bot", **WINDOW)
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records(found, spans)))
    run, _ = import_traces(dataset(), path, Settings(id_field="doc_id"))
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["error"]["code"] == "trace_no_retrieval"  # never guessed from a name


def test_each_trace_becomes_its_root_span_and_top_level_spans_hang_off_it():
    found, spans = project()
    out = records(found, spans[:4])
    roots = [r for r in out if r["parent_span_id"] is None]
    assert [r["id"] for r in roots] == ["0190-trace-1", "0190-trace-2"]
    assert roots[0]["input"] == {"question": "How do I reset my password?"}
    assert {r["parent_span_id"] for r in out if r["id"].startswith("0190-ret")} == {
        "0190-trace-1",
        "0190-trace-2",
    }
    assert all("feedback_scores" not in r for r in out)  # scores go to judgments only
    assert SECRET in json.dumps(out)  # passages stay in the local file, as in any import


def test_scores_map_without_guessing_and_reasons_are_dropped():
    found, spans = project()
    rows, skipped = scores_to_judgments(found, spans)
    assert rows == [
        {"trace_id": "0190-trace-1", "evaluator": "review", "kind": "human", "verdict": "correct"},
        {"trace_id": "0190-trace-1", "evaluator": "correctness", "kind": "model", "score": 0.4},
        {"trace_id": "0190-trace-2", "evaluator": "grade", "kind": "model", "verdict": "incorrect"},
    ]  # fmt: skip
    assert skipped == {"category_not_a_verdict": 1}  # span_feedback_scores averages unused
    odd = [{"id": "t", "feedback_scores": [
        {"name": "x", "value": True, "source": "ui"},
        {"name": "", "value": 1, "source": "ui"},
        {"name": "x", "value": float("inf"), "source": "ui"},
        "not a score",
    ]}]  # fmt: skip
    assert scores_to_judgments(odd, []) == ([], {"invalid_value": 3, "unnamed": 1})


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (401, b"", "check OPIK_API_KEY and OPIK_WORKSPACE"),
        (404, b"", "check OPIK_URL_OVERRIDE"),
        (429, b"", "rate limit"),
        (500, b'{"errors": ["PLANTED-SERVER-DETAIL"]}', "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b'{"data": []}', "Unexpected Opik response"),
        (200, b'{"content": [1]}', "Unexpected record"),
    ],
)
def test_bad_responses_are_refused_plainly(status, body, match):
    def send(url, headers, timeout):
        return status, body

    with pytest.raises(OpikAPIError, match=match) as info:
        fetch(source("https://opik.example/api", send=send), "p", **WINDOW)
    assert "PLANTED" not in str(info.value)


def test_an_oversized_window_stops_the_pull():
    page = json.dumps({"content": [{"id": f"t{i}"} for i in range(5)], "total": 5}).encode()
    with pytest.raises(OpikAPIError, match="More than 3 records"):
        fetch(
            source("https://opik.example/api", send=lambda u, h, t: (200, page)),
            "p",
            **WINDOW,
            max_records=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"url": "http://opik.example/api"}, "Refusing to send the Opik API key over http"),
        ({"url": "https://u:p@opik.example/api"}, "must not contain credentials"),
        ({"url": "ftp://opik.example"}, "Invalid Opik URL"),
        ({"url": "https://www.comet.com/opik/api", "api_key": None}, "needs an API key"),
    ],
)
def test_unsafe_or_incomplete_settings_are_refused(kwargs, match):
    with pytest.raises(OpikAPIError, match=match):
        OpikSource(**{"api_key": KEY, **kwargs})


def test_a_keyless_self_hosted_opik_is_allowed_over_http():
    seen = {}

    def send(url, headers, timeout):
        seen.update(headers)
        return 200, b'{"content": [], "total": 0}'

    found, spans = fetch(OpikSource(url="http://opik.internal:5173/api", send=send), "p", **WINDOW)
    assert (found, spans) == ([], [])
    assert "Authorization" not in seen
    assert "Comet-Workspace" not in seen


def test_window_and_grace_are_validated():
    with pytest.raises(ValueError, match="needs a time zone"):
        fetch(source("https://opik.example/api"), "p", start="2026-10-20T00:00", end=WINDOW["end"])
    with pytest.raises(OpikAPIError, match="grace-minutes"):
        fetch(source("https://opik.example/api"), "p", **WINDOW, grace_minutes=-1)


def test_cli_needs_an_output_and_a_project(tmp_path):
    runner = CliRunner()
    window = ["--since", WINDOW["start"], "--until", WINDOW["end"]]
    result = runner.invoke(app, ["workbench", "opik-pull", *window, "--project", "p"])
    assert result.exit_code == 2
    assert "give --traces, --judgments or both" in result.output
    result = runner.invoke(
        app,
        ["workbench", "opik-pull", *window, "--traces", str(tmp_path / "s.jsonl")],
        env={"OPIK_PROJECT_NAME": ""},
    )
    assert result.exit_code == 2
    assert "give --project or set OPIK_PROJECT_NAME" in result.output
    assert not (tmp_path / "s.jsonl").exists()


def test_an_opik_file_cannot_be_mixed_with_another_format(tmp_path):
    found, spans = project()
    langfuse_like = {"traceId": "x", "type": "SPAN", "startTime": "2026-10-20T10:00:00Z"}
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [*records(found, spans), langfuse_like]))
    with pytest.raises(TraceImportError, match="mixes export formats"):
        import_traces(dataset(), path, STAGES)


def test_production_intake_reads_an_opik_pull_when_the_retriever_is_named(opik, tmp_path):
    found, spans = fetch(source(opik), "support-bot", **WINDOW)
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records(found, spans)))
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "text": q.text, "relevant": q.relevant})
            for q in dataset().queries
        )
    )
    args = ["workbench", "intake", str(path), str(data), "--id-field", "doc_id"]
    named = CliRunner().invoke(
        app, [*args, "--state", str(tmp_path / "a"), "--stage-kind", "search_kb=vector",
              "--json", str(tmp_path / "a.json")],
    )  # fmt: skip
    assert named.exit_code == 0, named.output
    report = json.loads((tmp_path / "a.json").read_text())
    assert (report["traces_taken"], report["matched_benchmark"]) == (2, 2)
    # q2's ranking (d9) misses its labelled d2: a likely recurrence, seen only via the retriever
    assert [r["query_id"] for r in report["likely_recurrences"]] == ["q2"]
    bad = CliRunner().invoke(app, [*args, "--state", str(tmp_path / "b"), "--stage-kind", "x"])
    assert bad.exit_code == 2
    assert "--stage-kind expects SPAN=KIND" in bad.output
