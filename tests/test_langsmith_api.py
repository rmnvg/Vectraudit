"""Pulling runs and feedback from the LangSmith API."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from tests.test_trace_formats import docs, langsmith_trace
from tests.test_trace_import import dataset
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_file
from vectraudit.pipelines.langsmith_api import (
    LangSmithAPIError,
    LangSmithSource,
    feedback_to_judgments,
    fetch_feedback,
    fetch_runs,
    project_id,
)
from vectraudit.pipelines.traces import Settings, import_traces

KEY = "lsv2_pt_test"
PROJECT = {"id": "11111111-1111-1111-1111-111111111111", "name": "support-bot"}
COMMENT = "PLANTED-REVIEWER-COMMENT"
WINDOW = {"start": "2026-10-20T00:00:00Z", "end": "2026-10-21T00:00:00Z"}


def runs() -> list[dict]:
    """Two traces; the shared fixture reuses child IDs across traces, real run IDs do not."""
    out = []
    for n, question, retrieved, start in (
        (1, "How do I reset my password?", docs("d1", "d4"), 0),
        (2, "Where is the invoice?", docs("d9"), 5),
    ):
        for run in langsmith_trace(n, question, retrieved, start=start):
            if run["parent_run_id"]:
                run["id"] = f"{run['id']}-{n}"
            out.append(run)
    return out


def feedback() -> list[dict]:
    r = runs()
    root1, root2, retriever2 = r[0]["id"], r[3]["id"], r[4]["id"]

    def fb(i, run_id, key, source, **values):
        return {
            "id": f"f{i}",
            "run_id": run_id,
            "key": key,
            "comment": COMMENT,
            "correction": COMMENT,
            "feedback_source": {"type": source},
            **values,
        }

    return [
        fb(1, root1, "review", "app", value="Correct", score=None),
        fb(2, root1, "correctness", "model", score=0.4),
        fb(3, retriever2, "grade", "api", score=False),
        fb(4, root2, "tone", "app", value="friendly"),
        fb(5, root2, "note", "app"),
        fb(6, "99999999-9999-9999-9999-999999999999", "review", "app", score=True),
    ]


class FakeLangSmith(BaseHTTPRequestHandler):
    """Projects, the run query (cursor-paged) and feedback (offset-paged)."""

    run_pages: ClassVar[list[list[dict]]] = []
    feedback: ClassVar[list[dict]] = []
    requests: ClassVar[list[tuple[str, str, dict]]] = []

    def _check(self) -> bool:
        if self.headers.get("x-api-key") != KEY:
            self._send(401, {"detail": "Invalid token"})
            return False
        return True

    def do_GET(self):
        url = urlsplit(self.path)
        params = parse_qs(url.query)
        type(self).requests.append(("GET", url.path, params))
        if not self._check():
            return None
        if url.path == "/sessions":
            match = [PROJECT] if params.get("name") == [PROJECT["name"]] else []
            return self._send(200, match)
        if url.path == "/feedback":
            wanted = set(params.get("run", []))
            items = [f for f in self.feedback if f["run_id"] in wanted]
            offset, limit = int(params["offset"][0]), int(params["limit"][0])
            return self._send(200, items[offset : offset + limit])
        return self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(("POST", self.path, body))
        if not self._check():
            return None
        index = int(body.get("cursor", "0"))
        nxt = str(index + 1) if index + 1 < len(self.run_pages) else None
        return self._send(200, {"runs": self.run_pages[index], "cursors": {"next": nxt}})

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
def langsmith():
    r = runs()
    FakeLangSmith.run_pages = [r[:2], r[2:5], r[4:]]  # r[4] repeats across pages
    FakeLangSmith.feedback = feedback()
    FakeLangSmith.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLangSmith)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def source(url: str, **kw) -> LangSmithSource:
    return LangSmithSource(api_url=url, api_key=KEY, **kw)


def test_runs_are_selected_by_their_root_and_paged_without_repeats(langsmith):
    got = fetch_runs(source(langsmith), "support-bot", **WINDOW)
    assert sorted(r["id"] for r in got) == sorted(r["id"] for r in runs())
    assert len({r["trace_id"] for r in got}) == 2
    posts = [body for method, _, body in FakeLangSmith.requests if method == "POST"]
    assert [b.get("cursor") for b in posts] == [None, "1", "2"]
    first = posts[0]
    assert first["session"] == [PROJECT["id"]]
    assert first["trace_filter"] == (
        'and(gte(start_time, "2026-10-20T00:00:00Z"), lt(start_time, "2026-10-21T00:00:00Z"))'
    )
    assert {"inputs", "outputs", "extra", "parent_run_id"} <= set(first["select"])
    assert "filter" not in first  # the window is on the root, so children are never cut off


def test_pulled_runs_and_feedback_flow_into_import_and_attach_answers(langsmith, tmp_path):
    env = {"LANGSMITH_ENDPOINT": langsmith, "LANGSMITH_API_KEY": KEY}
    traces, judged = tmp_path / "runs.jsonl", tmp_path / "judgments.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "langsmith-pull", "--project", "support-bot",
            "--since", WINDOW["start"], "--until", WINDOW["end"],
            "--traces", str(traces), "--judgments", str(judged),
        ],
        env=env,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 6 runs from 2 traces" in result.output
    assert "Wrote 3 of 5 feedback records as judgments" in result.output
    assert "1 no score, 1 value not a verdict" in result.output
    assert "Numeric scores need --threshold" in result.output
    assert COMMENT not in judged.read_text()

    run, report = import_traces(dataset(), traces, Settings(id_field="doc_id"))
    assert report["source"]["shape"] == "langsmith"
    by = {q["query_id"]: q for q in run["queries"]}
    assert by["q1"]["ids"] == ["d1", "d4"]
    assert by["q2"]["ids"] == ["d9"]

    out = attach_file(run, judged, {"correctness": 0.7})
    answers = {q["query_id"]: q["answer"] for q in out["queries"] if "answer" in q}
    assert (answers["q1"]["verdict"], answers["q1"]["decided_by"]) == ("correct", "human")
    assert answers["q2"]["verdict"] == "incorrect"  # feedback on a child run counts for its trace
    assert out["answers"]["unmatched_judgments"] == 0


def test_feedback_maps_to_judgments_without_guessing():
    r = runs()
    rows, skipped = feedback_to_judgments(feedback(), r)
    trace1, trace2 = r[0]["trace_id"], r[3]["trace_id"]
    assert rows == [
        {"trace_id": trace1, "evaluator": "review", "kind": "human", "verdict": "correct"},
        {"trace_id": trace1, "evaluator": "correctness", "kind": "model", "score": 0.4},
        {"trace_id": trace2, "evaluator": "grade", "kind": "model", "verdict": "incorrect"},
    ]
    assert skipped == {"value_not_a_verdict": 1, "no_score": 1, "run_not_pulled": 1}
    as_human, _ = feedback_to_judgments(feedback(), r, human_sources=("app", "API", "model"))
    assert {row["kind"] for row in as_human} == {"human"}
    odd = [
        {"run_id": r[0]["id"], "key": "x", "score": float("nan")},
        {"run_id": r[0]["id"], "key": "", "score": 1},
    ]
    assert feedback_to_judgments(odd, r) == ([], {"no_score": 1, "unnamed": 1})


def stub(responses):
    calls = []

    def send(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body) if body else None))
        return responses(len(calls), method, url)

    return send, calls


def test_feedback_is_batched_by_run_and_paged_by_offset():
    def respond(n, method, url):
        params = parse_qs(urlsplit(url).query)
        offset = int(params["offset"][0])
        batch = params["run"]
        if batch[0] == "r00" and offset == 0:  # the first batch has a full page, then one more
            return 200, json.dumps([{"id": f"a{i}", "run_id": "r00"} for i in range(100)]).encode()
        if batch[0] == "r00":
            return 200, json.dumps([{"id": "a100", "run_id": "r00"}]).encode()
        return 200, json.dumps([{"id": "b0", "run_id": batch[0]}]).encode()

    send, calls = stub(respond)
    got = fetch_feedback(source("https://ls.example", send=send), [f"r{i:02d}" for i in range(51)])
    assert len(got) == 102
    pages = [(parse_qs(urlsplit(u).query)["offset"][0], len(parse_qs(urlsplit(u).query)["run"]))
             for _, u, _ in calls]  # fmt: skip
    assert pages == [("0", 50), ("100", 50), ("0", 1)]


def test_an_unknown_project_and_wrong_key_name_the_problem(langsmith):
    with pytest.raises(LangSmithAPIError, match="No LangSmith project named 'other'"):
        project_id(source(langsmith), "other")
    wrong = LangSmithSource(api_url=langsmith, api_key="nope")
    with pytest.raises(LangSmithAPIError, match="refused the key"):
        fetch_runs(wrong, "support-bot", **WINDOW)


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (429, b"", "rate limit"),
        (500, b'{"detail": "PLANTED-SERVER-DETAIL"}', "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b'{"items": []}', "Unexpected LangSmith response shape for /runs/query"),
        (200, b'{"runs": [1]}', "Unexpected record"),
    ],
)
def test_bad_run_responses_are_refused_plainly(status, body, match):
    project = json.dumps([PROJECT]).encode()

    def respond(n, method, url):
        return (200, project) if method == "GET" else (status, body)

    send, _ = stub(respond)
    with pytest.raises(LangSmithAPIError, match=match) as info:
        fetch_runs(source("https://ls.example", send=send), "support-bot", **WINDOW)
    assert "PLANTED" not in str(info.value)


def test_a_repeated_cursor_and_an_oversized_window_stop_the_pull():
    project = json.dumps([PROJECT]).encode()
    loop = json.dumps({"runs": runs()[:1], "cursors": {"next": "same"}}).encode()

    def respond(n, method, url):
        return (200, project) if method == "GET" else (200, loop)

    send, calls = stub(respond)
    with pytest.raises(LangSmithAPIError, match="repeated a pagination cursor"):
        fetch_runs(source("https://ls.example", send=send), "support-bot", **WINDOW)
    assert len(calls) == 3  # project, then two pages
    send, _ = stub(respond)
    with pytest.raises(LangSmithAPIError, match="More than 1 runs"):
        fetch_runs(source("https://ls.example", send=send), "support-bot", **WINDOW, max_records=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"api_url": "http://ls.example"}, "Refusing to send the LangSmith key over http"),
        ({"api_url": "https://user:pw@ls.example"}, "must not contain credentials"),
        ({"api_url": "ftp://ls.example"}, "Invalid LangSmith endpoint"),
        ({"api_url": "https://ls.example", "api_key": ""}, "key is missing"),
    ],
)
def test_unsafe_or_incomplete_settings_are_refused(kwargs, match):
    with pytest.raises(LangSmithAPIError, match=match):
        LangSmithSource(**{"api_key": KEY, **kwargs})


def test_times_need_a_zone():
    send, calls = stub(lambda n, m, u: (200, json.dumps([PROJECT]).encode()))
    with pytest.raises(ValueError, match="needs a time zone"):
        fetch_runs(
            source("https://ls.example", send=send),
            "support-bot",
            start="2026-10-20T00:00:00",
            end=WINDOW["end"],
        )
    assert calls == []  # refused before any request


def test_cli_needs_an_output_and_reports_errors(tmp_path):
    runner = CliRunner()
    args = ["workbench", "langsmith-pull", "--project", "p", "--since", WINDOW["start"]]
    args += ["--until", WINDOW["end"]]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "give --traces, --judgments or both" in result.output
    result = runner.invoke(
        app,
        [*args, "--traces", str(tmp_path / "r.jsonl")],
        env={"LANGSMITH_API_KEY": "", "LANGCHAIN_API_KEY": ""},
    )
    assert result.exit_code == 2
    assert "key is missing" in result.output
    assert not (tmp_path / "r.jsonl").exists()
