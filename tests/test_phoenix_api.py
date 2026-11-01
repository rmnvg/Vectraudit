"""Pulling spans and annotations from Phoenix, replayed from a real Phoenix 21 server.

``tests/fixtures/phoenix`` holds the server's answers to every request phoenix-pull made,
recorded by ``generate/record.py`` (see the README), once with normal page sizes and once
with tiny ones so cursor paging and ID batching are real too.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines import phoenix_api
from vectraudit.pipelines.phoenix_api import (
    PhoenixAPIError,
    PhoenixSource,
    annotations_to_judgments,
    fetch_annotations,
    fetch_spans,
    is_note,
)
from vectraudit.pipelines.traces import Settings, import_traces

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "http://127.0.0.1:6006"
WINDOW = {"start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z"}
RANKINGS = {"q1": ["d4", "d1", "d3"], "q2": ["d4", "d1", "d5"], "q3": ["d4", "d2", "d1"]}
Q1, Q2, Q3 = (
    "749ef00b3f93e29be06db44f3be4fc3d",
    "2d21338f346651d99980c31523ef78d8",
    "42e568da14943d772ea8038b410e77fa",
)


def replay(name: str = "responses.json", seen: list | None = None):
    recorded = json.loads((FIXTURES / "phoenix" / name).read_text())

    def send(url, headers, timeout):
        if seen is not None:
            seen.append((url, dict(headers)))
        path = url[len(BASE) :]
        if path not in recorded:
            return 404, b""
        return 200, json.dumps(recorded[path]).encode()

    return send


def pulled(name: str = "responses.json"):
    source = PhoenixSource(BASE, send=replay(name))
    spans = fetch_spans(source, "rag-demo", **WINDOW)
    on_traces, on_spans = fetch_annotations(source, "rag-demo", spans)
    return source, spans, on_traces, on_spans


def small_pages(monkeypatch):
    monkeypatch.setattr(phoenix_api, "SPAN_PAGE", 4)
    monkeypatch.setattr(phoenix_api, "ANNOTATION_PAGE", 2)
    monkeypatch.setattr(phoenix_api, "ID_BATCH", 2)


def test_the_pull_reads_every_span_of_the_traces_rooted_in_the_window():
    source, spans, on_traces, on_spans = pulled()
    assert len(spans) == 27
    assert {s["context"]["trace_id"] for s in spans} == {Q1, Q2, Q3}
    assert source.calls == 4  # roots, spans by trace, trace and span annotations
    assert len(on_traces) == 5  # four annotations and one note
    assert len(on_spans) == 2


def test_cursor_paging_and_id_batches_give_the_same_pull(monkeypatch):
    _, spans, on_traces, on_spans = pulled()
    small_pages(monkeypatch)
    source, paged, paged_traces, paged_spans = pulled("responses_paged.json")
    assert source.calls == 26
    assert paged == spans
    key = lambda a: a["id"]  # noqa: E731
    assert sorted(paged_traces, key=key) == sorted(on_traces, key=key)
    assert sorted(paged_spans, key=key) == sorted(on_spans, key=key)


def test_pulled_spans_import_like_the_direct_openinference_export(tmp_path):
    # Phoenix flattens document metadata into document.metadata.doc_id; the OTLP export
    # keeps it as a JSON string. Both must give the same rankings and model context.
    _, spans, _, _ = pulled()
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in spans))
    run, _ = import_traces(
        load_dataset(FIXTURES / "real_traces" / "questions.jsonl"),
        path,
        Settings(id_field="doc_id"),
    )
    queries = {q["query_id"]: q for q in run["queries"]}
    assert {qid: q["ids"] for qid, q in queries.items()} == RANKINGS
    for q in queries.values():
        assert q["error"] is None
        assert q["model_context"]["coverage"] == "observed"


def test_annotations_become_judgments_and_notes_are_skipped():
    _, spans, on_traces, on_spans = pulled()
    rows, skipped = annotations_to_judgments(on_traces, on_spans, spans)
    got = sorted((r["trace_id"], r["evaluator"], r["kind"], r.get("verdict"), r.get("score"))
                 for r in rows)  # fmt: skip
    assert got == sorted(
        [
            (Q1, "correctness", "human", "correct", 1.0),
            (Q2, "correctness", "human", "incorrect", None),  # label "Incorrect"
            (Q3, "relevance", "model", None, 0.4),  # CODE annotator, score only
            (Q2, "qa_correctness", "model", "incorrect", 0.0),  # on the root span
            (Q3, "qa_correctness", "model", "correct", 1.0),
        ]
    )
    assert skipped == {"notes": 1, "label_not_a_verdict": 1}  # the note, and "friendly"
    text = json.dumps(rows)
    assert "Billing" not in text
    assert "legacy plan" not in text  # the note's text
    assert "help page" not in text  # an explanation


def test_annotation_names_can_be_chosen_and_unpulled_spans_are_skipped():
    _, spans, on_traces, on_spans = pulled()
    rows, skipped = annotations_to_judgments(on_traces, on_spans, spans, names=["qa_correctness"])
    assert {r["evaluator"] for r in rows} == {"qa_correctness"}
    assert skipped["not_chosen"] == 4
    rows, skipped = annotations_to_judgments([], on_spans, [])
    assert rows == []
    assert skipped["span_not_pulled"] == 2


def test_notes_are_recognised_by_name_or_identifier():
    assert is_note({"name": "note"})
    assert is_note({"name": "x", "identifier": "px-span-note:1"})
    assert not is_note({"name": "correctness", "identifier": ""})
    rows, skipped = annotations_to_judgments(
        [
            {"trace_id": "t", "name": "c", "annotator_kind": "LLM", "result": None},
            {"trace_id": "t", "name": "c", "annotator_kind": "LLM", "result": {"score": True}},
            {"trace_id": "t", "name": "", "annotator_kind": "LLM", "result": {"score": 1}},
        ],
        [],
        [],
    )
    assert rows == []
    assert skipped == {"no_label_or_score": 2, "unnamed": 1}


def test_the_judgments_attach_and_disagreement_is_disputed(tmp_path):
    from vectraudit.evaluation.answers import attach_answers, read_judgments

    _, spans, on_traces, on_spans = pulled()
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in spans))
    run, _ = import_traces(
        load_dataset(FIXTURES / "real_traces" / "questions.jsonl"),
        path,
        Settings(id_field="doc_id"),
    )
    rows, _ = annotations_to_judgments(on_traces, on_spans, spans)
    judged = tmp_path / "j.jsonl"
    judged.write_text("\n".join(json.dumps(r) for r in rows))
    out = attach_answers(run, read_judgments(judged, {"relevance": 0.5}))
    verdicts = {q["query_id"]: q["answer"]["verdict"] for q in out["queries"]}
    # q3 has no human review; its model judgments disagree (relevance 0.4 < 0.5).
    assert verdicts == {"q1": "correct", "q2": "incorrect", "q3": "disputed"}


def test_requests_carry_the_key_as_a_bearer_token_and_repeat_list_parameters():
    seen: list = []
    source = PhoenixSource(BASE + "/v1/traces", api_key="k", send=replay(seen=seen))
    assert source.url == BASE  # the OTLP collector endpoint is accepted
    fetch_spans(source, "rag-demo", **WINDOW)
    url, headers = seen[1]
    assert headers["Authorization"] == "Bearer k"
    assert url.count("trace_id=") == 3
    assert "parent_id=null" in seen[0][0]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "check PHOENIX_API_KEY"),
        (404, "check the project name"),
        (429, "rate limit"),
        (500, "HTTP 500"),
    ],
)
def test_http_failures_are_explained(status, message):
    source = PhoenixSource(BASE, send=lambda u, h, t: (status, b"secret detail"))
    with pytest.raises(PhoenixAPIError, match=message) as info:
        fetch_spans(source, "rag-demo", **WINDOW)
    assert "secret" not in str(info.value)


def test_bad_bodies_and_unreachable_servers_are_errors():
    for body, message in ((b"<html>", "non-JSON"), (b'{"items": []}', "response shape")):
        source = PhoenixSource(BASE, send=lambda u, h, t, b=body: (200, b))
        with pytest.raises(PhoenixAPIError, match=message):
            fetch_spans(source, "rag-demo", **WINDOW)

    def down(url, headers, timeout):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(PhoenixAPIError, match="Cannot reach Phoenix"):
        fetch_spans(PhoenixSource(BASE, send=down), "rag-demo", **WINDOW)


def test_a_repeated_cursor_stops_and_the_record_limit_holds():
    calls = []

    def loop(url, headers, timeout):
        calls.append(url)
        span = {"context": {"trace_id": "t", "span_id": f"s{len(calls)}"}}
        return 200, json.dumps({"data": [span], "next_cursor": "same"}).encode()

    fetch_spans(PhoenixSource(BASE, send=loop), "p", **WINDOW)
    assert len(calls) == 2 + 2  # roots: first page, then "same" once; spans likewise
    with pytest.raises(PhoenixAPIError, match="More than 1 records"):
        fetch_spans(PhoenixSource(BASE, send=loop), "p", **WINDOW, max_records=1)


def test_unsafe_settings_are_refused():
    with pytest.raises(PhoenixAPIError, match="over http://"):
        PhoenixSource("http://phoenix.example.com", api_key="k")
    with pytest.raises(PhoenixAPIError, match="needs an API key"):
        PhoenixSource("https://app.phoenix.arize.com/s/team")
    with pytest.raises(PhoenixAPIError, match="credentials"):
        PhoenixSource("https://user:pw@phoenix.example.com")
    with pytest.raises(PhoenixAPIError, match="Invalid Phoenix URL"):
        PhoenixSource("phoenix.example.com")
    with pytest.raises(ValueError, match="--since"):
        fetch_spans(PhoenixSource(BASE, send=replay()), "p", start="yesterday", end="now")


def test_cli_pulls_spans_and_judgments(tmp_path, monkeypatch):
    monkeypatch.setattr(phoenix_api, "_urllib_send", replay())
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", BASE)
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "rag-demo")
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    runner = CliRunner()
    args = ["workbench", "phoenix-pull", "--since", WINDOW["start"], "--until", WINDOW["end"]]
    assert runner.invoke(app, args).exit_code == 2  # neither output chosen
    traces, judged = tmp_path / "spans.jsonl", tmp_path / "j.jsonl"
    done = runner.invoke(app, [*args, "--traces", str(traces), "--judgments", str(judged)])
    assert done.exit_code == 0, done.output
    text = " ".join(done.output.split())
    assert "Wrote 27 spans of 3 traces" in text
    assert "Wrote 5 annotations as judgments" in text
    assert "Skipped: 1 label not a verdict, 1 notes" in text
    assert "--threshold EVALUATOR=VALUE" in text
    assert "(4 API requests.)" in text
    assert len(traces.read_text().splitlines()) == 27
    missing = runner.invoke(app, [*args, "--project", "nope", "--traces", str(traces)])
    assert missing.exit_code == 2
    assert "HTTP 404" in missing.output


def test_the_recordings_hold_no_machine_details():
    for name in ("responses.json", "responses_paged.json"):
        text = (FIXTURES / "phoenix" / name).read_text()
        assert "/home/" not in text
        assert "/tmp/" not in text
