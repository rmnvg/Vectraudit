"""Writing diagnoses back to Phoenix as trace annotations.

The fake server below behaves as a running Phoenix 21 server did (see
``vectraudit.pipelines.phoenix_push``): with ``sync=true`` a batch naming an unknown trace
is refused whole with 404, and an annotation with the same trace, name and identifier
replaces the earlier one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_phoenix_api import BASE, Q1, Q2, Q3, pulled
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_answers, read_judgments
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines import phoenix_api, phoenix_push
from vectraudit.pipelines.phoenix_api import (
    PhoenixAPIError,
    PhoenixSource,
    annotations_to_judgments,
)
from vectraudit.pipelines.phoenix_push import (
    PhoenixPushError,
    diagnosis_annotations,
    preview,
    push_annotations,
)
from vectraudit.pipelines.traces import Settings, import_traces

FIXTURES = Path(__file__).parent / "fixtures"


class FakePhoenix:
    def __init__(self, traces=(Q1, Q2, Q3), status: int | None = None) -> None:
        self.traces = set(traces)
        self.status = status
        self.stored: dict[tuple, dict] = {}
        self.requests: list[tuple[str, dict, dict]] = []

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.requests.append((url, dict(headers), payload))
        if self.status:
            return self.status, b"detail"
        assert url.endswith("/v1/trace_annotations?sync=true")
        if any(a["trace_id"] not in self.traces for a in payload["data"]):
            return 404, b"Traces with IDs ... do not exist."
        for a in payload["data"]:
            self.stored[(a["trace_id"], a["name"], a["identifier"])] = a
        return 200, b'{"data": []}'


def judged_run(tmp_path) -> dict:
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
    return attach_answers(run, read_judgments(judged, {"relevance": 0.5}))


def test_one_annotation_per_judged_trace_with_nothing_private(tmp_path):
    annotations, skipped = diagnosis_annotations(judged_run(tmp_path))
    assert {a["trace_id"] for a in annotations} == {Q1, Q2, Q3}
    assert not skipped
    for a in annotations:
        assert set(a) == {"trace_id", "name", "annotator_kind", "identifier", "result", "metadata"}
        assert a["annotator_kind"] == "CODE"
        assert a["identifier"] == "vectraudit"
        assert set(a["result"]) == {"label"}
        assert set(a["metadata"]) == {"source", "format_version", "run_digest"}
    by = {a["trace_id"]: a["result"]["label"] for a in annotations}
    assert by[Q3] == "disputed"
    text = json.dumps(annotations)
    for private in ("password", "Billing", '"d1"', '"d4"'):  # IDs quoted: the digest is hex
        assert private not in text


def test_runs_from_other_tools_or_without_judgments_are_refused(tmp_path):
    run = judged_run(tmp_path)
    langfuse = {**run, "backend": {**run["backend"], "config": {**run["backend"]["config"],
                                                                "shape": "langfuse"}}}  # fmt: skip
    with pytest.raises(PhoenixPushError, match="trace IDs must be Phoenix's"):
        diagnosis_annotations(langfuse)
    with pytest.raises(PhoenixPushError, match="attach-answers first"):
        diagnosis_annotations({k: v for k, v in run.items() if k != "answers"})
    with pytest.raises(PhoenixPushError, match="annotation name"):
        diagnosis_annotations(run, name="has space")
    odd = {**run, "queries": [{**run["queries"][0], "trace_id": "run-123"}]}
    annotations, skipped = diagnosis_annotations(odd)
    assert annotations == []
    assert skipped["not_an_otel_trace_id"] == 1


def test_a_push_writes_in_one_request_and_a_repush_replaces(tmp_path):
    annotations, _ = diagnosis_annotations(judged_run(tmp_path))
    server = FakePhoenix()
    source = PhoenixSource(BASE, api_key="k", post_send=server)
    assert push_annotations(source, annotations) == {
        "sent": 3, "written": 3, "unknown_trace": 0, "requests": 1
    }  # fmt: skip
    _, headers, _ = server.requests[0]
    assert headers["Authorization"] == "Bearer k"
    assert headers["Content-Type"] == "application/json"
    push_annotations(source, annotations)
    assert len(server.stored) == 3  # replaced, not added


def test_unknown_traces_are_found_one_by_one_and_the_rest_still_written(tmp_path):
    annotations, _ = diagnosis_annotations(judged_run(tmp_path))
    server = FakePhoenix(traces=(Q1, Q2))
    totals = push_annotations(PhoenixSource(BASE, post_send=server), annotations)
    assert totals == {"sent": 3, "written": 2, "unknown_trace": 1, "requests": 4}
    assert {k[0] for k in server.stored} == {Q1, Q2}


def test_large_pushes_are_batched(tmp_path, monkeypatch):
    monkeypatch.setattr(phoenix_push, "BATCH", 2)
    annotations, _ = diagnosis_annotations(judged_run(tmp_path))
    server = FakePhoenix()
    totals = push_annotations(PhoenixSource(BASE, post_send=server), annotations)
    assert totals["requests"] == 2
    assert [len(p["data"]) for _, _, p in server.requests] == [2, 1]


@pytest.mark.parametrize(
    ("status", "message"), [(401, "write access"), (429, "rate limit"), (500, "HTTP 500")]
)
def test_write_failures_are_explained_without_echoing_the_body(tmp_path, status, message):
    annotations, _ = diagnosis_annotations(judged_run(tmp_path))
    source = PhoenixSource(BASE, post_send=FakePhoenix(status=status))
    with pytest.raises(PhoenixAPIError, match=message) as info:
        push_annotations(source, annotations)
    assert "detail" not in str(info.value)


def test_preview_shows_exactly_what_is_sent(tmp_path):
    annotations, skipped = diagnosis_annotations(judged_run(tmp_path))
    text = preview(annotations, {**skipped, "no_diagnosis": 2})
    assert "3 traces would get an annotation named 'vectraudit.diagnosis'." in text
    assert "Not included: 2 no diagnosis" in text
    assert json.dumps(annotations[0], indent=2) in text


def test_cli_previews_by_default_and_sends_with_send(tmp_path, monkeypatch):
    run = tmp_path / "judged.json"
    run.write_text(json.dumps(judged_run(tmp_path)))
    server = FakePhoenix(traces=(Q1, Q2))
    monkeypatch.setattr(phoenix_api, "_urllib_post", server)
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", BASE)
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    runner = CliRunner()
    shown = runner.invoke(app, ["workbench", "phoenix-push", str(run)])
    assert shown.exit_code == 0, shown.output
    assert "Preview only: nothing was sent" in shown.output
    assert server.requests == []
    sent = runner.invoke(app, ["workbench", "phoenix-push", str(run), "--send"])
    assert sent.exit_code == 3  # one trace is not on this server
    assert "Wrote 2 of 3 annotations in 4 requests." in sent.output
    assert "1 traces are not in this Phoenix server" in sent.output
