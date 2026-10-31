"""Writing Vectraudit's diagnosis back onto Langfuse traces as scores."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from tests.test_langfuse_api import KEYS, FakeLangfuse, observations
from tests.test_trace_formats import langsmith_trace
from tests.test_trace_import import SECRET
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_file
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.langfuse_api import LangfuseAPIError, LangfuseSource
from vectraudit.pipelines.langfuse_push import (
    BATCH,
    LangfusePushError,
    diagnosis_scores,
    push_scores,
    score_id,
)
from vectraudit.pipelines.traces import Settings, import_traces

QUESTIONS = [
    # tf1 retrieves d1 and d4 and the model sees d1: evidence reached the model.
    {"query_id": "q1", "text": "How do I reset my password?", "relevant": {"d1": 1},
     "required_evidence": [["d1"]]},
    # tf2 retrieves d9 only: the required d2 was never observed.
    {"query_id": "q2", "text": "Where is the invoice?", "relevant": {"d2": 1},
     "required_evidence": [["d2"]]},
    {"query_id": "q3", "text": "Cancel my plan", "relevant": {"d3": 1}},
]  # fmt: skip


def benchmark():
    return parse_dataset("\n".join(json.dumps(q) for q in QUESTIONS))


def judged_run(tmp_path, verdicts=None) -> dict:
    """A Langfuse import with answer judgments attached."""
    traces = tmp_path / "obs.jsonl"
    traces.write_text("\n".join(json.dumps(o) for o in observations()))
    run, _ = import_traces(benchmark(), traces, Settings(id_field="doc_id"))
    verdicts = verdicts or {"tf1": "incorrect", "tf2": "incorrect"}
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        "\n".join(
            json.dumps({"trace_id": t, "evaluator": "review", "kind": "human", "verdict": v})
            for t, v in verdicts.items()
        )
    )
    return attach_file(run, judgments)


class FakeLangfuseScores(FakeLangfuse):
    """Also accepts score batches, answering 202 or a configured 207."""

    posted: ClassVar[list[list[dict]]] = []
    partial: ClassVar[dict | None] = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, {}))
        if self.path != "/api/public/scores":
            return self._send(404, {"message": "not found"})
        if self.headers.get("Authorization") != FakeLangfuse.expected():
            return self._send(401, {"message": "Invalid credentials"})
        type(self).posted.append(body)
        if self.partial is not None:
            return self._send(207, self.partial)
        return self._send(202, {"message": "Accepted"})


@pytest.fixture
def langfuse():
    FakeLangfuseScores.posted = []
    FakeLangfuseScores.partial = None
    FakeLangfuse.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLangfuseScores)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def env(host: str) -> dict[str, str]:
    return {"LANGFUSE_HOST": host, "LANGFUSE_PUBLIC_KEY": KEYS[0], "LANGFUSE_SECRET_KEY": KEYS[1]}


def test_each_judged_trace_gets_one_categorical_diagnosis_score(tmp_path):
    scores, skipped = diagnosis_scores(judged_run(tmp_path), environment="production")
    by_trace = {s["traceId"]: s for s in scores}
    assert by_trace["tf1"]["value"] == "answered_wrong_despite_evidence"
    assert by_trace["tf2"]["value"] == "not_observed"
    assert by_trace["tf1"] == {
        "id": score_id("tf1", "vectraudit.diagnosis"),
        "traceId": "tf1",
        "name": "vectraudit.diagnosis",
        "value": "answered_wrong_despite_evidence",
        "dataType": "CATEGORICAL",
        "metadata": {
            "source": "vectraudit",
            "format_version": 1,
            "run_digest": by_trace["tf1"]["metadata"]["run_digest"],
        },
        "environment": "production",
    }
    assert skipped == {"no_trace": 1}  # q3 has no trace in this window


def test_nothing_private_is_in_the_payload(tmp_path):
    scores, _ = diagnosis_scores(judged_run(tmp_path))
    for score in scores:  # the ID and run digest are hashes; check everything else
        assert len(score["id"]) == 32
        assert len(score["metadata"]["run_digest"]) == 16
    sent = json.dumps(
        [{**s, "id": None, "metadata": {**s["metadata"], "run_digest": None}} for s in scores]
    )
    for private in (SECRET, "reset my password", "invoice", "d1", "d4", "d9", "review"):
        assert private not in sent


def test_preview_sends_nothing_and_send_writes_the_same_scores_idempotently(langfuse, tmp_path):
    path = tmp_path / "judged.run.json"
    path.write_text(json.dumps(judged_run(tmp_path)))
    runner = CliRunner()
    preview = runner.invoke(app, ["workbench", "langfuse-push", str(path)], env=env(langfuse))
    assert preview.exit_code == 0, preview.output
    assert "2 traces would get a score named 'vectraudit.diagnosis'" in preview.output
    assert "Not included: 1 no trace" in preview.output
    assert "Preview only: nothing was sent" in preview.output
    assert FakeLangfuse.requests == []  # not even a connection

    for _ in range(2):
        sent = runner.invoke(
            app, ["workbench", "langfuse-push", str(path), "--send"], env=env(langfuse)
        )
        assert sent.exit_code == 0, sent.output
        assert "Sent 2 scores in 1 requests: 2 accepted, 0 rejected." in sent.output
    first, second = FakeLangfuseScores.posted
    shown = preview.output.split("Example score (exactly as sent):\n", 1)[1]
    example, _ = json.JSONDecoder().raw_decode(shown)
    assert example in first
    assert [s["id"] for s in first] == [s["id"] for s in second]  # a re-push replaces


def test_partial_rejection_is_counted_not_retried_and_not_echoed(langfuse, tmp_path):
    path = tmp_path / "judged.run.json"
    path.write_text(json.dumps(judged_run(tmp_path)))
    FakeLangfuseScores.partial = {
        "accepted": 1,
        "rejected": 1,
        "errors": [{"message": "PLANTED-REJECTION-DETAIL"}],
    }
    result = CliRunner().invoke(
        app, ["workbench", "langfuse-push", str(path), "--send"], env=env(langfuse)
    )
    assert result.exit_code == 3
    assert "1 accepted, 1 rejected" in result.output
    assert "PLANTED" not in result.output
    assert len(FakeLangfuseScores.posted) == 1


def test_scores_are_sent_in_batches_and_odd_partial_answers_are_refused():
    sent = []

    def post(url, headers, body, timeout):
        sent.append(len(json.loads(body)))
        return 202, b'{"message": "Accepted"}'

    source = LangfuseSource(
        host="https://lf.example", public_key=KEYS[0], secret_key=KEYS[1], post_send=post
    )
    scores = [{"traceId": f"t{i}", "name": "n", "value": "x"} for i in range(BATCH * 2 + 5)]
    assert push_scores(source, scores) == {
        "sent": 205, "accepted": 205, "rejected": 0, "requests": 3,
    }  # fmt: skip
    assert sent == [100, 100, 5]

    def odd(url, headers, body, timeout):
        return 207, b'{"accepted": "all"}'

    source.post_send = odd
    with pytest.raises(LangfuseAPIError, match="Unexpected Langfuse response"):
        push_scores(source, scores[:1])


def test_questions_without_a_diagnosis_or_with_a_repeated_trace_are_left_out(tmp_path):
    run = judged_run(tmp_path, {"tf1": "incorrect"})  # tf2 not judged
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    run["queries"].append({**q1, "query_id": "q1-again"})
    scores, skipped = diagnosis_scores(run)
    assert [s["traceId"] for s in scores] == ["tf1"]
    assert skipped == {"no_trace": 1, "no_diagnosis": 1, "duplicate_trace": 1}


def test_runs_that_are_not_judged_langfuse_imports_are_refused(tmp_path):
    judged = judged_run(tmp_path)
    with pytest.raises(LangfusePushError, match="attach-answers first"):
        diagnosis_scores({k: v for k, v in judged.items() if k != "answers"})
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps(langsmith_trace(1, "How do I reset my password?", [])))
    langsmith_run, _ = import_traces(benchmark(), runs)
    with pytest.raises(LangfusePushError, match="Only runs imported from Langfuse"):
        diagnosis_scores({**langsmith_run, "answers": {}})
    with pytest.raises(LangfusePushError, match="score name"):
        diagnosis_scores(judged, name="has spaces")
    for bad in ("Production", "langfuse-prod", "a" * 41):
        with pytest.raises(LangfusePushError, match="environment"):
            diagnosis_scores(judged, environment=bad)


def test_cli_send_needs_keys_and_reports_refusals(tmp_path):
    path = tmp_path / "judged.run.json"
    path.write_text(json.dumps(judged_run(tmp_path)))
    result = CliRunner().invoke(
        app,
        ["workbench", "langfuse-push", str(path), "--send"],
        env={"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
    )
    assert result.exit_code == 2
    assert "keys are missing" in result.output
    result = CliRunner().invoke(app, ["workbench", "langfuse-push", str(path), "--name", "a b"])
    assert result.exit_code == 2
    assert "score name" in result.output
