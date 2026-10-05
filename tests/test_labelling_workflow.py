"""Browser transaction semantics and the local reviewer/export journey."""

import re

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.experiments.corpus import Document, make_corpus
from vectraudit.labelling.operations import judge, resolve_judgment
from vectraudit.labelling.pooling import pool
from vectraudit.labelling.storage import update_task
from vectraudit.labelling.task import Judgment, load_task, save_task
from vectraudit.labelling.web import create_app


@pytest.fixture
def local(tmp_path):
    queries = "\n".join(f'{{"query_id":"q{i}","text":"question {i}"}}' for i in range(5))
    task = pool(parse_dataset(queries), make_corpus([Document("doc", "A 😀 answer text")]), k=1)
    path = tmp_path / "task.json"
    save_task(task, path)
    return path, TestClient(create_app(path), base_url="http://127.0.0.1")


def fields(client, url):
    response = client.get(url)
    return dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', response.text))


def test_stale_tab_conflict_and_retry_preserves_resolution(local):
    path, client = local
    original = fields(client, "/?annotator=A&q=0")
    assert client.post("/judge", data={**original, "grade": "2"}).status_code == 200
    reviewed = Judgment(query_id="q0", doc_id="doc", annotator="Reviewer", grade=3)
    update_task(path, lambda task: resolve_judgment(task, reviewed))
    # Retrying the identical request is harmless and must not invalidate review.
    assert client.post("/judge", data={**original, "grade": "2"}).status_code == 200
    assert load_task(path).resolutions == [reviewed]
    response = client.post("/judge", data={**original, "grade": "1"})
    assert response.status_code == 409
    assert "another tab" in response.text
    assert load_task(path).judgments[0].grade == 2


def test_resume_and_saved_spans(local):
    path, client = local
    form = fields(client, "/?annotator=A&q=0")
    assert (
        client.post("/judge", data={**form, "grade": "2", "start": "2", "end": "3"}).status_code
        == 200
    )
    restored = client.get("/?annotator=A&q=0").text
    assert 'name="start" value="2"' in restored
    assert 'name="end" value="3"' in restored
    assert "question 1" in client.get("/?annotator=A").text
    assert "question 0" in client.get("/?annotator=B").text
    form = fields(client, "/?annotator=A&q=0")
    assert (
        client.post("/judge", data={**form, "grade": "0", "start": "2", "end": "3"}).status_code
        == 200
    )
    assert load_task(path).judgments[0].start is None


def test_browser_review_and_download(local):
    path, client = local
    for person, grade in [("Alice", 1), ("Bob", 3)]:
        update_task(
            path,
            lambda task, person=person, grade=grade: judge(
                task, Judgment(query_id="q0", doc_id="doc", annotator=person, grade=grade)
            ),
        )
    assert client.get("/dataset.jsonl").status_code == 409
    page = client.get("/review?reviewer=R")
    assert "Alice: grade 1" in page.text
    assert "Bob: grade 3" in page.text
    form = fields(client, "/review?reviewer=R")
    assert client.post("/resolve", data={**form, "grade": "2"}).status_code == 200
    status = client.get("/status?annotator=R")
    assert "1/5 questions with positive labels" in status.text
    assert "partial label set" in status.text
    exported = client.get("/dataset.jsonl")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    assert parse_dataset(exported.text).queries[0].relevant == {"doc": 2}
    assert "Cache-Control" in exported.headers


def test_stale_reviewer_cannot_override_new_votes(local):
    path, client = local
    for person, grade in [("Alice", 1), ("Bob", 3)]:
        update_task(
            path,
            lambda task, person=person, grade=grade: judge(
                task, Judgment(query_id="q0", doc_id="doc", annotator=person, grade=grade)
            ),
        )
    form = fields(client, "/review?reviewer=R")
    update_task(
        path,
        lambda task: judge(task, Judgment(query_id="q0", doc_id="doc", annotator="Bob", grade=2)),
    )
    assert client.post("/resolve", data={**form, "grade": "3"}).status_code == 409
    assert not load_task(path).resolutions


def test_malformed_forms_fail_cleanly(local):
    _, client = local
    form = fields(client, "/?annotator=A&q=0")
    assert client.post("/judge", data={**form, "grade": "2", "csrf": "😀"}).status_code == 403
    headers = {"content-type": "application/x-www-form-urlencoded"}
    assert client.post("/judge", content=b"\xff", headers=headers).status_code == 400
    assert client.post("/judge", content=b"x=" + b"a" * 16384, headers=headers).status_code == 413
    assert client.post("/resolve", data={**form, "grade": "2"}).status_code == 400


def test_cli_errors_and_exports_do_not_clobber(local, tmp_path):
    path, _ = local
    update_task(
        path,
        lambda task: judge(task, Judgment(query_id="q0", doc_id="doc", annotator="A", grade=2)),
    )
    runner = CliRunner()
    args = ["label", "export", str(path), "--out", str(tmp_path / "dataset.jsonl")]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, [*args, "--force"]).exit_code == 0
    result = runner.invoke(app, ["label", "sample", str(path), "B", "--fraction", "0"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output
    assert runner.invoke(app, ["label", "status", str(path), "--json"]).exit_code == 0
