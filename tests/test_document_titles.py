"""Document titles beside IDs: from the run's own attached corpus, checked, never persisted."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_guided_workbench import run, save
from vectraudit.integrations.shared_records import project_record, validate_record
from vectraudit.pipelines.local_display import run_document_titles
from vectraudit.pipelines.release import corpus_version
from vectraudit.pipelines.web import create_workbench
from vectraudit.pipelines.workflows import documents

TITLE = "Invoice export guide (PRIVATE title)"
ROWS = [
    {"query_id": "q", "text": "invoice", "relevant": {"a": 1}},
    {"query_id": "h", "text": "invoice", "relevant": {"a": 1}, "split": "heldout"},
]
DOCS = [
    {"doc_id": "a", "title": TITLE, "text": "invoice export", "metadata": {"tenant": "acme"}},
    {"doc_id": "b", "text": "invoice archive", "metadata": {"title": "Archived invoices"}},
    {"doc_id": "c", "text": "volcano"},
]
CORPUS = "\n".join(map(json.dumps, DOCS))


@pytest.fixture
def evidence(tmp_path):
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = save(client, rows=ROWS, corpus=CORPUS).json()
        baseline = run(client, draft, corpus=CORPUS)
        candidate = run(client, draft, corpus=CORPUS)
        yield client, draft, baseline, candidate


def compare(client, baseline, candidate):
    response = client.post(
        "/api/guided/compare",
        json={"baseline": baseline["run_id"], "candidate": candidate["run_id"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_comparison_shows_verified_titles_and_never_writes_them(evidence, tmp_path):
    client, _, baseline, candidate = evidence
    shown = compare(client, baseline, candidate)["document_display"]
    assert shown["verified"] is True
    assert shown["reason"] is None
    assert shown["titles"]["a"] == TITLE
    assert shown["titles"]["b"] == "Archived invoices"  # metadata.title fallback
    assert "c" not in shown["titles"]
    for path in tmp_path.glob("*.json"):
        assert TITLE not in path.read_text()


def test_case_titles_are_response_only(evidence, tmp_path):
    client, draft, baseline, _ = evidence
    # A candidate whose corpus lost the relevant document: a real finding citing "a".
    lost = "\n".join(json.dumps(d) for d in DOCS if d["doc_id"] != "a")
    candidate = run(client, draft, corpus=lost)
    case = client.post(
        "/api/guided/cases",
        json={"baseline": baseline["run_id"], "candidate": candidate["run_id"], "query_id": "q"},
    )
    assert case.status_code == 200, case.text
    record = case.json()
    assert record["document_display"]["titles"] == {"a": TITLE}  # the baseline's own corpus
    assert record["document_display"]["verified"] is True
    stored = (tmp_path / ".workbench" / "cases" / f"{record['id']}.json").read_text()
    assert "document_display" not in stored
    assert TITLE not in stored


def test_a_corpus_other_than_the_runs_withholds_titles(evidence, tmp_path):
    client, draft, baseline, _ = evidence
    from vectraudit.pipelines.guided import GuidedWorkspace

    workspace = GuidedWorkspace(tmp_path)
    other = json.loads(json.dumps(baseline))
    other["backend"]["corpus_version"] = "0" * 64
    withheld = run_document_titles(workspace, other, {"a"})
    assert withheld["titles"] == {}
    assert "not the one this run used" in withheld["reason"]
    assert withheld["code"] == "mismatch"
    external = json.loads(json.dumps(baseline))
    external["backend"].pop("corpus_version", None)
    shown = run_document_titles(workspace, external, {"a"})
    assert shown == {"titles": {"a": TITLE}, "reason": None, "code": None, "verified": False}
    save(client, rows=ROWS, corpus=CORPUS, id=draft["id"], revision=draft["revision"], name="x")
    changed = run_document_titles(workspace, baseline, {"a"})
    assert "saved benchmark has changed" in changed["reason"]


def test_untitled_corpora_say_so_and_keep_their_versions(tmp_path):
    untitled = "\n".join(json.dumps({"doc_id": d["doc_id"], "text": d["text"]}) for d in DOCS)
    assert all("title" not in d for d in documents(untitled))
    # Titles are optional: a corpus without them hashes exactly as before.
    assert corpus_version(documents(untitled)) == corpus_version(
        [{"doc_id": d["doc_id"], "text": d["text"], "metadata": {}} for d in DOCS]
    )
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = save(client, rows=ROWS, corpus=untitled).json()
        first, second = run(client, draft, corpus=untitled), run(client, draft, corpus=untitled)
        shown = compare(client, first, second)["document_display"]
    assert shown["titles"] == {}
    assert "has no titles" in shown["reason"]
    assert shown["code"] == "no_titles"


def case_with_titles():
    from tests.test_shared_records import case

    record = case()
    record["document_display"] = {"titles": {"doc1": TITLE, "uncited": "Other"}}
    return record


def test_shared_cases_carry_titles_only_with_text_sharing():
    without = project_record("case", case_with_titles())
    assert "document_titles" not in without["data"]
    with_text = project_record("case", case_with_titles(), share_query_text=True)
    assert with_text["data"]["document_titles"] == {"doc1": TITLE}  # cited documents only
    forged = json.loads(json.dumps(without))
    forged["data"]["document_titles"] = {"doc1": TITLE}
    with pytest.raises(ValueError, match="explicit sharing"):
        validate_record(forged)


def test_hosted_case_page_shows_titles_or_says_why(world, cloud_app):
    pytest.importorskip("sqlalchemy")
    from tests.test_cloud_runs import client_for
    from tests.test_cloud_web import sign_in

    client = client_for(world, "acme_runner")
    shared = client.upload_record(project_record("case", case_with_titles(), share_query_text=True))
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = browser.get(f"/app/acme/search/records/{shared['id']}").text
    assert "Invoice export guide (PRIVATE title)" in page
    record = case_with_titles()
    record["revision"] = "f" * 32
    plain = client.upload_record(project_record("case", record))
    page = browser.get(f"/app/acme/search/records/{plain['id']}").text
    assert "Document titles were not shared" in page
    assert TITLE not in page
