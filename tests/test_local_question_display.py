"""Local text must match historical evidence and never enter persisted run/case files."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_guided_workbench import run, save
from vectraudit.pipelines.web import create_workbench

QUESTION = "How do I export the PRIVATE invoice?"
HELDOUT = "PRIVATE held-out question"
ROWS = [
    {"query_id": "q", "text": QUESTION, "relevant": {"missing-doc": 1}},
    {"query_id": "heldout", "text": HELDOUT, "split": "heldout", "no_answer": True},
]


@pytest.fixture
def evidence(tmp_path):
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = save(client, rows=ROWS).json()
        baseline, candidate = run(client, draft), run(client, draft)
        yield client, draft, baseline, candidate


def compare(client, baseline, candidate):
    response = client.post(
        "/api/guided/compare",
        json={"baseline": baseline["run_id"], "candidate": candidate["run_id"]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def test_matching_text_is_local_and_split_scoped(evidence, tmp_path):
    client, draft, baseline, candidate = evidence
    result = compare(client, baseline, candidate)
    expected = {"texts": {"q": QUESTION}, "reason": None}
    assert result["question_display"] == {"baseline": expected, "candidate": expected}
    assert result["baseline"] == baseline
    assert result["candidate"] == candidate
    assert HELDOUT not in json.dumps(result)
    for path in tmp_path.glob("*.json"):
        assert "PRIVATE" not in path.read_text()
    assert "PRIVATE" not in client.get("/api/guided/runs").text
    heldout = run(client, draft, mode="heldout", confirm_heldout=True)
    result = compare(client, baseline, heldout)
    assert result["question_display"]["candidate"]["texts"] == {"heldout": HELDOUT}


@pytest.mark.parametrize("change", ["revision", "dataset", "missing", "corrupt", "symlink"])
def test_unavailable_drafts_keep_findings_without_wrong_text(evidence, tmp_path, change):
    client, draft, baseline, candidate = evidence
    path = tmp_path / ".workbench" / "drafts" / f"{draft['id']}.json"
    if change == "revision":
        save(client, rows=ROWS, id=draft["id"], revision=draft["revision"], name="New name")
    elif change == "dataset":
        draft["dataset"] = draft["dataset"].replace(QUESTION, "PRIVATE wrong question")
        path.write_text(json.dumps(draft))
    elif change == "missing":
        path.unlink()
    elif change == "corrupt":
        path.write_text("PRIVATE invalid JSON")
    else:
        outside = tmp_path / "other-draft.json"
        outside.write_text(json.dumps(draft))
        path.unlink()
        path.symlink_to(outside)
    result = compare(client, baseline, candidate)
    assert result["triage"]["items"]
    for display in result["question_display"].values():
        assert display["texts"] == {}
        assert display["reason"].startswith("Question text unavailable:")
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("change", ["legacy", "split", "source_hash", "run_hash", "path"])
def test_run_identity_must_match_before_joining(evidence, tmp_path, change):
    client, _, baseline, candidate = evidence
    if change == "legacy":
        baseline.pop("benchmark_source")
    elif change == "split":
        baseline["benchmark_source"]["split"] = "heldout"
    elif change == "source_hash":
        baseline["benchmark_source"]["sha256"] = "wrong"
    elif change == "run_hash":
        baseline["dataset"]["sha256"] = "wrong"
    else:
        baseline["benchmark_source"]["draft_id"] = "../outside"
    (tmp_path / f"{baseline['run_id']}.run.json").write_text(json.dumps(baseline))
    result = compare(client, baseline, candidate)
    assert result["question_display"]["baseline"]["texts"] == {}
    assert result["question_display"]["baseline"]["reason"]
    assert result["question_display"]["candidate"]["texts"] == {"q": QUESTION}


def test_changed_question_on_same_id_is_only_shown_for_its_own_run(evidence):
    client, draft, baseline, _ = evidence
    changed = save(
        client,
        rows=[{**ROWS[0], "text": "A different PRIVATE question"}],
        id=draft["id"],
        revision=draft["revision"],
    ).json()
    candidate = run(client, changed)
    result = compare(client, baseline, candidate)
    assert result["question_display"]["baseline"]["texts"] == {}
    assert result["question_display"]["candidate"]["texts"] == {"q": "A different PRIVATE question"}


def test_case_text_is_response_only_and_survives_update(evidence, tmp_path):
    client, _, baseline, candidate = evidence
    case = client.post(
        "/api/guided/cases",
        json={"baseline": baseline["run_id"], "candidate": candidate["run_id"], "query_id": "q"},
    ).json()
    assert case["question_display"] == {"text": QUESTION, "reason": None}
    assert client.get("/api/guided/cases").json() == [case]
    updated = client.post(
        f"/api/guided/cases/{case['id']}",
        json={"revision": case["revision"], "status": "investigating", "owner": "Engineer"},
    ).json()
    assert updated["question_display"] == case["question_display"]
    stored = (tmp_path / ".workbench" / "cases" / f"{case['id']}.json").read_text()
    assert "PRIVATE" not in stored
    assert "question_display" not in stored
    (tmp_path / f"{baseline['run_id']}.run.json").unlink()
    unavailable = client.get("/api/guided/cases").json()[0]
    assert unavailable["question_display"]["text"] is None
    assert "original run" in unavailable["question_display"]["reason"]
    assert unavailable["history"] == updated["history"]


def test_vector_only_query_has_no_invented_text(evidence):
    client, _, _, _ = evidence
    draft = save(client, rows=[{"query_id": "vector", "vector": [1.0, 0.0]}]).json()
    result = run(client, draft)
    assert compare(client, result, result)["question_display"]["baseline"] == {
        "texts": {},
        "reason": None,
    }
