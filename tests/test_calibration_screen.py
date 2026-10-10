"""A/A calibration from the workbench: measure, review the draft, save it as a new version."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.test_calibration import STRICT, noisy_runs
from vectraudit.evaluation.runfile import save_run
from vectraudit.pipelines.web import create_workbench


@pytest.fixture
def workbench(tmp_path):
    ids = []
    for i, run in enumerate(noisy_runs()):
        run["run_id"] = f"{i:032x}"
        save_run(run, tmp_path / f"{run['run_id']}.run.json")
        ids.append(run["run_id"])
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        policy = client.post("/api/guided/policies", json={"name": "Release", "policy": STRICT})
        assert policy.status_code == 200, policy.text
        yield client, ids, policy.json()


def test_calibration_reports_false_alarms_and_a_reviewable_draft(workbench, tmp_path):
    client, ids, policy = workbench
    before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    response = client.post("/api/guided/calibrate", json={"runs": ids, "policy": policy["id"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["policy"]["id"] == policy["id"]
    report = result["report"]
    assert report["current"]["comparisons"] == 20
    assert report["current"]["false_alarms"] > 0
    assert report["draft"]["false_alarms"] == 0
    assert {c["path"] for c in report["changes"]} >= {"metrics.ndcg.max_drop"}
    # Measuring writes nothing; the draft is only a proposal.
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == before

    saved = client.post(
        "/api/guided/policies",
        json={"name": "Release", "policy": report["draft_policy"], "note": "A/A calibration"},
    ).json()
    assert saved["version"] == 2
    assert saved["supersedes"] == policy["id"]
    again = client.post("/api/guided/calibrate", json={"runs": ids, "policy": saved["id"]}).json()[
        "report"
    ]
    assert again["current"]["false_alarms"] == 0
    assert again["draft_policy"] is None


@pytest.mark.parametrize(
    ("change", "status", "message"),
    [
        (lambda ids, p: {"runs": ids, "policy": None}, 422, "exploratory defaults"),
        (lambda ids, p: {"runs": ids[:1], "policy": p}, 422, None),  # one recording
        (lambda ids, p: {"runs": [ids[0], ids[0]], "policy": p}, 422, "each recording once"),
        (lambda ids, p: {"runs": ids, "policy": "f" * 32}, 404, None),  # no such policy
        (lambda ids, p: {"runs": [ids[0], "e" * 32], "policy": p}, 404, None),  # no such run
    ],
)
def test_bad_requests_are_refused(workbench, change, status, message):
    client, ids, policy = workbench
    response = client.post("/api/guided/calibrate", json=change(ids, policy["id"]))
    assert response.status_code == status, response.text
    if message:
        assert message in response.json()["detail"]


def test_mixed_benchmarks_are_refused_with_a_reason(workbench, tmp_path):
    client, ids, policy = workbench
    other = noisy_runs(1)[0]
    other["run_id"] = "d" * 32
    other["dataset"]["sha256"] = "e" * 64
    save_run(other, tmp_path / f"{other['run_id']}.run.json")
    response = client.post(
        "/api/guided/calibrate", json={"runs": [ids[0], other["run_id"]], "policy": policy["id"]}
    )
    assert response.status_code == 422
    assert "same dataset" in response.json()["detail"]
