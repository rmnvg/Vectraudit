"""Two workbench evaluations at once: one per registered pipeline, overlap recorded and shown."""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from tests.test_guided_workbench import options, save
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines import PipelineResult
from vectraudit.pipelines.web import create_workbench


class Gate:
    """Pipelines that block until released, counting how many evaluations are inside."""

    def __init__(self, expected: int) -> None:
        self.lock = threading.Lock()
        self.inside: set[str] = set()
        self.active = 0
        self.all_in = threading.Event()
        self.release = threading.Event()
        self.expected = expected

    def pipeline(self, name: str):
        def call(request):
            with self.lock:
                self.inside.add(name)
                self.active += 1
                if self.active >= self.expected:
                    self.all_in.set()
            assert self.release.wait(10)
            return PipelineResult(hits=[])

        return call


def workbench(tmp_path, monkeypatch, gate: Gate) -> TestClient:
    manifest = tmp_path / "pipelines.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "unused.jsonl",
                "candidates": [
                    {"name": name, "adapter": {"kind": "http", "target": f"https://{name}.test/r"}}
                    for name in ("a", "b")
                ],
            }
        )
    )
    monkeypatch.setattr(
        "vectraudit.pipelines.guided.open_adapter",
        lambda adapter, root: gate.pipeline(str(adapter.target)),
    )
    monkeypatch.setattr(
        "vectraudit.pipelines.guided.hybrid_pipeline", lambda _: gate.pipeline("builtin")
    )
    client = TestClient(create_workbench(tmp_path, manifest))
    client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
    return client


def finish(client, *jobs):
    runs = []
    for job in jobs:
        assert client.app.state.guided.jobs.get(job).wait(10)
        state = client.get(f"/api/guided/jobs/{job}").json()
        assert state["status"] == "done", state
        runs.append(state["run"])
    return runs


def test_two_registered_pipelines_run_at_the_same_time(tmp_path, monkeypatch):
    gate = Gate(expected=2)
    client = workbench(tmp_path, monkeypatch, gate)
    draft = save(client).json()
    remote = {"confirm_remote": True}
    first = client.post("/api/guided/jobs", json=options(draft, pipeline="configured-0", **remote))
    second = client.post("/api/guided/jobs", json=options(draft, pipeline="configured-1", **remote))
    assert (first.status_code, second.status_code) == (202, 202)
    try:
        assert gate.all_in.wait(5)  # both evaluations are inside their pipelines at once
        # The same registered pipeline never runs twice at once: its limits describe one run.
        again = client.post(
            "/api/guided/jobs", json=options(draft, pipeline="configured-0", **remote)
        )
        assert again.status_code == 409
        assert "already has an evaluation running" in again.json()["detail"]
    finally:
        gate.release.set()
    runs = finish(client, first.json()["id"], second.json()["id"])
    assert [r["timing"]["overlapped_runs"] for r in runs] == [1, 1]
    saved = [json.loads(p.read_text()) for p in tmp_path.glob("*.run.json")]
    assert sorted(r["timing"]["overlapped_runs"] for r in saved) == [1, 1]

    # Once finished, the pipeline is free again, and a run alone records no overlap.
    alone = client.post("/api/guided/jobs", json=options(draft, pipeline="configured-0", **remote))
    assert alone.status_code == 202
    (solo,) = finish(client, alone.json()["id"])
    assert solo["timing"]["overlapped_runs"] == 0


def test_built_in_runs_may_overlap_and_comparisons_say_so(tmp_path, monkeypatch):
    gate = Gate(expected=1)
    client = workbench(tmp_path, monkeypatch, gate)
    draft = save(client).json()
    first = client.post("/api/guided/jobs", json=options(draft)).json()["id"]
    second = client.post("/api/guided/jobs", json=options(draft, label="other"))
    assert second.status_code == 202
    gate.release.set()
    base, cand = finish(client, first, second.json()["id"])
    assert base["timing"]["overlapped_runs"] == cand["timing"]["overlapped_runs"] == 1
    warnings = compare_runs(base, cand, DEFAULT_POLICY)["warnings"]
    assert any(
        "Latency includes extra load" in w and "baseline beside 1 other" in w for w in warnings
    )
    # The hosted projection keeps the count, so a hosted check warns the same way.
    assert project_run(cand)["timing"]["overlapped_runs"] == 1
    hosted = compare_runs(project_run(base), project_run(cand), DEFAULT_POLICY)["warnings"]
    assert any("Latency includes extra load" in w for w in hosted)
    base["timing"]["overlapped_runs"] = cand["timing"]["overlapped_runs"] = 0
    quiet = compare_runs(base, cand, DEFAULT_POLICY)["warnings"]
    assert not any("extra load" in w for w in quiet)


def test_capacity_failures_free_the_pipeline(tmp_path, monkeypatch):
    gate = Gate(expected=2)
    client = workbench(tmp_path, monkeypatch, gate)
    draft = save(client).json()
    remote = {"confirm_remote": True}
    jobs = [
        client.post("/api/guided/jobs", json=options(draft)).json()["id"],
        client.post("/api/guided/jobs", json=options(draft)).json()["id"],
    ]
    try:
        assert gate.all_in.wait(5)
        full = client.post(
            "/api/guided/jobs", json=options(draft, pipeline="configured-0", **remote)
        )
        assert full.status_code == 429
    finally:
        gate.release.set()
    finish(client, *jobs)
    # The refused start did not leave configured-0 marked as running.
    later = client.post("/api/guided/jobs", json=options(draft, pipeline="configured-0", **remote))
    assert later.status_code == 202
    finish(client, later.json()["id"])
