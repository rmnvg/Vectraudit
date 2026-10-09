"""Default hosted transfers exclude private content without changing release gates."""

from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.integrations.cloud import CloudClient, CloudTarget
from vectraudit.integrations.sharing import project_run, transfer_preview

PRIVATE = "SEEDED_PRIVATE_EVIDENCE_DO_NOT_TRANSFER"


def rich_run():
    run = fake_run([1.0] * 30)
    run["future_field"] = PRIVATE
    run["dataset"]["meta"] = {"description": PRIVATE, "provenance": {"raw": PRIVATE}}
    run["backend"]["config"] = {"prompt": PRIVATE}
    run["connection"] = {"endpoint": PRIVATE}
    run["search"]["params"]["prompt"] = PRIVATE
    run["provenance"]["labels"] = {"private": PRIVATE}
    q = run["queries"][0]
    q.update(
        text=PRIVATE,
        history=[{"role": "user", "content": PRIVATE}],
        context={"tenant": PRIVATE},
        evidence={"future_context": PRIVATE},
        future_field=PRIVATE,
    )
    return run


def test_projection_is_separate_bounded_and_idempotent():
    run = rich_run()
    before = copy.deepcopy(run)
    projected = project_run(run)
    assert PRIVATE not in json.dumps(projected)
    assert run == before
    assert projected["sharing"]["source_sha256"] != projected["sharing"]["payload_sha256"]
    assert project_run(projected) == projected
    assert projected["summary"] == run["summary"]
    tampered = copy.deepcopy(projected)
    tampered["future_field"] = PRIVATE
    with pytest.raises(ValueError, match="outside its sharing profile"):
        project_run(tampered)


def test_text_sharing_is_explicit_and_does_not_include_context_or_raw_evidence():
    run = rich_run()
    projected = project_run(run, share_query_text=True)
    q = projected["queries"][0]
    assert q["text"] == PRIVATE
    assert q["history"] == run["queries"][0]["history"]
    assert "context" not in q
    assert q["evidence"] is None
    assert "future_field" not in q
    assert transfer_preview(projected)["query_text_and_history"] is True


def test_comparisons_and_critical_failed_attempts_survive_projection():
    base = fake_run([1.0] * 30)
    candidate = fake_run([1.0] * 25 + [0.0] * 5, latency=2)
    for run in (base, candidate):
        run["timing"]["repeats"] = 2
        for query in run["queries"]:
            query["repeat_ids"] = [query["ids"], query["ids"]]
            query["family_id"] = "family-" + str(int(query["query_id"][1:]) // 2)
            query["permission_checked"] = True
            query["forbidden_ids"] = []
    q = candidate["queries"][0]
    q["attempt_count"] = 2
    q["attempts"] = [
        {
            "attempt": 1,
            "status": "failed",
            "result": {
                "error": {"code": "timeout", "message": PRIVATE},
                "repeat_ids": [["wrong"]],
                "future_field": PRIVATE,
            },
        }
    ]
    policy = parse_policy(
        {"min_queries": 1, "critical_queries": [{"query_id": "q0", "doc_id": "a", "max_rank": 1}]}
    )
    raw = compare_runs(base, candidate, policy)
    shared = compare_runs(project_run(base), project_run(candidate), policy)
    for key in ("verdict", "reasons", "metrics", "latency", "critical_queries", "n_families"):
        assert shared[key] == raw[key], key
    assert shared["verdict"] == "fail"
    assert shared["critical_queries"][0]["failed_executions"] == 1
    assert PRIVATE not in json.dumps(project_run(candidate))


def test_client_sends_only_projection_and_preview_matches_payload_size():
    sent = []

    def send(method, url, headers, body):
        sent.append(body)
        return 201, b'{"run_id":"r"}'

    run = rich_run()
    CloudClient(CloudTarget("https://example.test", "token", "w", "p"), send).upload_run(run)
    assert PRIVATE.encode() not in sent[0]
    assert len(sent[0]) == transfer_preview(run)["bytes"]


def test_preview_has_no_credentials_or_network_and_never_overwrites(tmp_path, monkeypatch):
    for variable in ("VECTRAUDIT_CLOUD_URL", "VECTRAUDIT_CLOUD_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(rich_run()))
    before = path.read_bytes()
    out = tmp_path / "shared.json"
    runner = CliRunner()
    result = runner.invoke(app, ["cloud", "preview", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert PRIVATE not in result.output
    assert PRIVATE not in out.read_text()
    assert json.loads(result.output)["profile"] == "release-metadata-v1"
    assert path.read_bytes() == before
    assert runner.invoke(app, ["cloud", "preview", str(path), "--out", str(path)]).exit_code == 1
    assert runner.invoke(app, ["cloud", "preview", str(path), "--out", str(out)]).exit_code == 1


def test_server_projects_legacy_uploads_and_rejects_extra_projected_fields(world):
    from tests.test_cloud_runs import RUNS, client_for

    run = rich_run()
    response = world.client.post(RUNS, json=run, headers=world.as_("acme_runner"))
    assert response.status_code == 201, response.text
    stored = client_for(world, "acme_runner").get_run_file(run["run_id"])
    assert PRIVATE not in json.dumps(stored)
    forged = project_run({**run, "run_id": "another"})
    forged["future_field"] = PRIVATE
    response = world.client.post(RUNS, json=forged, headers=world.as_("acme_runner"))
    assert response.status_code == 422


def test_projection_keeps_what_changed_without_text_or_hosts():
    run = fake_run([1.0] * 30)
    run["search"]["params"] = {"nprobe": 16, "rerank": True, "reranker": "production-v2"}
    run["backend"]["config"] = {
        "adapter": "http",
        "name": "Support search",
        "sha256": "a" * 64,
        "pipeline_version": "v7",
        "destination": "https://internal.example",
        "code": {"module": "private_pkg.search", "sha256": "b" * 64},
    }
    projected = project_run(run)
    params = projected["search"]["params"]
    assert params["nprobe"] == 16
    assert params["rerank"] is True
    assert params["reranker"].startswith("sha256:")
    assert "production-v2" not in json.dumps(projected)
    assert projected["backend"]["config"] == {
        "adapter": "http",
        "name": "Support search",
        "sha256": "a" * 64,
        "pipeline_version": "v7",
    }
    assert "internal.example" not in json.dumps(projected)
    assert "private_pkg" not in json.dumps(projected)
    # Re-validating a projection keeps fingerprints; raw text smuggled in is refused.
    assert project_run(projected) == projected
    tampered = copy.deepcopy(projected)
    tampered["search"]["params"]["reranker"] = "production-v2"
    with pytest.raises(ValueError, match=r"fingerprint|digest"):
        project_run(tampered)


@pytest.mark.parametrize("regressed", [False, True])
def test_every_surface_reports_the_same_decision_and_changes(regressed):
    base = fake_run([1.0] * 30)
    cand = fake_run([0.5] * 30 if regressed else [1.0] * 30)
    base["search"]["params"] = {"depth": 50, "reranker": "none"}
    cand["search"]["params"] = {"depth": 20, "reranker": "phrase"}
    cand["backend"]["config"] = {"name": "candidate", "destination": "https://host.example"}
    policy = parse_policy({"metrics": {"ndcg": {"max_drop": 0.01}}, "min_queries": 30})
    local = compare_runs(base, cand, policy)
    # `check cloud:<name>` and the hosted check both compare projections on both sides.
    shared = compare_runs(project_run(base), project_run(cand), policy)
    assert shared["verdict"] == local["verdict"] == ("fail" if regressed else "pass")
    assert shared["reasons"] == local["reasons"]
    changed = {d["path"] for d in shared["config_diff"]}
    assert {"search.params.depth", "search.params.reranker"} <= changed
    assert not any("destination" in path for path in changed)
