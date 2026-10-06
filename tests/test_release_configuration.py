"""One release policy everywhere, and exported release checks that run from a clean checkout."""

import io
import json
import textwrap
import zipfile

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.test_guided_workbench import save
from vectraudit.cli import app
from vectraudit.evaluation.policy import parse_policy, policy_sha256
from vectraudit.pipelines.models import Hit, PipelineResult
from vectraudit.pipelines.release import (
    ReleaseError,
    ReleasePipeline,
    check_release,
    load_release,
    open_release_pipeline,
    render_release_workflow,
    run_release,
    update_release,
    validate_release,
)
from vectraudit.pipelines.web import create_workbench

LENIENT = {"metrics": {"ndcg": {"max_drop": 0.1}}, "min_queries": 30}
STRICT = {"metrics": {"ndcg": {"max_drop": 0.001}}, "min_queries": 30}
ROWS = [{"query_id": f"q{i}", "text": "invoice", "relevant": {"a": 2}} for i in range(35)]
CORPUS = "\n".join(
    json.dumps({"doc_id": d, "text": text})
    for d, text in [("a", "invoice export guide"), ("b", "password reset")]
)
runner = CliRunner()


def _client(path, **kwargs):
    client = TestClient(create_workbench(path, **kwargs))
    client.__enter__()
    client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
    return client


@pytest.fixture
def client(tmp_path):
    client = _client(tmp_path / "workspace")
    yield client
    client.__exit__(None, None, None)


def post(client, path, body, status=200):
    response = client.post("/api/guided/" + path, json=body)
    assert response.status_code == status, response.text
    return response.json()


def record(client, draft, **changes):
    body = {"draft_id": draft["id"], "revision": draft["revision"], "corpus": CORPUS, "k": 1}
    started = post(client, "jobs", {**body, **changes}, status=202)
    assert client.app.state.guided.jobs.get(started["id"]).wait(10)
    state = client.get(f"/api/guided/jobs/{started['id']}").json()
    assert state["status"] == "done", state
    return state["run"]


@pytest.fixture
def lossy(monkeypatch):
    """The phrase reranker loses q0's relevant document; everything else is unchanged."""

    def pipeline(request):
        lost = request.params["reranker"] != "none" and request.query_id == "q0"
        return PipelineResult(hits=[Hit(id="b" if lost else "a")])

    monkeypatch.setattr("vectraudit.pipelines.guided.hybrid_pipeline", lambda _: pipeline)


def test_policies_are_versioned_validated_and_exploratory_is_labelled(tmp_path):
    startup = tmp_path / "ci-policy.json"
    startup.write_text(json.dumps(STRICT))
    client = _client(tmp_path / "w", policy=startup)
    listing = client.get("/api/guided/policies").json()
    assert listing["exploratory"]["source"] == "exploratory"
    assert listing["startup"]["sha256"] == policy_sha256(parse_policy(STRICT))
    first = post(client, "policies", {"name": "Release", "policy": LENIENT})
    assert post(client, "policies", {"name": "release ", "policy": LENIENT}) == first
    second = post(client, "policies", {"name": "Release", "policy": STRICT})
    assert (second["version"], second["supersedes"]) == (2, first["id"])
    assert second["sha256"] == listing["startup"]["sha256"]
    bad = client.post(
        "/api/guided/policies", json={"name": "Bad", "policy": {"metrics": {"ndcg": {}}}}
    )
    assert bad.status_code == 422
    assert "sets no limit" in bad.json()["detail"]
    latest = {p["id"]: p["latest"] for p in client.get("/api/guided/policies").json()["policies"]}
    assert latest == {first["id"]: False, second["id"]: True}
    # A policy edited on disk no longer matches its hash and is refused.
    path = tmp_path / "w" / ".workbench" / "policies" / f"{first['id']}.json"
    tampered = json.loads(path.read_text())
    tampered["policy"] = STRICT
    path.write_text(json.dumps(tampered))
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    one, two = record(client, draft), record(client, draft)
    compared = client.post(
        "/api/guided/compare",
        json={"baseline": one["run_id"], "candidate": two["run_id"], "policy": first["id"]},
    )
    assert compared.status_code == 409
    client.__exit__(None, None, None)


def test_every_surface_gives_the_same_verdict_and_a_stricter_policy_changes_all(
    client, lossy, tmp_path
):
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    baseline = record(client, draft, label="Baseline")
    original = record(client, draft, label="Broken", reranker="phrase")
    candidate = record(client, draft, label="Candidate", reranker="phrase")
    workspace = tmp_path / "workspace"
    base_file = workspace / f"{baseline['run_id']}.run.json"
    cand_file = workspace / f"{candidate['run_id']}.run.json"
    case = post(
        client,
        "cases",
        {"baseline": baseline["run_id"], "candidate": original["run_id"], "query_id": "q0"},
    )
    assert case["policy"]["source"] == "exploratory"

    for rules, expected, code in [(LENIENT, "pass", 0), (STRICT, "fail", 3)]:
        policy = post(client, "policies", {"name": "Release", "policy": rules})
        browser = post(
            client,
            "compare",
            {
                "baseline": baseline["run_id"],
                "candidate": candidate["run_id"],
                "policy": policy["id"],
            },
        )
        assert browser["check"]["verdict"] == expected
        assert browser["policy"]["sha256"] == policy["sha256"]

        policy_file = tmp_path / f"{expected}.json"
        policy_file.write_text(json.dumps(rules))
        cli = runner.invoke(
            app, ["check", str(base_file), str(cand_file), "--policy", str(policy_file)]
        )
        assert cli.exit_code == code, cli.output

        case = post(
            client,
            f"cases/{case['id']}/verify",
            {
                "revision": case["revision"],
                "candidate": candidate["run_id"],
                "policy": policy["id"],
            },
        )
        event = case["history"][-1]
        assert event["verdict"] == expected
        assert event["policy"]["sha256"] == policy["sha256"]
        assert event["verified"] is False  # q0 is still lost under either policy

    # The stricter version did not rewrite the earlier verification's evidence.
    versions = [e["policy"]["version"] for e in case["history"] if e["kind"] == "verification"]
    assert versions == [1, 2]

    exploratory = post(
        client, "compare", {"baseline": baseline["run_id"], "candidate": candidate["run_id"]}
    )
    assert exploratory["policy"]["source"] == "exploratory"


def _export(client, tmp_path, draft, baseline, candidate, *, include_corpus=True, rules=LENIENT):
    policy = post(client, "policies", {"name": "Release", "policy": rules})
    created = post(
        client,
        "releases",
        {
            "baseline": baseline["run_id"],
            "candidate": candidate["run_id"],
            "policy": policy["id"],
            "name": "Support search",
            "repeats": 1,
            "include_corpus": include_corpus,
        },
    )
    return policy, created


def test_exported_release_check_runs_from_a_clean_checkout(client, tmp_path):
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    baseline = record(client, draft, label="Baseline")
    candidate = record(client, draft, label="Candidate", reranker="phrase")
    policy, created = _export(client, tmp_path, draft, baseline, candidate)
    assert created["version"] == 1
    files = created["files"]
    assert set(files) == {
        "retrieval/release/release.json",
        "retrieval/release/benchmark.jsonl",
        "retrieval/release/policy.json",
        "retrieval/release/corpus.jsonl",
        "retrieval/release/README.md",
        ".github/workflows/retrieval-release-check.yml",
    }
    release = json.loads(files["retrieval/release/release.json"])
    assert release["policy"]["sha256"] == policy["sha256"]
    assert release["benchmark"]["sha256"] == candidate["dataset"]["sha256"]
    assert release["pipeline"]["params"]["reranker"] == "phrase"
    assert release["evidence"]["candidate_run"] == candidate["run_id"]
    assert (tmp_path / "workspace" / release["evidence"]["check_file"]).exists()
    workflow = files[".github/workflows/retrieval-release-check.yml"]
    assert "release check" in workflow
    assert "--approved" in workflow
    assert "secrets." in workflow

    # The zip download and the CLI export carry the same files.
    archive = client.get(f"/api/guided/releases/{created['id']}/export.zip")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        assert {n: z.read(n).decode() for n in z.namelist()} == files
    repo = tmp_path / "repo"
    exported = runner.invoke(
        app,
        [
            "workbench",
            "export",
            created["id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--out",
            str(repo),
        ],
    )
    assert exported.exit_code == 0, exported.output
    again = runner.invoke(
        app,
        [
            "workbench",
            "export",
            created["id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--out",
            str(repo),
        ],
    )
    assert again.exit_code == 1
    assert "skipped" in again.output

    path = repo / "retrieval/release/release.json"
    assert runner.invoke(app, ["release", "validate", str(path)]).exit_code == 0
    runs = tmp_path / "runs"
    for name in ("baseline", "candidate"):
        result = runner.invoke(
            app, ["release", "run", str(path), "--out", str(runs / f"{name}.run.json")]
        )
        assert result.exit_code == 0, result.output
    checked = runner.invoke(
        app,
        [
            "release",
            "check",
            str(path),
            str(runs / "baseline.run.json"),
            str(runs / "candidate.run.json"),
            "--json",
            str(runs / "result.json"),
            "--markdown",
            str(runs / "summary.md"),
        ],
    )
    assert checked.exit_code == 0, checked.output
    result = json.loads((runs / "result.json").read_text())
    assert result["release"]["policy"]["sha256"] == policy["sha256"]
    summary = (runs / "summary.md").read_text()
    assert f"Support search v1 · policy Release v1 ({policy['sha256'][:12]})" in summary
    # CI judged the same rules the workbench exported, and the same verdict on the same runs.
    same_runs = check_release(path, baseline, candidate)
    assert same_runs["verdict"] == "pass"

    # A reviewed, stricter policy becomes a new configuration version that CI then applies.
    (repo / "retrieval/release/policy.json").write_text(json.dumps(STRICT))
    stale = runner.invoke(app, ["release", "validate", str(path)])
    assert stale.exit_code == 1
    assert "changed" in stale.output
    assert runner.invoke(app, ["release", "update", str(path)]).exit_code == 0
    assert load_release(path).version == 2
    assert check_release(path, baseline, candidate)["verdict"] == "pass"  # nothing was lost
    assert check_release(path, baseline, candidate)["policy"] == parse_policy(STRICT).as_dict()


def test_export_refuses_exploratory_failing_and_unreproducible_comparisons(client, lossy, tmp_path):
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    baseline = record(client, draft)
    broken = record(client, draft, reranker="phrase")
    body = {"baseline": baseline["run_id"], "candidate": broken["run_id"], "name": "X"}
    assert client.post("/api/guided/releases", json=body).status_code == 422
    strict = post(client, "policies", {"name": "Strict", "policy": STRICT})
    refused = client.post("/api/guided/releases", json={**body, "policy": strict["id"]})
    assert refused.status_code == 422
    assert "is fail" in refused.json()["detail"]
    lenient = post(client, "policies", {"name": "Lenient", "policy": LENIENT})
    # The benchmark changed after the runs and was never frozen: nothing reproduces them.
    save(client, rows=ROWS[:-1], corpus=CORPUS, id=draft["id"], revision=draft["revision"])
    stale = client.post("/api/guided/releases", json={**body, "policy": lenient["id"]})
    assert stale.status_code == 422
    assert "Freeze" in stale.json()["detail"]


def test_heldout_exports_stay_separate_and_warn(client, tmp_path):
    rows = [
        *ROWS,
        {"query_id": "final", "text": "invoice", "relevant": {"a": 1}, "split": "heldout"},
    ]
    draft = save(client, rows=rows, corpus=CORPUS).json()
    snapshot = post(client, "snapshots", {"draft_id": draft["id"], "revision": draft["revision"]})
    runs = [record(client, draft, mode="heldout", confirm_heldout=True) for _ in range(2)]
    policy = post(
        client, "policies", {"name": "Final", "policy": {"metrics": {"ndcg": {"max_drop": 0.1}}}}
    )
    created = post(
        client,
        "releases",
        {
            "baseline": runs[0]["run_id"],
            "candidate": runs[1]["run_id"],
            "policy": policy["id"],
            "name": "Final",
            "include_corpus": True,
        },
    )
    release = json.loads(created["files"]["retrieval/release/release.json"])
    assert release["benchmark"]["split"] == "heldout"
    assert release["benchmark"]["n_queries"] == 1
    assert release["evidence"]["snapshot"] == snapshot["id"]
    assert any("held-out" in w for w in created["issues"]["warnings"])
    benchmark = created["files"]["retrieval/release/benchmark.jsonl"]
    assert '"q0"' not in benchmark
    # Mixing splits into the release benchmark is refused, even through update.
    repo = tmp_path / "repo"
    for rel, content in created["files"].items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    bench = repo / "retrieval/release/benchmark.jsonl"
    bench.write_text(bench.read_text() + json.dumps(ROWS[0]) + "\n")
    path = repo / "retrieval/release/release.json"
    with pytest.raises(ReleaseError, match="exactly one split"):
        update_release(path)


def test_python_pipelines_execute_each_checkout_and_ci_uses_the_approved_policy(tmp_path):
    def checkout(name, delivered):
        root = tmp_path / name
        (root / "retrieval/release").mkdir(parents=True)
        (root / "acme_search.py").write_text(
            textwrap.dedent(
                f"""
                from vectraudit.pipelines.models import Hit, PipelineResult

                def pipeline(request):
                    lost = request.query_id == "q0"
                    return PipelineResult(hits=[Hit(id={delivered!r} if lost else "a")])
                """
            )
        )
        return root

    manifest_root = checkout("registered", "a")
    (manifest_root / "queries.jsonl").write_text("")
    manifest = manifest_root / "pipelines.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "candidates": [
                    {
                        "name": "Acme",
                        "adapter": {"kind": "python", "target": "acme_search:pipeline"},
                    }
                ],
            }
        )
    )
    client = _client(tmp_path / "workspace", manifest=manifest)
    draft = save(client, rows=ROWS).json()
    runs = [record(client, draft, pipeline="configured-0", corpus="") for _ in range(2)]
    _, created = _export(client, tmp_path, draft, *runs, include_corpus=False)
    client.__exit__(None, None, None)
    release = json.loads(created["files"]["retrieval/release/release.json"])
    assert release["pipeline"] == {
        **release["pipeline"],
        "kind": "python",
        "target": "acme_search:pipeline",
        "python_path": ["../.."],
    }
    assert any("acme_search" in p for p in created["issues"]["prerequisites"])
    assert (
        "Install pipeline dependencies"
        in created["files"][".github/workflows/retrieval-release-check.yml"]
    )

    base, pr = checkout("base", "a"), checkout("pr", "b")  # the pull request loses q0
    for root in (base, pr):
        for rel, content in created["files"].items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(content)
    # The pull request also tries to loosen its own gate.
    (pr / "retrieval/release/policy.json").write_text(json.dumps({"metrics": {"ndcg": {"min": 0}}}))
    assert (
        runner.invoke(
            app, ["release", "update", str(pr / "retrieval/release/release.json")]
        ).exit_code
        == 0
    )

    approved = base / "retrieval/release/release.json"
    proposed = pr / "retrieval/release/release.json"
    assert validate_release(approved) == []
    baseline = run_release(approved)
    candidate = run_release(proposed, approved_path=approved)
    assert baseline["backend"]["config"]["code"] != candidate["backend"]["config"]["code"]
    assert baseline["queries"][0]["ids"] == ["a"]
    assert candidate["queries"][0]["ids"] == ["b"]
    result = check_release(approved, baseline, candidate, proposed_path=proposed)
    # 1/35 nDCG lost is inside the approved 0.1 tolerance: the proposal is reported, not used.
    assert result["policy"] == parse_policy(LENIENT).as_dict()
    assert result["release"]["proposed_changes"]
    assert any("proposes policy" in w for w in result["warnings"])


def test_missing_and_mismatched_artifacts_fail_clearly(tmp_path):
    with pytest.raises(ReleaseError, match="not found"):
        load_release(tmp_path / "missing.json")
    (tmp_path / "release.json").write_text('{"format": "vectraudit.release"}')
    issues = validate_release(tmp_path / "release.json")
    assert issues[0]["level"] == "error"
    assert "configuration:" in issues[0]["message"]
    http = ReleasePipeline(
        name="API", kind="http", target_env="RETRIEVAL_PIPELINE_URL", token_env="API_TOKEN"
    )
    with pytest.raises(ReleaseError, match="RETRIEVAL_PIPELINE_URL"):
        open_release_pipeline(http, tmp_path, env={})
    with pytest.raises(ReleaseError, match="API_TOKEN"):
        open_release_pipeline(http, tmp_path, env={"RETRIEVAL_PIPELINE_URL": "https://x.test"})
    workflow = render_release_workflow(http)
    assert "RETRIEVAL_PIPELINE_URL: ${{ secrets.RETRIEVAL_BASELINE_URL }}" in workflow
    assert "RETRIEVAL_PIPELINE_URL: ${{ secrets.RETRIEVAL_CANDIDATE_URL }}" in workflow
    assert "API_TOKEN: ${{ secrets.API_TOKEN }}" in workflow
    with pytest.raises(ValueError, match="module:function"):
        ReleasePipeline(name="X", kind="python", target="no_function")


def test_release_check_rejects_runs_of_another_benchmark(client, tmp_path):
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    baseline, candidate = record(client, draft), record(client, draft)
    _, created = _export(client, tmp_path, draft, baseline, candidate)
    repo = tmp_path / "repo"
    for rel, content in created["files"].items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    other = save(client, rows=ROWS[:31], corpus=CORPUS).json()
    a, b = record(client, other), record(client, other)
    result = check_release(repo / "retrieval/release/release.json", a, b)
    assert result["verdict"] == "not_comparable"
    assert any("not the approved benchmark" in r["message"] for r in result["reasons"])
