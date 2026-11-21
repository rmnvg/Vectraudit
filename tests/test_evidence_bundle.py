"""Evidence bundles a reviewer outside the team can verify (Phase 50)."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.bundle import BundleError, build_bundle, verify_bundle
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.exceptions import apply_exceptions
from vectraudit.evaluation.policy import DEFAULT_POLICY, parse_policy
from vectraudit.evaluation.runfile import save_run
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

SECRET_QUESTION = "how do refunds work for the platinum plan"
SECRET_DOC = "Platinum refunds are processed within nine days"
SECRET_PROMPT = "You are a careful assistant for Acme"
DOCS = [
    {"doc_id": f"d{i}", "text": f"{SECRET_DOC} topic {i} item {i % 4}" if i == 0 else f"topic {i}"}
    for i in range(30)
]
ROWS = [
    {
        "query_id": f"q{i}",
        "text": SECRET_QUESTION if i == 0 else f"topic {i}",
        "relevant": {f"d{i}": 1},
    }
    for i in range(24)
]


def runs(regress: bool = False):
    data = parse_dataset("\n".join(json.dumps(r) for r in ROWS))
    options = {"k": 3, "warmup": 0, "params": {"prompt": SECRET_PROMPT}}
    base = execute_pipeline(data, hybrid_pipeline(DOCS), PipelineOptions(label="base", **options))
    # The regressing candidate indexes filler text for half the relevant documents.
    docs = [
        {**d, "text": "unrelated filler"} if regress and 1 <= i <= 12 else d
        for i, d in enumerate(DOCS)
    ]
    cand = execute_pipeline(data, hybrid_pipeline(docs), PipelineOptions(label="cand", **options))
    return base, cand


def write(tmp_path: Path, regress: bool = False) -> tuple[Path, Path]:
    base, cand = runs(regress)
    b, c = tmp_path / "base.json", tmp_path / "cand.json"
    save_run(base, b)
    save_run(cand, c)
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    return b, c


def check(tmp_path: Path, *extra: str, regress: bool = False):
    b, c = write(tmp_path, regress)
    out = tmp_path / "bundle.zip"
    result = CliRunner().invoke(app, ["check", str(b), str(c), "--bundle", str(out), *extra])
    return result, out, b, c


def rewrite(src: Path, dst: Path, change) -> None:
    """Copy a bundle, letting ``change`` edit the {name: bytes} entries."""
    with zipfile.ZipFile(src) as z:
        entries = {i.filename: z.read(i) for i in z.infolist()}
    change(entries)
    with zipfile.ZipFile(dst, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


def test_check_writes_a_bundle_of_ids_and_numbers(tmp_path):
    result, out, b, _ = check(tmp_path, "--bundle-runs")
    assert result.exit_code == 0, result.output
    assert "manifest SHA-256" in result.output
    with zipfile.ZipFile(out) as z:
        names = sorted(z.namelist())
        blob = b"".join(z.read(n) for n in names)
        manifest = json.loads(z.read("manifest.json"))
    assert names == [
        "check.json",
        "dataset.json",
        "exceptions.json",
        "inputs.json",
        "manifest.json",
        "policy.json",
        "runs/baseline.header.json",
        "runs/baseline.json",
        "runs/candidate.header.json",
        "runs/candidate.json",
        "versions.json",
    ]
    for secret in (SECRET_QUESTION, SECRET_DOC, SECRET_PROMPT):
        assert secret.encode() not in blob
    assert manifest["verdict"] == "pass"
    assert manifest["includes_runs"] is True
    assert {f["path"] for f in manifest["files"]} == set(names) - {"manifest.json"}
    inputs = json.loads(zipfile.ZipFile(out).read("inputs.json"))
    from vectraudit.evaluation.bundle import file_sha256

    assert inputs["baseline_file"] == file_sha256(b)
    assert inputs["policy_file"] is None


def test_a_failing_check_still_writes_its_bundle(tmp_path):
    result, out, _, _ = check(tmp_path, regress=True)
    assert result.exit_code == 3
    assert json.loads(zipfile.ZipFile(out).read("manifest.json"))["verdict"] == "fail"
    assert "runs/baseline.json" not in zipfile.ZipFile(out).namelist()


def test_bundle_runs_needs_a_bundle(tmp_path):
    b, c = write(tmp_path)
    result = CliRunner().invoke(app, ["check", str(b), str(c), "--bundle-runs"])
    assert result.exit_code == 1
    assert "--bundle-runs needs --bundle" in result.output


def test_a_bundle_with_runs_reproduces_its_verdict(tmp_path):
    result, out, _, _ = check(tmp_path, "--bundle-runs", regress=True)
    assert result.exit_code == 3
    report = verify_bundle(out)
    assert report["status"] == "verified"
    assert report["verdict"] == "fail"
    verdict = next(c for c in report["checks"] if c["check"] == "verdict")
    assert "reproduces fail" in verdict["detail"]
    ok = CliRunner().invoke(app, ["verify", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "Verified" in ok.output


def test_it_reproduces_in_another_process(tmp_path):
    """Phase acceptance: a bundle with runs reproduces its verdict on another machine."""
    _, out, _, _ = check(tmp_path, "--bundle-runs", regress=True)
    moved = tmp_path / "elsewhere" / "evidence.zip"
    moved.parent.mkdir()
    moved.write_bytes(out.read_bytes())
    done = subprocess.run(
        [sys.executable, "-m", "vectraudit.cli", "verify", str(moved)],
        capture_output=True,
        text=True,
        cwd=moved.parent,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_without_runs_hashes_are_consistent_but_nothing_is_verified(tmp_path):
    _, out, b, c = check(tmp_path)
    report = verify_bundle(out)
    assert report["status"] == "hashes_consistent"
    cli = CliRunner().invoke(app, ["verify", str(out)])
    assert cli.exit_code == 2
    assert "Hashes consistent, verdict not reproduced" in cli.output
    # Supplying the original run files re-runs it.
    again = verify_bundle(out, baseline=b, candidate=c, dataset=tmp_path / "queries.jsonl")
    assert again["status"] == "verified"
    assert {x["check"] for x in again["checks"] if x["status"] == "ok"} >= {
        "baseline file",
        "candidate file",
        "dataset",
        "verdict",
    }


def test_one_changed_byte_fails_and_names_the_file(tmp_path):
    """Phase acceptance: a bundle with one byte changed fails verification naming the file."""
    _, out, _, _ = check(tmp_path, "--bundle-runs")
    bad = tmp_path / "bad.zip"

    def flip(entries):
        data = bytearray(entries["runs/candidate.json"])
        at = data.index(b'"label"')
        data[at + 1] = ord("L")
        entries["runs/candidate.json"] = bytes(data)

    rewrite(out, bad, flip)
    report = verify_bundle(bad)
    assert report["status"] == "failed"
    assert [c["detail"] for c in report["checks"]] == [
        "runs/candidate.json changed: its SHA-256 differs from the manifest"
    ]
    cli = CliRunner().invoke(app, ["verify", str(bad)])
    assert cli.exit_code == 1
    assert "runs/candidate.json changed" in cli.output


def test_missing_and_added_files_fail(tmp_path):
    _, out, _, _ = check(tmp_path)
    gone, added = tmp_path / "gone.zip", tmp_path / "added.zip"
    rewrite(out, gone, lambda e: e.pop("exceptions.json"))
    rewrite(out, added, lambda e: e.update({"notes.txt": b"trust me"}))
    assert "exceptions.json is listed in the manifest but missing" in [
        c["detail"] for c in verify_bundle(gone)["checks"]
    ]
    assert "notes.txt is in the bundle but not in the manifest" in [
        c["detail"] for c in verify_bundle(added)["checks"]
    ]


def test_a_rewritten_verdict_with_a_rewritten_manifest_is_caught_by_rerunning(tmp_path):
    _, out, _, _ = check(tmp_path, "--bundle-runs", regress=True)
    forged = tmp_path / "forged.zip"

    def forge(entries):
        import hashlib

        result = json.loads(entries["check.json"])
        result["verdict"] = "pass"
        entries["check.json"] = json.dumps(result).encode()
        manifest = json.loads(entries["manifest.json"])
        manifest["verdict"] = "pass"
        for f in manifest["files"]:
            f["sha256"] = hashlib.sha256(entries[f["path"]]).hexdigest()
        entries["manifest.json"] = json.dumps(manifest).encode()

    rewrite(out, forged, forge)
    report = verify_bundle(forged)
    assert report["status"] == "failed"
    assert any("gives fail, not pass" in c["detail"] for c in report["checks"])
    # The manifest hash differs from the one printed when the bundle was written.
    assert report["manifest_sha256"] != verify_bundle(out)["manifest_sha256"]


def test_the_wrong_run_file_is_named(tmp_path):
    _, out, _, c = check(tmp_path)
    report = verify_bundle(out, baseline=c, candidate=c)
    assert report["status"] == "failed"
    assert any("is not the file the check was run on" in x["detail"] for x in report["checks"])


def test_exceptions_are_recorded_and_reapplied(tmp_path):
    base, cand = runs(regress=True)
    policy = parse_policy(
        {
            "metrics": {"recall": {"max_drop": 0.01}},
            "exceptions": {"allow": ["metric"], "approver": "engineer"},
        }
    )
    plain = compare_runs(base, cand, policy)
    assert plain["verdict"] == "fail"
    accepted = apply_exceptions(
        plain,
        [
            {
                "id": f"exc-{n}",
                "rule": rule,
                "accepted_by": "reviewer@example.com",
                "reason": "known drop, fixed next sprint",
                "expires_at": "2026-12-31T00:00:00+00:00",
            }
            for n, rule in enumerate(
                r["rule"] for r in plain["reasons"] if r.get("verdict") == "fail"
            )
        ],
        policy,
    )
    assert accepted["verdict"] == "accepted"
    out = tmp_path / "b.zip"
    build_bundle(accepted, base, cand, policy, out, include_runs=True)
    recorded = json.loads(zipfile.ZipFile(out).read("exceptions.json"))
    assert {e["accepted_by"] for e in recorded} == {"reviewer@example.com"}
    report = verify_bundle(out)
    assert report["status"] == "verified"
    assert report["verdict"] == "accepted"


def test_a_bundle_that_would_not_reproduce_is_refused(tmp_path):
    base, cand = runs()
    result = compare_runs(base, cand, DEFAULT_POLICY)
    with pytest.raises(BundleError, match="verdict of record"):
        build_bundle({**result, "verdict": "fail"}, base, cand, DEFAULT_POLICY, tmp_path / "x.zip")


def test_unsafe_or_foreign_zips_are_refused(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../manifest.json", b"{}")
    with pytest.raises(BundleError, match="unsafe"):
        verify_bundle(evil)
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("manifest.json", b'{"format": "something"}')
    with pytest.raises(BundleError, match="not an evidence bundle"):
        verify_bundle(other)
    (tmp_path / "plain.zip").write_text("not a zip")
    cli = CliRunner().invoke(app, ["verify", str(tmp_path / "plain.zip")])
    assert cli.exit_code == 1
    assert "Not verified" in cli.output


def test_a_policy_file_is_hashed_and_its_rules_round_trip(tmp_path):
    rules = {
        "version": 1,
        "metrics": {"ndcg": {"max_drop": 0.01, "min": 0.3}, "recall": {"max_drop": 0.02}},
        "latency": {"p95_max_increase_pct": 30, "p95_max_ms": 200},
        "min_queries": 5,
        "max_regressed_queries": 3,
        "primary_metric": "ndcg",
        "cohorts": {"min_queries": 3, "control": "holm", "metrics": {"recall": {"max_drop": 0.05}}},
        "critical_queries": [{"query_id": "q1", "doc_id": "d1", "max_rank": 3}],
        "exceptions": {"allow": ["metric", "latency"], "approver": "engineer", "max_days": 30},
        "forbidden_stages": ["rerank"],
        "prompt_changes": "fail",
    }
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(rules))
    result, out, _, _ = check(tmp_path, "--policy", str(policy_file), "--bundle-runs")
    assert result.exit_code == 4, result.output  # prompt rule without prompt versions
    from vectraudit.evaluation.bundle import file_sha256

    inputs = json.loads(zipfile.ZipFile(out).read("inputs.json"))
    assert inputs["policy_file"] == file_sha256(policy_file)
    assert verify_bundle(out)["status"] == "verified"
