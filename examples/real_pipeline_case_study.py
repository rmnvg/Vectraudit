"""Measure actual BM25/filter changes in a fictional support-search application."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from vectraudit.pipelines.web import create_workbench

INPUTS = Path(__file__).resolve().with_name("reference_search")
VARIANTS = ("baseline", "candidate", "fixed")


def revision(variant: str) -> str:
    """Produce a standalone application revision, without modifying the source fixture."""
    if variant not in VARIANTS:
        raise ValueError("Unknown revision")
    source = (INPUTS / "app.py").read_text(encoding="utf-8")
    marker = 'SEARCH_MODE = "baseline"'
    assert source.count(marker) == 1
    return source.replace(marker, f'SEARCH_MODE = "{variant}"')


def run_case_study(root: Path) -> None:
    """Create guided runs, export checks, and verify independent CLI executions."""
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((INPUTS / "inventory.json").read_text())
    for name, digest in inventory["sha256"].items():
        assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == digest, name
    corpus = (INPUTS / "corpus.jsonl").read_text()
    dataset = (INPUTS / "queries.jsonl").read_text()
    registered = root / "registered"
    registered.mkdir()
    (registered / "corpus.jsonl").write_text(corpus)
    manifest = registered / "manifest.json"
    for variant in VARIANTS:
        (registered / f"reference_{variant}.py").write_text(revision(variant))
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(INPUTS / "queries.jsonl"),
                "corpus_version": inventory["sha256"]["corpus.jsonl"],
                "k": 2,
                "repeats": 3,
                "candidates": [
                    {
                        "name": variant,
                        "pipeline_version": hashlib.sha256(revision(variant).encode()).hexdigest(),
                        "adapter": {"kind": "python", "target": f"reference_{variant}:retrieve"},
                    }
                    for variant in VARIANTS
                ],
            },
            indent=2,
        )
    )
    workspace = root / "workspace"
    with TestClient(create_workbench(workspace, manifest=manifest)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]

        def post(route: str, body: dict, status: int = 200) -> dict:
            response = client.post(f"/api/guided/{route}", json=body)
            assert response.status_code == status, response.text
            return response.json()

        draft = post(
            "drafts",
            {
                "name": "Reference support search — labels pending human review",
                "dataset": dataset,
                "corpus": corpus,
            },
        )
        post("snapshots", {"draft_id": draft["id"], "revision": draft["revision"]})
        policy = post(
            "policies",
            {
                "name": "Reference search regression policy",
                "policy": json.loads((INPUTS / "policy.json").read_text()),
                "note": "Engineering example only; proposed labels await human review.",
            },
        )
        runs = {}
        for i, variant in enumerate(VARIANTS):
            job = post(
                "jobs",
                {
                    "draft_id": draft["id"],
                    "revision": draft["revision"],
                    "pipeline": f"configured-{i}",
                    "label": variant,
                    "k": 2,
                    "repeats": 3,
                },
                202,
            )
            assert client.app.state.guided.jobs.get(job["id"]).wait(30)
            state = client.get(f"/api/guided/jobs/{job['id']}").json()
            assert state["status"] == "done", state
            runs[variant] = state["run"]
        guided = {}
        for variant, expected in [("candidate", "fail"), ("fixed", "pass")]:
            result = post(
                "compare",
                {
                    "baseline": runs["baseline"]["run_id"],
                    "candidate": runs[variant]["run_id"],
                    "policy": policy["id"],
                },
            )
            guided[variant] = result["check"]
            assert guided[variant]["verdict"] == expected
        assert guided["candidate"]["metrics"]["ndcg"]["delta"] < 0
        assert guided["candidate"]["critical_queries"][0]["failed_executions"] == 3
        exported = post(
            "releases",
            {
                "baseline": runs["baseline"]["run_id"],
                "candidate": runs["fixed"]["run_id"],
                "policy": policy["id"],
                "name": "Reference support search",
                "repeats": 3,
            },
        )

    def cli(cwd: Path, *args: str, expected: int = 0) -> None:
        result = subprocess.run(
            [sys.executable, "-I", "-m", "vectraudit", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr

    release = "retrieval/release/release.json"
    approved = root / "baseline" / release
    fresh = {}
    for variant in VARIANTS:
        checkout = root / variant
        cli(
            root,
            "workbench",
            "export",
            exported["id"],
            "--workspace",
            str(workspace),
            "--out",
            str(checkout),
        )
        # The exported Python entry point is the same; source bytes differ per checkout.
        (checkout / "reference_fixed.py").write_text(revision(variant))
        (checkout / "corpus.jsonl").write_text(corpus)
        cli(checkout, "release", "validate", release)
        cli(
            checkout,
            "release",
            "run",
            release,
            "--approved",
            str(approved),
            "--label",
            variant,
            "--out",
            "run.json",
        )
        fresh[variant] = json.loads((checkout / "run.json").read_text())
        assert [q["ids"] for q in fresh[variant]["queries"]] == [
            q["ids"] for q in runs[variant]["queries"]
        ]
    assert len({r["backend"]["config"]["code"]["sha256"] for r in fresh.values()}) == 3
    for variant, run in fresh.items():
        assert (
            run["backend"]["config"]["code"]["sha256"]
            == hashlib.sha256(revision(variant).encode()).hexdigest()
        )

    comparisons = {}
    for variant, code in [("candidate", 3), ("fixed", 0)]:
        checkout = root / variant
        cli(
            checkout,
            "release",
            "check",
            str(approved),
            str(root / "baseline/run.json"),
            "run.json",
            "--json",
            "report.json",
            "--html",
            "report.html",
            "--markdown",
            "report.md",
            "--no-github-summary",
            expected=code,
        )
        checked = json.loads((checkout / "report.json").read_text())
        assert checked["verdict"] == guided[variant]["verdict"]
        assert checked["policy"] == guided[variant]["policy"]
        assert checked["critical_queries"] == guided[variant]["critical_queries"]
        assert checked["release"]["policy"]["sha256"] == policy["sha256"]
        # Also compare exactly the artifacts shown by the guided UI.
        cli(
            checkout,
            "release",
            "check",
            str(approved),
            str(workspace / f"{runs['baseline']['run_id']}.run.json"),
            str(workspace / f"{runs[variant]['run_id']}.run.json"),
            "--no-github-summary",
            expected=code,
        )
        comparisons[variant] = {
            "verdict": checked["verdict"],
            "metrics": checked["metrics"],
            "critical_queries": checked["critical_queries"],
        }

    evidence = {}
    for variant, run in fresh.items():
        critical = next(q for q in run["queries"] if q["query_id"] == "recover-account")
        evidence[variant] = critical["evidence"]
    (root / "investigation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for name in ("inventory.json", "judgments.json"):
        shutil.copyfile(INPUTS / name, root / name)
    summary = {
        "label_status": "pending_human_review",
        "data": "fictional reference application",
        "input_inventory": inventory,
        "policy_sha256": policy["sha256"],
        "automated_elapsed_seconds": time.perf_counter() - started,
        "human_setup_minutes": None,
        "human_labeling_minutes": None,
        "second_engineer_reproduced": False,
        "runs": {
            name: {
                "run_id": run["run_id"],
                "code": run["backend"]["config"]["code"],
                "summary": run["summary"],
            }
            for name, run in fresh.items()
        },
        "comparisons": comparisons,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Reference search case study passed: {root}")
    print("candidate/report.html: shallow retrieval loses current guidance; exit 3")
    print("fixed/report.html: filter before ranking restores quality; exit 0")
    print(f"Inspect in browser: vectraudit workbench serve {workspace} --manifest {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    run_case_study(parser.parse_args().out.resolve())
