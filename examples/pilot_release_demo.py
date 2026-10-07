"""Synthetic release rehearsal; requires vectraudit and httpx, no models or services."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from vectraudit.pipelines.web import create_workbench

ADAPTER = """\
from collections import defaultdict
from vectraudit.pipelines.models import Hit, PipelineResult, StageTrace

calls = defaultdict(int)
VARIANT = {variant!r}

def retrieve(request):
    calls[request.query_id] += 1
    critical = request.query_id == "account-recovery"
    ids = ["account-guide", "unrelated"]
    if VARIANT == "baseline" and not critical:
        ids.reverse()
    retrieved = [Hit(id=id) for id in ids]
    # Simulated intermittent reranker defect: only the second critical execution fails.
    if VARIANT == "candidate" and critical and calls[request.query_id] % 2 == 0:
        ids = ["unrelated"]
    delivered = [Hit(id=id) for id in ids]
    return PipelineResult(hits=delivered, stages=[
        StageTrace(name="retrieve", kind="keyword", hits=retrieved, latency_ms=0),
        StageTrace(name="rerank", kind="rerank", parents=["retrieve"],
                   hits=delivered, latency_ms=0),
    ])
"""


def rehearse(root: Path) -> None:
    """Save reproducible fail/fix reports and verify guided/exported verdict parity."""
    root.mkdir(parents=True, exist_ok=False)
    registered = root / "registered"
    registered.mkdir()
    variants = ("baseline", "candidate", "fixed")
    for variant in variants:
        (registered / f"{variant}.py").write_text(ADAPTER.format(variant=variant))
    manifest = registered / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "unused.jsonl",
                "corpus_version": "synthetic-support-v1",
                "candidates": [
                    {
                        "name": variant,
                        "pipeline_version": f"synthetic-{variant}-v1",
                        "adapter": {"kind": "python", "target": f"{variant}:retrieve"},
                    }
                    for variant in variants
                ],
            }
        )
    )
    workspace = root / "workspace"
    with TestClient(create_workbench(workspace, manifest=manifest)) as client:
        assert client.get("/").status_code == 200
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]

        def post(route: str, body: dict, status: int = 200) -> dict:
            response = client.post(f"/api/guided/{route}", json=body)
            assert response.status_code == status, response.text
            return response.json()

        rows = [
            {
                "query_id": "account-recovery" if i == 0 else f"question-{i}",
                "text": "How do I recover my account?",
                "relevant": {"account-guide": 2},
            }
            for i in range(40)
        ]
        draft = post(
            "drafts", {"name": "Synthetic pilot", "dataset": "\n".join(map(json.dumps, rows))}
        )
        post("snapshots", {"draft_id": draft["id"], "revision": draft["revision"]})
        policy = post(
            "policies",
            {
                "name": "Synthetic pilot policy",
                "policy": {
                    "metrics": {"ndcg": {"max_drop": 0}},
                    "min_queries": 30,
                    "critical_queries": [
                        {"query_id": "account-recovery", "doc_id": "account-guide", "max_rank": 1}
                    ],
                },
            },
        )
        runs = {}
        for i, variant in enumerate(variants):
            job = post(
                "jobs",
                {
                    "draft_id": draft["id"],
                    "revision": draft["revision"],
                    "pipeline": f"configured-{i}",
                    "label": variant,
                    "k": 2,
                    "repeats": 2,
                },
                202,
            )
            assert client.app.state.guided.jobs.get(job["id"]).wait(30)
            state = client.get(f"/api/guided/jobs/{job['id']}").json()
            assert state["status"] == "done", state
            runs[variant] = state["run"]
        guided = {}
        for variant, verdict in [("candidate", "fail"), ("fixed", "pass")]:
            comparison = post(
                "compare",
                {
                    "baseline": runs["baseline"]["run_id"],
                    "candidate": runs[variant]["run_id"],
                    "policy": policy["id"],
                },
            )
            guided[variant] = comparison["check"]
            assert guided[variant]["verdict"] == verdict
            assert guided[variant]["metrics"]["ndcg"]["delta"] > 0
        rule = guided["candidate"]["critical_queries"][0]
        assert (rule["executions"], rule["failed_executions"]) == (2, 1)
        assert rule["failures"][0]["repeat"] == 2
        exported = post(
            "releases",
            {
                "baseline": runs["baseline"]["run_id"],
                "candidate": runs["fixed"]["run_id"],
                "policy": policy["id"],
                "name": "Synthetic pilot",
                "repeats": 2,
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

    # Each directory represents an independent customer checkout. Only exported files
    # and the explicitly customer-owned adapter are present, with no source-tree imports.
    release = "retrieval/release/release.json"
    approved = root / "baseline" / release
    for variant in variants:
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
        (checkout / "fixed.py").write_text(ADAPTER.format(variant=variant))
        cli(checkout, "release", "validate", release)
        cli(checkout, "release", "run", release, "--approved", str(approved), "--out", "run.json")

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
        assert checked["critical_queries"] == guided[variant]["critical_queries"]
        assert checked["policy"] == guided[variant]["policy"]
        assert checked["release"]["policy"]["sha256"] == policy["sha256"]
        assert checked["metrics"]["ndcg"]["delta"] > 0
        # Exact guided artifacts must also yield the same exported CLI verdict.
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

    # A missing benchmark must fail clearly rather than falling back to demo data.
    benchmark = root / "fixed/retrieval/release/benchmark.jsonl"
    contents = benchmark.read_bytes()
    benchmark.unlink()
    try:
        cli(root / "fixed", "release", "validate", release, expected=1)
    finally:
        benchmark.write_bytes(contents)
    print(f"Synthetic rehearsal passed: {root}")
    print("candidate/report.html: improved mean nDCG, critical repeat 2 fails (exit 3)")
    print("fixed/report.html: both critical executions pass (exit 0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; never overwritten"
    )
    rehearse(parser.parse_args().out.resolve())
