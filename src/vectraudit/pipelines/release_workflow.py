"""Workbench release policies and release configurations, exported for review.

Policies are immutable records: saving a policy under an existing name creates the next
version, so earlier comparisons and case verifications keep the exact rules they used.
Without a selected policy the workbench uses the built-in defaults, labelled exploratory;
those defaults can never be exported as a release contract.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import Field

from vectraudit import __version__
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, parse_policy, policy_sha256
from vectraudit.evaluation.runfile import load_run
from vectraudit.labelling.storage import task_lock
from vectraudit.pipelines.manifest import candidate_config
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.release import (
    PIPELINE_URL_ENV,
    RELEASE_DIR,
    BenchmarkRef,
    PolicyRef,
    ReleaseError,
    ReleaseFile,
    ReleasePipeline,
    _local_host,
    corpus_version,
    read_corpus,
    render_release_files,
)
from vectraudit.pipelines.workflows import LocalRecords, _event, digest, freeze_draft

if TYPE_CHECKING:
    from vectraudit.pipelines.guided import GuidedWorkspace

STARTUP = "startup"
EXPLORATORY: dict[str, Any] = {
    "id": None,
    "name": "Exploratory defaults",
    "version": None,
    "sha256": policy_sha256(DEFAULT_POLICY),
    "source": "exploratory",
}


def policy_ref(record: dict[str, Any]) -> dict[str, Any]:
    """What a comparison, case or release records about the policy that judged it."""
    return {k: record[k] for k in ("id", "name", "version", "sha256", "source")}


class PolicyIn(Contract):
    name: str = Field(min_length=1, max_length=128)
    policy: dict[str, Any]
    note: str = Field(default="", max_length=2000)


class PolicyStore:
    """Versioned local policies plus an optional read-only policy file given at startup."""

    def __init__(self, workspace: GuidedWorkspace, startup: Path | None = None) -> None:
        self.records = LocalRecords(workspace, "policies")
        self.startup: dict[str, Any] | None = None
        if startup is not None:
            # Fail at startup: a customer's CI policy must not silently fall back to defaults.
            raw = json.loads(startup.read_text(encoding="utf-8"))
            parsed = parse_policy(raw)
            self.startup = {
                "id": STARTUP,
                "name": startup.name,
                "version": 1,
                "sha256": policy_sha256(parsed),
                "source": "startup",
                "policy": raw,
                "effective": parsed.as_dict(),
                "note": f"Registered at startup from {startup}",
            }

    def resolve(self, identifier: str | None) -> tuple[Policy, dict[str, Any]]:
        if identifier is None:
            return DEFAULT_POLICY, dict(EXPLORATORY)
        record = self.record(identifier)
        policy = parse_policy(record["policy"])
        if policy_sha256(policy) != record["sha256"]:
            raise HTTPException(409, "Saved policy does not match its recorded hash")
        return policy, policy_ref(record)

    def record(self, identifier: str) -> dict[str, Any]:
        if identifier == STARTUP:
            if self.startup is None:
                raise HTTPException(404, "No policy was registered at startup")
            return self.startup
        return self.records.read(identifier)

    def save(self, body: PolicyIn) -> dict[str, Any]:
        """Validate with the CLI's parser; identical rules under one name are one version."""
        parsed = parse_policy(body.policy)
        sha = policy_sha256(parsed)
        name = body.name.strip()
        lineage = digest(name.casefold())[:32]
        with task_lock(self.records.path(lineage)):
            versions = [r for r in self.records.list() if r.get("lineage") == lineage]
            latest = max(versions, key=lambda r: int(r["version"]), default=None)
            if latest is not None and latest["sha256"] == sha:
                return latest
            return self.records.write(
                {
                    "id": uuid.uuid4().hex,
                    "format_version": 1,
                    "lineage": lineage,
                    "name": name,
                    "version": 1 if latest is None else int(latest["version"]) + 1,
                    "supersedes": None if latest is None else latest["id"],
                    "created_at": _event("policy")["at"],
                    "note": body.note,
                    "source": "workspace",
                    "sha256": sha,
                    "policy": body.policy,
                    "effective": parsed.as_dict(),
                }
            )

    def describe(self) -> dict[str, Any]:
        records = sorted(self.records.list(), key=lambda r: (r["name"], -int(r["version"])))
        latest = {r["lineage"]: r["id"] for r in reversed(records)}
        for r in records:
            r["latest"] = latest[r["lineage"]] == r["id"]
        return {
            "exploratory": {**EXPLORATORY, "effective": DEFAULT_POLICY.as_dict()},
            "startup": self.startup,
            "policies": records,
        }


class ReleaseIn(Contract):
    baseline: str
    candidate: str
    policy: str
    name: str = Field(min_length=1, max_length=128)
    repeats: int = Field(default=3, ge=1, le=100)
    include_corpus: bool = False


def _pipeline(
    workspace: GuidedWorkspace, run: dict[str, Any], draft: dict[str, Any] | None, include: bool
) -> tuple[ReleasePipeline, str | None, str | None, list[str], list[str], str | None]:
    """The adopted pipeline as portable configuration, with what the customer must supply."""
    config = run["backend"]["config"]
    params = run["search"]["params"]
    prerequisites: list[str] = []
    warnings: list[str] = []
    corpus_file = f"{RELEASE_DIR}/corpus.jsonl"
    if config.get("adapter") == "hybrid" and "sha256" not in config:
        # The built-in pipeline measured the draft's local corpus.
        measured = run["backend"].get("corpus_version")
        pipeline = ReleasePipeline(
            name="Local BM25 / hybrid", kind="hybrid", corpus="corpus.jsonl", params=params
        )
        text = (draft or {}).get("corpus") or ""
        from vectraudit.pipelines.guided import Document

        docs = [
            Document.model_validate_json(line).model_dump(exclude_none=True)
            for line in text.splitlines()
            if line.strip()
        ]
        same = bool(docs) and corpus_version(docs) == measured
        if include and not same:
            raise ValueError(
                "The saved draft's corpus differs from the one this comparison measured; "
                "export without the corpus or re-run both sides."
            )
        if not include:
            prerequisites.append(
                f"Place the corpus the comparison measured at `{corpus_file}` "
                f"(version {str(measured)[:12]}). The export does not copy corpora."
            )
        return pipeline, measured, text if include else None, prerequisites, warnings, None
    key = next(
        (
            k
            for k, c in workspace.registry.items()
            if candidate_config(c, workspace.manifest_root)["sha256"] == config.get("sha256")
        ),
        None,
    )
    if key is None:
        raise ValueError(
            "The candidate's pipeline is no longer registered with the configuration it ran "
            "with. Restart the workbench with the same --manifest and record the run again."
        )
    candidate = workspace.registry[key]
    adapter = candidate.adapter
    if adapter.kind == "python":
        module = str(adapter.target).partition(":")[0]
        prerequisites += [
            f"Make `{module}` importable from the repository root, or edit `python_path` in "
            "release.json (relative to the release folder). Each side imports its own checkout.",
            "Install the adapter's runtime dependencies in the workflow's "
            "'Install pipeline dependencies' step.",
        ]
        pipeline = ReleasePipeline(
            name=candidate.name,
            kind="python",
            target=adapter.target,
            python_path=["../.."],
            timeout=adapter.timeout,
            params=candidate.params,
        )
        return pipeline, run["backend"].get("corpus_version"), None, prerequisites, warnings, None
    if adapter.kind == "http":
        destination = str(adapter.target)
        prerequisites.append(
            "Add repository secrets `RETRIEVAL_BASELINE_URL` (the deployment the target branch "
            "represents) and `RETRIEVAL_CANDIDATE_URL` (the pull request's deployment)."
        )
        if adapter.token_env:
            prerequisites.append(f"Add repository secret `{adapter.token_env}` (read-only).")
        if _local_host(destination):
            warnings.append(
                "The comparison measured a local endpoint, which a CI runner cannot reach."
            )
        pipeline = ReleasePipeline(
            name=candidate.name,
            kind="http",
            target_env=PIPELINE_URL_ENV,
            token_env=adapter.token_env,
            timeout=adapter.timeout,
            params=candidate.params,
        )
        version = run["backend"].get("corpus_version")
        return pipeline, version, None, prerequisites, warnings, destination
    source = workspace.manifest_root / str(adapter.corpus)
    try:
        docs = read_corpus(source)
    except ReleaseError as e:
        raise ValueError(str(e)) from e
    if not include:
        prerequisites.append(
            f"Place the registered corpus at `{corpus_file}`. The export does not copy corpora."
        )
    pipeline = ReleasePipeline(
        name=candidate.name, kind="hybrid", corpus="corpus.jsonl", params=candidate.params
    )
    text = source.read_text(encoding="utf-8") if include else None
    return pipeline, corpus_version(docs), text, prerequisites, warnings, None


def release_router(workspace: GuidedWorkspace) -> APIRouter:
    """Mounted under the guided router's token/origin protection."""
    router = APIRouter()
    policies: PolicyStore = workspace.policies
    releases = LocalRecords(workspace, "releases")
    snapshots = LocalRecords(workspace, "benchmarks")

    @router.get("/policies")
    def list_policies() -> dict[str, Any]:
        return policies.describe()

    @router.post("/policies")
    def save_policy(body: PolicyIn) -> dict[str, Any]:
        return policies.save(body)

    @router.get("/policies/{identifier}")
    def read_policy(identifier: str) -> dict[str, Any]:
        return policies.record(identifier)

    @router.get("/releases")
    def list_releases() -> list[dict[str, Any]]:
        return [
            {
                k: r[k]
                for k in ("id", "name", "version", "created_at", "policy", "issues", "pipeline")
            }
            for r in releases.list()
        ]

    @router.get("/releases/{identifier}")
    def read_release(identifier: str) -> dict[str, Any]:
        return releases.read(identifier)

    @router.get("/releases/{identifier}/export.zip")
    def download(identifier: str) -> Response:
        record = releases.read(identifier)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, content in sorted(record["files"].items()):
                info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        name = f"release-check-{record['id'][:8]}-v{record['version']}.zip"
        return Response(
            buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @router.post("/releases")
    def create(body: ReleaseIn) -> dict[str, Any]:
        """Bind a passing comparison into a versioned, exportable release configuration."""
        from vectraudit.pipelines.guided import _write

        base = load_run(workspace.run_path(body.baseline))
        cand = load_run(workspace.run_path(body.candidate))
        policy, ref = policies.resolve(body.policy)
        check = compare_runs(base, cand, policy)
        if check["verdict"] != "pass":
            raise ValueError(
                f"Export needs a passing comparison under {ref['name']} v{ref['version']}; "
                f"this comparison is {check['verdict'].replace('_', ' ')}."
            )
        source = cand.get("benchmark_source")
        if not source or (base.get("benchmark_source") or {}).get("sha256") != source["sha256"]:
            raise ValueError("Export needs two guided runs of the same saved benchmark.")
        if (base["benchmark_source"]["split"], base["search"]["k"]) != (
            source["split"],
            cand["search"]["k"],
        ):
            raise ValueError("Both runs must use the same split and k.")
        snapshot = next(
            (
                snapshots.read(s["id"])
                for s in snapshots.list()
                if s["dataset_sha256"] == source["sha256"]
            ),
            None,
        )
        draft: dict[str, Any] | None = None
        try:
            draft = workspace.load_draft(source["draft_id"])
        except HTTPException:
            draft = None
        if snapshot is None:
            if draft is None or draft["revision"] != source["revision"]:
                raise ValueError(
                    "The benchmark changed after these runs and no frozen snapshot holds the "
                    "measured version. Freeze the benchmark, then record both runs again."
                )
            snapshot = freeze_draft(snapshots, draft)
        ds = parse_dataset(snapshot["dataset"])
        rows = [r for r in dataset_rows(ds) if r.get("split", "explore") == source["split"]]
        benchmark = render_dataset(rows, ds.meta)
        measured = parse_dataset(benchmark)
        if measured.sha256 != cand["dataset"]["sha256"]:
            raise ValueError("The frozen benchmark does not reproduce the measured query set.")
        pipeline, version, corpus, prerequisites, warnings, destination = _pipeline(
            workspace, cand, draft, body.include_corpus
        )
        if source["split"] == "heldout":
            warnings.append(
                "This release check runs held-out queries on every change, which gradually "
                "tunes against them. Prefer an exploratory release check."
            )
        if ref["source"] == "startup":
            warnings.append(
                "The policy came from the startup file; the export copies its current rules."
            )
        name = body.name.strip()
        lineage = digest(name.casefold())[:32]
        check_file = f"{uuid.uuid4().hex}.check.json"
        _write(workspace.root / check_file, check)
        with task_lock(releases.path(lineage)):
            versions = [r for r in releases.list() if r.get("lineage") == lineage]
            latest = max(versions, key=lambda r: int(r["version"]), default=None)
            identifier = uuid.uuid4().hex
            release = ReleaseFile(
                configuration=identifier,
                name=name,
                version=1 if latest is None else int(latest["version"]) + 1,
                benchmark=BenchmarkRef(
                    path="benchmark.jsonl",
                    sha256=measured.sha256,
                    split=source["split"],
                    n_queries=len(measured),
                ),
                policy=PolicyRef(
                    path="policy.json",
                    sha256=ref["sha256"],
                    name=ref["name"],
                    version=int(ref["version"]),
                ),
                pipeline=pipeline,
                k=cand["search"]["k"],
                repeats=body.repeats,
                corpus_version=version,
                evidence={
                    "verdict": check["verdict"],
                    "baseline_run": base["run_id"],
                    "baseline_label": base["label"],
                    "candidate_run": cand["run_id"],
                    "candidate_label": cand["label"],
                    "check_file": check_file,
                    "snapshot": snapshot["id"],
                },
                vectraudit_version=__version__,
            )
            files = render_release_files(
                release,
                benchmark=benchmark,
                policy=policies.record(body.policy)["policy"],
                corpus=corpus,
                prerequisites=prerequisites,
                evaluated_destination=destination,
            )
            return releases.write(
                {
                    "id": identifier,
                    "format_version": 1,
                    "lineage": lineage,
                    "name": name,
                    "version": release.version,
                    "supersedes": None if latest is None else latest["id"],
                    "created_at": _event("release")["at"],
                    "policy": ref,
                    "pipeline": {"name": pipeline.name, "kind": pipeline.kind},
                    "release": release.model_dump(),
                    "issues": {"prerequisites": prerequisites, "warnings": warnings},
                    "files": files,
                }
            )

    return router
