"""Local benchmark preparation, immutable snapshots and evidence-backed case tracking."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import Field

from vectraudit.contracts import Context, Filters
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import ColumnMap, dataset_rows, import_rows, render_dataset
from vectraudit.evaluation.runfile import load_run
from vectraudit.evaluation.triage import triage
from vectraudit.labelling.storage import task_lock
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.local_display import CaseQuestionDisplay
from vectraudit.pipelines.models import Contract

if TYPE_CHECKING:
    from vectraudit.integrations.cloud import CloudClientError
    from vectraudit.pipelines.guided import GuidedWorkspace

LIMIT = 4 * 1024 * 1024


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def documents(text: str) -> list[dict[str, Any]]:
    """Validate a customer-supplied local corpus; never fetch document content."""
    from vectraudit.pipelines.guided import Document

    docs = [
        Document.model_validate_json(line).model_dump(exclude_none=True)
        for line in text.splitlines()
        if line.strip()
    ]
    if len(docs) > 5000 or len({d["doc_id"] for d in docs}) != len(docs):
        raise ValueError("Provide at most 5000 documents with unique IDs")
    return docs


def inventory(text: str) -> dict[str, str] | None:
    """Content and metadata hashes; absent inventory remains unknown."""
    docs = documents(text)
    return {
        d["doc_id"]: digest({"text": d["text"], "metadata": d["metadata"]}) for d in docs
    } or None


class ImportIn(Contract):
    content: str = Field(max_length=LIMIT)
    format: Literal["csv", "json", "jsonl"]
    text_column: str = "text"
    id_column: str = "query_id"
    relevant_column: str = "relevant"
    incident: bool = False
    source: str = Field(default="Local export", max_length=256)


def prepare_import(body: ImportIn) -> dict[str, Any]:
    """Preview mapped rows; incidents never promote feedback into relevance labels."""
    if body.format == "csv":
        records: Any = list(csv.DictReader(io.StringIO(body.content.lstrip("\ufeff"))))
    elif body.format == "json":
        records = json.loads(body.content)
        if isinstance(records, dict):
            records = next(
                (records[k] for k in ("examples", "items", "data") if k in records), None
            )
    else:
        records = [json.loads(line) for line in body.content.splitlines() if line.strip()]
    if not isinstance(records, list) or not records or len(records) > 5000:
        raise ValueError("Import 1-5000 rows from a CSV, JSON array or JSONL file")
    if any(not isinstance(r, dict) for r in records):
        raise ValueError("Every imported row must be an object")
    source_meta: dict[str, Any] = {}
    if records and "dataset" in records[0]:
        source_meta = records.pop(0)["dataset"]
        if not isinstance(source_meta, dict):
            raise ValueError("Dataset header must be an object")
        if not isinstance(source_meta.get("provenance", {}), dict):
            raise ValueError("Dataset provenance must be an object")
    columns = sorted({str(k) for r in records for k in r})
    # import_rows already supports dotted JSON paths and native permission context.
    prepared = []
    for rec in records:
        value = dict(rec)
        reference: Any = rec.get(body.id_column)
        if reference is None:
            reference = rec
            for part in body.id_column.split("."):
                reference = reference.get(part) if isinstance(reference, dict) else None
        value["_import_id"] = str(reference) if reference is not None else "pending"
        if body.incident:
            for key in (
                "relevant",
                "relevant_spans",
                "no_answer",
                "forbidden",
                "split",
                "required_evidence",
            ):
                value.pop(key, None)
        prepared.append(value)
    mapped = import_rows(
        prepared,
        ColumnMap(
            query_id="_import_id",
            text=body.text_column,
            relevant=None if body.incident else body.relevant_column,
            tags="tags",
        ),
    )
    for row, rec in zip(mapped, prepared, strict=True):
        # These native fields are not column-mapped by import_rows. Preserve their
        # semantics instead of silently losing spans or merging distinct filters.
        for field in ("filter", "vector", "relevant_spans"):
            if field in rec and (not body.incident or field == "filter"):
                value = rec[field]
                row[field] = json.loads(value) if isinstance(value, str) else value
    unique: dict[str, dict[str, Any]] = {}
    refs: dict[str, list[str]] = {}
    for row in mapped:
        key = digest(
            {
                **{k: row.get(k) for k in ("text", "context", "filter")},
                **({"history": row["history"]} if row.get("history") else {}),
            }
        )
        if body.incident:
            reference = row["query_id"]
            row["query_id"] = "incident-" + key[:20]
            row["tags"] = sorted(set(row.get("tags", [])) | {"incident-review"})
            row["split"] = "explore"
            refs.setdefault(row["query_id"], []).append(reference)
            unique.setdefault(key, row)
        else:
            if row["query_id"] == "pending":
                row["query_id"] = "q-" + key[:20]
            # Preserve separate identities and conflicting judgments for validation.
            unique[str(len(unique))] = row
    rows = list(unique.values())
    from vectraudit.pipelines.guided import MAX_QUERIES

    if len(rows) > MAX_QUERIES:
        raise ValueError(f"Select at most {MAX_QUERIES} distinct questions before importing")
    meta = {
        **({} if body.incident else source_meta),
        "provenance": {
            **(source_meta.get("provenance", {}) if not body.incident else {}),
            "source": body.source,
            "source_sha256": digest(body.content),
            "incident": body.incident,
            "references": refs,
        },
    }
    dataset = render_dataset(rows, meta)
    parse_dataset(dataset)
    return {
        "dataset": dataset,
        "columns": columns,
        "count": len(rows),
        "duplicates": len(mapped) - len(rows),
        "references": refs,
    }


class PermissionScenario(Contract):
    name: str = Field(min_length=1, max_length=80)
    context: Context = Field(default_factory=dict)
    filters: Filters = Field(default_factory=dict)
    allowed: list[str] = Field(default_factory=list, max_length=500)
    forbidden: list[str] = Field(default_factory=list, max_length=500)
    no_answer: bool = False


class PermissionIn(Contract):
    dataset: str = Field(max_length=LIMIT)
    query_id: str
    scenarios: list[PermissionScenario] = Field(min_length=1, max_length=50)


def permission_rows(body: PermissionIn) -> dict[str, Any]:
    ds = parse_dataset(body.dataset)
    original = next((r for r in dataset_rows(ds) if r["query_id"] == body.query_id), None)
    if original is None:
        raise ValueError("Select an existing question")
    if original.get("split") == "heldout":
        raise ValueError("Create permission scenarios from an exploratory question")
    rows = []
    for scenario in body.scenarios:
        if not scenario.context:
            raise ValueError("Each scenario needs explicit user or tenant context")
        if not (scenario.allowed or scenario.forbidden or scenario.no_answer):
            raise ValueError("Each scenario needs an expected result or no-answer assertion")
        key = digest({"source": body.query_id, "scenario": scenario.model_dump()})[:16]
        row = {
            k: v
            for k, v in original.items()
            if k
            not in (
                "relevant",
                "relevant_spans",
                "forbidden",
                "no_answer",
                "context",
                "filter",
                "required_evidence",
            )
        }
        row.update(
            family_id=original.get("family_id", body.query_id),
            query_id=f"permission-{key}",
            context=scenario.context,
            filter=scenario.filters,
            relevant=dict.fromkeys(scenario.allowed, 1),
            forbidden=scenario.forbidden,
            no_answer=scenario.no_answer,
            tags=sorted(set(original.get("tags", [])) | {"permission", scenario.name}),
        )
        rows.append(row)
    parse_dataset(render_dataset(rows))
    return {"rows": rows}


class SnapshotIn(Contract):
    draft_id: str
    revision: str


class FreshnessIn(Contract):
    snapshot: str
    corpus: str = Field(max_length=LIMIT)


class CaseIn(Contract):
    baseline: str
    candidate: str
    query_id: str
    policy: str | None = None
    """Policy of the comparison that found this case; ``None`` is exploratory defaults."""


class CaseUpdate(Contract):
    revision: str
    status: Literal["open", "investigating", "fix_proposed"]
    owner: str = Field(default="", max_length=128)
    note: str = Field(default="", max_length=4000)


class VerifyIn(Contract):
    revision: str
    candidate: str
    policy: str | None = None
    """Release policy to verify under; ``None`` uses the exploratory defaults."""


class LocalRecords:
    """Bounded records with optimistic writes and the same local path protections as drafts."""

    def __init__(self, workspace: GuidedWorkspace, category: str):
        self.workspace = workspace
        self.folder = workspace.root / ".workbench" / category

    def path(self, identifier: str) -> Path:
        from vectraudit.pipelines.guided import _identifier

        path = self.folder / f"{_identifier(identifier)}.json"
        for p in (self.folder.parent, self.folder, path, path.with_suffix(".json.lock")):
            if p.is_symlink():
                raise HTTPException(400, "Workspace paths must not be symbolic links")
        return path

    def read(self, identifier: str) -> dict[str, Any]:
        path = self.path(identifier)
        if not path.is_file():
            raise HTTPException(404, "Local record not found")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise HTTPException(413, "Local record exceeds size limit")
        result: dict[str, Any] = json.loads(path.read_text())
        return result

    def write(self, value: dict[str, Any]) -> dict[str, Any]:
        value["revision"] = uuid.uuid4().hex
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if len(payload.encode()) > 20 * 1024 * 1024:
            raise HTTPException(413, "Local record exceeds size limit")
        atomic_write(self.path(value["id"]), payload)
        return value

    def list(self) -> list[dict[str, Any]]:
        self.path("0" * 32)  # Check ancestors before enumerating.
        paths = sorted(self.folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [self.read(p.stem) for p in paths[:200]]


def _event(kind: str, **details: Any) -> dict[str, Any]:
    return {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **details}


def freeze_draft(snapshots: LocalRecords, draft: dict[str, Any]) -> dict[str, Any]:
    """An immutable snapshot of a saved draft: dataset version plus inventory hashes."""
    ds = parse_dataset(draft["dataset"])
    return snapshots.write(
        {
            "id": uuid.uuid4().hex,
            "format_version": 1,
            "name": draft["name"],
            "created_at": _event("freeze")["at"],
            "draft_id": draft["id"],
            "draft_revision": draft["revision"],
            "dataset_sha256": ds.sha256,
            "dataset": draft["dataset"],
            "irrelevant": draft.get("irrelevant", {}),
            "inventory": inventory(draft.get("corpus", "")),
        }
    )


def _revision(record: dict[str, Any], expected: str) -> None:
    if record["revision"] != expected:
        raise HTTPException(409, "Record changed. Reload before saving.")


class ShareIn(Contract):
    kind: Literal["case", "benchmark", "policy", "investigation"]
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    share_query_text: bool = False
    confirm: bool = False
    """Upload only: the person confirmed the previewed transfer to the named destination."""


class HostedExceptionsIn(Contract):
    check: str = Field(pattern=r"^[a-f0-9]{32}$")
    """The comparison's check file ID (``files.check`` without ``.check.json``)."""


def _hosted_error(exc: CloudClientError) -> HTTPException:
    # No status: not configured or unreachable before any response; otherwise relay the
    # hosted service's refusal (wrong role, missing project, conflict, too large).
    status = (
        422
        if exc.status is None
        else exc.status
        if exc.status in (401, 403, 404, 409, 413)
        else 502
    )
    return HTTPException(status, exc.message + (f" {exc.hint}" if exc.hint else ""))


def share_destination() -> dict[str, Any]:
    """Where an upload would go, from this server's environment; never the token."""
    from vectraudit.integrations.cloud import CloudClientError, CloudTarget

    try:
        target = CloudTarget.from_env()
    except CloudClientError as exc:
        return {"configured": False, "message": exc.message, "hint": exc.hint}
    return {
        "configured": True,
        "url": target.url,
        "project": f"{target.workspace}/{target.project}",
    }


def workflow_router(workspace: GuidedWorkspace) -> APIRouter:
    """Mounted under the guided router's existing token/origin protection."""
    router = APIRouter()
    snapshots = LocalRecords(workspace, "benchmarks")
    cases = LocalRecords(workspace, "cases")

    @router.post("/import")
    def preview(body: ImportIn) -> dict[str, Any]:
        return prepare_import(body)

    @router.post("/permissions")
    def permissions(body: PermissionIn) -> dict[str, Any]:
        return permission_rows(body)

    @router.get("/snapshots")
    def list_snapshots() -> list[dict[str, Any]]:
        return [
            {k: v[k] for k in ("id", "name", "created_at", "dataset_sha256", "draft_id")}
            for v in snapshots.list()
        ]

    @router.get("/snapshots/{identifier}")
    def snapshot(identifier: str) -> dict[str, Any]:
        return snapshots.read(identifier)

    @router.post("/snapshots")
    def freeze(body: SnapshotIn) -> dict[str, Any]:
        draft = workspace.load_draft(body.draft_id)
        _revision(draft, body.revision)
        return freeze_draft(snapshots, draft)

    @router.post("/freshness")
    def freshness(body: FreshnessIn) -> dict[str, Any]:
        frozen = snapshots.read(body.snapshot)
        before, after = frozen["inventory"], inventory(body.corpus)
        if before is None or after is None:
            return {"status": "unknown", "items": [], "changed": [], "deleted": [], "added": []}
        changed = sorted(d for d in before if d in after and before[d] != after[d])
        deleted, added = sorted(set(before) - set(after)), sorted(set(after) - set(before))
        affected = set(changed + deleted)
        items: list[dict[str, Any]] = []
        for q in parse_dataset(frozen["dataset"]).queries:
            referenced = set(q.relevant) | {s.doc_id for s in q.relevant_spans} | set(q.forbidden)
            referenced |= set(frozen.get("irrelevant", {}).get(q.query_id, []))
            for doc in sorted(referenced & affected):
                items.append(
                    {
                        "query_id": q.query_id,
                        "doc_id": doc,
                        "reason": "deleted" if doc in deleted else "changed",
                    }
                )
            if q.no_answer and (changed or added):
                items.append(
                    {
                        "query_id": q.query_id,
                        "doc_id": None,
                        "reason": "new or changed content may answer this question",
                    }
                )
        return {
            "status": "review" if changed or deleted or added else "unchanged",
            "items": items,
            "changed": changed,
            "deleted": deleted,
            "added": added,
        }

    @router.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Newest runs as summaries; per-query records load one run at a time."""
        paths = sorted(
            workspace.root.glob("*.run.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        result = []
        for path in paths[:100]:
            if path.is_symlink():
                continue
            try:
                result.append(workspace.run_summary(path.name.removesuffix(".run.json")))
            except (ValueError, HTTPException):
                continue
        return result

    @router.get("/runs/{identifier}")
    def get_run(identifier: str) -> dict[str, Any]:
        return load_run(workspace.run_path(identifier))

    def _projection(body: ShareIn) -> dict[str, Any]:
        from vectraudit.pipelines.share import GENERIC_ERROR, projection

        try:
            return projection(workspace, body.kind, body.id, share_query_text=body.share_query_text)
        except HTTPException:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            # Validation errors can contain the rejected private text: never echo them.
            raise HTTPException(422, GENERIC_ERROR) from exc

    @router.post("/share/preview")
    def share_preview(body: ShareIn) -> dict[str, Any]:
        """Exactly what sharing would send, and to which hosted project. Sends nothing."""
        from vectraudit.pipelines.share import preview

        return {**preview(_projection(body)), "destination": share_destination()}

    @router.post("/share")
    def share(body: ShareIn) -> dict[str, Any]:
        """Upload the previewed projection to the hosted project (explicitly confirmed)."""
        from vectraudit.integrations.cloud import CloudClient, CloudClientError, CloudTarget
        from vectraudit.pipelines.share import preview

        if not body.confirm:
            raise HTTPException(422, "Review the preview and confirm sharing explicitly")
        projected = _projection(body)
        try:
            target = CloudTarget.from_env()
            uploaded = CloudClient(target).upload_record(projected)
        except CloudClientError as exc:
            raise _hosted_error(exc) from None
        link = (
            f"{target.url}/app/{quote(target.workspace)}/{quote(target.project)}"
            f"/records/{quote(uploaded['id'])}"
        )
        return {**preview(projected), "uploaded": uploaded, "url": link}

    @router.post("/exceptions/hosted")
    def hosted_exceptions(body: HostedExceptionsIn) -> dict[str, Any]:
        """The hosted project's audited exceptions against one local comparison. Read only."""
        from vectraudit.evaluation.policy import PolicyError
        from vectraudit.integrations.cloud import CloudClient, CloudClientError, CloudTarget
        from vectraudit.pipelines.hosted_exceptions import hosted_policy, hosted_view

        path = workspace.root / f"{body.check}.check.json"
        try:
            check = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(404, "Comparison not found; compare again") from None
        try:
            target = CloudTarget.from_env()
        except CloudClientError as exc:
            raise HTTPException(
                422,
                f"Not connected to a hosted project: {exc.message} Set VECTRAUDIT_CLOUD_URL, "
                "VECTRAUDIT_CLOUD_TOKEN and VECTRAUDIT_CLOUD_PROJECT before starting the "
                "workbench.",
            ) from None
        try:
            client = CloudClient(target)
            policy = client.get_policy()
            listed = client.list_exceptions()
        except CloudClientError as exc:
            raise _hosted_error(exc) from None
        try:
            view = hosted_view(
                check,
                listed,
                hosted_policy(policy["policy"]),
                hosted_default=bool(policy.get("is_default")),
            )
        except (KeyError, TypeError, ValueError, PolicyError) as exc:
            raise HTTPException(502, "The hosted project returned an unexpected response") from exc
        return {**view, "destination": share_destination()}

    @router.get("/cases")
    def list_cases() -> list[dict[str, Any]]:
        display = CaseQuestionDisplay(workspace)
        return [display.decorate(record) for record in cases.list()]

    @router.post("/cases")
    def create_case(body: CaseIn) -> dict[str, Any]:
        baseline = load_run(workspace.run_path(body.baseline))
        candidate = load_run(workspace.run_path(body.candidate))
        if baseline["dataset"]["sha256"] != candidate["dataset"]["sha256"]:
            raise ValueError("Cases require the same benchmark version in both runs")
        finding = next(
            (i for i in triage(baseline, candidate)["items"] if i["query_id"] == body.query_id),
            None,
        )
        if finding is None or finding["severity"] not in (
            "blocking",
            "error",
            "regression",
            "info",
        ):
            raise ValueError("Choose an unresolved finding from this comparison")
        identifier = digest([body.baseline, baseline["dataset"]["sha256"], body.query_id])[:32]
        _, policy = workspace.policies.resolve(body.policy)
        with task_lock(cases.path(identifier)):
            if cases.path(identifier).exists():
                return CaseQuestionDisplay(workspace).decorate(cases.read(identifier))
            record = cases.write(
                {
                    "id": identifier,
                    "format_version": 1,
                    "query_id": body.query_id,
                    "dataset_sha256": baseline["dataset"]["sha256"],
                    "baseline": body.baseline,
                    "candidate": body.candidate,
                    "status": "open",
                    "owner": "",
                    "finding": finding,
                    "policy": policy,
                    "history": [_event("created", candidate=body.candidate, policy=policy)],
                }
            )
            return CaseQuestionDisplay(workspace).decorate(record)

    @router.post("/cases/{identifier}")
    def update_case(identifier: str, body: CaseUpdate) -> dict[str, Any]:
        with task_lock(cases.path(identifier)):
            record = cases.read(identifier)
            _revision(record, body.revision)
            record.update(status=body.status, owner=body.owner)
            record["history"].append(
                _event("updated", status=body.status, owner=body.owner, note=body.note)
            )
            return CaseQuestionDisplay(workspace).decorate(cases.write(record))

    @router.post("/cases/{identifier}/verify")
    def verify_case(identifier: str, body: VerifyIn) -> dict[str, Any]:
        with task_lock(cases.path(identifier)):
            record = cases.read(identifier)
            _revision(record, body.revision)
            baseline = load_run(workspace.run_path(record["baseline"]))
            candidate = load_run(workspace.run_path(body.candidate))
            if body.candidate in (record["baseline"], record["candidate"]):
                raise ValueError("Verify with a new candidate run, not the original evidence")
            rules, policy = workspace.policies.resolve(body.policy)
            check = compare_runs(baseline, candidate, rules)
            if check["verdict"] in ("not_comparable", "error"):
                raise ValueError("Verification requires a complete, comparable full benchmark run")
            finding = next(
                (
                    i
                    for i in triage(baseline, candidate)["items"]
                    if i["query_id"] == record["query_id"]
                ),
                None,
            )
            unresolved = finding is not None and finding["severity"] != "improvement"
            verified = check["verdict"] == "pass" and not unresolved
            check_file = f"{uuid.uuid4().hex}.check.json"
            atomic_write(
                workspace.root / check_file,
                json.dumps(check, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            )
            record["status"] = "verified" if verified else "open"
            record["history"].append(
                _event(
                    "verification",
                    candidate=body.candidate,
                    verdict=check["verdict"],
                    check_file=check_file,
                    verified=verified,
                    finding=finding,
                    # Each verification keeps the policy it used; later policy versions never
                    # rewrite this evidence.
                    policy=policy,
                )
            )
            return CaseQuestionDisplay(workspace).decorate(cases.write(record))

    return router
