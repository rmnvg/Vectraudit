"""Allowlisted workbench records for the hosted archive; no raw evidence or notes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import parse_policy, policy_sha256

MAX_BYTES = 4 * 1024 * 1024
PROFILE = "workbench-metadata-v1"
Kind = Literal["case", "benchmark", "policy", "investigation"]
ID = Annotated[str, Field(min_length=1, max_length=512)]
Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
LocalID = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PolicyRef(Metadata):
    id: ID | None = None
    version: int | None = None
    sha256: Hash
    source: Literal["workspace", "startup", "exploratory"]


class Finding(Metadata):
    kind: ID
    severity: Literal["blocking", "error", "regression", "info", "improvement"]
    doc_ids: list[ID] = Field(default_factory=list, max_length=5000)


class Event(Metadata):
    at: Annotated[str, Field(max_length=64)]
    kind: Literal["created", "updated", "verification"]
    candidate: ID | None = None
    status: Literal["open", "investigating", "fix_proposed", "verified"] | None = None
    verdict: Literal["pass", "fail", "not_comparable", "error"] | None = None
    verified: bool | None = None
    check_file: ID | None = None
    policy: PolicyRef | None = None


class Case(Metadata):
    query_id: ID
    dataset_sha256: Hash
    baseline: ID
    candidate: ID
    status: Literal["open", "investigating", "fix_proposed", "verified"]
    findings: list[Finding] = Field(max_length=100)
    policy: PolicyRef
    history: list[Event] = Field(min_length=1, max_length=2000)
    question_text: Annotated[str, Field(max_length=65536)] | None = None
    document_titles: dict[ID, Annotated[str, Field(max_length=512)]] | None = Field(
        default=None, max_length=500
    )
    """Titles of the findings' documents; only with explicit text sharing. Omitted from the
    canonical form when absent, so records from clients without titles keep their digest."""


def _dump_case(case: Case) -> dict[str, Any]:
    data = case.model_dump()
    if not data.get("document_titles"):
        data.pop("document_titles", None)
    return data


class Span(Metadata):
    doc_id: ID
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    grade: float = Field(ge=0)


class Judgment(Metadata):
    query_id: ID
    family_id: ID | None = None
    split: Literal["explore", "heldout"]
    relevant: dict[ID, Annotated[float, Field(ge=0)]]
    irrelevant: list[ID]
    relevant_spans: list[Span]
    forbidden: list[ID]
    no_answer: bool
    text: Annotated[str, Field(max_length=65536)] | None = None


class Benchmark(Metadata):
    dataset_sha256: Hash
    draft_id: LocalID
    draft_revision: LocalID
    created_at: Annotated[str, Field(max_length=64)]
    inventory: dict[ID, Hash] | None
    judgments: list[Judgment] = Field(max_length=1000)


class SavedPolicy(Metadata):
    sha256: Hash
    lineage: LocalID
    version: int = Field(ge=1)
    supersedes: LocalID | None
    created_at: Annotated[str, Field(max_length=64)]
    rules: dict[str, Any]


class TrialLink(Metadata):
    id: LocalID
    fix: ID | None
    recovered_focus: int | None = Field(default=None, ge=0)
    regressed: int | None = Field(default=None, ge=0)
    at: Annotated[str, Field(max_length=64)]


class ValidationLink(Metadata):
    id: LocalID
    fix: ID | None
    status: Literal["approved_under_policy", "not_approved", "exploratory_only"] | None
    approved: bool
    at: Annotated[str, Field(max_length=64)]


class Protection(Metadata):
    policy_id: LocalID
    version: int = Field(ge=1)
    sha256: Hash
    rules: list[Literal["context", "rank"]] = Field(max_length=10)
    at: Annotated[str, Field(max_length=64)]


class Investigation(Metadata):
    """One investigated question: states, counts and times. Names, dispute reasons,
    evidence requests and notes stay local."""

    query_id: ID
    run: LocalID
    baseline: LocalID | None
    status: Literal["open", "closed"]
    opened_at: Annotated[str, Field(max_length=64)]
    closed_at: Annotated[str, Field(max_length=64)] | None
    losses: list[ID] = Field(max_length=100)
    diagnosis: Literal["pending", "accepted", "disputed"]
    evidence_requests: int = Field(ge=0)
    trials: list[TrialLink] = Field(max_length=200)
    validations: list[ValidationLink] = Field(max_length=200)
    outcome: Literal["fixed", "not_fixed", "no_fix_needed", "abandoned"] | None
    protections: list[Protection] = Field(max_length=200)
    minutes_to_accepted_diagnosis: float | None = Field(ge=0)
    minutes_to_verified_fix: float | None = Field(ge=0)
    reproduced_by_second_engineer: bool


class Sharing(Metadata):
    profile: Literal["workbench-metadata-v1"]
    query_text: bool
    source_sha256: Hash
    payload_sha256: Hash


class Envelope(Metadata):
    format: Literal["vectraudit.shared-record"]
    format_version: Literal[1]
    kind: Kind
    id: LocalID
    revision: LocalID
    data: dict[str, Any]
    sharing: Sharing


def encoded(value: Any) -> bytes:
    """Canonical JSON for both transfer integrity and immutable revision identity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def _select(value: Mapping[str, Any], keys: str) -> dict[str, Any]:
    return {key: value[key] for key in keys.split() if key in value}


def _policy(value: Mapping[str, Any]) -> dict[str, Any]:
    return _select(value, "id version sha256 source")


def validate_record(value: Any) -> dict[str, Any]:
    """Validate the complete allowlist, including nested fields, on both transfer ends."""
    if len(encoded(value)) > MAX_BYTES:
        raise ValueError("Shared record exceeds 4 MiB")
    envelope = Envelope.model_validate(value)
    models: dict[str, type[Metadata]] = {
        "case": Case,
        "benchmark": Benchmark,
        "policy": SavedPolicy,
        "investigation": Investigation,
    }
    model = models[envelope.kind]
    validated = model.model_validate(envelope.data)
    data = _dump_case(validated) if isinstance(validated, Case) else validated.model_dump()
    if envelope.kind == "policy":
        rules = parse_policy(data["rules"])
        if policy_sha256(rules) != data["sha256"]:
            raise ValueError("Saved policy rules do not match their identity")
    if not envelope.sharing.query_text:
        texts = (
            [data.get("question_text")]
            if envelope.kind == "case"
            else [q.get("text") for q in data.get("judgments", [])]
        )
        if any(text is not None for text in texts) or data.get("document_titles"):
            raise ValueError("Question text and document titles require explicit sharing")
    canonical = {**envelope.model_dump(exclude={"sharing"}), "data": data}
    if digest(canonical) != envelope.sharing.payload_sha256:
        raise ValueError("Shared record payload digest mismatch")
    return {**canonical, "sharing": envelope.sharing.model_dump()}


def project_record(
    kind: Kind, record: Mapping[str, Any], *, share_query_text: bool = False
) -> dict[str, Any]:
    """Project a local case, snapshot or saved policy without modifying the source."""
    if record.get("format") == "vectraudit.shared-record":
        result = validate_record(record)
        if result["kind"] != kind:
            raise ValueError("Shared record kind mismatch")
        return result
    if record.get("format_version") != 1:
        raise ValueError("Unsupported local record version")
    if kind == "case":
        data = _select(record, "query_id dataset_sha256 baseline candidate status")
        data["findings"] = [
            _select(f, "kind severity doc_ids") for f in record["finding"]["findings"]
        ]
        data["policy"] = _policy(record["policy"])
        data["history"] = []
        for event in record["history"]:
            item = _select(event, "at kind candidate status verdict verified check_file")
            if event.get("policy") is not None:
                item["policy"] = _policy(event["policy"])
            data["history"].append(item)
        data["question_text"] = (
            (record.get("question_display") or {}).get("text") if share_query_text else None
        )
        if share_query_text:
            cited = {d for f in data["findings"] for d in f["doc_ids"]}
            titles = (record.get("document_display") or {}).get("titles") or {}
            data["document_titles"] = {d: t for d, t in titles.items() if d in cited} or None
        data = _dump_case(Case.model_validate(data))
    elif kind == "benchmark":
        ds = parse_dataset(record["dataset"])
        if ds.sha256 != record["dataset_sha256"]:
            raise ValueError("Snapshot dataset does not match its identity")
        data = _select(record, "dataset_sha256 draft_id draft_revision created_at inventory")
        data["judgments"] = [
            dict(
                query_id=q.query_id,
                family_id=q.family_id,
                split=q.split,
                relevant=q.relevant,
                irrelevant=record.get("irrelevant", {}).get(q.query_id, []),
                relevant_spans=[asdict(s) for s in q.relevant_spans],
                forbidden=list(q.forbidden),
                no_answer=q.no_answer,
                text=q.text if share_query_text else None,
            )
            for q in ds.queries
        ]
        data = Benchmark.model_validate(data).model_dump()
    elif kind == "policy":
        data = _select(record, "sha256 lineage version supersedes created_at")
        parse_policy(record["policy"])
        data["rules"] = record["policy"]
        data = SavedPolicy.model_validate(data).model_dump()
    elif kind == "investigation":
        from vectraudit.pipelines.investigation_records import measures

        found = measures(record)
        data = _select(record, "query_id run baseline status opened_at losses outcome")
        data |= {
            "closed_at": record.get("closed_at"),
            "diagnosis": record["diagnosis"]["status"],
            "evidence_requests": len(record["requests"]),
            "trials": [_select(t, "id fix recovered_focus regressed at") for t in record["trials"]],
            "validations": [_select(v, "id fix status approved at") for v in record["validations"]],
            "protections": [
                {
                    **_select(p, "policy_id version sha256 at"),
                    "rules": [rule["kind"] for rule in p["rules"]],
                }
                for p in record.get("protections") or []
            ],
            **{
                key: found[key]
                for key in (
                    "minutes_to_accepted_diagnosis",
                    "minutes_to_verified_fix",
                    "reproduced_by_second_engineer",
                )
            },
        }
        data = Investigation.model_validate(data).model_dump()
    else:
        raise ValueError("Unknown shared record kind")
    payload = dict(
        format="vectraudit.shared-record",
        format_version=1,
        kind=kind,
        id=record["id"],
        revision=record["revision"],
        data=data,
    )
    return validate_record(
        {
            **payload,
            "sharing": dict(
                profile=PROFILE,
                query_text=share_query_text,
                source_sha256=digest(record),
                payload_sha256=digest(payload),
            ),
        }
    )


def preview_record(record: dict[str, Any]) -> dict[str, Any]:
    """Describe exactly which projection would leave the machine, without its content."""
    projected = validate_record(record)
    return {
        "kind": projected["kind"],
        "id": projected["id"],
        "revision": projected["revision"],
        "bytes": len(encoded(projected)),
        **projected["sharing"],
    }
