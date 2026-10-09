"""The decision ledger: compact release decisions that outlive raw-run retention.

Runs, and the checks that reference them, are purged after the plan's retention period.
Each check also writes a small ``Decision`` row (verdict, policy digest, run identities,
per-metric changes, first reasons and changed setting names) that is kept until its project
or workspace is deleted, so a project's release history and quality trend survive retention.
Rows hold no query text, document text or raw evidence: only what the uploaded sharing
projection already contains.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vectraudit.cloud.models import AuditEvent, Baseline, Check, Decision, Project, Run

MAX_REASONS = 10
MAX_CHANGES = 50
SUMMARY_VERSION = 1


def summarize_check(result: dict[str, Any]) -> dict[str, Any]:
    """The decision-relevant part of a check result, bounded and free of evidence."""
    metrics = {
        name: {
            key: stats.get(key)
            for key in ("baseline", "candidate", "delta", "ci_low", "ci_high", "n", "n_families")
        }
        for name, stats in (result.get("metrics") or {}).items()
    }
    latency = result.get("latency") or None
    critical = result.get("critical_queries") or []
    return {
        "version": SUMMARY_VERSION,
        "n_queries": result.get("n_queries"),
        "n_families": result.get("n_families"),
        "metrics": metrics,
        "primary_metric": (result.get("queries") or {}).get("primary_metric"),
        "regressed_queries": (result.get("queries") or {}).get("regressed_count"),
        "latency": {
            "baseline_p95": latency["baseline"].get("p95"),
            "candidate_p95": latency["candidate"].get("p95"),
            "compared": latency.get("compared"),
        }
        if latency
        else None,
        "critical": {
            "total": len(critical),
            "failed": sum(rule.get("verdict") != "pass" for rule in critical),
        },
        "reasons": [r.get("message", "") for r in result.get("reasons", [])][:MAX_REASONS],
        "n_reasons": len(result.get("reasons", [])),
        "n_warnings": len(result.get("warnings", [])),
        "changed_settings": [d.get("path") for d in result.get("config_diff", [])][:MAX_CHANGES],
    }


def record_decision(
    session: Session, check: Check, base: Run, cand: Run, baseline_name: str | None
) -> Decision:
    """Write the ledger row for ``check`` once; later calls return the existing row."""
    existing = session.scalars(select(Decision).where(Decision.check_id == check.id)).first()
    if existing is not None:
        return existing
    decision = Decision(
        workspace_id=check.workspace_id,
        project_id=check.project_id,
        check_id=check.id,
        baseline_name=baseline_name,
        verdict=check.verdict,
        policy_digest=check.policy_digest,
        dataset_sha256=cand.dataset_sha256,
        baseline_run_id=base.run_id,
        baseline_digest=base.digest,
        candidate_run_id=cand.run_id,
        candidate_digest=cand.digest,
        candidate_label=cand.label,
        candidate_commit=cand.commit_sha,
        decided_at=check.created_at,
        decided_by=check.created_by,
        summary=summarize_check(check.result),
    )
    try:
        with session.begin_nested():  # a concurrent writer may record the same check
            session.add(decision)
    except IntegrityError:
        found = session.scalars(select(Decision).where(Decision.check_id == check.id)).one()
        return found
    return decision


def record_missing(session: Session, checks: list[Check]) -> int:
    """Back-fill ledger rows for checks that predate the ledger; returns rows written."""
    written = 0
    for check in checks:
        if session.scalars(select(Decision.id).where(Decision.check_id == check.id)).first():
            continue
        base = session.get(Run, check.baseline_run_pk)
        cand = session.get(Run, check.candidate_run_pk)
        if base is None or cand is None:
            continue
        record_decision(session, check, base, cand, None)
        written += 1
    return written


def timeline(
    session: Session, project: Project, *, before: datetime | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Decisions and baseline pins for one project, newest first.

    Whether each decision's full check (and its report) still exists is included, so views
    can link to evidence or say it expired under retention.
    """
    decisions_stmt = select(Decision).where(Decision.project_id == project.id)
    if before is not None:
        decisions_stmt = decisions_stmt.where(Decision.decided_at < before)
    decisions = list(
        session.scalars(decisions_stmt.order_by(Decision.decided_at.desc()).limit(limit))
    )
    live = set(
        session.scalars(
            select(Check.id).where(Check.id.in_([d.check_id for d in decisions if d.check_id]))
        )
    )
    pins_stmt = (
        select(AuditEvent)
        .join(Baseline, Baseline.id == AuditEvent.target_id)
        .where(
            AuditEvent.workspace_id == project.workspace_id,
            AuditEvent.action == "baseline.pin",
            Baseline.project_id == project.id,
        )
    )
    if before is not None:
        pins_stmt = pins_stmt.where(AuditEvent.created_at < before)
    pins = list(session.scalars(pins_stmt.order_by(AuditEvent.created_at.desc()).limit(limit)))
    entries: list[dict[str, Any]] = [
        {
            "kind": "decision",
            "at": d.decided_at,
            "verdict": d.verdict,
            "baseline_name": d.baseline_name,
            "baseline_run_id": d.baseline_run_id,
            "candidate_run_id": d.candidate_run_id,
            "candidate_label": d.candidate_label,
            "candidate_commit": d.candidate_commit,
            "policy_digest": d.policy_digest,
            "decided_by": d.decided_by,
            "check_id": d.check_id,
            "evidence_available": d.check_id in live,
            "summary": d.summary,
        }
        for d in decisions
    ]
    entries += [
        {
            "kind": "baseline_pin",
            "at": p.created_at,
            "baseline_name": p.details.get("name"),
            "run_id": p.details.get("run_id"),
            "previous_run_id": p.details.get("previous_run_id"),
            "note": p.details.get("note", ""),
            "actor": p.actor,
        }
        for p in pins
    ]
    entries.sort(key=lambda e: e["at"], reverse=True)
    return entries[:limit]
