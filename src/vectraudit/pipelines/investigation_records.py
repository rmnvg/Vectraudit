"""Investigation records: what pilots must measure about the repair loop, kept as it happens.

The strategy's acceptance for the investigation workspace is measured, not asserted: time
to an accepted explanation, accepted and disputed diagnoses, verified fixes, and repairs
reproduced by a second engineer. A record follows one failed question from opening to
closing: the diagnosis is accepted or disputed (with a reason), missing evidence can be
requested, fix trials and validations are linked as they finish, and closing states the
outcome. Names are self-reported in the local workbench; there is no sign-in to check them.

Records hold IDs, names, reasons and timestamps; never question or document text.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

DIAGNOSIS = ("accepted", "disputed")
OUTCOMES = ("fixed", "not_fixed", "no_fix_needed", "abandoned")
RECORD_FORMAT = "vectraudit.investigation_record"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _minutes(start: str, end: str | None) -> float | None:
    if end is None:
        return None
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    return round(delta.total_seconds() / 60, 1)


def new_record(
    identifier: str, run: str, query_id: str, baseline: str | None, by: str, losses: list[str]
) -> dict[str, Any]:
    """A freshly opened investigation; ``losses`` are the loss classes shown at opening."""
    at = now()
    return {
        "format": RECORD_FORMAT,
        "format_version": 1,
        "id": identifier,
        "run": run,
        "query_id": query_id,
        "baseline": baseline,
        "status": "open",
        "opened_by": by,
        "opened_at": at,
        "losses": losses,
        "diagnosis": {"status": "pending"},
        "requests": [],
        "trials": [],
        "validations": [],
        "outcome": None,
        "events": [{"at": at, "kind": "opened", "by": by}],
    }


def set_diagnosis(record: dict[str, Any], status: str, by: str, reason: str | None) -> None:
    """Accept or dispute the diagnosis; a dispute needs a reason. The latest decision holds,
    and every decision stays in the event history."""
    if status not in DIAGNOSIS:
        raise ValueError(f"diagnosis must be one of {', '.join(DIAGNOSIS)}")
    if status == "disputed" and not (reason or "").strip():
        raise ValueError("say why the diagnosis is disputed")
    at = now()
    record["diagnosis"] = {"status": status, "by": by, "reason": reason, "at": at}
    record["events"].append({"at": at, "kind": f"diagnosis_{status}", "by": by, "reason": reason})


def request_evidence(record: dict[str, Any], what: str, by: str) -> None:
    """Note evidence the diagnosis is missing (a trace, a corpus version, a judgment)."""
    if not what.strip():
        raise ValueError("describe the missing evidence")
    at = now()
    record["requests"].append({"what": what, "by": by, "at": at})
    record["events"].append({"at": at, "kind": "evidence_requested", "by": by})


def link_result(record: dict[str, Any], result: Mapping[str, Any], by: str) -> None:
    """Attach a finished fix trial or validation, read from the workspace's saved result."""
    at = now()
    kind = result.get("kind")
    entry = {"id": result["id"], "fix": (result.get("fix") or {}).get("name"), "by": by, "at": at}
    if kind == "trial":
        entry |= {
            "recovered_focus": result.get("recovered_focus"),
            "regressed": len(result.get("regressed") or []),
        }
        target = record["trials"]
    elif kind == "validation":
        entry |= {"status": result.get("status"), "approved": bool(result.get("approved"))}
        target = record["validations"]
    else:
        raise ValueError("only fix trials and validations can be linked")
    if any(item["id"] == entry["id"] for item in target):
        raise ValueError("this result is already linked")
    target.append(entry)
    record["events"].append({"at": at, "kind": f"{kind}_linked", "by": by, "id": result["id"]})


def close(record: dict[str, Any], outcome: str, by: str, note: str | None) -> None:
    """Close with an outcome. ``fixed`` needs an approved validation: a trial alone is
    exploratory, so it cannot verify a repair."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    if outcome == "fixed" and not any(v["approved"] for v in record["validations"]):
        raise ValueError("a fix is verified only by an approved validation; link one first")
    at = now()
    record.update(status="closed", outcome=outcome, closed_by=by, closed_at=at, note=note)
    record["events"].append({"at": at, "kind": "closed", "by": by, "outcome": outcome})


def measures(record: Mapping[str, Any]) -> dict[str, Any]:
    """The pilot measures for one investigation (see the discovery ledger's columns)."""
    diagnosis = record["diagnosis"]
    approved = [v for v in record["validations"] if v["approved"]]
    first_accept = next(
        (e["at"] for e in record["events"] if e["kind"] == "diagnosis_accepted"), None
    )
    by_fix: dict[str | None, set[str]] = {}
    for v in approved:
        by_fix.setdefault(v["fix"], set()).add(v["by"])
    reproduced = any(
        len(people) >= 2 and people - {record["opened_by"]} for people in by_fix.values()
    )
    return {
        "minutes_to_accepted_diagnosis": _minutes(record["opened_at"], first_accept),
        "minutes_to_verified_fix": _minutes(record["opened_at"], approved[0]["at"])
        if approved
        else None,
        "diagnosis": diagnosis["status"],
        "fix_verified": bool(approved),
        "reproduced_by_second_engineer": reproduced,
        "evidence_requests": len(record["requests"]),
        "trials": len(record["trials"]),
    }


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts with denominators and medians with their sample sizes, never bare rates."""
    rows = [
        {
            "id": r["id"],
            "query_id": r["query_id"],
            "status": r["status"],
            "outcome": r.get("outcome"),
            **measures(r),
        }
        for r in records
    ]

    def median(key: str) -> dict[str, Any]:
        values = [row[key] for row in rows if row[key] is not None]
        return {"median": statistics.median(values) if values else None, "n": len(values)}

    return {
        "investigations": len(rows),
        "open": sum(row["status"] == "open" for row in rows),
        "diagnosis": {
            status: sum(row["diagnosis"] == status for row in rows)
            for status in ("accepted", "disputed", "pending")
        },
        "fixes_verified": sum(row["fix_verified"] for row in rows),
        "reproduced_by_second_engineer": sum(row["reproduced_by_second_engineer"] for row in rows),
        "minutes_to_accepted_diagnosis": median("minutes_to_accepted_diagnosis"),
        "minutes_to_verified_fix": median("minutes_to_verified_fix"),
        "rows": rows,
        "note": "Names are self-reported in the local workbench. Small counts are stories, "
        "not rates; report them with their denominators.",
    }
