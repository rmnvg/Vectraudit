"""Plain-language release summaries and quality trends, read from the decision ledger.

Written for an engineering lead: what changed, why it matters, what is uncertain and what to
do next, without learning every ranking metric. Everything here is derived from the compact
``Decision`` rows, so it stays available after raw runs and checks expire under retention
and holds no text the ledger does not already hold. Pure functions over plain mappings; the
routes and pages supply the rows.

Trends are grouped by benchmark version (dataset SHA-256): scores on different questions
are not comparable, so a new benchmark version starts a new series instead of drawing a
misleading line across it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

SMALL_SAMPLE = 30
"""Matches ``evaluation.compare.SMALL_SAMPLE``: fewer families make intervals unreliable."""
UNCHANGED = 0.0005
METRICS = {
    "ndcg": "nDCG (ranking quality)",
    "recall": "Recall (relevant documents found)",
    "mrr": "MRR (rank of the first relevant document)",
    "precision": "Precision (share of results that are relevant)",
    "ann_recall": "ANN recall (agreement with exact search)",
}
HEADLINES = {
    "pass": "{candidate} passes the release policy.",
    "fail": "{candidate} is blocked: {n} release rule{s} fail{verb}.",
    "accepted": "{candidate} fails the release policy; every failure is accepted by exception.",
    "error": "The check of {candidate} could not be completed.",
    "not_comparable": "{candidate} could not be compared with the baseline.",
}
ACTIONS = {
    "pass": "Nothing in the policy blocks this release.",
    "fail": (
        "Do not release until the failing rules are fixed, or accepted deliberately through an "
        "audited, expiring exception."
    ),
    "accepted": (
        "Released deliberately with known regressions. Review the exception owners and expiry "
        "dates; the failures block again when the exceptions end."
    ),
    "error": "Re-record the incomplete run and check again. An error is never a pass.",
    "not_comparable": (
        "Re-record both sides on the same benchmark version and settings before deciding."
    ),
}


MONITOR_HEADLINES = {
    "pass": "Scheduled monitor: {candidate} still passes the release policy.",
    "fail": "Scheduled monitor: {candidate} now breaks {n} release rule{s}.",
    "accepted": (
        "Scheduled monitor: {candidate} breaks the release policy; every failure is accepted "
        "by exception."
    ),
    "error": "Scheduled monitor: the run of {candidate} could not be completed.",
    "not_comparable": "Scheduled monitor: {candidate} could not be compared with the baseline.",
}
MONITOR_ACTIONS = {
    "pass": "No action needed: the monitored setup still meets the policy.",
    "fail": (
        "Nothing new was released, yet the monitored setup no longer meets the policy. Check "
        "what drifted (corpus, configuration or an external service) before the next release."
    ),
    "accepted": "The failures are accepted by exception; review them before they expire.",
    "error": "Fix the monitor run (endpoint, credentials or adapter); an error is never a pass.",
    "not_comparable": (
        "The monitor no longer measures the frozen benchmark it was set up with; re-check its "
        "baseline and dataset."
    ),
}
ALERTING = ("fail", "error", "not_comparable")
"""Monitor verdicts that open or keep an alert; ``accepted`` was deliberately allowed."""


def metric_label(name: str) -> str:
    """A readable name for a metric key, e.g. ``ndcg`` → ``nDCG (ranking quality)``."""
    return METRICS.get(name, name)


def candidate_name(entry: Mapping[str, Any]) -> str:
    """The candidate's label, or a short run ID, with its commit when known."""
    name = entry.get("candidate_label") or (entry.get("candidate_run_id") or "candidate")[:12]
    if entry.get("candidate_commit"):
        name += f" @ {entry['candidate_commit'][:10]}"
    return str(name)


def _ordered(summary: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    metrics = summary.get("metrics") or {}
    primary = summary.get("primary_metric")
    names = sorted(metrics, key=lambda name: (name != primary, name))
    return [(name, metrics[name]) for name in names]


def _change(name: str, stats: Mapping[str, Any]) -> str | None:
    base, cand = stats.get("baseline"), stats.get("candidate")
    if base is None or cand is None:
        return None
    delta = cand - base if stats.get("delta") is None else stats["delta"]
    if abs(delta) < UNCHANGED:
        movement = f"unchanged at {cand:.3f}"
    else:
        word = "improved" if delta > 0 else "worsened"
        movement = f"{word} from {base:.3f} to {cand:.3f} ({delta:+.3f})"
    sample = ""
    if stats.get("n") is not None:
        sample = f" on {stats['n']} questions"
        families = stats.get("n_families")
        if families is not None and families != stats["n"]:
            sample += f" ({families} independent question families)"
    return f"{metric_label(name)} {movement}{sample}."


def _latency(latency: Mapping[str, Any] | None) -> str | None:
    if not latency or not latency.get("compared"):
        return None
    base, cand = latency.get("baseline_p95"), latency.get("candidate_p95")
    if base is None or cand is None:
        return None
    if base > 0:
        pct = (cand - base) / base * 100
        if abs(pct) < 1:
            return f"p95 latency unchanged at about {cand:.1f} ms."
        word = "rose" if pct > 0 else "fell"
        return f"p95 latency {word} from {base:.1f} ms to {cand:.1f} ms ({pct:+.0f}%)."
    return f"p95 latency {base:.1f} ms → {cand:.1f} ms."


def _uncertain(entry: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    notes = []
    noisy = []
    for name, stats in _ordered(summary):
        low, high = stats.get("ci_low"), stats.get("ci_high")
        tolerance = stats.get("tolerance") or {}
        if tolerance.get("decision") == "inconclusive":
            notes.append(
                f"Inconclusive for {metric_label(name)}: the 95% interval of the change "
                f"({low:+.3f} to {high:+.3f}) reaches past the allowed drop of "
                f"{tolerance['margin']:.3f}. Under the non-inferiority policy this fails the "
                "check; more query families would narrow the interval."
            )
        if tolerance or low is None or high is None or (stats.get("n_families") or 0) < 2:
            continue  # non-inferiority states its own conclusion for this metric
        if low < 0 < high:
            noisy.append(f"{metric_label(name).split(' (')[0]} ({low:+.3f} to {high:+.3f})")
    if noisy:
        note = (
            f"The 95% interval of the change includes zero for {', '.join(noisy)}: this "
            "check cannot separate those changes from measurement noise."
        )
        if entry.get("verdict") in ("fail", "accepted"):
            note += " The policy judges the measured change, so the failure still stands."
        else:
            note += " Not detecting harm is not proof of safety."
        notes.append(note)
    families = summary.get("n_families")
    if families is not None and 0 < families < SMALL_SAMPLE:
        notes.append(
            f"Only {families} independent question families were compared; small benchmarks "
            "miss rare failures and give wide intervals."
        )
    if summary.get("metrics") == {} and entry.get("verdict") not in ("error", "not_comparable"):
        notes.append("No quality metric was judged; the benchmark may be unlabelled.")
    cohorts = summary.get("cohorts") or {}
    if cohorts.get("failed_rules") and cohorts.get("control") == "none":
        rules = cohorts["judged_rules"]
        note = (
            f"{cohorts['failed_rules']} of {rules} cohort rules failed, each judged on its own: "
            "with many cohorts some fail by chance even without a real change."
        )
        if entry.get("verdict") in ("fail", "accepted"):
            note += " The failure still stands; A/A calibration shows how often this happens."
        notes.append(note)
    warnings = summary.get("n_warnings") or 0
    if warnings:
        notes.append(f"The full report lists {warnings} warning{'s' * (warnings != 1)}.")
    if entry.get("evidence_available") is False:
        notes.append(
            "The full report has expired under retention; this summary is the remaining record."
        )
    return notes


def _drift(drift: Mapping[str, Any] | None) -> list[str]:
    """Monitor drift in words; ``unknown`` says what was not recorded."""
    if not drift:
        return []
    words = {
        "changed": "changed since the baseline",
        "unchanged": "unchanged",
        "unknown": "not recorded by one of the runs",
    }
    lines = [
        f"Corpus version {words[str(drift.get('corpus') or 'unknown')]}.",
        f"Retrieval configuration {words[str(drift.get('configuration') or 'unknown')]}.",
    ]
    n = drift.get("changed_results") or 0
    lines.append(
        f"Returned documents changed for {n} question{'s' * (n != 1)}."
        if n
        else "Every question returned the same documents as the baseline."
    )
    return lines


def monitor_status(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The project's scheduled-monitor state from its monitor decisions, newest first.

    ``alerting`` while the newest runs fail, error or are not comparable: the alert opens at
    the oldest run of that unbroken streak, and its ``alert_id`` (that run's check ID) stays
    the same until a run passes, so a notifier can deliver it once. ``healthy`` after a pass,
    with the most recently resolved alert if there is one in ``decisions``. ``complete`` is
    False when the streak reaches the oldest decision supplied, so it may have started
    earlier.
    """
    runs = [d for d in decisions if d.get("trigger") == "monitor"]
    if not runs:
        return {"state": "none", "last": None, "alert": None}
    latest = runs[0]
    last = {"at": latest["at"], "check_id": latest.get("check_id"), "verdict": latest["verdict"]}
    streak = 0
    while streak < len(runs) and runs[streak]["verdict"] in ALERTING:
        streak += 1
    if streak:
        opened = runs[streak - 1]
        return {
            "state": "alerting",
            "last": last,
            "alert": {
                "alert_id": opened.get("check_id"),
                "opened_at": opened["at"],
                "failing_runs": streak,
                "complete": streak < len(runs),
                "resolved_at": None,
                "resolved_by": None,
            },
        }
    passes = 0
    while passes < len(runs) and runs[passes]["verdict"] not in ALERTING:
        passes += 1
    alert = None
    if passes < len(runs):
        end = passes
        while end < len(runs) and runs[end]["verdict"] in ALERTING:
            end += 1
        resolver = runs[passes - 1]
        alert = {
            "alert_id": runs[end - 1].get("check_id"),
            "opened_at": runs[end - 1]["at"],
            "failing_runs": end - passes,
            "complete": end < len(runs),
            "resolved_at": resolver["at"],
            "resolved_by": resolver.get("check_id"),
        }
    return {"state": "healthy", "last": last, "alert": alert}


def monitor_overdue(
    *,
    every_minutes: int,
    grace_minutes: int,
    expected_since: datetime,
    last_at: datetime | None,
    now: datetime,
) -> dict[str, Any]:
    """Whether a scheduled monitor is overdue: no run within its interval plus grace.

    The clock starts at the latest monitor run, or at ``expected_since`` if that is later
    (a newly declared schedule is not overdue for silence that came before it). Returns the
    ``reference`` time, ``due_at`` and ``overdue``.
    """
    reference = expected_since if last_at is None or last_at < expected_since else last_at
    due_at = reference + timedelta(minutes=every_minutes + grace_minutes)
    return {"reference": reference, "due_at": due_at, "overdue": now > due_at}


def interval_label(minutes: int) -> str:
    """``every 24 h``-style wording: days, hours or minutes, whichever is whole."""
    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days} day{'s' * (days != 1)}"
    if minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


def explain(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A plain-language summary of one ledger decision (a ``timeline`` decision entry).

    Returns ``headline``, ``action`` and lists ``changes`` (what changed), ``matters`` (why
    it matters for the release) and ``uncertain`` (what the evidence cannot show), plus the
    same as ``markdown`` for pasting into an issue, chat or CI job summary.
    """
    summary: Mapping[str, Any] = entry.get("summary") or {}
    verdict = str(entry.get("verdict"))
    candidate = candidate_name(entry)
    accepted = {a.get("rule") for a in summary.get("accepted") or []}
    failing = summary.get("n_reasons") or 0
    if verdict == "fail":
        failing = max(failing - len(accepted), 1)  # accepted rules no longer block
    monitor = entry.get("trigger") == "monitor"
    headlines = MONITOR_HEADLINES if monitor else HEADLINES
    headline = headlines.get(verdict, "{candidate}: " + verdict).format(
        candidate=candidate,
        n=failing,
        s="s" * (failing != 1),
        verb="" if failing != 1 else "s",
    )
    changes = [line for name, stats in _ordered(summary) if (line := _change(name, stats))]
    if latency := _latency(summary.get("latency")):
        changes.append(latency)
    settings = summary.get("changed_settings") or []
    if settings:
        shown = ", ".join(settings[:8])
        more = f" and {len(settings) - 8} more" if len(settings) > 8 else ""
        changes.append(f"Changed settings: {shown}{more}.")
    elif verdict not in ("error", "not_comparable"):
        changes.append("No configuration change was recorded between the two runs.")
    changes += _drift(summary.get("drift"))

    matters = []
    if verdict in ("fail", "accepted", "error", "not_comparable"):
        reasons = summary.get("reasons") or []
        matters += [str(r) for r in reasons]
        hidden = (summary.get("n_reasons") or 0) - len(reasons)
        if hidden > 0:
            matters.append(f"…and {hidden} more in the full report.")
    if accepted:
        matters.append(
            f"Accepted by exception: {', '.join(sorted(str(r) for r in accepted if r))}."
        )
    within = [
        f"{metric_label(name)} (allowed drop {stats['tolerance']['margin']:.3f})"
        for name, stats in _ordered(summary)
        if (stats.get("tolerance") or {}).get("decision") == "within_tolerance"
    ]
    if within:
        matters.append(
            "Shown within tolerance under the non-inferiority policy (the whole 95% interval "
            "stays above the allowed drop): " + ", ".join(within) + "."
        )
    critical = summary.get("critical") or {}
    if critical.get("total"):
        if critical.get("failed"):
            matters.append(
                f"{critical['failed']} of {critical['total']} critical questions failed their "
                "required ranking."
            )
        else:
            matters.append(f"All {critical['total']} critical questions kept their ranking.")
    regressed = summary.get("regressed_queries")
    if regressed:
        metric = summary.get("primary_metric")
        on = f" on {metric_label(metric)}" if metric else ""
        matters.append(
            f"{regressed} question{'s' * (regressed != 1)} scored lower than with the "
            f"baseline{on}; the report lists which."
        )
    uncertain = _uncertain(entry, summary)
    result = {
        "check_id": entry.get("check_id"),
        "verdict": verdict,
        "candidate": candidate,
        "headline": headline,
        "trigger": entry.get("trigger") or "release",
        "action": (MONITOR_ACTIONS if monitor else ACTIONS).get(verdict, ""),
        "changes": changes,
        "matters": matters,
        "uncertain": uncertain,
        "evidence_available": entry.get("evidence_available"),
    }
    result["markdown"] = to_markdown(entry, result)
    return result


def _md(text: str) -> str:
    """Neutralize Markdown that uploaded labels or reasons could inject (links, HTML, code)."""
    for char in "\\`*_[]<>|#":
        text = text.replace(char, "\\" + char)
    return text


VERDICT_WORDS = {
    "pass": "Pass",
    "fail": "Fail",
    "accepted": "Fail, accepted",
    "error": "Error",
    "not_comparable": "Not comparable",
}


def to_markdown(entry: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """The summary as Markdown; every uploaded string is escaped."""
    lines = [
        f"**{VERDICT_WORDS.get(result['verdict'], _md(result['verdict']))}**: "
        f"{_md(result['headline'])}",
        "",
        _md(result["action"]),
    ]
    for title, items in (
        ("What changed", result["changes"]),
        ("Why it matters", result["matters"]),
        ("What is uncertain", result["uncertain"]),
    ):
        if items:
            lines += ["", f"**{title}**", *[f"- {_md(item)}" for item in items]]
    at = entry.get("at")
    when = at.strftime("%Y-%m-%d %H:%M UTC") if isinstance(at, datetime) else str(at or "")
    baseline = entry.get("baseline_name") or (entry.get("baseline_run_id") or "")[:12]
    policy = (entry.get("policy_digest") or "")[:12]
    lines += [
        "",
        f"Baseline {_md(str(baseline))}, policy {policy}, checked {when}, "
        f"check {_md(str(entry.get('check_id') or ''))}.",
    ]
    return "\n".join(lines) + "\n"


def _point(decision: Mapping[str, Any], value: float, baseline: float | None) -> dict[str, Any]:
    return {
        "at": decision["at"],
        "check_id": decision.get("check_id"),
        "verdict": decision["verdict"],
        "trigger": decision.get("trigger") or "release",
        "candidate": candidate_name(decision),
        "policy_digest": decision.get("policy_digest"),
        "value": value,
        "baseline": baseline,
    }


def trends(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Candidate quality and p95 latency per benchmark version, oldest point first.

    ``decisions`` are ledger rows with ``at``, ``verdict``, ``dataset_sha256``,
    ``policy_digest``, candidate identity and ``summary``. Returns one series per benchmark
    version, the most recently checked first, each with per-metric points (candidate value,
    baseline value, 95% interval), latency points, the first-to-latest change and the times
    the policy changed within the series (a verdict before and after a policy change was
    judged by different rules).
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for decision in sorted(decisions, key=lambda d: d["at"]):
        groups.setdefault(decision["dataset_sha256"], []).append(decision)
    series = []
    for dataset, rows in groups.items():
        metrics: dict[str, list[dict[str, Any]]] = {}
        latency: list[dict[str, Any]] = []
        for row in rows:
            summary = row.get("summary") or {}
            for name, stats in _ordered(summary):
                if stats.get("candidate") is None:
                    continue
                point = _point(row, stats["candidate"], stats.get("baseline"))
                point.update(ci_low=stats.get("ci_low"), ci_high=stats.get("ci_high"))
                metrics.setdefault(name, []).append(point)
            lat = summary.get("latency") or {}
            if lat.get("candidate_p95") is not None:
                latency.append(_point(row, lat["candidate_p95"], lat.get("baseline_p95")))
        policy_changes = [
            later["at"]
            for earlier, later in pairwise(rows)
            if earlier.get("policy_digest") != later.get("policy_digest")
        ]
        primary = next(
            (r["summary"].get("primary_metric") for r in reversed(rows) if r.get("summary")),
            None,
        )
        series.append(
            {
                "dataset_sha256": dataset,
                "decisions": len(rows),
                "first_at": rows[0]["at"],
                "last_at": rows[-1]["at"],
                "primary_metric": primary,
                "verdicts": {
                    v: sum(r["verdict"] == v for r in rows)
                    for v in sorted({r["verdict"] for r in rows})
                },
                "metrics": {
                    name: {
                        "label": metric_label(name),
                        "points": points,
                        "change": points[-1]["value"] - points[0]["value"],
                    }
                    for name, points in sorted(
                        metrics.items(), key=lambda kv: (kv[0] != primary, kv[0])
                    )
                },
                "latency_p95": {
                    "points": latency,
                    "change": latency[-1]["value"] - latency[0]["value"] if latency else None,
                },
                "policy_changes": policy_changes,
            }
        )
    series.sort(key=lambda s: s["last_at"], reverse=True)
    return series


KINDS = ("regressed", "critical", "safety")


def recurring(decisions: Sequence[Mapping[str, Any]], *, min_failures: int = 2) -> dict[str, Any]:
    """Questions that failed in more than one check on the current benchmark version.

    ``decisions`` are ledger rows, newest first, with ``dataset_sha256`` and ``summary``.
    Only the benchmark version of the newest decision is counted, because question IDs on
    another version may mean other questions. Each question reports how many of the checks
    that list questions it failed in (the denominator), by kind (regressed on the primary
    metric, a failing critical rule, a safety failure), whether it still fails in the newest
    check and for how many checks in a row. Decisions recorded before questions were kept
    (ledger summary version 6) are counted separately, never as passes.
    """
    if not decisions:
        return {"dataset_sha256": None, "checks": 0, "without_questions": 0, "questions": []}
    dataset = decisions[0]["dataset_sha256"]
    rows = [d for d in decisions if d["dataset_sha256"] == dataset]
    listed = [d for d in rows if (d.get("summary") or {}).get("failed_questions") is not None]
    stats: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(listed):  # newest first
        failed = row["summary"]["failed_questions"]
        seen = set()
        for kind in KINDS:
            for qid in failed.get(kind) or []:
                entry = stats.setdefault(
                    qid,
                    {
                        "query_id": qid,
                        "failures": 0,
                        "kinds": {k: 0 for k in KINDS},
                        "triggers": {"release": 0, "monitor": 0},
                        "positions": [],
                        "last_failed_at": row["at"],
                        "last_check_id": row.get("check_id"),
                        "first_failed_at": row["at"],
                    },
                )
                entry["kinds"][kind] += 1
                if qid in seen:
                    continue
                seen.add(qid)
                entry["failures"] += 1
                entry["triggers"][row.get("trigger") or "release"] += 1
                entry["positions"].append(position)
                entry["first_failed_at"] = row["at"]
    questions = []
    for entry in stats.values():
        if entry["failures"] < min_failures:
            continue
        positions = entry.pop("positions")
        streak = 0
        while streak < len(positions) and positions[streak] == streak:
            streak += 1
        entry["still_failing"] = streak > 0
        entry["streak"] = streak
        entry["checks"] = len(listed)
        questions.append(entry)
    questions.sort(
        key=lambda e: (not e["still_failing"], -e["failures"], -e["last_failed_at"].timestamp())
    )
    return {
        "dataset_sha256": dataset,
        "checks": len(listed),
        "without_questions": len(rows) - len(listed),
        "questions": questions,
    }
