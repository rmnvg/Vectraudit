"""Dashboard views for an engineering lead: decision summaries and quality trends.

Both read only the decision ledger, so they keep working after raw runs and checks expire.
Charts are inline SVG rendered on the server; pages stay script-free under the dashboard CSP.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, Response

from vectraudit.cloud import lead, ledger, web
from vectraudit.cloud.deps import SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError, not_found

router = APIRouter(include_in_schema=False)
e = web.e
WIDTH, HEIGHT, PAD = 640, 170, 28
MONITOR_TAG = ' <span class="muted">Monitor</span>'
COLORS = {
    "pass": "var(--pass)",
    "fail": "var(--fail)",
    "accepted": "var(--error)",
    "error": "var(--error)",
    "not_comparable": "var(--nc)",
}


def chart(
    points: Sequence[Mapping[str, Any]],
    *,
    title: str,
    fmt: Callable[[float], str],
    policy_changes: Sequence[Any] = (),
) -> str:
    """An accessible inline-SVG line chart of candidate values, with the baseline dashed.

    Points are spaced by check order, not time. Each dot is coloured by verdict and has a
    hover title; a vertical rule marks a policy change.
    """
    values = [p["value"] for p in points] + [
        p["baseline"] for p in points if p.get("baseline") is not None
    ]
    low, high = float(min(values)), float(max(values))
    if high - low < 1e-9:
        low, high = low - 0.01 * max(abs(low), 1), high + 0.01 * max(abs(high), 1)
    span = high - low
    step = (WIDTH - 2 * PAD) / max(len(points) - 1, 1)

    def x(i: int) -> float:
        return PAD + i * step if len(points) > 1 else WIDTH / 2

    def y(v: float) -> float:
        return HEIGHT - PAD - (v - low) / span * (HEIGHT - 2 * PAD)

    parts = [
        f'<svg role="img" viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" '
        f'style="max-width:{WIDTH}px;display:block" aria-label="{e(title)}">',
        f"<title>{e(title)}</title>",
        f'<line x1="{PAD}" y1="{HEIGHT - PAD}" x2="{WIDTH - PAD}" y2="{HEIGHT - PAD}" '
        'stroke="var(--line)"/>',
        f'<text x="2" y="{PAD - 8}" font-size="11" fill="var(--muted)">{e(fmt(high))}</text>',
        f'<text x="2" y="{HEIGHT - 8}" font-size="11" fill="var(--muted)">{e(fmt(low))}</text>',
    ]
    changes = set(policy_changes)
    for i, p in enumerate(points):
        if p["at"] in changes:
            parts.append(
                f'<line x1="{x(i) - step / 2:.1f}" y1="{PAD - 6}" x2="{x(i) - step / 2:.1f}" '
                f'y2="{HEIGHT - PAD}" stroke="var(--muted)" stroke-dasharray="2 3">'
                "<title>Policy changed</title></line>"
            )
    base = [(x(i), y(p["baseline"])) for i, p in enumerate(points) if p.get("baseline") is not None]
    if len(base) > 1:
        coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in base)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="var(--muted)" '
            'stroke-dasharray="5 4"><title>Baseline</title></polyline>'
        )
    if len(points) > 1:
        coords = " ".join(f"{x(i):.1f},{y(p['value']):.1f}" for i, p in enumerate(points))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="var(--link)"/>')
    for i, p in enumerate(points):
        tip = (
            f"{p['at'].strftime('%Y-%m-%d %H:%M')} · {p['candidate']} · "
            f"{lead.VERDICT_WORDS.get(p['verdict'], p['verdict'])} · {fmt(p['value'])}"
        )
        if p.get("baseline") is not None:
            tip += f" (baseline {fmt(p['baseline'])})"
        color = COLORS.get(p["verdict"], "var(--nc)")
        if p.get("trigger") == "monitor":  # hollow: a scheduled run, not a proposed change
            tip = "Scheduled monitor · " + tip
            paint = f'fill="var(--bg)" stroke="{color}" stroke-width="2"'
        else:
            paint = f'fill="{color}"'
        parts.append(
            f'<circle cx="{x(i):.1f}" cy="{y(p["value"]):.1f}" r="4" {paint}>'
            f"<title>{e(tip)}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _quality(value: float) -> str:
    return f"{value:.3f}"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _list(items: Sequence[str]) -> str:
    return "<ul>" + "".join(f"<li>{e(item)}</li>" for item in items) + "</ul>"


def summary_html(result: Mapping[str, Any], *, heading: str = "h2") -> str:
    """The lead summary as HTML sections (shared by the decision page and the project page)."""
    verdict = result["verdict"]
    body = (
        f"<{heading}>{web._badge(verdict)} {e(result['headline'])}</{heading}>"
        f"<p><strong>{e(result['action'])}</strong></p>"
    )
    for title, key in (
        ("What changed", "changes"),
        ("Why it matters", "matters"),
        ("What is uncertain", "uncertain"),
    ):
        if result[key]:
            body += f"<h3>{title}</h3>{_list(result[key])}"
    return body


@router.get("/app/{workspace}/{project}/decisions/{check_id}")
def decision_page(
    workspace: str,
    project: str,
    check_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> HTMLResponse:
    try:
        identity, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
        entry = ledger.decision(session, access.project, check_id)
        if entry is None:
            raise not_found("Decision")
    except CloudError as err:
        return web._error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    result = lead.explain(entry)
    evidence = (
        f'<a href="{base}/checks/{quote(check_id)}">Full report</a>'
        if entry["evidence_available"]
        else '<span class="muted">Full report expired under retention</span>'
    )
    body = (
        f'<div class="crumbs"><a href="{base}">{e(project)}</a> / decision</div>'
        '<div class="card">' + summary_html(result, heading="h1") + "</div>"
        f"<p>{evidence} · "
        f'<a href="{base}/decisions/{quote(check_id)}/summary.md">Download as Markdown</a> · '
        f'<a href="{base}/trends">Quality trends</a></p>'
        f'<p class="muted">Baseline {e(entry["baseline_name"] or entry["baseline_run_id"][:12])}, '
        f"policy {e((entry['policy_digest'] or '')[:12])}, checked "
        f"{e(entry['at'].strftime('%Y-%m-%d %H:%M'))} UTC by "
        f"{e(web._actor(session, entry['decided_by']))}. Generated from the decision ledger; "
        "the full report has every rule and question.</p>"
        "<h2>Paste into an issue or chat</h2>"
        f'<pre style="white-space:pre-wrap">{e(result["markdown"])}</pre>'
    )
    return web.page("Release decision", body, identity)


@router.get("/app/{workspace}/{project}/decisions/{check_id}/summary.md")
def decision_markdown(
    workspace: str,
    project: str,
    check_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    try:
        _, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
        entry = ledger.decision(session, access.project, check_id)
        if entry is None:
            raise not_found("Decision")
    except CloudError as err:
        return web._error_page(err, None)
    filename = "".join(c for c in check_id if c.isalnum())[:32] or "decision"
    return Response(
        lead.explain(entry)["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="decision-{filename}.md"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )


@router.get("/app/{workspace}/{project}/trends")
def trends_page(
    workspace: str,
    project: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=200, ge=2, le=1000),
) -> HTMLResponse:
    try:
        identity, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
        rows = ledger.history(session, access.project, limit=limit)
    except CloudError as err:
        return web._error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    body = (
        f'<div class="crumbs"><a href="{base}">{e(project)}</a> / trends</div>'
        "<h1>Quality and latency trends</h1>"
        '<p class="muted">Each check, oldest on the left: the candidate (solid, dot coloured '
        "by verdict; hollow for a scheduled monitor run) against its baseline (dashed). Each "
        "benchmark version is a "
        "separate series, because scores on different questions are not comparable. A dotted "
        "rule marks a policy change: verdicts on either side were judged by different rules. "
        "Read from the decision ledger, so checks whose runs expired still appear. "
        f"Showing the newest {len(rows)} decisions.</p>"
    )
    series = lead.trends(rows)
    if not series:
        body += (
            '<p class="empty">No release decisions yet: run '
            "<code>vectraudit check cloud:main CANDIDATE --upload</code> in CI.</p>"
        )
    for number, s in enumerate(series):
        current = " (current)" if number == 0 else ""
        verdicts = ", ".join(
            f"{count} {lead.VERDICT_WORDS.get(v, v).lower()}" for v, count in s["verdicts"].items()
        )
        body += (
            f'<h2>Benchmark <span class="mono">{e(s["dataset_sha256"][:12])}</span>{current}</h2>'
            f'<p class="muted">{s["decisions"]} checks from '
            f"{e(s['first_at'].strftime('%Y-%m-%d'))} to {e(s['last_at'].strftime('%Y-%m-%d'))}: "
            f"{e(verdicts)}.</p>"
        )
        metrics = list(s["metrics"].values())
        sections = [
            f"<h3>{e(m['label'])}</h3>"
            f'<p class="muted">Latest {m["points"][-1]["value"]:.3f}; '
            f"{m['change']:+.3f} since the first check in this series.</p>"
            + chart(
                m["points"],
                title=f"{m['label']} per release check",
                fmt=_quality,
                policy_changes=s["policy_changes"],
            )
            for m in metrics
        ]
        body += "".join(sections[:1])
        if len(sections) > 1 and number == 0:  # older benchmark versions: primary metric only
            body += (
                f"<details><summary>Other metrics ({len(sections) - 1})</summary>"
                + "".join(sections[1:])
                + "</details>"
            )
        latency = s["latency_p95"]["points"]
        if latency:
            body += (
                "<h3>p95 latency</h3>"
                f'<p class="muted">Latest {latency[-1]["value"]:.1f} ms; '
                f"{s['latency_p95']['change']:+.1f} ms since the first check in this series. "
                "Lower is better.</p>"
                + chart(
                    latency,
                    title="p95 latency per release check",
                    fmt=_ms,
                    policy_changes=s["policy_changes"],
                )
            )
        recent = list(reversed(metrics[0]["points"]))[:10] if metrics else []
        if recent:
            body += web._table(
                ["When", "Verdict", "Candidate", "Value", "Baseline", ""],
                [
                    [
                        e(p["at"].strftime("%Y-%m-%d %H:%M")),
                        web._badge(p["verdict"]) + MONITOR_TAG * (p["trigger"] == "monitor"),
                        e(p["candidate"]),
                        f"{p['value']:.3f}",
                        f"{p['baseline']:.3f}" if p.get("baseline") is not None else "",
                        f'<a href="{base}/decisions/{quote(p["check_id"])}">Summary</a>'
                        if p.get("check_id")
                        else "",
                    ]
                    for p in recent
                ],
                "",
            )
    return web.page("Trends", body, identity)


KIND_WORDS = {"regressed": "scored lower", "critical": "critical rule", "safety": "safety rule"}


def recurring_rows(
    session: Any, project: Any, base: str, limit: int, top: int | None = None
) -> tuple[dict[str, Any], list[list[str]]]:
    """The recurring-question table rows (shared by the project page and the full page)."""
    found = lead.recurring(ledger.history(session, project, limit=limit))
    questions = found["questions"][:top] if top else found["questions"]
    cases = ledger.cases_by_question(session, project, {q["query_id"] for q in questions})
    rows = []
    for q in questions:
        linked = cases.get(q["query_id"], [])
        text = next((c["question_text"] for c in linked if c["question_text"]), None)
        question = f'<span class="mono">{e(q["query_id"])}</span>'
        if text:
            question += f"<br>{e(text)}"
        kinds = ", ".join(f"{KIND_WORDS[k]} ({n})" for k, n in q["kinds"].items() if n)
        state = (
            f"{web._badge('fail')} {q['streak']} in a row"
            if q["still_failing"]
            else '<span class="muted">Not in the latest check</span>'
        )
        last = e(q["last_failed_at"].strftime("%Y-%m-%d"))
        if q["last_check_id"]:
            last = f'<a href="{base}/decisions/{quote(q["last_check_id"])}">{last}</a>'
        case_links = (
            " ".join(
                f'<a href="{base}/records/{quote(c["record_id"])}">{e(c["source_id"][:8])}</a> '
                f'<span class="muted">({e(c["status"] or "unknown")} when shared)</span>'
                for c in linked
            )
            or '<span class="muted">None shared</span>'
        )
        rows.append(
            [question, f"{q['failures']} of {q['checks']}", e(kinds), state, last, case_links]
        )
    return found, rows


RECURRING_HEADERS = ["Question", "Failed in", "How", "Now", "Last failed", "Regression case"]


@router.get("/app/{workspace}/{project}/recurring")
def recurring_page(
    workspace: str,
    project: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=200, ge=2, le=1000),
) -> HTMLResponse:
    try:
        identity, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
    except CloudError as err:
        return web._error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    found, rows = recurring_rows(session, access.project, base, limit)
    note = (
        f"Questions that failed in at least two of the newest {found['checks']} checks on the "
        "current benchmark version, still-failing first. A question counts once per check: "
        "it scored lower than the baseline on the primary metric, broke a critical rule, or "
        "broke a safety rule. Question text appears only when a regression case was shared "
        "with it; otherwise IDs are shown."
    )
    if found["without_questions"]:
        note += (
            f" {found['without_questions']} older decisions on this benchmark were recorded "
            "before questions were kept and are not counted."
        )
    body = (
        f'<div class="crumbs"><a href="{base}">{e(project)}</a> / recurring problems</div>'
        f'<h1>Recurring problems</h1><p class="muted">{e(note)}</p>'
        + web._table(RECURRING_HEADERS, rows, "No question has failed more than once.")
    )
    return web.page("Recurring problems", body, identity)
