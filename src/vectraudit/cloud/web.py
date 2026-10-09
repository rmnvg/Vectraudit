"""The dashboard: server-rendered pages over the same access rules as the API.

No tokens live in browser storage: sign-in creates a server-side session (see
:mod:`vectraudit.cloud.web_auth`). Every value is HTML-escaped, every state-changing form
carries a CSRF token, and pages send a strict Content-Security-Policy with no scripts.
"""

from __future__ import annotations

import html
import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, quote, urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from vectraudit import __version__
from vectraudit.cloud import exceptions, ledger, quotas, repo, service
from vectraudit.cloud.auth import Principal, authenticate
from vectraudit.cloud.deps import SessionDep, SettingsDep, client_key, record_failure
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import Run, User, Workspace, utcnow
from vectraudit.cloud.settings import Settings
from vectraudit.cloud.web_auth import (
    INVITE_HOURS,
    SESSION_COOKIE,
    SESSION_DAYS,
    create_invite,
    create_session,
    csrf_ok,
    csrf_token,
    redeem_invite,
    resolve_session,
    revoke_session,
    session_principal,
)
from vectraudit.evaluation.report import render_html

router = APIRouter(include_in_schema=False)
LOGIN_CSRF = "va_login_csrf"
OAUTH_STATE = "va_oauth_state"
MAX_FORM = 16_384
GITHUB = "https://github.com"
GITHUB_API = "https://api.github.com"
PAGE_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; frame-src 'self'; form-action 'self'; "
    "frame-ancestors 'none'; base-uri 'none'"
)
VERDICT = {
    "pass": "Pass",
    "fail": "Fail",
    "error": "Error",
    "not_comparable": "Not comparable",
    "accepted": "Fail, accepted",
}
Http = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]

CSS = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b636d;--line:#d9dde3;--card:#f6f8fa;--link:#0b62d6;
--pass:#1a7f37;--fail:#cf222e;--error:#9a6700;--nc:#6e7781}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--line:#30363d;--card:#161b22;--link:#58a6ff;--pass:#3fb950;--fail:#f85149;--error:#d29922;
--nc:#8b949e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;
align-items:center;flex-wrap:wrap}header .brand{font-weight:700}header .spacer{flex:1}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}
.muted{color:var(--muted)}.crumbs{color:var(--muted);margin-bottom:8px}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;color:#fff;font-size:12px;
font-weight:600}.pass{background:var(--pass)}.fail{background:var(--fail)}
.error{background:var(--error)}.not_comparable{background:var(--nc)}
.accepted{background:var(--error)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}.mono{font-family:ui-monospace,monospace;font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
margin:10px 0}form.inline{display:inline}input,button,select{font:inherit;padding:5px 8px;
border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{cursor:pointer}button.primary{background:var(--fg);color:var(--bg)}
.notice{border-left:3px solid var(--link);padding:8px 12px;background:var(--card);margin:12px 0}
.secret{font-family:ui-monospace,monospace;word-break:break-all;padding:8px;
background:var(--card);border:1px dashed var(--line);border-radius:6px}
iframe{width:100%;height:1600px;border:1px solid var(--line);border-radius:8px;
background:var(--bg)}
"""


class _Redirect(Exception):
    def __init__(self, location: str) -> None:
        self.location = location


@dataclass
class Web:
    """The signed-in user for a dashboard request."""

    user: User
    principal: Principal
    csrf: str
    raw_session: str


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _secure(settings: Settings) -> bool:
    return settings.public_url.startswith("https://")


def page(title: str, body: str, web: Web | None = None, status: int = 200) -> HTMLResponse:
    nav = ""
    if web is not None:
        nav = (
            f'<a href="/account">Settings</a><span class="muted">{e(web.user.email)}</span>'
            f'<form class="inline" method="post" action="/logout">'
            f'<input type="hidden" name="csrf" value="{e(web.csrf)}">'
            f"<button>Sign out</button></form>"
        )
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)} · Vectraudit</title><style>{CSS}</style></head><body>"
        f'<header><a class="brand" href="/app">Vectraudit</a><span class="spacer"></span>{nav}'
        f"</header><main>{body}</main></body></html>"
    )
    return HTMLResponse(
        doc,
        status_code=status,
        headers={
            "Content-Security-Policy": PAGE_CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "same-origin",
            "Cache-Control": "private, no-store",
        },
    )


def _table(header: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="muted">{e(empty)}</p>'
    head = "".join(f"<th>{e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def _actor(session: Session, actor: str | None) -> str:
    """``user:<id>`` as the person's email; token prefixes as they are."""
    if actor and actor.startswith("user:"):
        user = session.get(User, actor[5:])
        if user is not None:
            return user.email
    return actor or ""


def _badge(verdict: str) -> str:
    return f'<span class="badge {e(verdict)}">{e(VERDICT.get(verdict, verdict))}</span>'


def _metric_change(summary: Mapping[str, Any]) -> str:
    """``ndcg 0.812 → 0.799 (-0.013)`` for the decision's primary (or first) metric."""
    metrics = summary.get("metrics") or {}
    name = summary.get("primary_metric") or next(iter(metrics), None)
    stats = metrics.get(name) if name else None
    if not stats or stats.get("baseline") is None or stats.get("candidate") is None:
        return ""
    delta = stats.get("delta") or 0.0
    return f"{e(name)} {stats['baseline']:.3f} → {stats['candidate']:.3f} ({delta:+.3f})"


def _timeline_rows(session: Session, entries: list[dict[str, Any]], base: str) -> list[list[str]]:
    rows = []
    for entry in entries:
        when = e(entry["at"].strftime("%Y-%m-%d %H:%M"))
        if entry["kind"] == "exception":
            rows.append(
                [
                    when,
                    f'<span class="muted">Exception {e(entry["action"])}</span>',
                    f'<span class="mono">{e(entry["rule"] or "")}</span>',
                    "",
                    e(
                        f"until {entry['expires_at'][:10]}, owner {entry['owner']}"
                        if entry["action"] == "granted" and entry["expires_at"]
                        else ""
                    ),
                    e(_actor(session, entry["actor"])),
                ]
            )
            continue
        if entry["kind"] == "baseline_pin":
            rows.append(
                [
                    when,
                    '<span class="muted">Baseline pinned</span>',
                    f"{e(entry['baseline_name'] or '')} → "
                    f'<span class="mono">{e((entry["run_id"] or "")[:12])}</span>',
                    "",
                    e(entry["note"] or ""),
                    e(_actor(session, entry["actor"])),
                ]
            )
            continue
        summary = entry["summary"] or {}
        reasons = summary.get("reasons") or []
        candidate = entry["candidate_label"] or entry["candidate_run_id"][:12]
        if entry["candidate_commit"]:
            candidate += f" @ {entry['candidate_commit'][:10]}"
        evidence = (
            f'<a href="{base}/checks/{quote(entry["check_id"])}">Report</a>'
            if entry["evidence_available"]
            else '<span class="muted">Expired under retention</span>'
        )
        rows.append(
            [
                when,
                _badge(entry["verdict"]),
                e(candidate),
                _metric_change(summary),
                e(reasons[0] if reasons else "All policy rules hold."),
                evidence,
            ]
        )
    return rows


def _accept_section(web: Web, access: repo.Access, check: Any, base: str) -> str:
    """Failing rules of a check, accepted ones and, for allowed approvers, an accept form."""
    from vectraudit.evaluation.exceptions import required_role, rule_kind

    reasons = [r for r in check.result.get("reasons", []) if r.get("verdict") == "fail"]
    if not reasons:
        return ""
    policy = service.project_policy(access.project) if access.project else None
    rows = []
    for reason in reasons:
        rule = reason.get("rule")
        accepted = reason.get("accepted")
        if accepted:
            action = (
                f"Accepted by {e(accepted['accepted_by'])} until "
                f"{e(str(accepted['expires_at'])[:10])}: {e(accepted['reason'])}"
            )
        elif rule is None or policy is None:
            action = '<span class="muted">Cannot be accepted</span>'
        else:
            need = required_role(policy, rule)
            if need is None:
                action = (
                    f'<span class="muted">The policy does not allow exceptions for '
                    f"{e(rule_kind(rule))} rules</span>"
                )
            elif repo.ROLE_RANK.get(access.role, 0) < repo.ROLE_RANK[need]:
                action = f'<span class="muted">Needs the {e(need)} role to accept</span>'
            else:
                action = (
                    f'<form method="post" action="{base}/checks/{quote(check.id)}/accept">'
                    f"{_csrf_field(web)}"
                    f'<input type="hidden" name="rule" value="{e(rule)}">'
                    '<input name="reason" placeholder="Why this is acceptable" required '
                    'maxlength="2000"> <input name="owner" placeholder="Who follows up" '
                    'required maxlength="200"> <input name="expires" type="date" required> '
                    '<input name="max_checks" placeholder="Max checks (optional)" size="8"> '
                    "<button>Accept until expiry</button></form>"
                )
        rows.append([e(reason["message"]), f'<span class="mono">{e(rule or "")}</span>', action])
    return (
        "<h2>Failing rules</h2>"
        '<p class="muted">An accepted rule makes the check read "fail, accepted", never pass. '
        "Every acceptance is audited and expires.</p>"
        + _table(["Reason", "Rule", "Acceptance"], rows, "")
    )


def _exception_rows(
    session: Session, web: Web, rows: list[Any], base: str, can_write: bool
) -> list[list[str]]:
    out = []
    for row in rows:
        data = exceptions.payload(session, row)
        uses = f"{data['uses']}/{data['max_checks']}" if data["max_checks"] else str(data["uses"])
        revoke = ""
        if can_write and data["status"] == "active":
            revoke = (
                f'<form class="inline" method="post" '
                f'action="{base}/exceptions/{quote(data["id"])}/revoke">{_csrf_field(web)}'
                "<button>Revoke</button></form>"
            )
        out.append(
            [
                f'<span class="mono">{e(data["rule"])}</span>',
                e(data["status"]),
                e(data["accepted_by"]),
                e(data["owner"]),
                e(data["expires_at"][:10]),
                e(uses),
                e(data["reason"]),
                revoke,
            ]
        )
    return out


def _csrf_field(web: Web) -> str:
    return f'<input type="hidden" name="csrf" value="{e(web.csrf)}">'


async def _form(request: Request) -> dict[str, str]:
    raw = await request.body()
    if len(raw) > MAX_FORM:
        raise CloudError(413, "payload_too_large", "Form too large.")
    return {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace")).items()}


def _web(request: Request, session: Session, settings: Settings) -> Web:
    raw = request.cookies.get(SESSION_COOKIE)
    found = resolve_session(session, raw)
    if found is None or raw is None:
        raise _Redirect("/login?next=" + quote(request.url.path))
    return Web(found[1], session_principal(found[1]), csrf_token(settings.secret_key, raw), raw)


def _require_csrf(web: Web, form: Mapping[str, str], settings: Settings) -> None:
    if not csrf_ok(settings.secret_key, web.raw_session, form.get("csrf")):
        raise CloudError(403, "csrf", "This form expired. Go back, reload and try again.")


def _safe_next(target: str | None) -> str:
    """Only same-site paths: never redirect to another host after sign-in."""
    if target and target.startswith("/app") and not target.startswith("//"):
        return target
    return "/app"


def _signed_in(response: Response, raw: str, settings: Settings) -> Response:
    response.set_cookie(
        SESSION_COOKIE,
        raw,
        max_age=SESSION_DAYS * 86_400,
        httponly=True,
        secure=_secure(settings),
        samesite="lax",
        path="/",
    )
    return response


# --- sign-in -----------------------------------------------------------------------------


@router.get("/login")
def login_page(
    request: Request, settings: SettingsDep, next: str = "/app", error: str = ""
) -> HTMLResponse:
    state = secrets.token_urlsafe(24)
    github = (
        '<p><a class="badge pass" href="/auth/sso">Enterprise sign-in</a></p>'
        if settings.sso
        else ""
    )
    if settings.github_client_id:
        github += (
            f'<p><a class="badge pass" href="/auth/github?next={quote(_safe_next(next))}">'
            "Sign in with GitHub</a></p>"
        )
    notice = f'<div class="notice">{e(error)}</div>' if error else ""
    body = (
        f"<h1>Sign in</h1>{notice}{github}"
        '<div class="card"><p>Or paste a personal API token (from your workspace owner or '
        "<code>vectraudit-cloud create-token</code>):</p>"
        '<form method="post" action="/login/token">'
        f'<input type="hidden" name="csrf" value="{e(state)}">'
        f'<input type="hidden" name="next" value="{e(_safe_next(next))}">'
        '<input type="password" name="token" placeholder="va_pat_…" size="48" required> '
        '<button class="primary">Sign in</button></form></div>'
        '<p class="muted">Access is by invitation: a workspace owner adds you by email.</p>'
    )
    response = page("Sign in", body)
    response.set_cookie(
        LOGIN_CSRF,
        state,
        max_age=900,
        httponly=True,
        secure=_secure(settings),
        samesite="lax",
        path="/login",
    )
    return response


@router.post("/login/token")
async def login_token(request: Request, session: SessionDep, settings: SettingsDep) -> Response:
    retry = request.app.state.auth_failures.retry_after(client_key(request))
    if retry is not None:
        return RedirectResponse(
            "/login?error=" + quote(f"Too many attempts; wait {retry} seconds."), status_code=303
        )
    form = await _form(request)
    expected = request.cookies.get(LOGIN_CSRF)
    if not expected or not secrets.compare_digest(expected, form.get("csrf", "")):
        return RedirectResponse(
            "/login?error=" + quote("The sign-in form expired; try again."), status_code=303
        )
    try:
        principal = authenticate(session, f"Bearer {form.get('token', '').strip()}")
    except CloudError:
        record_failure(request, "sign-in")
        return RedirectResponse(
            "/login?error=" + quote("That token is not valid."), status_code=303
        )
    if principal.kind != "user":
        return RedirectResponse(
            "/login?error=" + quote("Runner tokens can't sign in; use a personal token."),
            status_code=303,
        )
    user = session.get(User, principal.user_id)
    assert user is not None
    raw = create_session(session, user, "token")
    response = RedirectResponse(_safe_next(form.get("next")), status_code=303)
    response.delete_cookie(LOGIN_CSRF, path="/login")
    return _signed_in(response, raw, settings)


@router.get("/auth/github")
def github_start(settings: SettingsDep, next: str = "/app") -> Response:
    if not settings.github_client_id:
        raise CloudError(404, "not_found", "GitHub sign-in is not configured.")
    state = secrets.token_urlsafe(24)
    query = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": f"{settings.public_url}/auth/github/callback",
            "scope": "read:user user:email",
            "state": state,
            "allow_signup": "false",
        }
    )
    response = RedirectResponse(f"{GITHUB}/login/oauth/authorize?{query}", status_code=303)
    response.set_cookie(
        OAUTH_STATE,
        f"{state}|{_safe_next(next)}",
        max_age=600,
        httponly=True,
        secure=_secure(settings),
        samesite="lax",
        path="/auth/github",
    )
    return response


def github_http(request: Request) -> Http:
    http: Http = getattr(request.app.state, "github_http", None) or _urllib_http
    return http


def _urllib_http(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    from vectraudit.integrations.cloud import urllib_send

    return urllib_send(method, url, headers, body)


@router.get("/auth/github/callback")
def github_callback(
    request: Request, session: SessionDep, settings: SettingsDep, code: str = "", state: str = ""
) -> Response:
    """Exchange the code, read the verified emails, and sign in an invited user."""

    def fail(message: str) -> Response:
        record_failure(request, "github")
        return RedirectResponse("/login?error=" + quote(message), status_code=303)

    retry = request.app.state.auth_failures.retry_after(client_key(request))
    if retry is not None:
        return RedirectResponse(
            "/login?error=" + quote(f"Too many attempts; wait {retry} seconds."), status_code=303
        )

    stored, _, next_path = (request.cookies.get(OAUTH_STATE) or "").partition("|")
    if not settings.github_client_id or not settings.github_client_secret:
        return fail("GitHub sign-in is not configured.")
    if not code or not stored or not secrets.compare_digest(stored, state):
        return fail("GitHub sign-in expired; try again.")
    http = github_http(request)
    status, raw = http(
        "POST",
        f"{GITHUB}/login/oauth/access_token",
        {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        urlencode(
            {
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": f"{settings.public_url}/auth/github/callback",
            }
        ).encode(),
    )
    token = json.loads(raw or b"{}").get("access_token") if status == 200 else None
    if not token:
        return fail("GitHub did not confirm the sign-in.")
    status, raw = http(
        "GET",
        f"{GITHUB_API}/user/emails",
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": f"vectraudit/{__version__}",
        },
        None,
    )
    emails = json.loads(raw or b"[]") if status == 200 else []
    # Verified addresses only, primary first: an unverified email proves nothing.
    entries = [m for m in emails if isinstance(m, dict) and m.get("verified") and m.get("email")]
    entries.sort(key=lambda m: not m.get("primary"))
    verified = [str(m["email"]).lower() for m in entries]
    user = None
    for email in verified:
        user = session.scalars(select(User).where(User.email == email)).first()
        if user is not None and not user.disabled:
            break
        user = None
    if user is None:
        shown = verified[0] if verified else "your GitHub account"
        return fail(f"No invitation for {shown}. Ask a workspace owner to add you.")
    raw_session = create_session(session, user, "github")
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    response.delete_cookie(OAUTH_STATE, path="/auth/github")
    return _signed_in(response, raw_session, settings)


@router.post("/logout")
async def logout(request: Request, session: SessionDep, settings: SettingsDep) -> Response:
    form = await _form(request)
    raw = request.cookies.get(SESSION_COOKIE)
    if raw and csrf_ok(settings.secret_key, raw, form.get("csrf")):
        revoke_session(session, raw)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


# --- pages -------------------------------------------------------------------------------


def _error_page(err: CloudError, web: Web | None) -> HTMLResponse:
    hint = f'<p class="muted">{e(err.hint)}</p>' if err.hint else ""
    return page(
        "Error",
        f'<h1>{e(err.message)}</h1>{hint}<p><a href="/app">Back</a></p>',
        web,
        status=err.status,
    )


@router.get("/")
def root() -> Response:
    return RedirectResponse("/app", status_code=303)


@router.get("/app")
def home(request: Request, session: SessionDep, settings: SettingsDep) -> HTMLResponse:
    web = _web(request, session, settings)
    rows = [
        [f'<a href="/app/{quote(ws.slug)}">{e(ws.name)}</a>', e(ws.slug), e(role), e(ws.plan)]
        for ws, role in repo.workspaces_of(session, web.user.id)
    ]
    body = "<h1>Workspaces</h1>" + _table(
        ["Workspace", "Slug", "Your role", "Plan"],
        rows,
        "You are not a member of any workspace yet.",
    )
    return page("Workspaces", body, web)


@router.get("/app/{workspace}")
def workspace_page(
    workspace: str, request: Request, session: SessionDep, settings: SettingsDep
) -> HTMLResponse:
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "read")
    except CloudError as err:
        return _error_page(err, web)
    ws = access.workspace
    projects = [
        [
            f'<a href="/app/{quote(ws.slug)}/{quote(p.slug)}">{e(p.name)}</a>',
            e(p.slug),
            e(p.created_at.date()),
        ]
        for p in repo.projects(session, ws, access)
    ]
    owner = access.role == "owner"
    members = [_member_row(ws.slug, u, m.role, web, owner) for u, m in repo.members(session, ws)]
    q = quotas.status(session, ws)
    lim, used = q["limits"], q["usage"]
    plan = (
        f'<div class="card">Plan <strong>{e(ws.plan)}</strong> · '
        f"{used['runs']:,} / {lim['runs_per_month']:,} runs this month · "
        f"{used['stored_runs']:,} / {lim['stored_runs']:,} runs stored · "
        f"up to {lim['queries_per_run']:,} queries per run · runs kept "
        f"{lim['retention_days']} days (pinned baselines are kept)</div>"
    )
    body = (
        f'<div class="crumbs"><a href="/app">Workspaces</a> / {e(ws.slug)}</div>'
        f'<h1>{e(ws.name)}</h1><p class="muted">Your role: {e(access.role)}</p>{plan}'
        "<h2>Projects</h2>"
        + _table(["Project", "Slug", "Created"], projects, "No projects yet.")
        + "<h2>Members</h2>"
        + _table(["Email", "Role", ""] if owner else ["Email", "Role"], members, "No members.")
        + (_invite_form(ws.slug, web) + _danger_zone(ws.slug, web) if owner else "")
    )
    return page(ws.name, body, web)


ROLE_OPTIONS = ("viewer", "engineer", "owner")


def _member_row(slug: str, user: User, role: str, web: Web, owner: bool) -> list[str]:
    if not owner:
        return [e(user.email), e(role)]
    base = f"/app/{quote(slug)}/members/{quote(user.id)}"
    options = "".join(
        f'<option value="{r}"{" selected" if r == role else ""}>{r}</option>' for r in ROLE_OPTIONS
    )
    return [
        e(user.email),
        f'<form class="inline" method="post" action="{base}/role">{_csrf_field(web)}'
        f'<select name="role">{options}</select> <button>Change</button></form>',
        f'<form class="inline" method="post" action="{base}/invite">{_csrf_field(web)}'
        "<button>New sign-in link</button></form> "
        f'<form class="inline" method="post" action="{base}/remove">{_csrf_field(web)}'
        "<button>Remove</button></form>",
    ]


def _invite_form(slug: str, web: Web) -> str:
    options = "".join(f'<option value="{r}">{r}</option>' for r in ROLE_OPTIONS)
    return (
        f'<form method="post" action="/app/{quote(slug)}/members">{_csrf_field(web)}'
        '<input type="email" name="email" placeholder="colleague@company.com" required '
        f'maxlength="320"> <select name="role">{options}</select> '
        "<button>Invite</button></form>"
    )


def _invite_page(
    settings: Settings, web: Web, slug: str, email: str, raw: str, status: int = 200
) -> HTMLResponse:
    link = f"{settings.public_url}/invite/{raw}"
    github = (
        f"<p>They can also sign in with GitHub if <strong>{e(email)}</strong> is a verified "
        "email on their account.</p>"
        if settings.github_client_id
        else ""
    )
    body = (
        f"<h1>Sign-in link for {e(email)}</h1>"
        "<p>Send this link to them privately. It works once and expires in "
        f"{INVITE_HOURS} hours; it's not shown again.</p>"
        f'<p class="secret">{e(link)}</p>{github}'
        f'<p><a href="/app/{quote(slug)}">Back to the workspace</a></p>'
    )
    return page("Invite", body, web, status=status)


@router.post("/app/{workspace}/members")
async def invite_member(
    workspace: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "manage_members")
        _require_csrf(web, form, settings)
        user, _ = repo.add_member(
            session, access, form.get("email", ""), form.get("role", "viewer")
        )
        raw = create_invite(session, access.workspace.id, user, web.principal.actor)
    except CloudError as err:
        return _error_page(err, web)
    return _invite_page(settings, web, workspace, user.email, raw, status=201)


@router.post("/app/{workspace}/members/{user_id}/invite")
async def reinvite_member(
    workspace: str, user_id: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "manage_members")
        _require_csrf(web, form, settings)
        user = next(
            (u for u, _ in repo.members(session, access.workspace) if u.id == user_id), None
        )
        if user is None:
            raise CloudError(404, "not_found", "Member not found.")
        raw = create_invite(session, access.workspace.id, user, web.principal.actor)
        repo.audit(
            session, access.workspace.id, web.principal.actor, "member.invite_link", "user", user.id
        )
    except CloudError as err:
        return _error_page(err, web)
    return _invite_page(settings, web, workspace, user.email, raw)


@router.post("/app/{workspace}/members/{user_id}/role")
async def change_member_role(
    workspace: str, user_id: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "manage_members")
        _require_csrf(web, form, settings)
        repo.change_role(session, access, user_id, form.get("role", ""))
    except CloudError as err:
        return _error_page(err, web)
    return RedirectResponse(f"/app/{quote(workspace)}", status_code=303)


@router.post("/app/{workspace}/members/{user_id}/remove")
async def remove_member_action(
    workspace: str, user_id: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "manage_members")
        _require_csrf(web, form, settings)
        repo.remove_member(session, access, user_id)
    except CloudError as err:
        return _error_page(err, web)
    return RedirectResponse(f"/app/{quote(workspace)}", status_code=303)


@router.get("/invite/{token}")
def accept_invite(
    token: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    """Redeem a single-use sign-in link."""
    retry = request.app.state.auth_failures.retry_after(client_key(request))
    if retry is not None:
        return RedirectResponse(
            "/login?error=" + quote(f"Too many attempts; wait {retry} seconds."), status_code=303
        )
    found = redeem_invite(session, token)
    if found is None:
        record_failure(request, "invite")
        return RedirectResponse(
            "/login?error="
            + quote(
                "This sign-in link is invalid, used or expired. Ask the "
                "workspace owner for a new one."
            ),
            status_code=303,
        )
    user, workspace_id = found
    slug = session.scalar(select(Workspace.slug).where(Workspace.id == workspace_id))
    raw = create_session(session, user, "invite")
    response = RedirectResponse(f"/app/{quote(slug or '')}" if slug else "/app", status_code=303)
    return _signed_in(response, raw, settings)


# --- personal settings -------------------------------------------------------------------


@router.get("/account")
def settings_page(
    request: Request, session: SessionDep, settings: SettingsDep, msg: str = ""
) -> HTMLResponse:
    web = _web(request, session, settings)
    rows = []
    for t in repo.personal_tokens(session, web.user.id):
        action = (
            "revoked"
            if t.revoked_at
            else (
                f'<form class="inline" method="post" action="/account/tokens/{quote(t.id)}'
                f'/revoke">{_csrf_field(web)}<button>Revoke</button></form>'
            )
        )
        rows.append(
            [
                e(t.name),
                f'<span class="mono">{e(t.prefix)}</span>',
                e(t.created_at.strftime("%Y-%m-%d")),
                e(t.last_used_at.strftime("%Y-%m-%d %H:%M") if t.last_used_at else "never"),
                e(t.expires_at.strftime("%Y-%m-%d") if t.expires_at else "never"),
                action,
            ]
        )
    notice = f'<div class="notice">{e(msg)}</div>' if msg else ""
    body = (
        f'<h1>Settings</h1>{notice}<p class="muted">Signed in as {e(web.user.email)}.</p>'
        '<h2>Personal API tokens</h2><p class="muted">They act as you in every workspace you '
        "belong to. Use runner tokens (per project) for CI instead.</p>"
        + _table(
            ["Name", "Prefix", "Created", "Last used", "Expires", ""], rows, "No personal tokens."
        )
        + f'<form method="post" action="/account/tokens">{_csrf_field(web)}'
        '<input name="name" placeholder="laptop" required maxlength="200"> '
        '<select name="days"><option value="30">30 days</option>'
        '<option value="90" selected>90 days</option><option value="365">1 year</option>'
        "</select> <button>Create token</button></form>"
    )
    return page("Settings", body, web)


@router.post("/account/tokens")
async def create_personal_token_action(
    request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        _require_csrf(web, form, settings)
        days = int(form.get("days", "90")) if form.get("days", "90").isdigit() else 90
        expires = utcnow() + timedelta(days=min(max(days, 1), 365))
        row, new = repo.create_personal_token(
            session,
            web.user,
            form.get("name", "").strip()[:200] or "token",
            web.principal.actor,
            expires,
        )
    except CloudError as err:
        return _error_page(err, web)
    body = (
        "<h1>Token created</h1><p>Copy it now; it won't be shown again.</p>"
        f'<p class="secret">{e(new.token)}</p><p class="muted">{e(row.name)} · {e(row.prefix)}'
        f" · expires {e(row.expires_at.strftime('%Y-%m-%d') if row.expires_at else 'never')}</p>"
        '<p><a href="/account">Back to settings</a></p>'
    )
    return page("Token", body, web, status=201)


@router.post("/account/tokens/{token_id}/revoke")
async def revoke_personal_token_action(
    token_id: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        _require_csrf(web, form, settings)
        row = repo.revoke_personal_token(session, web.principal, token_id)
    except CloudError as err:
        return _error_page(err, web)
    return RedirectResponse("/account?msg=" + quote(f"Revoked {row.prefix}."), status_code=303)


def _danger_zone(slug: str, web: Web) -> str:
    return (
        "<h2>Data</h2>"
        f'<p><a href="/api/v1/workspaces/{quote(slug)}/export">Download everything (JSON)</a>'
        " · projects, runs, checks, baselines, members and the audit log.</p>"
        '<div class="card"><strong>Delete this workspace</strong><p class="muted">Removes every '
        "project, run, check, baseline, token and membership. It can't be undone; backups "
        "expire on their own schedule.</p>"
        f'<form method="post" action="/app/{quote(slug)}/delete">{_csrf_field(web)}'
        f'<input name="confirm" placeholder="type {e(slug)} to confirm" required> '
        "<button>Delete workspace</button></form></div>"
    )


@router.post("/app/{workspace}/delete")
async def delete_workspace_action(
    workspace: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    web = _web(request, session, settings)
    try:
        access = repo.authorize(session, web.principal, workspace, "delete")
        _require_csrf(web, form, settings)
        service.delete_workspace(session, access, form.get("confirm"))
    except CloudError as err:
        return _error_page(err, web)
    return RedirectResponse("/app", status_code=303)


def _project_access(
    request: Request,
    session: Session,
    settings: Settings,
    workspace: str,
    project: str,
    action: str = "read",
) -> tuple[Web, repo.Access]:
    web = _web(request, session, settings)
    return web, repo.authorize(session, web.principal, workspace, action, project)


@router.get("/app/{workspace}/{project}")
def project_page(
    workspace: str,
    project: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    msg: str = "",
) -> HTMLResponse:
    try:
        web, access = _project_access(request, session, settings, workspace, project)
    except CloudError as err:
        return _error_page(err, None)
    assert access.project is not None
    base = f"/app/{quote(workspace)}/{quote(project)}"
    can_write = repo.ROLE_RANK.get(access.role, 0) >= repo.ROLE_RANK["engineer"]
    notice = f'<div class="notice">{e(msg)}</div>' if msg else ""
    runs_by_pk: dict[str, Run] = {}
    baseline_rows = []
    for b, r in service.baselines(session, access.project):
        baseline_rows.append(
            [
                e(b.name),
                f'<span class="mono">{e(r.run_id[:12])}</span>',
                e(r.label),
                e(_actor(session, b.pinned_by)),
                e(b.pinned_at.strftime("%Y-%m-%d %H:%M")),
                e(b.note),
            ]
        )
    check_rows = []
    for c in service.recent_checks(session, access):
        cand = runs_by_pk.get(c.candidate_run_pk) or session.get(Run, c.candidate_run_pk)
        assert cand is not None
        runs_by_pk[cand.id] = cand
        reasons = c.result.get("reasons") or []
        check_rows.append(
            [
                f'<a href="{base}/checks/{quote(c.id)}">{_badge(c.verdict)}</a>',
                e(cand.label or cand.run_id[:12]),
                e((cand.commit_sha or "")[:10]),
                e(reasons[0]["message"] if reasons else "All policy rules hold."),
                e(c.created_at.strftime("%Y-%m-%d %H:%M")),
            ]
        )
    run_rows = []
    main = {r.id for b, r in service.baselines(session, access.project) if b.name == "main"}
    for r in service.recent_runs(session, access):
        pin = ""
        if r.id in main:
            pin = '<span class="muted">baseline main</span>'
        elif can_write and r.status == "complete":
            pin = (
                f'<form class="inline" method="post" action="{base}/pin">{_csrf_field(web)}'
                f'<input type="hidden" name="run_id" value="{e(r.run_id)}">'
                '<input type="hidden" name="name" value="main">'
                '<button title="Make this run the baseline named main">Pin as main</button>'
                "</form>"
            )
        run_rows.append(
            [
                f'<span class="mono">{e(r.run_id[:12])}</span>',
                e(r.status),
                e(r.label),
                e((r.commit_sha or "")[:10]),
                e(r.backend),
                e(r.uploaded_at.strftime("%Y-%m-%d %H:%M")),
                pin,
            ]
        )
    tokens = ""
    if can_write:
        token_rows = [
            [
                e(t.name),
                f'<span class="mono">{e(t.prefix)}</span>',
                e(", ".join(t.scopes or [])),
                e(t.last_used_at.strftime("%Y-%m-%d %H:%M") if t.last_used_at else "never"),
                "revoked"
                if t.revoked_at
                else (
                    f'<form class="inline" method="post" action="{base}/runner-tokens/{quote(t.id)}'
                    f'/revoke">{_csrf_field(web)}<button>Revoke</button></form>'
                ),
            ]
            for t in repo.runner_tokens(session, access)
        ]
        tokens = (
            "<h2>Runner tokens</h2>"
            + _table(["Name", "Prefix", "Scopes", "Last used", ""], token_rows, "No tokens yet.")
            + f'<form method="post" action="{base}/runner-tokens">{_csrf_field(web)}'
            '<input name="name" placeholder="github-actions" required maxlength="200"> '
            "<button>Create runner token</button></form>"
        )
    body = (
        f'<div class="crumbs"><a href="/app">Workspaces</a> / '
        f'<a href="/app/{quote(workspace)}">{e(workspace)}</a> / {e(project)}</div>'
        f"<h1>{e(access.project.name)}</h1>{notice}"
        "<h2>Baselines</h2>"
        + _table(
            ["Name", "Run", "Label", "Pinned by", "Pinned at", "Note"],
            baseline_rows,
            "No baseline yet: pin a complete run below.",
        )
        + "<h2>Release timeline</h2>"
        + '<p class="muted">Decisions and baseline changes. Decisions stay after their runs '
        "expire under retention; the full report is available while the check exists.</p>"
        + _table(
            ["When", "Event", "Candidate", "Quality change", "Why", "Evidence"],
            _timeline_rows(session, ledger.timeline(session, access.project, limit=30), base),
            "No release decisions yet.",
        )
        + "<h2>Accepted regressions</h2>"
        + '<p class="muted">Audited exceptions. A check they cover reads "fail, accepted"; '
        "they stop applying when revoked, expired or used up.</p>"
        + _table(
            ["Rule", "Status", "Accepted by", "Owner", "Expires", "Uses", "Reason", ""],
            _exception_rows(
                session, web, exceptions.listing(session, access.project)[:30], base, can_write
            ),
            "No exceptions.",
        )
        + "<h2>Recent checks</h2>"
        + _table(
            ["Verdict", "Candidate", "Commit", "Why", "When"],
            check_rows,
            "No checks yet: run `vectraudit check cloud:main CANDIDATE --upload` in CI.",
        )
        + "<h2>Recent runs</h2>"
        + _table(
            ["Run", "Status", "Label", "Commit", "Backend", "Uploaded", ""],
            run_rows,
            "No runs yet: `vectraudit record --upload`.",
        )
        + tokens
    )
    return page(access.project.name, body, web)


@router.get("/app/{workspace}/{project}/checks/{check_id}")
def check_page(
    workspace: str,
    project: str,
    check_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    msg: str = "",
) -> HTMLResponse:
    try:
        web, access = _project_access(request, session, settings, workspace, project)
        check, base_run, cand = service.load_check(session, access, check_id)
    except CloudError as err:
        return _error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    findings = "".join(
        f'<div class="card"><strong>{e(f["title"])}</strong> <span class="muted">'
        f"({e(f['level'])})</span>"
        + "".join(f'<div class="muted">{e(fact)}</div>' for fact in f.get("facts", []))
        + (f"<div>→ {e(f['suggestion'])}</div>" if f.get("suggestion") else "")
        + "</div>"
        for f in check.result.get("findings") or []
    )
    accept = _accept_section(web, access, check, base)
    body = (
        f'<div class="crumbs"><a href="/app">Workspaces</a> / '
        f'<a href="/app/{quote(workspace)}">{e(workspace)}</a> / '
        f'<a href="{base}">{e(project)}</a> / check</div>'
        f"<h1>{_badge(check.verdict)} {e(cand.label or cand.run_id)} vs "
        f"{e(base_run.label or base_run.run_id)}</h1>"
        + (f'<div class="notice">{e(msg)}</div>' if msg else "")
        + f'<p class="muted">Commit {e((cand.commit_sha or "unknown")[:12])} · '
        f"{e(check.created_at.strftime('%Y-%m-%d %H:%M'))} UTC · by "
        f"{e(_actor(session, check.created_by))}</p>"
        + (f"<h2>Diagnosis</h2>{findings}" if findings else "")
        + accept
        + "<h2>Full report</h2>"
        f'<iframe sandbox="" title="Investigation report" '
        f'src="{base}/checks/{quote(check.id)}/report"></iframe>'
    )
    return page("Check", body, web)


@router.get("/app/{workspace}/{project}/checks/{check_id}/report")
def check_report_frame(
    workspace: str,
    project: str,
    check_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> HTMLResponse:
    """The report document embedded by the check page (same origin only)."""
    try:
        _, access = _project_access(request, session, settings, workspace, project)
        check, _, _ = service.load_check(session, access, check_id)
    except CloudError as err:
        return _error_page(err, None)
    return HTMLResponse(
        render_html(check.result),
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
            "frame-ancestors 'self'",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )


@router.post("/app/{workspace}/{project}/pin")
async def pin_action(
    workspace: str, project: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    try:
        web, access = _project_access(
            request, session, settings, workspace, project, "pin_baseline"
        )
        _require_csrf(web, form, settings)
        b, r = service.pin_baseline(
            session,
            access,
            form.get("name", "main"),
            form.get("run_id", ""),
            form.get("note", "")[:2000],
        )
    except CloudError as err:
        return _error_page(err, None)
    msg = f"Baseline {b.name} now points to run {r.run_id[:12]}."
    return RedirectResponse(
        f"/app/{quote(workspace)}/{quote(project)}?msg={quote(msg)}", status_code=303
    )


@router.post("/app/{workspace}/{project}/checks/{check_id}/accept")
async def accept_action(
    workspace: str,
    project: str,
    check_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    """Grant an audited exception for one failing rule of this check."""
    form = await _form(request)
    try:
        web, access = _project_access(
            request, session, settings, workspace, project, "grant_exception"
        )
        _require_csrf(web, form, settings)
        try:
            day = datetime.strptime(form.get("expires", ""), "%Y-%m-%d").date()
        except ValueError as exc:
            raise CloudError(422, "invalid_expiry", "Give an expiry date (YYYY-MM-DD).") from exc
        uses = form.get("max_checks", "").strip()
        granted = exceptions.grant(
            session,
            access,
            check_id,
            rule=form.get("rule", ""),
            reason=form.get("reason", ""),
            owner=form.get("owner", ""),
            expires_at=datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc),
            max_checks=int(uses) if uses.isdigit() else None,
        )
    except CloudError as err:
        return _error_page(err, None)
    msg = f"Accepted {granted.rule} until {granted.expires_at:%Y-%m-%d}."
    return RedirectResponse(
        f"/app/{quote(workspace)}/{quote(project)}/checks/{quote(check_id)}?msg={quote(msg)}",
        status_code=303,
    )


@router.post("/app/{workspace}/{project}/exceptions/{exception_id}/revoke")
async def revoke_exception_action(
    workspace: str,
    project: str,
    exception_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    form = await _form(request)
    try:
        web, access = _project_access(
            request, session, settings, workspace, project, "grant_exception"
        )
        _require_csrf(web, form, settings)
        revoked = exceptions.revoke(session, access, exception_id)
    except CloudError as err:
        return _error_page(err, None)
    msg = f"Revoked the exception for {revoked.rule}."
    return RedirectResponse(
        f"/app/{quote(workspace)}/{quote(project)}?msg={quote(msg)}", status_code=303
    )


@router.post("/app/{workspace}/{project}/runner-tokens")
async def create_token_action(
    workspace: str, project: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await _form(request)
    try:
        web, access = _project_access(
            request, session, settings, workspace, project, "manage_runner_tokens"
        )
        _require_csrf(web, form, settings)
        name = form.get("name", "").strip()[:200] or "runner"
        row, new = repo.create_runner_token(session, access, name)
    except CloudError as err:
        return _error_page(err, None)
    back = f"/app/{quote(workspace)}/{quote(project)}"
    body = (
        f"<h1>Runner token created</h1><p>Store it now as the CI secret "
        f"<code>VECTRAUDIT_CLOUD_TOKEN</code>; it won't be shown again.</p>"
        f'<p class="secret">{e(new.token)}</p>'
        f'<p class="muted">{e(row.name)} · {e(row.prefix)} · scopes '
        f'{e(", ".join(row.scopes))}</p><p><a href="{back}">Back to the project</a></p>'
    )
    return page("Runner token", body, web, status=201)


@router.post("/app/{workspace}/{project}/runner-tokens/{token_id}/revoke")
async def revoke_token_action(
    workspace: str,
    project: str,
    token_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    form = await _form(request)
    try:
        web, access = _project_access(
            request, session, settings, workspace, project, "manage_runner_tokens"
        )
        _require_csrf(web, form, settings)
        row = repo.revoke_runner_token(session, access, token_id)
    except CloudError as err:
        return _error_page(err, None)
    msg = f"Revoked {row.prefix}."
    return RedirectResponse(
        f"/app/{quote(workspace)}/{quote(project)}?msg={quote(msg)}", status_code=303
    )
