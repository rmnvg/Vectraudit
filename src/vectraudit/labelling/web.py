"""Private local annotation, reviewer decisions and dataset download."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vectraudit.labelling.agreement import agreement
from vectraudit.labelling.export import export_dataset
from vectraudit.labelling.operations import (
    judge,
    judgment_for,
    resolve_judgment,
    revision,
    task_status,
)
from vectraudit.labelling.storage import TaskBusy, TaskConflict, update_task
from vectraudit.labelling.task import Judgment, Task, load_task

CSS = """
body{font:16px system-ui;background:#101827;color:#e5edf9;margin:0;line-height:1.6}
main{max-width:980px;margin:32px auto;padding:24px}header{color:#9cbbdd}
article,.summary{background:#1c293e;border:1px solid #3a506d;border-radius:12px;
padding:24px;margin:20px 0;overflow-wrap:anywhere}
.passage{white-space:pre-wrap}button,input{font:inherit;padding:8px;margin:4px;max-width:100%;box-sizing:border-box}
button{cursor:pointer;border:1px solid #6088aa;border-radius:6px;background:#dbeafe;color:#122038}
button:focus,article:focus,a:focus{outline:3px solid #58d7bd}a{color:#84d8ff}
small{color:#b4c7de}.saved{color:#83e6b7}input[type=number]{width:100px}
nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}.warning{color:#ffcf83}
progress{width:100%;height:12px;accent-color:#58d7bd}label{display:inline-block}
"""
JS = """
document.addEventListener('DOMContentLoaded',()=>{
 const card=document.querySelector('article[data-state="pending"]');
 if(card)card.focus({preventScroll:true});
});
document.addEventListener('keydown',e=>{
 if(e.altKey||e.ctrlKey||e.metaKey||e.repeat||e.target.matches('input,textarea')||
    !['0','1','2','3','s'].includes(e.key.toLowerCase()))return;
 const card=document.activeElement.closest('article');
 const button=card?.querySelector('button[name="grade"][value="'+
  (e.key.toLowerCase()==='s'?'skip':e.key)+'"]');
 if(button){e.preventDefault();button.click();}
});
document.addEventListener('click',e=>{
 if(e.target.matches('[data-clear-span]')){
  const f=e.target.closest('form');f.elements.start.value='';f.elements.end.value='';
 }
});
document.addEventListener('mouseup',()=>{
 const s=window.getSelection();if(!s||s.isCollapsed||s.rangeCount!==1)return;
 const p=s.anchorNode?.parentElement.closest('.passage');
 if(!p||!p.contains(s.focusNode))return;
 const r=s.getRangeAt(0),before=r.cloneRange();before.selectNodeContents(p);
 before.setEnd(r.startContainer,r.startOffset);
 const start=Array.from(before.toString()).length,end=start+Array.from(r.toString()).length;
 const f=p.closest('article').querySelector('form');
 f.elements.start.value=start;f.elements.end.value=end;
});
"""
GRADES = ("Not relevant", "Partly", "Relevant", "Perfect")


def _kappa(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def _url(path: str, **params: str | int) -> str:
    return escape(path + "?" + urlencode(params), quote=True)


def _page(title: str, body: str, annotator: str = "") -> str:
    nav = ""
    if annotator:
        nav = (
            '<nav aria-label="Workflow">'
            f'<a href="{_url("/", annotator=annotator)}">Label / resume</a>'
            f'<a href="{_url("/review", reviewer=annotator)}">Review disagreements</a>'
            f'<a href="{_url("/status", annotator=annotator)}">Dataset status</a></nav>'
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)} · Vectraudit</title><link rel="stylesheet" href="/style.css">'
        '<script src="/keys.js" defer></script></head><body><main>'
        "<header>VECTRAUDIT / LOCAL LABELLING</header>"
        f"<h1>{escape(title)}</h1><p>Private on this computer. Your choices save locally.</p>"
        f"{nav}{body}</main></body></html>"
    )


def _fields(values: dict[str, str | int]) -> str:
    return "".join(
        f'<input type="hidden" name="{key}" value="{escape(str(value), quote=True)}">'
        for key, value in values.items()
    )


def _span_fields(previous: Judgment | None) -> str:
    start = previous.start if previous is not None and previous.start is not None else ""
    end = previous.end if previous is not None and previous.end is not None else ""
    return (
        f'<label>Span start <input type="number" min="0" name="start" value="{start}"></label>'
        f'<label>End <input type="number" min="1" name="end" value="{end}"></label>'
        '<button type="button" data-clear-span>Clear span</button><br>'
    )


def _buttons(*, skip: bool) -> str:
    buttons = "".join(
        f'<button name="grade" value="{g}" aria-label="Grade {g}: {label}">{g}</button>'
        for g, label in enumerate(GRADES)
    )
    return buttons + ('<button name="grade" value="skip">Skip</button>' if skip else "")


async def _form(request: Request, token: str) -> dict[str, str]:
    if (
        request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/x-www-form-urlencoded"
    ):
        raise HTTPException(415, "Use the labelling form.")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 16384:
            raise HTTPException(413, "Form is too large.")
    try:
        values = parse_qs(
            data.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=20,
        )
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, "Invalid form encoding.") from exc
    if any(len(v) != 1 for v in values.values()):
        raise HTTPException(400, "Duplicate form fields.")
    form = {key: value[0] for key, value in values.items()}
    if not secrets.compare_digest(form.get("csrf", "").encode(), token.encode()):
        raise HTTPException(403, "This form expired. Reload and try again.")
    return form


def _decision(form: dict[str, str], qid: str, *, reviewer: bool = False) -> Judgment:
    grade = int(form["grade"]) if form["grade"] != "skip" else None
    return Judgment(
        query_id=qid,
        doc_id=form["doc"],
        annotator=form["reviewer"] if reviewer else form["annotator"],
        grade=grade,
        # A zero/skip is a document decision, not a positive span label.
        start=int(form["start"]) if grade and form.get("start") else None,
        end=int(form["end"]) if grade and form.get("end") else None,
    )


def create_app(path: Path) -> FastAPI:
    """Serve one task on loopback; process-wide file locks protect all writers."""
    load_task(path)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    token = secrets.token_urlsafe(32)

    @app.exception_handler(HTTPException)
    async def error_page(request: Request, exc: HTTPException) -> HTMLResponse:
        body = (
            f'<p class="warning">{escape(str(exc.detail))}</p>'
            '<p><a href="/">Reload labelling</a></p>'
        )
        return HTMLResponse(_page("Action needs attention", body), status_code=exc.status_code)

    @app.middleware("http")
    async def headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(
            {
                "Content-Security-Policy": "default-src 'none'; script-src 'self'; "
                "style-src 'self'; "
                "form-action 'self'; frame-ancestors 'none'; base-uri 'none'; connect-src 'none'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            }
        )
        return response

    @app.get("/style.css")
    def style() -> Response:
        return Response(CSS, media_type="text/css")

    @app.get("/keys.js")
    def script() -> Response:
        return Response(JS, media_type="text/javascript")

    @app.get("/", response_class=HTMLResponse)
    def page(annotator: str = "", q: int | None = None) -> str:
        task = load_task(path)
        if not annotator.strip():
            body = (
                '<form><label>Your annotator name <input name="annotator" required maxlength="100">'
                "</label><button>Start / resume labelling</button></form>"
            )
            return _page("Human relevance judgments", body)
        if len(annotator) > 100:
            raise HTTPException(400, "Annotator name is too long.")
        assigned = task.assignments.get(annotator)
        eligible = [
            i for i, row in enumerate(task.queries) if assigned is None or row.query_id in assigned
        ]
        if q is None:
            q = next(
                (
                    i
                    for i in eligible
                    if any(
                        judgment_for(task, task.queries[i].query_id, doc, annotator) is None
                        for doc in task.queries[i].documents
                    )
                ),
                eligible[0],
            )
        if not 0 <= q < len(task.queries):
            raise HTTPException(404, "Question not found.")
        if q not in eligible:
            q = eligible[0]
        query = task.queries[q]
        eligible_ids = {task.queries[i].query_id for i in eligible}
        done = sum(
            j.annotator == annotator and j.grade is not None and j.query_id in eligible_ids
            for j in task.judgments
        )
        total = sum(len(task.queries[i].documents) for i in eligible)
        body = (
            f"<p>{escape(annotator)} · {done}/{total} judged · "
            f"Question {eligible.index(q) + 1}/{len(eligible)}</p>"
            f'<progress value="{done}" max="{max(total, 1)}" '
            'aria-label="Judgments saved"></progress>'
            f"<h2>{escape(query.text)}</h2>"
            "<p>0 not relevant · 1 partly · 2 relevant · 3 perfect.</p>"
            "<p>Focus a passage and press 0-3 or S. "
            "Select text for an optional span, then grade it.</p>"
        )
        for doc in query.documents:
            previous = judgment_for(task, query.query_id, doc, annotator)
            status = (
                "Unjudged"
                if previous is None
                else "Skipped"
                if previous.grade is None
                else f"Saved grade {previous.grade}"
            )
            state = (
                "pending" if previous is None else "skipped" if previous.grade is None else "saved"
            )
            body += (
                f'<article tabindex="0" data-state="{state}"><small class="saved">{status}</small>'
                f'<div class="passage">{escape(task.documents[doc])}</div>'
                '<form method="post" action="/judge">'
            )
            body += _fields(
                {
                    "csrf": token,
                    "annotator": annotator,
                    "q": q,
                    "doc": doc,
                    "version": revision(task, query.query_id, doc, annotator=annotator),
                }
            )
            suggestion = next(
                (s for s in task.suggestions if (s.query_id, s.doc_id) == (query.query_id, doc)),
                None,
            )
            if suggestion is not None:
                body += (
                    f"<p>Machine suggestion: {suggestion.grade} ({escape(suggestion.judge)}). "
                    "Requires your decision.</p>"
                    + _fields({"suggestion": suggestion.grade, "judge": suggestion.judge})
                    + '<button name="grade" value="accept">Accept suggestion</button>'
                )
            body += _span_fields(previous) + _buttons(skip=True) + "</form></article>"
        body += '<nav aria-label="Questions">'
        for label, offset in [("Previous", -1), ("Next", 1)]:
            position = eligible.index(q) + offset
            if 0 <= position < len(eligible):
                body += (
                    f'<a href="{_url("/", annotator=annotator, q=eligible[position])}">{label}</a>'
                )
        body += f'<a href="{_url("/", annotator=annotator)}">Next unfinished</a></nav>'
        return _page("Human relevance judgments", body, annotator)

    @app.get("/review", response_class=HTMLResponse)
    def review(reviewer: str = "") -> str:
        if not reviewer.strip():
            return _page(
                "Review disagreements",
                '<form><label>Reviewer name <input name="reviewer" required maxlength="100">'
                "</label><button>Review</button></form>",
            )
        if len(reviewer) > 100:
            raise HTTPException(400, "Reviewer name is too long.")
        task = load_task(path)
        report = agreement(task)
        conflicts = [d for d in report["disagreements"] if not d["resolved"]]
        body = (
            f"<p>{len(conflicts)} unresolved disagreements. "
            "Reviewer decisions preserve the original votes.</p>"
        )
        for pair in report["pairs"]:
            body += (
                f"<p>{escape(' / '.join(pair['annotators']))}: {pair['n']} shared judgments; "
                f"binary kappa {_kappa(pair['binary_kappa'])}, "
                f"graded kappa {_kappa(pair['graded_kappa'])}.</p>"
            )

        body += (
            "<p>Undefined agreement means there are no shared labels or no label variation. "
            "Reviewer grades: 0 not relevant · 1 partly · 2 relevant · 3 perfect.</p>"
        )
        for item in conflicts:
            query = next(q for q in task.queries if q.query_id == item["query_id"])
            doc = item["doc_id"]
            body += (
                f'<article tabindex="0"><h2>{escape(query.text)}</h2>'
                f'<div class="passage">{escape(task.documents[doc])}</div>'
            )
            for row in item["judgments"]:
                span = f" · span [{row['start']}, {row['end']})" if row["start"] is not None else ""
                body += f"<p>{escape(row['annotator'])}: grade {row['grade']}{span}</p>"
            body += '<form method="post" action="/resolve">' + _fields(
                {
                    "csrf": token,
                    "reviewer": reviewer,
                    "query_id": query.query_id,
                    "doc": doc,
                    "version": revision(task, query.query_id, doc),
                }
            )
            body += _span_fields(None) + _buttons(skip=False) + "</form></article>"
        if not conflicts:
            body += "<p>No unresolved disagreements. Review dataset status before exporting.</p>"
        return _page("Review disagreements", body, reviewer)

    @app.get("/status", response_class=HTMLResponse)
    def status(annotator: str = "") -> str:
        task = load_task(path)
        stats = task_status(task)
        body = (
            '<section class="summary"><h2>Label readiness</h2>'
            f"<p>{stats['queries_with_positive_labels']}/{stats['queries']} "
            "questions with positive labels.</p>"
            f"<p>{stats['resolved_pairs']}/{stats['pooled_pairs']} pooled pairs resolved; "
            f"{stats['unjudged_pairs']} unjudged or skipped; "
            f"{stats['unresolved_pairs']} disagreements need review.</p>"
            "<p>Unjudged documents count as non-relevant in metrics for labelled questions. "
            "Questions without a positive label have no semantic quality metric.</p>"
        )
        if stats["export_ready"]:
            body += '<a href="/dataset.jsonl" download>Download labelled dataset</a>'
        else:
            body += (
                '<p class="warning">Add a positive human label and resolve '
                "all disagreements before export.</p>"
            )
        if not stats["fully_judged"]:
            body += (
                '<p class="warning">This is a partial label set. '
                "Expand coverage before trusting release decisions.</p>"
            )
        return _page("Dataset status", body + "</section>", annotator)

    @app.get("/dataset.jsonl")
    def download() -> Response:
        try:
            text = export_dataset(load_task(path))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(
            text,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="dataset.jsonl"'},
        )

    @app.post("/judge")
    async def save(request: Request) -> RedirectResponse:
        form = await _form(request, token)

        def change(task: Task) -> Judgment:
            q = int(form["q"])
            if not 0 <= q < len(task.queries):
                raise ValueError("Unknown question")
            query = task.queries[q]
            suggestion = next(
                (
                    s
                    for s in task.suggestions
                    if (s.query_id, s.doc_id) == (query.query_id, form["doc"])
                ),
                None,
            )
            if (form.get("suggestion"), form.get("judge")) != (
                str(suggestion.grade) if suggestion else None,
                suggestion.judge if suggestion else None,
            ):
                raise TaskConflict("Suggestion changed; reload before judging.")
            accepting = form["grade"] == "accept"
            if accepting:
                if suggestion is None:
                    raise ValueError("There is no suggestion to accept")
                decision_form = {**form, "grade": str(suggestion.grade)}
            else:
                decision_form = form
            judgment = _decision(decision_form, query.query_id)
            judgment.suggestion_grade = suggestion.grade if suggestion else None
            judgment.suggestion_judge = suggestion.judge if suggestion else None
            judgment.accepted_suggestion = accepting
            if (
                form.get("version")
                != revision(task, query.query_id, form["doc"], annotator=judgment.annotator)
                and judgment_for(task, query.query_id, form["doc"], judgment.annotator) != judgment
            ):
                raise TaskConflict("Your judgment changed in another tab. Reload before saving.")
            judge(task, judgment)
            return judgment

        judgment = await persist(change)
        return RedirectResponse(
            "/?" + urlencode({"annotator": judgment.annotator, "q": form["q"]}), status_code=303
        )

    @app.post("/resolve")
    async def save_resolution(request: Request) -> RedirectResponse:
        form = await _form(request, token)

        def change(task: Task) -> Judgment:
            judgment = _decision(form, form["query_id"], reviewer=True)
            if form.get("version") != revision(task, judgment.query_id, judgment.doc_id):
                raise TaskConflict(
                    "Judgments changed during review. Reload and review the new votes."
                )
            resolve_judgment(task, judgment)
            return judgment

        judgment = await persist(change)
        return RedirectResponse(
            "/review?" + urlencode({"reviewer": judgment.annotator}), status_code=303
        )

    async def persist(change: Callable[[Task], Judgment]) -> Judgment:
        try:
            return await run_in_threadpool(update_task, path, change)
        except TaskConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except TaskBusy as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, KeyError, StopIteration) as exc:
            raise HTTPException(
                400, "Invalid decision. Check the grade, identity and character offsets."
            ) from exc

    return app
