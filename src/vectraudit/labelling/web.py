"""Loopback-only, server-rendered labelling with CSRF-safe atomic autosave."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Awaitable, Callable
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vectraudit.labelling.task import Judgment, Task, load_task, save_task

CSS = """
body{font:16px system-ui;background:#101827;color:#e5edf9;margin:0;line-height:1.6}
main{max-width:980px;margin:40px auto;padding:24px}header{color:#9cbbdd}
article{background:#1c293e;border:1px solid #3a506d;border-radius:12px;padding:24px;margin:20px 0}
.passage{white-space:pre-wrap}button,input{font:inherit;padding:8px;margin:4px}
button{cursor:pointer;border:1px solid #6088aa;border-radius:6px;background:#dbeafe;color:#122038}
button:focus,article:focus{outline:3px solid #58d7bd}a{color:#84d8ff}
small{color:#b4c7de}.saved{color:#83e6b7}input[type=number]{width:100px}
"""
JS = """
document.addEventListener('keydown',e=>{
 if(e.target.matches('input,textarea')||!['0','1','2','3','s'].includes(e.key))return;
 const card=document.activeElement.closest('article');
 if(card){e.preventDefault();card.querySelector('[value="'+(e.key==='s'?'skip':e.key)+'"]').click();}
});
document.addEventListener('mouseup',()=>{
 const s=window.getSelection();if(!s||s.isCollapsed)return;
 const p=s.anchorNode?.parentElement.closest('.passage');
 if(!p||!p.contains(s.focusNode))return;
 const r=s.getRangeAt(0),before=r.cloneRange();before.selectNodeContents(p);
 before.setEnd(r.startContainer,r.startOffset);
 const start=Array.from(before.toString()).length,end=start+Array.from(r.toString()).length;
 const f=p.closest('article').querySelector('form');
 f.elements.start.value=start;f.elements.end.value=end;
});
"""


def judge(task: Task, judgment: Judgment) -> None:
    """Save one annotator's latest decision, after validating document and span identity."""
    query = next((q for q in task.queries if q.query_id == judgment.query_id), None)
    if query is None or judgment.doc_id not in query.documents:
        raise ValueError("Judgment is outside this pool")
    if judgment.end is not None and judgment.end > len(task.documents[judgment.doc_id]):
        raise ValueError("Span exceeds source document")
    task.judgments = [
        j
        for j in task.judgments
        if (j.query_id, j.doc_id, j.annotator)
        != (judgment.query_id, judgment.doc_id, judgment.annotator)
    ] + [judgment]


def create_app(path: Path) -> FastAPI:
    """Serve one local task; one process serializes reload/modify/replace transactions."""
    load_task(path)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()

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
    def page(annotator: str = "", q: int = 0) -> str:
        task = load_task(path)
        head = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Vectraudit · Labelling</title><link rel="stylesheet" href="/style.css">'
            '<script src="/keys.js" defer></script><main><header>VECTRAUDIT / LOCAL LABELLING'
            "</header><h1>Human relevance judgments</h1>"
            "<p>Private on this computer. 0 not relevant · 1 partly · 2 relevant · 3 perfect.</p>"
        )
        if not annotator.strip():
            return head + (
                '<form><label>Your annotator name <input name="annotator" required '
                'maxlength="100"></label><button>Start labelling</button></form></main></html>'
            )
        if not 0 <= q < len(task.queries):
            raise HTTPException(404)
        query = task.queries[q]
        done = sum(j.annotator == annotator and j.grade is not None for j in task.judgments)
        total = sum(len(row.documents) for row in task.queries)
        body = (
            f"<p>{escape(annotator)} · {done}/{total} judged · "
            f"Question {q + 1}/{len(task.queries)}</p>"
            f"<h2>{escape(query.text)}</h2><p>Focus a passage, then press 0-3 or S to skip. "
            "Select text to fill optional character offsets; choose a grade to save.</p>"
        )
        for doc in query.documents:
            previous = next(
                (
                    j
                    for j in task.judgments
                    if (j.query_id, j.doc_id, j.annotator) == (query.query_id, doc, annotator)
                ),
                None,
            )
            status = (
                "Unjudged"
                if previous is None
                else ("Skipped" if previous.grade is None else f"Saved grade {previous.grade}")
            )
            body += (
                f'<article tabindex="0"><small class="saved">{status}</small>'
                f'<div class="passage">{escape(task.documents[doc])}</div>'
                '<form method="post" action="/judge">'
            )
            fields = {"csrf": token, "annotator": annotator, "q": str(q), "doc": doc}
            body += "".join(
                f'<input type="hidden" name="{key}" value="{escape(value, quote=True)}">'
                for key, value in fields.items()
            )
            body += (
                '<label>Span start <input type="number" min="0" name="start"></label>'
                '<label>End <input type="number" min="1" name="end"></label><br>'
            )
            body += "".join(f'<button name="grade" value="{g}">{g}</button>' for g in range(4))
            body += '<button name="grade" value="skip">Skip</button></form></article>'
        for label, index in [("Previous", q - 1), ("Next", q + 1)]:
            if 0 <= index < len(task.queries):
                url = escape(urlencode({"annotator": annotator, "q": index}))
                body += f'<a href="/?{url}">{label}</a> '
        return head + body + "</main></html>"

    @app.post("/judge")
    async def save(request: Request) -> RedirectResponse:
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/x-www-form-urlencoded"
        ):
            raise HTTPException(415)
        data = await request.body()
        if len(data) > 16384:
            raise HTTPException(413)
        form = {k: v[-1] for k, v in parse_qs(data.decode(), keep_blank_values=True).items()}
        if not secrets.compare_digest(form.get("csrf", ""), token):
            raise HTTPException(403, "Invalid CSRF token")
        try:
            with lock:
                task = load_task(path)
                q = int(form["q"])
                if not 0 <= q < len(task.queries):
                    raise ValueError("Unknown question")
                judgment = Judgment(
                    query_id=task.queries[q].query_id,
                    doc_id=form["doc"],
                    annotator=form["annotator"],
                    grade=None if form["grade"] == "skip" else int(form["grade"]),
                    start=int(form["start"]) if form.get("start") else None,
                    end=int(form["end"]) if form.get("end") else None,
                )
                judge(task, judgment)
                save_task(task, path)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(
            "/?" + urlencode({"annotator": judgment.annotator, "q": q}), status_code=303
        )

    return app
