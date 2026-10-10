"""Read-only question text and document title joins for the authenticated local workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.runfile import load_run

if TYPE_CHECKING:
    from vectraudit.pipelines.guided import GuidedWorkspace


def unavailable(reason: str) -> dict[str, Any]:
    return {"texts": {}, "reason": reason}


def run_questions(workspace: GuidedWorkspace, run: dict[str, Any]) -> dict[str, Any]:
    """Join only the exact saved benchmark, never query IDs from an unrelated draft.

    The returned projection belongs beside a run in a local API response, not inside
    the run artifact or a persisted case. Missing or changed drafts remain explicit.
    """
    source = run.get("benchmark_source")
    if not isinstance(source, dict):
        return unavailable("Question text unavailable: this run has no saved benchmark reference.")
    try:
        draft = workspace.load_draft(source["draft_id"])
        if draft["revision"] != source["revision"]:
            return unavailable("Question text unavailable: the saved benchmark has changed.")
        dataset = parse_dataset(draft["dataset"])
        split = source["split"]
        if split not in {"explore", "heldout"} or dataset.sha256 != source["sha256"]:
            return unavailable("Question text unavailable: the benchmark does not match this run.")
        rows = [r for r in dataset_rows(dataset) if r.get("split", "explore") == split]
        evaluated = parse_dataset(render_dataset(rows, dataset.meta))
        if evaluated.sha256 != run["dataset"]["sha256"]:
            return unavailable("Question text unavailable: the benchmark does not match this run.")
        recorded = {q["query_id"] for q in run["queries"]}
        return {
            "texts": {
                q.query_id: q.text
                for q in evaluated.queries
                if q.query_id in recorded and q.text is not None
            },
            "reason": None,
        }
    except (HTTPException, OSError, ValueError, KeyError, TypeError):
        # Display lookup must not hide otherwise valid evidence or echo corrupt local data.
        return unavailable("Question text unavailable: the saved benchmark could not be loaded.")


def no_titles(reason: str, code: str) -> dict[str, Any]:
    """``code`` lets a view stay quiet when there was nothing to show (``no_corpus``,
    ``no_titles``) and explain when titles exist but were withheld."""
    return {"titles": {}, "reason": reason, "code": code, "verified": False}


def title_of(doc: dict[str, Any]) -> str | None:
    title = doc.get("title") or (doc.get("metadata") or {}).get("title")
    return title if isinstance(title, str) and title.strip() else None


def run_document_titles(
    workspace: GuidedWorkspace, run: dict[str, Any], doc_ids: set[str]
) -> dict[str, Any]:
    """Titles for ``doc_ids`` from the corpus attached to the run's exact saved benchmark.

    ``verified`` is True when the run recorded a corpus version equal to that corpus; a
    recorded version that differs withholds titles, because the same ID may name another
    document. Without a recorded version (external pipelines) titles are shown unverified.
    """
    from vectraudit.pipelines.release import corpus_version
    from vectraudit.pipelines.workflows import documents

    source = run.get("benchmark_source")
    if not isinstance(source, dict):
        return no_titles(
            "Document titles unavailable: this run has no saved benchmark reference.", "no_corpus"
        )
    try:
        draft = workspace.load_draft(source["draft_id"])
        if draft["revision"] != source["revision"]:
            return no_titles(
                "Document titles unavailable: the saved benchmark has changed.", "changed"
            )
        docs = documents(draft.get("corpus") or "")
    except (HTTPException, OSError, ValueError, KeyError, TypeError):
        return no_titles(
            "Document titles unavailable: the saved benchmark could not be loaded.", "unreadable"
        )
    if not docs:
        return no_titles(
            "Document titles unavailable: no corpus is attached to the benchmark.", "no_corpus"
        )
    recorded = (run.get("backend") or {}).get("corpus_version")
    if recorded is not None and recorded != corpus_version(docs):
        return no_titles(
            "Document titles unavailable: the attached corpus is not the one this run used.",
            "mismatch",
        )
    titles = {d["doc_id"]: title for d in docs if d["doc_id"] in doc_ids and (title := title_of(d))}
    if not any(title_of(d) for d in docs):
        return no_titles(
            "Document titles unavailable: the attached corpus has no titles.", "no_titles"
        )
    return {"titles": titles, "reason": None, "code": None, "verified": recorded is not None}


def run_doc_ids(*runs: dict[str, Any]) -> set[str]:
    return {i for run in runs for q in run.get("queries") or [] for i in q.get("ids") or []}


class CaseQuestionDisplay:
    """Resolve each original baseline once per API request; never write display text."""

    def __init__(self, workspace: GuidedWorkspace):
        self.workspace = workspace
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}

    def decorate(self, record: dict[str, Any]) -> dict[str, Any]:
        key = (record["baseline"], record["dataset_sha256"])
        if key not in self.cache:
            try:
                run = load_run(self.workspace.run_path(key[0]))
                display = (
                    run_questions(self.workspace, run)
                    if run["dataset"]["sha256"] == key[1]
                    else unavailable(
                        "Question text unavailable: the case benchmark does not match."
                    )
                )
                titles = run_document_titles(self.workspace, run, run_doc_ids(run))
            except (HTTPException, OSError, ValueError):
                display = unavailable("Question text unavailable: the original run is unavailable.")
                titles = no_titles(
                    "Document titles unavailable: the original run is unavailable.", "unreadable"
                )
            self.cache[key] = {**display, "document_titles": titles}
        display = self.cache[key]
        wanted = {
            d
            for f in (record.get("finding") or {}).get("findings") or []
            for d in f.get("doc_ids") or []
        }
        titles = display["document_titles"]
        return {
            **record,
            "question_display": {
                "text": display["texts"].get(record["query_id"]),
                "reason": display["reason"],
            },
            "document_display": {
                "titles": {d: t for d, t in titles["titles"].items() if d in wanted},
                "reason": titles["reason"],
                "code": titles["code"],
                "verified": titles["verified"],
            },
        }
