"""Read-only question text joins for the authenticated local workbench."""

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
            except (HTTPException, OSError, ValueError):
                display = unavailable("Question text unavailable: the original run is unavailable.")
            self.cache[key] = display
        display = self.cache[key]
        return {
            **record,
            "question_display": {
                "text": display["texts"].get(record["query_id"]),
                "reason": display["reason"],
            },
        }
