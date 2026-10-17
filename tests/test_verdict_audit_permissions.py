"""Second verdict audit (§9 gate): permission and freshness paths added since Step 38."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from tests.test_freshness import UPDATE, check, chunk, run_with
from tests.test_permission_path import DATA, METRIC, Assistant, run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import check_run
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines import (
    ContextItem,
    Hit,
    ModelContext,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    execute_pipeline,
)
from vectraudit.pipelines.errors import PipelineCallError

POLICY = parse_policy(METRIC)


class LeaksThenFails:
    """Repeat 1 serves the forbidden document; repeat 2 fails."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        self.calls += 1
        if self.calls == 2:
            raise PipelineCallError("invalid_response", "bad payload")
        return PipelineResult(hits=[Hit(id="guide"), Hit(id="private")])


def test_a_failed_execution_keeps_the_forbidden_documents_it_served():
    record = execute_pipeline(DATA, LeaksThenFails(), PipelineOptions(k=2, warmup=0, repeats=2))
    (query,) = record["queries"]
    assert query["error"]["code"] == "invalid_response"
    assert (query["forbidden_ids"], query["permission_checked"]) == (["private"], True)
    assert query["permission_path"]["documents"][0]["results"] is True


def retried(clean: dict, failed: dict) -> dict:
    """A run whose question failed once (``failed``) and then succeeded cleanly."""
    out = copy.deepcopy(clean)
    query = out["queries"][0]
    query["attempt_count"] = 2
    query["attempts"] = [
        {"attempt": 1, "status": "failed", "started_at": "2026-10-18T09:00:00+00:00",
         "result": failed},
        {"attempt": 2, "status": "complete", "started_at": "2026-10-18T09:01:00+00:00",
         "result": {"error": None}},
    ]  # fmt: skip
    return out


def test_a_clean_retry_cannot_hide_a_leak_from_the_failed_attempt():
    """The defect: only the final attempt's forbidden documents were judged."""
    clean = run(Assistant())
    failed = execute_pipeline(DATA, LeaksThenFails(), PipelineOptions(k=2, warmup=0, repeats=2))[
        "queries"
    ][0]
    candidate = retried(clean, failed)
    check_run(candidate)
    result = compare_runs(clean, candidate, POLICY)
    assert result["verdict"] == "fail"
    assert any("failed attempt" in r["message"] for r in result["reasons"])
    # Hosted uploads keep the attempt's forbidden documents, so the hosted check agrees.
    assert compare_runs(project_run(clean), project_run(candidate), POLICY)["verdict"] == "fail"


def test_a_context_leak_in_a_failed_attempt_also_counts():
    class ContextThenFails(LeaksThenFails):
        def __call__(self, request):
            self.calls += 1
            if self.calls == 2:
                raise PipelineCallError("invalid_response", "bad payload")
            return PipelineResult(
                hits=[Hit(id="guide")],
                context=ModelContext(items=[ContextItem(id="guide"), ContextItem(id="private")]),
            )

    clean = run(Assistant())
    failed = execute_pipeline(DATA, ContextThenFails(), PipelineOptions(k=2, warmup=0, repeats=2))[
        "queries"
    ][0]
    assert failed["forbidden_ids"] == []
    assert compare_runs(clean, retried(clean, failed), POLICY)["verdict"] == "fail"


def test_dropping_context_recording_is_not_comparable_rather_than_a_silent_pass():
    """The defect: a candidate that stopped recording context skipped the leak check."""
    baseline = run(Assistant())
    blind = run(Assistant(context=("guide", "private"), record_context=False))
    assert blind["queries"][0]["permission_path"]["context_recorded"] is False
    assert compare_runs(baseline, blind, POLICY)["verdict"] == "not_comparable"
    assert compare_runs(blind, blind, POLICY)["verdict"] == "pass"  # never could check: unchanged


def test_old_content_inside_the_window_is_pending_not_a_failure(tmp_path):
    """The defect: a run inside the propagation window failed for expected old content."""
    sources = [
        {
            "source_id": "policy",
            "revision": "r9",
            "chunks": [chunk("policy#0@r8", "r8"), chunk("policy#0@r9", "r9", 100, 200)],
        }
    ]
    served = run_with(["policy#0@r8", "policy#0@r9"], context_ids=["policy#0@r8"])
    early = check(
        tmp_path,
        [UPDATE],
        sources,
        run=served,
        as_of=datetime(2026, 10, 18, 9, 30, tzinfo=timezone.utc),
    )
    assert (early["failing"], early["served"]["stale"], early["served"]["conflicts"]) == (0, [], [])
    assert len(early["served"]["pending"]) == 2  # the old chunk and the revision pair
    late = check(tmp_path, [UPDATE], sources, run=served)
    assert late["failing"] == 3  # lingering, stale in context, conflicting revisions
