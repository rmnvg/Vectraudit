"""Knowledge freshness: changes reach the index, results and context in time."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.freshness import check_freshness, read_changes
from vectraudit.evaluation.lineage import LineageError, read_manifest

CHANGED = "2026-10-18T09:00:00Z"
BEFORE_DEADLINE = datetime(2026, 10, 18, 9, 30, tzinfo=timezone.utc)
AFTER_DEADLINE = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def chunk(chunk_id, revision, start=0, end=100, indexed=True):
    return {
        "chunk_id": chunk_id,
        "start": start,
        "end": end,
        "indexed": indexed,
        "indexed_revision": revision,
    }


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def check(tmp_path, changes, sources, *, as_of=AFTER_DEADLINE, **kwargs):
    return check_freshness(
        read_changes(write(tmp_path, "changes.jsonl", changes)),
        read_manifest(write(tmp_path, "lineage.jsonl", sources)),
        as_of=as_of,
        **kwargs,
    )


UPDATE = {"source_id": "policy", "change": "update", "to_revision": "r9", "changed_at": CHANGED}


@pytest.mark.parametrize(
    ("chunks", "as_of", "status"),
    [
        ([chunk("policy#0@r9", "r9")], AFTER_DEADLINE, "propagated"),
        ([chunk("policy#0@r8", "r8")], BEFORE_DEADLINE, "pending"),
        ([chunk("policy#0@r8", "r8", indexed=False)], AFTER_DEADLINE, "overdue"),
        ([chunk("policy#0@r8", "r8"), chunk("policy#0@r9", "r9")], AFTER_DEADLINE, "lingering"),
    ],
)
def test_update_status_depends_on_the_index_and_the_window(tmp_path, chunks, as_of, status):
    report = check(
        tmp_path,
        [UPDATE],
        [{"source_id": "policy", "revision": "r9", "chunks": chunks}],
        as_of=as_of,
    )
    assert report["changes"][0]["status"] == status
    assert report["failing"] == (status in ("overdue", "lingering"))


def test_replacements_and_deletions(tmp_path):
    changes = [
        {
            "source_id": "old-faq",
            "change": "replace",
            "replaced_by": "faq-2026",
            "changed_at": CHANGED,
        },
        {"source_id": "promo", "change": "delete", "changed_at": CHANGED},
        {"source_id": "manual-v1", "change": "archive", "changed_at": CHANGED},
    ]
    sources = [
        {"source_id": "old-faq", "chunks": [chunk("old-faq#0", None)]},
        {"source_id": "faq-2026", "chunks": [chunk("faq-2026#0", None)]},
        {"source_id": "manual-v1", "chunks": [chunk("manual-v1#0", None, indexed=False)]},
    ]
    rows = {r["source_id"]: r for r in check(tmp_path, changes, sources)["changes"]}
    assert rows["old-faq"]["status"] == "lingering"  # replaced, but the old one still answers
    assert rows["old-faq"]["new_content_indexed"] is True
    assert rows["promo"]["status"] == "propagated"  # gone from the export entirely
    assert rows["manual-v1"]["status"] == "propagated"  # present but no longer indexed


def test_a_change_may_declare_its_own_window_and_the_latest_change_wins(tmp_path):
    slow = {**UPDATE, "window_minutes": 24 * 60}
    newer = {**UPDATE, "to_revision": "r10", "changed_at": "2026-10-18T10:00:00Z"}
    sources = [{"source_id": "policy", "chunks": [chunk("policy#0@r8", "r8")]}]
    assert check(tmp_path, [slow], sources)["changes"][0]["status"] == "pending"
    rows = check(tmp_path, [slow, newer], sources)["changes"]
    assert len(rows) == 1
    assert (rows[0]["status"], rows[0]["new_content_indexed"]) == ("lingering", False)


def run_with(ids, context_ids=(), query="refund"):
    return {
        "run_id": "r",
        "created_at": "2026-10-18T12:00:00+00:00",
        "queries": [
            {
                "query_id": query,
                "error": None,
                "ids": list(ids),
                "model_context": {
                    "coverage": "observed",
                    "items": [{"id": i, "truncated": False} for i in context_ids],
                },
            }
        ],
    }


def test_a_stale_answer_is_found_then_cleared_by_reindexing(tmp_path):
    """Stale content served, then the repair verified."""
    stale = [
        {
            "source_id": "policy",
            "revision": "r9",
            "chunks": [chunk("policy#0@r8", "r8"), chunk("policy#0@r9", "r9", 100, 200)],
        }
    ]
    run = run_with(["policy#0@r8", "policy#0@r9"], context_ids=["policy#0@r8"])
    before = check(tmp_path, [UPDATE], stale, run=run)
    assert before["changes"][0]["status"] == "lingering"
    assert before["served"]["stale"] == [
        {
            "query_id": "refund",
            "chunk_id": "policy#0@r8",
            "source_id": "policy",
            "change": "update",
            "where": "context",
        }
    ]
    assert before["served"]["conflicts"] == [
        {"query_id": "refund", "source_id": "policy", "revisions": ["r8", "r9"]}
    ]
    reindexed = [
        {"source_id": "policy", "revision": "r9", "chunks": [chunk("policy#0@r9", "r9", 100, 200)]}
    ]
    after = check(tmp_path, [UPDATE], reindexed, run=run_with(["policy#0@r9"], ["policy#0@r9"]))
    assert (after["changes"][0]["status"], after["failing"]) == ("propagated", 0)


def test_questions_tagged_historical_may_reach_old_content(tmp_path):
    stale = [{"source_id": "policy", "chunks": [chunk("policy#0@r8", "r8")]}]
    data = parse_dataset(
        json.dumps(
            {"query_id": "refund", "text": "what was the old policy", "tags": ["historical"]}
        )
    )
    report = check(tmp_path, [UPDATE], stale, run=run_with(["policy#0@r8"]), dataset=data)
    assert report["served"]["stale"] == []
    assert report["served"]["historical_questions_exempt"] == ["refund"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_id": "a", "change": "update", "changed_at": CHANGED}, "to_revision"),
        ({"source_id": "a", "change": "replace", "changed_at": CHANGED}, "replaced_by"),
        (
            {"source_id": "a", "change": "delete", "changed_at": CHANGED, "to_revision": "r2"},
            "no new revision",
        ),
        (
            {"source_id": "a", "change": "update", "to_revision": "r", "changed_at": "yesterday"},
            "ISO 8601",
        ),
        ({"source_id": "a", "change": "rename", "changed_at": CHANGED}, "change"),
    ],
)
def test_invalid_change_logs_are_refused(tmp_path, change, message):
    with pytest.raises(LineageError, match=message):
        read_changes(write(tmp_path, "changes.jsonl", [change]))


def test_cli_reports_pending_changes_and_can_fail_ci(tmp_path):
    changes = write(tmp_path, "changes.jsonl", [UPDATE])
    lineage = write(
        tmp_path, "lineage.jsonl", [{"source_id": "policy", "chunks": [chunk("policy#0@r8", "r8")]}]
    )
    pending = CliRunner().invoke(
        app,
        [
            "workbench",
            "freshness",
            str(changes),
            str(lineage),
            "--as-of",
            "2026-10-18T09:10:00Z",
            "--fail-on-stale",
        ],
    )
    assert pending.exit_code == 0, pending.output
    assert "1 pending" in pending.output
    late = CliRunner().invoke(
        app,
        [
            "workbench",
            "freshness",
            str(changes),
            str(lineage),
            "--as-of",
            "2026-10-19T00:00:00Z",
            "--fail-on-stale",
            "--json",
            str(tmp_path / "out.json"),
        ],
    )
    assert late.exit_code == 3
    assert "policy (update): lingering" in late.output
    assert json.loads((tmp_path / "out.json").read_text())["counts"] == {"lingering": 1}
