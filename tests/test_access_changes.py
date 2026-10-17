"""Permission lifecycle: access change logs become expected access, and runs are checked."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.access import (
    apply_access_changes,
    check_access,
    expected_access,
    matches,
    read_access_changes,
)
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.lineage import LineageError
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines import (
    ContextItem,
    Hit,
    ModelContext,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    execute_pipeline,
)

CHANGED = "2026-10-18T09:00:00Z"
LATER = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
SOON = datetime(2026, 10, 18, 9, 5, tzinfo=timezone.utc)
ROWS = [
    {
        "query_id": "payroll-u17",
        "text": "payroll schedule",
        "relevant": {"payroll-2026": 2, "handbook": 1},
        "required_evidence": [["payroll-2026"]],
        "context": {"user": "u-17", "groups": ["finance", "staff"]},
    },
    {
        "query_id": "payroll-u2",
        "text": "payroll schedule",
        "relevant": {"payroll-2026": 2},
        "context": {"user": "u-2", "groups": ["finance"]},
    },
]
REVOKE = {
    "change": "revoke",
    "principal": {"user": "u-17"},
    "doc_ids": ["payroll-2026"],
    "changed_at": CHANGED,
    "window_minutes": 15,
}
POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 1.0}}})


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def dataset(rows=ROWS):
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def changes(tmp_path, *rows):
    return read_access_changes(write(tmp_path, "access.jsonl", rows))


class Assistant:
    """Serves documents its own access index allows; the index may lag the change log."""

    def __init__(self, acl, context_only=False):
        self.acl, self.context_only = acl, context_only

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        visible = [
            d for d in ("payroll-2026", "handbook") if request.context["user"] in self.acl[d]
        ]
        hits = [h for h in visible if not (self.context_only and h == "payroll-2026")]
        return PipelineResult(
            hits=[Hit(id=h) for h in hits],
            context=ModelContext(items=[ContextItem(id=d) for d in visible]),
        )


STALE = {"payroll-2026": {"u-17", "u-2"}, "handbook": {"u-17", "u-2"}}
FIXED = {"payroll-2026": {"u-2"}, "handbook": {"u-17", "u-2"}}


def test_principals_match_scalars_and_list_valued_context():
    context = {"user": "u-17", "groups": ["finance", "staff"]}
    assert matches({"user": "u-17"}, context)
    assert matches({"groups": "finance"}, context)
    assert matches({"groups": ["finance", "staff"]}, context)
    assert not matches({"groups": "legal"}, context)
    assert not matches({"tenant": "acme"}, context)
    assert matches({}, context)  # an archive applies to everyone


def test_a_revocation_becomes_expected_access_and_flags_the_stale_label(tmp_path):
    text, report = apply_access_changes(dataset(), changes(tmp_path, REVOKE), as_of=LATER)
    rows = {q.query_id: q for q in parse_dataset(text).queries}
    revoked = rows["payroll-u17"]
    assert revoked.forbidden == ("payroll-2026",)
    assert revoked.relevant == {"handbook": 1}
    assert revoked.required_evidence == ()
    assert "access-change" in revoked.tags
    assert rows["payroll-u2"].forbidden == ()  # another finance member keeps access
    assert report["labels_to_review"] == [
        {"query_id": "payroll-u17", "removed_labels": ["payroll-2026"]}
    ]
    early_text, early = apply_access_changes(dataset(), changes(tmp_path, REVOKE), as_of=SOON)
    assert parse_dataset(early_text).queries[0].forbidden == ()  # inside the window
    assert early["pending"][0]["query_id"] == "payroll-u17"


def test_a_planted_revocation_defect_is_caught_and_passes_after_correction(tmp_path):
    """The strategy's 2.2 acceptance: a customer-built revocation scenario."""
    text, _ = apply_access_changes(dataset(), changes(tmp_path, REVOKE), as_of=LATER)
    expected = parse_dataset(text)
    options = PipelineOptions(k=2, warmup=0)
    stale = execute_pipeline(expected, Assistant(STALE), options)
    fixed = execute_pipeline(expected, Assistant(FIXED), options)
    assert compare_runs(fixed, stale, POLICY)["verdict"] == "fail"
    assert compare_runs(fixed, fixed, POLICY)["verdict"] == "pass"
    log = changes(tmp_path, REVOKE)
    leak = check_access(log, stale, dataset(), as_of=LATER)
    assert leak["violations"] == [
        {
            "query_id": "payroll-u17",
            "doc_id": "payroll-2026",
            "reached": ["results", "model context"],
        }
    ]
    assert check_access(log, fixed, dataset(), as_of=LATER)["failing"] == 0


def test_a_revoked_document_handed_only_to_the_model_is_still_a_leak(tmp_path):
    text, _ = apply_access_changes(dataset(), changes(tmp_path, REVOKE), as_of=LATER)
    hidden = execute_pipeline(
        parse_dataset(text), Assistant(STALE, context_only=True), PipelineOptions(k=2, warmup=0)
    )
    assert hidden["queries"][0]["forbidden_ids"] == []
    assert compare_runs(hidden, hidden, POLICY)["verdict"] == "fail"
    report = check_access(changes(tmp_path, REVOKE), hidden, dataset(), as_of=LATER)
    assert report["violations"][0]["reached"] == ["model context"]


def test_inside_the_window_old_access_is_reported_not_failed(tmp_path):
    stale = execute_pipeline(dataset(), Assistant(STALE), PipelineOptions(k=2, warmup=0))
    report = check_access(changes(tmp_path, REVOKE), stale, dataset(), as_of=SOON)
    assert report["failing"] == 0
    assert report["pending"][0]["still_reached"] == {"payroll-2026": ["results", "model context"]}


def test_moves_grants_archives_and_the_latest_change_wins(tmp_path):
    log = changes(
        tmp_path,
        {
            "change": "move",
            "doc_ids": ["roadmap"],
            "from": {"tenant": "acme"},
            "to": {"tenant": "beta"},
            "changed_at": CHANGED,
        },
        {"change": "archive", "doc_ids": ["manual-v1"], "changed_at": CHANGED},
        {**REVOKE, "principal": {"groups": "finance"}, "window_minutes": 0},
        {
            "change": "grant",
            "principal": {"user": "u-17"},
            "doc_ids": ["payroll-2026"],
            "changed_at": "2026-10-18T10:00:00Z",
            "window_minutes": 0,
        },
    )
    rows = parse_dataset(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "query_id": "acme",
                    "text": "plan",
                    "context": {"tenant": "acme"},
                    "forbidden": ["secret"],
                },
                {
                    "query_id": "beta",
                    "text": "plan",
                    "context": {"tenant": "beta"},
                    "forbidden": ["roadmap"],
                },
                *ROWS,
            ]
        )
    )
    by = {q.query_id: q for q in rows.queries}
    access = {i: expected_access(log, by[i], as_of=LATER) for i in by}
    assert access["acme"]["revoked"] == ["manual-v1", "roadmap"]
    assert access["beta"]["granted"] == ["roadmap"]
    assert access["payroll-u17"]["granted"] == ["payroll-2026"]  # re-granted after the group revoke
    assert access["payroll-u2"]["revoked"] == ["manual-v1", "payroll-2026"]
    text, _ = apply_access_changes(rows, log, as_of=LATER)
    applied = {q.query_id: q for q in parse_dataset(text).queries}
    assert applied["beta"].forbidden == ("manual-v1",)  # moved in; the archive applies to all
    assert applied["acme"].forbidden == ("manual-v1", "roadmap", "secret")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"change": "revoke", "doc_ids": ["a"], "changed_at": CHANGED}, "principal"),
        (
            {"change": "move", "doc_ids": ["a"], "from": {"t": "a"}, "changed_at": CHANGED},
            "from and to",
        ),
        (
            {"change": "archive", "doc_ids": ["a"], "principal": {"u": "1"}, "changed_at": CHANGED},
            "no principal",
        ),
        (
            {"change": "grant", "doc_ids": [], "principal": {"u": "1"}, "changed_at": CHANGED},
            "doc_ids",
        ),
        (
            {"change": "revoke", "doc_ids": ["a"], "principal": {"u": "1"}, "changed_at": "soon"},
            "ISO 8601",
        ),
    ],
)
def test_invalid_access_logs_are_refused(tmp_path, row, message):
    with pytest.raises(LineageError, match=message):
        changes(tmp_path, row)


def test_cli_applies_the_log_checks_a_run_and_can_fail_ci(tmp_path):
    log = write(tmp_path, "access.jsonl", [REVOKE])
    data = write(tmp_path, "questions.jsonl", ROWS)
    stale = execute_pipeline(dataset(), Assistant(STALE), PipelineOptions(k=2, warmup=0))
    (tmp_path / "run.json").write_text(json.dumps(stale))
    out = tmp_path / "expected.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "access",
            str(log),
            str(data),
            "--run",
            str(tmp_path / "run.json"),
            "--write-dataset",
            str(out),
            "--as-of",
            "2026-10-18T12:00:00Z",
            "--fail-on-leak",
        ],
    )
    assert result.exit_code == 3, result.output
    assert "review payroll-u17: removed revoked label(s) payroll-2026" in result.output
    assert "payroll-u17: payroll-2026 reached results, model context" in result.output
    assert parse_dataset(out.read_text()).queries[0].forbidden == ("payroll-2026",)
    neither = CliRunner().invoke(app, ["workbench", "access", str(log), str(data)])
    assert neither.exit_code == 2
