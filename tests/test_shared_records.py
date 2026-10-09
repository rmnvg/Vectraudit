"""Shared workbench records preserve evidence identities without leaking private content."""

import copy
import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, parse_policy, policy_sha256
from vectraudit.integrations.shared_records import digest, project_record, validate_record

PRIVATE = "PRIVATE_CONTEXT_PASSAGE_NOTE"
TEXT = "Question explicitly shared <script>alert(1)</script>"
POLICY = {
    "id": None,
    "name": PRIVATE,
    "version": None,
    "sha256": policy_sha256(DEFAULT_POLICY),
    "source": "exploratory",
}


def case():
    return dict(
        id="a" * 32,
        revision="b" * 32,
        format_version=1,
        query_id="q1",
        dataset_sha256="c" * 64,
        baseline="baseline",
        candidate="candidate",
        status="open",
        owner=PRIVATE,
        policy=POLICY,
        finding={
            "findings": [
                {
                    "kind": "lost_relevant",
                    "severity": "regression",
                    "doc_ids": ["doc1"],
                    "message": PRIVATE,
                    "documents": [{"text": PRIVATE}],
                }
            ]
        },
        history=[
            {
                "at": "2026-10-09T00:00:00+00:00",
                "kind": "created",
                "candidate": "candidate",
                "policy": POLICY,
            },
            {
                "at": "2026-10-09T00:01:00+00:00",
                "kind": "updated",
                "status": "investigating",
                "owner": PRIVATE,
                "note": PRIVATE,
            },
        ],
        question_display={"text": TEXT},
        future_field=PRIVATE,
    )


def benchmark():
    dataset = json.dumps(
        {
            "query_id": "q1",
            "text": TEXT,
            "context": {"tenant": PRIVATE},
            "relevant": {"doc1": 1},
            "relevant_spans": [{"doc_id": "doc1", "start": 0, "end": 8}],
            "history": [{"role": "user", "content": PRIVATE}],
        }
    )
    return dict(
        id="d" * 32,
        revision="e" * 32,
        format_version=1,
        name=PRIVATE,
        draft_id="f" * 32,
        draft_revision="a" * 32,
        created_at="2026-10-09T00:00:00+00:00",
        dataset=dataset,
        dataset_sha256=parse_dataset(dataset).sha256,
        inventory={"doc1": "b" * 64},
        irrelevant={"q1": ["doc2"]},
        future=PRIVATE,
    )


def policy():
    return dict(
        id="c" * 32,
        revision="d" * 32,
        format_version=1,
        name=PRIVATE,
        note=PRIVATE,
        lineage="a" * 32,
        version=1,
        supersedes=None,
        created_at="2026-10-09T00:00:00+00:00",
        policy={"metrics": {"ndcg": {"max_drop": 0.02}}},
        sha256=policy_sha256(parse_policy({"metrics": {"ndcg": {"max_drop": 0.02}}})),
    )


@pytest.mark.parametrize(
    ("kind", "factory"), [("case", case), ("benchmark", benchmark), ("policy", policy)]
)
def test_projection_is_allowlisted_nonmutating_and_idempotent(kind, factory):
    original = factory()
    before = copy.deepcopy(original)
    projected = project_record(kind, original)
    assert original == before
    assert PRIVATE not in json.dumps(projected)
    assert TEXT not in json.dumps(projected)
    assert validate_record(projected) == projected
    assert project_record(kind, projected) == projected
    assert projected["sharing"]["source_sha256"] == digest(original)
    assert projected["id"] == original["id"]


@pytest.mark.parametrize(
    ("kind", "factory", "text_key"),
    [("case", case, "question_text"), ("benchmark", benchmark, "text")],
)
def test_text_requires_opt_in_and_never_shares_context(kind, factory, text_key):
    projected = project_record(kind, factory(), share_query_text=True)
    data = projected["data"] if kind == "case" else projected["data"]["judgments"][0]
    assert data[text_key] == TEXT
    assert PRIVATE not in json.dumps(projected)
    projected["sharing"]["query_text"] = False
    with pytest.raises(ValueError, match="explicit"):
        validate_record(projected)


def test_nested_injection_and_identity_mismatches_rejected():
    projected = project_record("case", case())
    projected["data"]["history"][0]["note"] = PRIVATE
    projected["sharing"]["payload_sha256"] = digest(
        {k: v for k, v in projected.items() if k != "sharing"}
    )
    with pytest.raises(ValueError, match="note"):
        validate_record(projected)
    snapshot = benchmark()
    snapshot["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        project_record("benchmark", snapshot)
    rules = policy()
    rules["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        project_record("policy", rules)


def test_cli_preview_needs_no_credentials_and_does_not_upload(tmp_path, monkeypatch):
    from vectraudit.integrations.cloud import CloudClient

    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not connect")

    monkeypatch.setattr(CloudClient, "upload_record", forbidden)
    folder = tmp_path / ".workbench" / "cases"
    folder.mkdir(parents=True)
    raw = json.dumps(case())
    path = folder / ("a" * 32 + ".json")
    path.write_text(raw)
    result = CliRunner().invoke(
        app, ["workbench", "share", str(tmp_path), "case", "a" * 32, "--preview"]
    )
    assert result.exit_code == 0, result.output
    preview = json.loads(result.output)["preview"]
    assert preview["query_text"] is False
    assert PRIVATE not in result.output
    assert path.read_text() == raw
    assert (
        CliRunner()
        .invoke(app, ["workbench", "share", str(tmp_path), "case", "../escape", "--preview"])
        .exit_code
        == 2
    )
