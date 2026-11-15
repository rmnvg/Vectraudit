"""Review items from claim evidence: queued for a human, never labels, never held-out."""

from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.dataset import load_dataset, parse_dataset
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.claim_review import accept_evidence, claim_review


def support(reference=0, positions=None, **extra):
    row = {"reference": reference, "response": 0, "contradicts_reference": 0, **extra}
    if positions is not None:
        row["reference_claims"] = positions
    return row


def dataset(**changes):
    rows = [
        {
            "query_id": "q1",
            "text": "refund policy",
            "relevant": {"labelled": 2, "stale": 1},
            "forbidden": ["secret"],
            **changes,
        },
        {"query_id": "q2", "text": "held out question", "relevant": {"x": 1}, "split": "heldout"},
    ]
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


CLAIMS = {
    "reference": 6, "response": 3, "reference_in_context": 5, "reference_in_response": 3,
    "unsupported_incorrect": 0, "unsupported_correct": 0, "from_irrelevant_incorrect": 0,
    "from_relevant_incorrect": 0,
}  # fmt: skip
DOCS = {
    "labelled": support(3, [0, 1, 2]),
    "stale": support(0, []),
    "new-a": support(2, [2, 3]),
    "new-b": support(1, [4]),
    "secret": support(1, [5]),
    "undelivered": support(2, [0, 1]),
}


def run_for(ds, evidence=None, ids=None, qid="q1", error=None, **entry):
    run = fake_run([1.0, 1.0], run_id="r")
    run["dataset"]["sha256"] = ds.sha256
    q = run["queries"][0] | {
        "query_id": qid,
        "ids": ids or ["labelled", "stale", "new-a", "new-b", "secret"],
    }
    q["answer"] = {"claim_evidence": evidence or {"claims": CLAIMS, "claim_support": DOCS}}
    q["error"] = error
    q.update(entry)
    heldout = run["queries"][1] | {"query_id": "q2", "ids": ["x"]}
    heldout["answer"] = {
        "claim_evidence": {"claims": CLAIMS, "claim_support": {"x": support(2, [0, 1])}}
    }
    run["queries"] = [q, heldout]
    return run


def test_unjudged_documents_that_entail_claims_are_queued_most_claims_first():
    ds = dataset()
    queue, _ = claim_review(ds, [run_for(ds)])
    unjudged = [i for i in queue if i["kind"] == "possibly_relevant_unjudged"]
    assert [i["doc_id"] for i in unjudged] == ["new-a", "new-b"]
    first = unjudged[0]
    assert first["reason"] == "possibly relevant, unjudged: entails 2 of 6 reference claims"
    assert (first["claims_entailed"], first["reference_claims"], first["priority"]) == (2, 6, 3)
    assert (first["evidence"], first["source"]) == ("supported", "claim_evidence")
    judged = {i["doc_id"] for i in queue}
    assert "labelled" not in judged  # already judged
    assert "secret" not in judged  # forbidden documents are never suggested
    assert "undelivered" not in judged  # the run did not deliver it


def test_a_draft_irrelevant_mark_keeps_a_document_out_of_the_queue():
    ds = dataset()
    queue, _ = claim_review(ds, [run_for(ds)], judged_extra={"q1": ["new-b"]})
    assert [i["doc_id"] for i in queue if i["kind"] == "possibly_relevant_unjudged"] == ["new-a"]
    _, proposals = claim_review(ds, [run_for(ds)], judged_extra={"q1": ["new-b"]})
    assert proposals[0]["docs"] == ["labelled", "new-a"]  # a reviewed-irrelevant document is out


def test_a_relevant_label_that_entails_nothing_while_another_document_does_may_be_stale():
    ds = dataset()
    queue, _ = claim_review(ds, [run_for(ds)])
    stale = [i for i in queue if i["kind"] == "label_may_be_stale"]
    assert [i["doc_id"] for i in stale] == ["stale"]
    assert stale[0]["priority"] == 4
    assert stale[0]["grade"] == 1
    assert "label may be stale or wrong" in stale[0]["reason"]
    assert queue[0]["doc_id"] == "stale"  # a possibly wrong label outranks a new suggestion
    # No document entails any claim: nothing says the label is worse than the others.
    silent = {
        "claims": CLAIMS,
        "claim_support": {"stale": support(0, []), "labelled": support(0, [])},
    }
    assert claim_review(ds, [run_for(ds, silent)])[0] == []


def test_the_proposed_group_is_a_greedy_cover_with_its_coverage_and_what_needs_judging():
    ds = dataset()
    _, proposals = claim_review(ds, [run_for(ds)])
    (proposal,) = proposals
    # labelled covers 0-2, new-a adds 3, new-b adds 4; claim 5 is only in a forbidden document,
    # which is never proposed, so the group covers five of six.
    assert proposal["docs"] == ["labelled", "new-a", "new-b"]
    assert proposal["claims_covered"] == 5
    assert proposal["coverage"] == pytest.approx(5 / 6)
    assert proposal["status"] == "proposed"
    assert proposal["already_relevant"] == ["labelled"]
    assert proposal["needs_judgment"] == ["new-a", "new-b"]
    assert proposal["required_evidence"] == [proposal["docs"]]
    assert proposal["evidence"] == "supported"


def test_ties_break_by_document_id_and_partial_cover_reports_the_fraction():
    ds = dataset()
    tie = {"claims": CLAIMS, "claim_support": {"b": support(2, [0, 1]), "a": support(2, [2, 3])}}
    _, proposals = claim_review(ds, [run_for(ds, tie, ids=["a", "b"])])
    assert proposals[0]["docs"] == ["a", "b"]
    assert proposals[0]["coverage"] == pytest.approx(4 / 6)
    one = {"claims": CLAIMS, "claim_support": {"a": support(2, [0, 1])}}
    assert claim_review(ds, [run_for(ds, one, ids=["a"])])[1][0]["claims_covered"] == 2


def test_a_group_the_question_already_requires_is_not_proposed_again_and_no_positions_no_group():
    ds = dataset(required_evidence=[["labelled", "stale"]])
    only = {"claims": CLAIMS, "claim_support": {"labelled": support(3, [0, 1, 2]),
                                                "stale": support(1, [3])}}  # fmt: skip
    assert claim_review(ds, [run_for(ds, only, ids=["labelled", "stale"])])[1] == []
    counts_only = {"claims": CLAIMS, "claim_support": {"new-a": support(2)}}
    ds2 = dataset()
    assert claim_review(ds2, [run_for(ds2, counts_only, ids=["new-a"])])[1] == []


def test_held_out_questions_failed_questions_and_runs_without_evidence_give_nothing():
    ds = dataset()
    queue, proposals = claim_review(ds, [run_for(ds)])
    assert {i["query_id"] for i in queue} == {"q1"}
    assert {p["query_id"] for p in proposals} == {"q1"}  # q2 is held out and had evidence
    assert claim_review(ds, [run_for(ds, error={"code": "x"})]) == ([], [])
    plain = run_for(ds)
    del plain["queries"][0]["answer"]
    del plain["queries"][1]["answer"]
    assert claim_review(ds, [plain]) == ([], [])
    assert claim_review(ds, []) == ([], [])


def test_across_runs_the_larger_count_and_the_better_cover_win():
    ds = dataset()
    smaller = {
        "claims": CLAIMS,
        "claim_support": {"new-a": support(1, [2]), "labelled": support(3, [0, 1, 2])},
    }
    queue, proposals = claim_review(
        ds, [run_for(ds, smaller, ids=["new-a", "labelled"]), run_for(ds)]
    )
    assert [i["claims_entailed"] for i in queue if i["doc_id"] == "new-a"] == [2]
    assert proposals[0]["coverage"] == pytest.approx(5 / 6)


def test_items_carry_ids_counts_and_flags_only():
    ds = dataset()
    queue, proposals = claim_review(ds, [run_for(ds)])
    allowed = {
        "query_id", "doc_id", "kind", "reason", "priority", "claims_entailed",
        "reference_claims", "grade", "evidence", "source",
    }  # fmt: skip
    assert all(set(item) <= allowed for item in queue)
    assert all(set(p) <= {
        "query_id", "docs", "claims_covered", "reference_claims", "coverage", "status",
        "already_relevant", "needs_judgment", "required_evidence", "evidence", "source",
    } for p in proposals)  # fmt: skip


def test_the_benchmark_audit_merges_claim_items_with_the_pool_and_lists_proposals():
    ds = dataset()
    run = run_for(ds)
    audit = audit_benchmark(ds, [run])
    queue = audit["judgment_queue"]
    pooled = {i["doc_id"]: i for i in queue}
    # new-a was both an unjudged pooled result (priority 1) and a claim suggestion (3): one item.
    assert [i["doc_id"] for i in queue].count("new-a") == 1
    assert pooled["new-a"]["priority"] == 3
    assert pooled["new-a"]["claims_entailed"] == 2
    assert "possibly relevant" in pooled["new-a"]["reason"]
    assert [i["doc_id"] for i in queue][:3] == ["stale", "new-a", "new-b"]
    assert audit["required_evidence_proposals"][0]["docs"][0] == "labelled"
    assert "held-out" in audit["note"].lower()
    assert all(i["query_id"] == "q1" for i in queue)
    # Without claim evidence the audit is what it was, plus an empty proposal list.
    plain = copy.deepcopy(run)
    for q in plain["queries"]:
        q.pop("answer")
    before = audit_benchmark(ds, [plain])
    assert before["required_evidence_proposals"] == []
    assert all("kind" not in i for i in before["judgment_queue"])
    assert [i["doc_id"] for i in before["judgment_queue"]] == ["new-a", "new-b"]
    assert [i["priority"] for i in before["judgment_queue"]] == [1, 1]


def test_the_audit_never_changes_the_dataset_or_labels():
    ds = dataset()
    audit_benchmark(ds, [run_for(ds)])
    assert dict(ds.queries[0].relevant) == {"labelled": 2, "stale": 1}
    assert ds.queries[0].required_evidence == ()


def judged_dataset():
    """The reviewer has judged the proposed documents relevant (forbidden stays out)."""
    return dataset(relevant={"labelled": 2, "stale": 1, "new-a": 1, "new-b": 1})


def audit_for(ds):
    return audit_benchmark(ds, [run_for(ds)])


def test_accepting_a_proposal_adds_a_required_group_and_records_it():
    ds = judged_dataset()
    text, accepted = accept_evidence(ds, audit_for(ds), ["q1"])
    new = parse_dataset(text)
    assert [p["query_id"] for p in accepted] == ["q1"]
    assert new.queries[0].required_evidence == (("labelled", "new-a", "new-b"),)
    assert new.sha256 != ds.sha256
    # Everything else is as it was: labels, forbidden documents and the held-out question.
    assert new.queries[0].relevant == ds.queries[0].relevant
    assert new.queries[0].forbidden == ("secret",)
    assert new.queries[1] == ds.queries[1]
    entry = new.meta["provenance"]["required_evidence_accepted"][0]
    assert entry["docs"] == ["labelled", "new-a", "new-b"]
    assert (entry["source"], entry["evidence"]) == ("claim_evidence", "supported")
    assert ds.queries[0].required_evidence == ()  # the original is untouched


def test_an_accepted_group_joins_existing_groups_and_is_not_added_twice():
    ds = dataset(
        relevant={"labelled": 2, "stale": 1, "new-a": 1, "new-b": 1},
        required_evidence=[["labelled"]],
    )
    text, _ = accept_evidence(ds, audit_for(ds), ["q1"])
    again = parse_dataset(text)
    assert set(again.queries[0].required_evidence) == {
        ("labelled",),
        ("labelled", "new-a", "new-b"),
    }
    # The audit of the new version no longer proposes the group that is now required.
    assert audit_benchmark(again, [run_for(again)])["required_evidence_proposals"] == []


@pytest.mark.parametrize(
    ("questions", "message"),
    [([], "Name the questions"), (["nope"], "No proposed evidence"), (["q2"], "No proposed")],
)
def test_nothing_is_accepted_unless_a_proposed_question_is_named(questions, message):
    ds = judged_dataset()
    with pytest.raises(ValueError, match=message):
        accept_evidence(ds, audit_for(ds), questions)


def test_documents_still_to_judge_a_stale_audit_and_other_files_are_refused():
    ds = dataset()  # new-a and new-b are not labelled relevant yet
    with pytest.raises(ValueError, match="judge new-a, new-b as relevant first"):
        accept_evidence(ds, audit_for(ds), ["q1"])
    judged = judged_dataset()
    with pytest.raises(ValueError, match="another version of this dataset"):
        accept_evidence(judged, audit_for(ds), ["q1"])
    with pytest.raises(ValueError, match="not a benchmark audit"):
        accept_evidence(judged, {"format": "vectraudit.run"}, ["q1"])


def test_a_held_out_question_is_refused_even_if_a_hand_edited_proposal_names_it():
    ds = judged_dataset()
    audit = audit_for(ds)
    audit["required_evidence_proposals"].append(
        {**audit["required_evidence_proposals"][0], "query_id": "q2"}
    )
    with pytest.raises(ValueError, match="held out"):
        accept_evidence(ds, audit, ["q2"])


def test_the_cli_writes_a_new_version_and_leaves_the_original(tmp_path):
    ds = judged_dataset()
    rows = [json.dumps(r) for r in _rows(ds)]
    original = tmp_path / "benchmark.jsonl"
    original.write_text("\n".join(rows) + "\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(audit_for(load_dataset(original))))
    out = tmp_path / "next.jsonl"
    before = original.read_text()
    args = ["workbench", "accept-evidence", str(original), str(audit), "--out", str(out)]
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code == 2
    assert "Name the questions" in refused.output
    result = CliRunner().invoke(app, [*args, "--question", "q1"])
    assert result.exit_code == 0, result.output
    assert (
        "q1: required evidence labelled, new-a, new-b (covers 5 of 6 reference claims"
        in result.output
    )
    assert original.read_text() == before
    assert load_dataset(out).queries[0].required_evidence == (("labelled", "new-a", "new-b"),)
    same = CliRunner().invoke(
        app, ["workbench", "accept-evidence", str(original), str(audit),
              "--out", str(original), "--question", "q1"],
    )  # fmt: skip
    assert same.exit_code == 2
    assert "must differ" in same.output


def _rows(ds):
    from vectraudit.evaluation.dataset_tools import dataset_rows

    return dataset_rows(ds)
