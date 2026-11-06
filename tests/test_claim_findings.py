"""Claim findings: what claim-level evidence says beside a question's diagnosis."""

from __future__ import annotations

import json

from tests.test_eval_compare import fake_run
from tests.test_ragchecker_import import lines
from vectraudit.evaluation.answers import (
    CLAIM_FINDINGS,
    MAX_FINDING_DOCUMENTS,
    attach_file,
    render_summary,
)

CLAIMS = {
    "reference": 6, "response": 5, "reference_in_context": 4, "reference_in_response": 3,
    "reference_used": 3, "unsupported_incorrect": 1, "unsupported_correct": 1,
    "from_irrelevant_incorrect": 1, "from_relevant_incorrect": 1, "response_incorrect": 3,
}  # fmt: skip
SUPPORT = {
    "good": {"reference": 3, "response": 2, "response_incorrect": 1, "contradicts_reference": 0},
    "noise": {"reference": 0, "response": 1, "response_incorrect": 1, "contradicts_reference": 0},
    "stale": {"reference": 1, "response": 0, "response_incorrect": 0, "contradicts_reference": 2},
    "quiet": {"reference": 0, "response": 0, "response_incorrect": 0, "contradicts_reference": 0},
}  # fmt: skip


def run_with(**query):
    run = fake_run([1.0])
    run["queries"][0] |= {"query_id": "q", "ids": ["good", "noise", "stale"], **query}
    return run


def attach(tmp_path, judgments, **query):
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps({"query_id": "q", **j}) for j in judgments))
    return attach_file(run_with(**query), path)["queries"][0]["answer"]


def claim_line(claims=CLAIMS, support=SUPPORT, **extra):
    return {
        "evaluator": "ragchecker.claims", "kind": "model",
        "claims": claims, "claim_support": support, **extra,
    }  # fmt: skip


def findings(answer):
    return {f["finding"]: f for f in answer["claim_findings"]}


def test_every_finding_names_its_count_documents_and_evidence_level(tmp_path):
    answer = attach(tmp_path, [claim_line()])
    found = findings(answer)
    assert set(found) == set(CLAIM_FINDINGS)
    assert found["reference_not_in_context"]["count"] == 2
    assert found["evidence_unused"]["count"] == 1  # 4 in context, 3 of them used
    assert found["misled_by_irrelevant_context"]["documents"] == ["noise"]
    assert found["misled_by_relevant_context"]["documents"] == ["good"]
    assert found["unsupported_incorrect"]["count"] == 1
    assert found["unsupported_correct"]["count"] == 1
    assert found["contradicting_document"]["documents"] == ["stale"]
    assert found["contradicting_document"]["count"] == 1
    for finding in found.values():
        assert finding["evidence"] == "supported"
        assert finding["disputed"] is False
        assert finding["points_to"] == CLAIM_FINDINGS[finding["finding"]]
    assert answer["claim_evidence"]["claims"]["reference"] == 6


def test_findings_never_change_the_verdict_or_the_diagnosis(tmp_path):
    scored = {"evaluator": "x.score", "kind": "model", "verdict": "incorrect"}
    with_claims = attach(tmp_path, [scored, claim_line()], sufficiency=_sufficiency())
    without = attach(tmp_path, [scored], sufficiency=_sufficiency())
    assert with_claims["verdict"] == without["verdict"] == "incorrect"
    assert with_claims["diagnosis"] == without["diagnosis"]
    assert "claim_findings" not in without


def _sufficiency(delivered="sufficient"):
    return {"groups": 1, "delivered": delivered, "context": "sufficient",
            "missing_from_delivered": []}  # fmt: skip


def test_a_missing_reference_claim_is_a_gap_unless_a_complete_group_was_delivered(tmp_path):
    gap = findings(attach(tmp_path, [claim_line()], sufficiency=_sufficiency("insufficient")))
    assert "note" not in gap["reference_not_in_context"]
    stale = findings(attach(tmp_path, [claim_line()], sufficiency=_sufficiency()))
    assert (
        "label or the required evidence may be incomplete or stale"
        in (stale["reference_not_in_context"]["note"])
    )


def test_findings_are_disputed_when_human_review_disagrees_with_the_model(tmp_path):
    model = {"evaluator": "ragchecker.f1", "kind": "model", "verdict": "incorrect"}

    def disputed(*extra):
        answer = attach(tmp_path, [model, claim_line(), *extra])
        return {f["disputed"] for f in answer["claim_findings"]}

    assert disputed() == {False}
    human_agrees = {"evaluator": "review", "kind": "human", "verdict": "incorrect"}
    assert disputed(human_agrees) == {False}
    human_disagrees = {"evaluator": "review", "kind": "human", "verdict": "correct"}
    assert disputed(human_disagrees) == {True}
    two_humans = [human_agrees, {**human_disagrees, "evaluator": "review2"}]
    assert disputed(*two_humans) == {True}


def test_documents_the_run_never_delivered_are_listed_apart_and_never_named(tmp_path):
    support = {**SUPPORT, "elsewhere": {**SUPPORT["noise"], "contradicts_reference": 1}}
    answer = attach(tmp_path, [claim_line(support=support)])
    assert answer["claim_evidence"]["unmatched_documents"] == ["elsewhere", "quiet"]
    found = findings(answer)
    assert found["misled_by_irrelevant_context"]["documents"] == ["noise"]
    assert found["contradicting_document"]["documents"] == ["stale"]
    # Documents that reached only the model context count as delivered.
    context = {"coverage": "observed", "items": [{"id": "elsewhere"}]}
    inside = attach(tmp_path, [claim_line(support=support)], model_context=context)
    assert inside["claim_evidence"]["unmatched_documents"] == ["quiet"]
    assert "elsewhere" in findings(inside)["contradicting_document"]["documents"]


def test_counts_without_per_document_incorrect_claims_name_no_documents(tmp_path):
    bare = {
        d: {k: v for k, v in s.items() if k != "response_incorrect"} for d, s in SUPPORT.items()
    }
    found = findings(attach(tmp_path, [claim_line(support=bare)]))
    assert found["misled_by_irrelevant_context"]["count"] == 1
    assert "documents" not in found["misled_by_irrelevant_context"]
    assert "documents" in found["contradicting_document"]


def test_without_the_optional_reference_used_count_unused_evidence_is_not_guessed(tmp_path):
    claims = {k: v for k, v in CLAIMS.items() if k != "reference_used"}
    assert "evidence_unused" not in findings(attach(tmp_path, [claim_line(claims=claims)]))


def test_a_clean_answer_has_no_findings_and_a_long_list_is_capped(tmp_path):
    clean = dict.fromkeys(CLAIMS, 0) | {"reference": 3, "reference_in_context": 3,
                                        "reference_in_response": 3, "reference_used": 3,
                                        "response": 3}  # fmt: skip
    assert attach(tmp_path, [claim_line(claims=clean, support={})])["claim_findings"] == []
    many = {
        f"d{i:03}": {"reference": 0, "response": 1, "response_incorrect": 1,
                     "contradicts_reference": 0}
        for i in range(MAX_FINDING_DOCUMENTS + 5)
    }  # fmt: skip
    claims = dict.fromkeys(CLAIMS, 0) | {
        "reference": 1,
        "response": 25,
        "response_incorrect": 25,
        "from_irrelevant_incorrect": 25,
    }
    ids = list(many)
    path = tmp_path / "j.jsonl"
    path.write_text(json.dumps({"query_id": "q"} | claim_line(claims=claims, support=many)))
    answer = attach_file(run_with(ids=ids), path)["queries"][0]["answer"]
    capped = findings(answer)["misled_by_irrelevant_context"]
    assert len(capped["documents"]) == MAX_FINDING_DOCUMENTS
    assert capped["more_documents"] == 5


def test_a_failed_question_keeps_its_claim_evidence_but_gets_no_findings(tmp_path):
    answer = attach(tmp_path, [claim_line()], error={"code": "x", "message": "failed"})
    assert "claim_evidence" in answer
    assert "claim_findings" not in answer


def test_the_summary_counts_findings_per_question_and_the_terminal_says_what_they_mean(tmp_path):
    rows, _ = lines(("f1",))
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    run = fake_run([1.0, 1.0])
    for q, docs in zip(run["queries"], (["000", "001", "002", "003"], ["000", "001", "002"]),
                       strict=True):  # fmt: skip
        q["ids"] = docs
    run["search"]["k"] = run["metric_contract"]["k"] = 4
    attached = attach_file(run, path)
    summary = attached["answers"]["claim_evidence"]
    assert summary["questions"] == 2
    assert summary["findings"] == {
        "contradicting_document": 2,
        "evidence_unused": 1,  # only the second question left reference claims unused
        "misled_by_relevant_context": 2,
        "reference_not_in_context": 1,
        "unsupported_correct": 1,
        "unsupported_incorrect": 1,
    }
    q0, q1 = (q["answer"]["claim_findings"] for q in attached["queries"])
    by0 = {f["finding"]: f for f in q0}
    assert by0["contradicting_document"]["documents"] == ["000", "001"]
    assert by0["misled_by_relevant_context"]["documents"] == ["001", "003"]
    assert by0["reference_not_in_context"]["count"] == 17
    assert {f["finding"] for f in q1} == {
        "contradicting_document", "evidence_unused", "misled_by_relevant_context",
    }  # fmt: skip
    text = render_summary(attached)
    assert "Claim-level evidence attached to 2 questions." in text
    assert "2: contradicting document." in text
    assert "A delivered document contradicts the reference answer" in text
