"""Triage ranks real recorded runs into evidence-linked findings."""

import json

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.runner import RunOptions
from vectraudit.evaluation.triage import triage
from vectraudit.pipelines import Hit, PythonPipeline, Stage, execute_pipeline

ROWS = [
    {"query_id": "lost", "text": "a", "relevant": {"good": 2}},
    {"query_id": "perm", "text": "b", "relevant": {"ok": 1}, "forbidden": ["secret"]},
    {"query_id": "none", "text": "c", "no_answer": True},
    {"query_id": "same", "text": "d", "relevant": {"x": 1}},
    {"query_id": "recover", "text": "e", "relevant": {"r": 1}},
    {"query_id": "still", "text": "f", "relevant": {"zz": 1}},
]
DATASET = parse_dataset("\n".join(map(json.dumps, ROWS)))


def pipeline(candidate: bool) -> PythonPipeline:
    """Same retrieval for both runs, except where the candidate's change is under test."""
    table = {
        "lost": ["good", "other"],
        "perm": ["ok", "secret"] if candidate else ["ok"],
        "none": ["junk"] if candidate else [],
        "same": ["x"],
        "recover": ["r"] if candidate else ["n"],
        "still": ["w"],
    }

    def rerank(q, p):
        # The candidate's reranker discards the relevant document for "lost".
        return [h for h in p["keyword"] if not (candidate and h.id == "good")]

    return PythonPipeline(
        [
            Stage("keyword", "keyword", lambda q, p: [Hit(id=i) for i in table[q.query_id]]),
            Stage("rerank", "rerank", rerank, ("keyword",)),
            Stage("final", "truncate", lambda q, p: p["rerank"][: q.k], ("rerank",)),
        ]
    )


def record(candidate: bool, **options):
    return execute_pipeline(DATASET, pipeline(candidate), RunOptions(k=2, warmup=0, **options))


def test_findings_are_ranked_and_point_at_recorded_stages():
    base, cand = record(False), record(True)
    result = triage(base, cand)
    assert result["format"] == "vectraudit.triage"
    assert result["primary_metric"] == "ndcg"
    assert result["baseline_run"] == base["run_id"]
    assert [i["query_id"] for i in result["items"]] == [
        "none",
        "perm",
        "lost",
        "still",
        "recover",
    ]
    assert result["counts"] == {
        "blocking": 2,
        "error": 0,
        "regression": 1,
        "info": 1,
        "improvement": 1,
        "unchanged": 1,
    }
    by_id = {i["query_id"]: i for i in result["items"]}

    permission = by_id["perm"]["findings"][0]
    assert permission["kind"] == "permission"
    assert permission["new_doc_ids"] == ["secret"]
    assert permission["documents"] == [
        {"doc_id": "secret", "stages": ["keyword", "rerank", "final"]}
    ]

    no_answer = by_id["none"]["findings"][0]
    assert no_answer["kind"] == "no_answer"
    assert no_answer["doc_ids"] == ["junk"]
    assert no_answer["baseline_correct"] is True

    lost = by_id["lost"]
    assert lost["delta"] < 0
    assert lost["stages_recorded"]
    finding = lost["findings"][0]
    assert finding["kind"] == "lost_relevant"
    assert finding["doc_ids"] == ["good"]
    assert finding["documents"][0]["stages"] == [
        {
            "stage": "rerank",
            "parents": ["keyword"],
            "evidence": "measured",
            "reason": "present in parent output; absent from this stage",
        }
    ]

    assert finding["documents"][0]["diverged_at"] == "rerank"
    assert finding["documents"][0]["baseline_stages"] == ["keyword", "rerank", "final"]
    assert finding["documents"][0]["candidate_stages"] == ["keyword"]

    still = by_id["still"]["findings"][0]
    assert still["kind"] == "still_missing"
    assert still["documents"][0]["reason"].startswith("not present in recorded stages")
    assert by_id["recover"]["findings"][0]["kind"] == "recovered_relevant"


def test_without_stage_evidence_reports_dropped_results_not_located_losses():
    result = triage(record(False, evidence=False), record(True, evidence=False))
    lost = next(i for i in result["items"] if i["query_id"] == "lost")
    assert not lost["stages_recorded"]
    assert [f["kind"] for f in lost["findings"]] == ["ranking_regressed"]
    assert lost["findings"][0]["doc_ids"] == ["good"]
    # Safety assertions do not depend on stage evidence.
    assert result["counts"]["blocking"] == 2


def test_errors_and_unpaired_queries_are_reported_not_paired():
    base, cand = record(False), record(True)
    cand["queries"][3]["error"] = {"code": "pipeline_error", "message": "failed"}
    cand["queries"] = cand["queries"][:-1]
    result = triage(base, cand)
    assert result["n_paired"] == 5
    assert result["unpaired"] == {"baseline_only": ["still"], "candidate_only": []}
    failed = next(i for i in result["items"] if i["query_id"] == "same")
    assert failed["severity"] == "error"
    assert failed["findings"][0]["runs"] == ["candidate"]
    assert failed["delta"] is None


def test_identical_runs_have_only_shared_gaps():
    base = record(False)
    result = triage(base, base)
    assert [i["query_id"] for i in result["items"]] == ["recover", "still"]
    assert {i["severity"] for i in result["items"]} == {"info"}
    assert result["counts"]["unchanged"] == 4


def test_unobserved_loss_is_located_by_comparing_both_runs():
    """A shallower candidate never sees the document; the baseline's trace locates the loss."""
    data = parse_dataset(json.dumps({"query_id": "q", "text": "a", "relevant": {"rel": 1}}))

    def make(depth):
        pool = ["noise", "rel"]
        return PythonPipeline(
            [
                Stage("keyword", "keyword", lambda q, p: [Hit(id=i) for i in pool[:depth]]),
                Stage("final", "truncate", lambda q, p: p["keyword"][-1:], ("keyword",)),
            ]
        )

    options = RunOptions(k=1, warmup=0)
    base = execute_pipeline(data, make(2), options)
    cand = execute_pipeline(data, make(1), options)
    finding = triage(base, cand)["items"][0]["findings"][0]
    assert finding["kind"] == "lost_relevant"
    document = finding["documents"][0]
    assert document["stages"] == []
    assert document["candidate_stages"] == []
    assert document["diverged_at"] == "keyword"
