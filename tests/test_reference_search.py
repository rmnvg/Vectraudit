"""Reference application rankings come from text and filters, never question IDs."""

import importlib.util
import json
from pathlib import Path

import pytest

from vectraudit.pipelines.models import PipelineRequest

ROOT = Path(__file__).resolve().parents[1] / "examples" / "reference_search"


@pytest.fixture
def application():
    spec = importlib.util.spec_from_file_location("reference_app_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(text="Recover account without phone", **changes):
    return PipelineRequest.model_validate(
        {
            "query_id": "unseen-id",
            "text": text,
            "k": 2,
            "context": {"tenant": "acme"},
            "filters": {"current": True},
            **changes,
        }
    )


def test_actual_text_controls_ranking_not_query_identifier(application):
    first = application.retrieve(request())
    renamed = application.retrieve(request(query_id="completely-new-id"))
    changed = application.retrieve(request("Download monthly invoice PDF"))
    assert first.hits == renamed.hits
    assert first.hits[0].id == "account-recovery"
    assert changed.hits[0].id == "invoice-download"
    assert all(stage.source == "measured" and stage.latency_ms >= 0 for stage in first.stages)


def test_shallow_selection_loses_current_passage_and_prefilter_recovers_it(application):
    application.SEARCH_MODE = "candidate"
    broken = application.retrieve(request())
    keyword = next(s for s in broken.stages if s.name == "keyword")
    assert {h.id for h in keyword.hits} == {"account-recovery-archive", "account-recovery-other"}
    assert broken.hits == []
    assert broken.abstained
    application.SEARCH_MODE = "fixed"
    fixed = application.retrieve(request())
    assert fixed.hits[0].id == "account-recovery"
    assert not fixed.abstained
    assert all("account-recovery-other" not in {h.id for h in s.hits} for s in fixed.stages)


@pytest.mark.parametrize("mode", ["baseline", "candidate", "fixed"])
def test_context_tenant_wins_over_conflicting_filter_and_unknown_tenant_is_empty(application, mode):
    application.SEARCH_MODE = mode
    result = application.retrieve(request(filters={"tenant": "other", "current": True}))
    by_id = {d["doc_id"]: d for d in application.DOCUMENTS}
    assert all(by_id[h.id]["metadata"]["tenant"] == "acme" for h in result.hits)
    assert application.retrieve(request(context={"tenant": "unknown"})).hits == []
    assert application.retrieve(request("Quantum entanglement calibration")).abstained


@pytest.mark.parametrize("mode", ["baseline", "fixed"])
def test_reference_benchmark_and_review_evidence_are_consistent(application, mode):
    application.SEARCH_MODE = mode
    queries = [json.loads(line) for line in (ROOT / "queries.jsonl").read_text().splitlines()]
    assert len({q["text"] for q in queries}) == len(queries)
    by_id = {d["doc_id"]: d for d in application.DOCUMENTS}
    for query in queries:
        result = application.retrieve(
            request(
                query["text"],
                query_id=query["query_id"],
                context=query["context"],
                filters=query["filter"],
            )
        )
        if query.get("no_answer"):
            assert result.abstained
        else:
            assert result.hits[0].id in query["relevant"]
        assert not set(query.get("forbidden", [])) & {h.id for h in result.hits}
        assert set(query.get("forbidden", [])) <= by_id.keys()
    review = json.loads((ROOT / "judgments.json").read_text())
    assert review["review_status"] == "pending_human_review"
    for judgment in review["judgments"]:
        assert judgment["evidence"] in by_id[judgment["doc_id"]]["text"]
