"""Exports from real instrumentation libraries import as expected.

The fixtures in ``tests/fixtures/real_traces`` were produced by the libraries themselves
(see its README), not written by hand. One pipeline, three instrumentations: they must
agree on what was retrieved, and each gap must be reported, not filled in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines.openllmetry import _question
from vectraudit.pipelines.traces import Settings, import_traces

FIXTURES = Path(__file__).parent / "fixtures" / "real_traces"
RANKINGS = {"q1": ["d4", "d1", "d3"], "q2": ["d4", "d1", "d5"], "q3": ["d4", "d2", "d1"]}


def imported(name: str, settings: Settings | None = None) -> dict:
    run, _ = import_traces(
        load_dataset(FIXTURES / "questions.jsonl"),
        FIXTURES / name,
        settings or Settings(id_field="doc_id"),
    )
    return {q["query_id"]: q for q in run["queries"]}


@pytest.mark.parametrize("name", ["openinference.json", "openllmetry.json"])
def test_openinference_and_openllmetry_import_fully_with_model_context(name):
    queries = imported(name)
    assert {qid: q["ids"] for qid, q in queries.items()} == RANKINGS
    for q in queries.values():
        assert q["error"] is None
        assert q["model_context"]["coverage"] == "observed"
        assert [i["id"] for i in q["model_context"]["items"]] == q["ids"]


def test_openlit_imports_rankings_but_not_context_across_its_separate_traces():
    # OpenLIT put the retriever, vector-store and chat spans of one question in separate
    # traces, so what reached the model cannot be tied to the retrieval.
    queries = imported("openlit_ids.json", Settings())
    assert {qid: q["ids"] for qid, q in queries.items()} == RANKINGS
    assert all(q.get("model_context") is None for q in queries.values())


def test_openlit_without_metadata_ids_is_a_gap_not_guessed_from_text():
    # OpenLIT records a document ID only from metadata "id" or "source".
    queries = imported("openlit.json", Settings())
    assert {q["error"]["code"] for q in queries.values()} == {"document_id_missing"}


def test_the_fixtures_hold_no_machine_details():
    for name in ("openinference.json", "openllmetry.json", "openlit.json", "openlit_ids.json"):
        text = (FIXTURES / name).read_text()
        assert "/home/" not in text
        assert "/tmp/" not in text
        resource = json.loads(text)["resourceSpans"][0]["resource"]["attributes"]
        assert {a["key"] for a in resource} <= {
            "telemetry.sdk.language", "telemetry.sdk.name", "telemetry.sdk.version",
            "service.instance.id", "service.name",
        }  # fmt: skip


@pytest.mark.parametrize(
    ("raw", "expected"),
    [  # OpenLLMetry's LangChain instrumentation, as recorded in openllmetry.json
        (
            {"inputs": "Q?", "tags": [], "metadata": {}, "kwargs": {"name": "RunnableSequence"}},
            "Q?",
        ),
        ({"query": "Q?", "tags": ["seq:step:1"], "metadata": {}}, "Q?"),
        ({"inputs": {"question": "Q?", "context": "c"}, "kwargs": {"name": "x"}}, "Q?"),
        ({"inputs": [{"lc": 1}], "kwargs": {"name": "x"}}, None),
        # the Traceloop SDK's decorator form still wins
        ({"args": ["first"], "kwargs": {}, "inputs": "other"}, "first"),
    ],
)
def test_questions_in_the_langchain_instrumentations_entity_input(raw, expected):
    assert _question(json.dumps(raw)) == expected
