"""Migration assurance: two systems, one benchmark, and the checks a migration needs."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.migration import (
    MigrationError,
    filter_conformance,
    migration_report,
    read_inventory,
    render_migration,
)
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines import (
    Hit,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    execute_pipeline,
)

DOCS = {f"d{i}": {"sha256": f"{i:064x}", "metadata": {"lang": "en"}} for i in range(9)}
DOCS["d9"] = {"sha256": f"{9:064x}", "metadata": {"lang": "fr"}}
ROWS = [
    {
        "query_id": f"q{i}",
        "text": f"topic {i}",
        "relevant": {f"d{i}": 1},
        **({"filter": {"lang": "en"}} if i % 2 else {}),
    }
    for i in range(8)
]
POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 0.05}}})


def dataset():
    return parse_dataset("\n".join(json.dumps(r) for r in ROWS))


class System:
    """Answers ``topic i`` with d<i>; ``defect`` plants one migration failure."""

    def __init__(self, defect=None):
        self.defect = defect

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        i = int((request.text or "").split()[-1])
        ids = [f"d{i}", "d9"]
        if request.filters and self.defect != "ignores_filter":
            ids = [d for d in ids if DOCS[d]["metadata"]["lang"] == request.filters["lang"]]
        if self.defect == "uuid_ids":
            ids = [f"uuid-{d}" for d in ids]
        if self.defect == "lost_document" and i == 3:
            ids = [d for d in ids if d != "d3"]
        return PipelineResult(hits=[Hit(id=d) for d in ids])


def run(system, label, data=None):
    return execute_pipeline(data or dataset(), system, PipelineOptions(k=2, warmup=0, label=label))


def report(proposed, *, proposed_docs=DOCS, **kwargs):
    return migration_report(
        run(System(), "pgvector"),
        proposed,
        dataset(),
        POLICY,
        current_inventory=DOCS,
        proposed_inventory=proposed_docs,
        **kwargs,
    )


def test_an_equivalent_migration_is_ready():
    result = report(run(System(), "qdrant"))
    assert (result["verdict"], result["concerns"], result["migration_ready"]) == ("pass", [], True)
    assert result["corpus"]["equivalent"] is True
    assert result["filters"]["proposed"]["n_violations"] == 0


def test_a_backend_that_ignores_a_filter_is_caught_even_if_metrics_hold():
    result = report(run(System("ignores_filter"), "qdrant"))
    assert result["filters"]["proposed"]["violations"][0] == {
        "query_id": "q1",
        "doc_id": "d9",
        "fields": ["lang"],
    }
    assert result["filters"]["current"]["n_violations"] == 0
    assert not result["migration_ready"]


def test_unmapped_ids_are_named_as_a_mapping_problem_not_worse_retrieval():
    result = report(run(System("uuid_ids"), "qdrant"))
    proposed = result["id_mapping"]["proposed"]
    assert proposed["suspect"] is True
    assert proposed["n_not_in_inventory"] == 9
    assert result["verdict"] == "fail"
    assert any("look unmapped" in c for c in result["concerns"])


def test_a_labelled_document_missing_from_the_new_corpus_is_a_corpus_gap():
    smaller = {d: v for d, v in DOCS.items() if d != "d3"}
    result = report(run(System("lost_document"), "qdrant"), proposed_docs=smaller)
    assert result["corpus"]["labelled_missing_in_proposed"] == ["d3"]
    assert result["corpus"]["n_missing_in_proposed"] == 1
    assert any("corpus gap, not retrieval" in c for c in result["concerns"])
    changed = {**DOCS, "d4": {**DOCS["d4"], "sha256": "f" * 64}}
    assert report(run(System(), "qdrant"), proposed_docs=changed)["corpus"]["content_changed"] == [
        "d4"
    ]


def test_without_inventories_the_checks_say_what_is_unverified():
    result = migration_report(run(System(), "pgvector"), run(System(), "qdrant"), dataset(), POLICY)
    assert result["corpus"] is None
    assert result["filters"]["proposed"] == {"checked": False, "filtered_questions": 4}
    assert not result["migration_ready"]
    assert any("Corpus equivalence is unverified" in c for c in result["concerns"])


def test_environment_and_costs_are_disclosed_not_corrected():
    proposed = execute_pipeline(
        dataset(), System(), PipelineOptions(k=2, warmup=0, label="qdrant", concurrency=2)
    )
    result = report(proposed, declared_costs={"proposed": {"reindex_usd": 1200}})
    assert "concurrency" in result["environment_differs"]
    assert result["costs"]["declared"]["proposed"] == {"reindex_usd": 1200}
    assert result["costs"]["measured"]["proposed"]["spend_usd"] is None  # unpriced calls
    text = render_migration(result)
    assert "concurrency **(differs)**" in text
    assert "declared estimate: reindex_usd $1,200.00" in text
    assert "unknown (unpriced calls)" in text
    with pytest.raises(MigrationError, match="non-negative"):
        report(proposed, declared_costs={"proposed": {"reindex_usd": -1}})


def test_runs_on_different_benchmarks_are_refused():
    other = parse_dataset(json.dumps({"query_id": "x", "text": "topic 1", "relevant": {"d1": 1}}))
    with pytest.raises(MigrationError, match="same labelled suite"):
        report(run(System(), "qdrant", data=other))


def test_inventories_hold_ids_hashes_and_metadata_only(tmp_path):
    path = tmp_path / "inv.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "private passage"}))
    with pytest.raises(MigrationError, match="no text"):
        read_inventory(path)
    path.write_text("\n".join(json.dumps({"id": "a"}) for _ in range(2)))
    with pytest.raises(MigrationError, match="appears twice"):
        read_inventory(path)


def test_dotted_filter_fields_and_ranges_follow_portable_semantics():
    data = parse_dataset(
        json.dumps(
            {
                "query_id": "q",
                "text": "x",
                "relevant": {"a": 1},
                "filter": {"meta.year": {"gte": 2024}},
            }
        )
    )
    runs = {"queries": [{"query_id": "q", "ids": ["a", "b"], "error": None}]}
    inventory = {
        "a": {"metadata": {"meta": {"year": 2025}}},
        "b": {"metadata": {"meta": {"year": 2020}}},
    }
    result = filter_conformance(runs, data, inventory)
    assert result["violations"] == [{"query_id": "q", "doc_id": "b", "fields": ["meta.year"]}]


def test_cli_writes_the_report_and_can_gate_the_migration(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    data = write("questions.jsonl", "\n".join(json.dumps(r) for r in ROWS))
    inventory = write("docs.jsonl", "\n".join(json.dumps({"id": d, **v}) for d, v in DOCS.items()))
    current = write("current.run.json", run(System(), "pgvector"))
    proposed = write("proposed.run.json", run(System("ignores_filter"), "qdrant"))
    out = tmp_path / "migration.md"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "migration", str(current), str(proposed), str(data),
            "--current-inventory", str(inventory),
            "--proposed-inventory", str(inventory),
            "--out", str(out),
            "--fail-unless-ready",
        ],
    )  # fmt: skip
    assert result.exit_code == 3, result.output
    assert "Not ready to migrate" in result.output
    assert "do not satisfy the question's filter" in result.output
    assert out.read_text().startswith("# Migration report: pgvector → qdrant")
