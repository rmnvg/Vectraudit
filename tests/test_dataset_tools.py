"""Dataset import, embed, diff and stats (library and CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit import release_cli
from vectraudit.cli import app
from vectraudit.evaluation.dataset import DatasetError, load_dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import (
    ColumnMap,
    dataset_stats,
    diff_datasets,
    embed_dataset,
    import_rows,
    read_records,
    render_dataset,
)

runner = CliRunner()
CSV = """query_id,text,relevant,segment,lang,year,public
q1,How do I get a refund?,doc-1:2; doc-7,billing,en,2024,true
q2,Reset my password,doc-3,account,en,2023,false
q3,Wo ist meine Rechnung?,,billing,de,,yes
"""


def fake_embed(texts):
    return np.array([[len(t), 1.0, 0.5] for t in texts], dtype=np.float32)


def test_import_csv(tmp_path) -> None:
    (tmp_path / "q.csv").write_text(CSV)
    columns = ColumnMap(tags="segment", filters={"lang": "str", "year": "int", "public": "bool"})
    rows = import_rows(read_records(tmp_path / "q.csv"), columns)
    assert rows[0] == {
        "query_id": "q1",
        "text": "How do I get a refund?",
        "relevant": {"doc-1": 2.0, "doc-7": 1.0},
        "tags": ["billing"],
        "filter": {"lang": "en", "year": 2024, "public": True},
    }
    assert "relevant" not in rows[2]
    assert rows[2]["filter"] == {"lang": "de", "public": True}
    ds = parse_dataset(render_dataset(rows, {"name": "support"}))
    assert ds.meta == {"name": "support"}
    assert ds.n_labelled == 2


def test_import_jsonl_and_errors(tmp_path) -> None:
    (tmp_path / "q.jsonl").write_text(
        json.dumps({"id": "a", "question": "hi", "gold": ["d1", "d2"], "vec": [1, 2]}) + "\n"
    )
    columns = ColumnMap(query_id="id", text="question", relevant="gold", vector="vec")
    rows = import_rows(read_records(tmp_path / "q.jsonl"), columns)
    assert rows == [
        {"query_id": "a", "text": "hi", "vector": [1, 2], "relevant": {"d1": 1.0, "d2": 1.0}}
    ]
    with pytest.raises(DatasetError, match="Row 1: missing column 'query_id'"):
        import_rows([{"text": "x"}], ColumnMap())
    with pytest.raises(DatasetError, match="not a number"):
        import_rows([{"query_id": "q", "relevant": "d:high"}], ColumnMap())
    with pytest.raises(DatasetError, match="not a int"):
        import_rows([{"query_id": "q", "y": "soon"}], ColumnMap(filters={"y": "int"}))
    with pytest.raises(DatasetError, match="must be one of"):
        import_rows([], ColumnMap(filters={"y": "date"}))
    with pytest.raises(DatasetError, match="not found"):
        read_records(tmp_path / "missing.csv")
    (tmp_path / "bad.jsonl").write_text("{oops\n")
    with pytest.raises(DatasetError, match="Line 1"):
        read_records(tmp_path / "bad.jsonl")


def test_embed_dataset() -> None:
    ds = parse_dataset(
        json.dumps({"query_id": "a", "text": "hello"})
        + "\n"
        + json.dumps({"query_id": "b", "text": "hi", "relevant": {"d": 1}})
    )
    embedded = parse_dataset(embed_dataset(ds, fake_embed, model="m1", normalized=False))
    assert embedded.dim == 3
    assert embedded.meta == {"embedding_model": "m1", "normalized": False}
    assert embedded.queries[0].vector is not None
    assert embedded.queries[0].vector[0] == 5.0
    with pytest.raises(DatasetError, match="already has vectors from 'm1'"):
        embed_dataset(embedded, fake_embed, model="m2", normalized=False)
    again = parse_dataset(embed_dataset(embedded, fake_embed, model="m1", normalized=False))
    assert again.sha256 == embedded.sha256
    switched = embed_dataset(embedded, fake_embed, model="m2", normalized=True, overwrite=True)
    assert parse_dataset(switched).meta["embedding_model"] == "m2"
    with pytest.raises(DatasetError, match="wrong shape"):
        embed_dataset(ds, lambda texts: np.ones((1, 3)), model="m", normalized=False)


def test_diff_and_stats() -> None:
    a = parse_dataset(
        "\n".join(
            json.dumps(r)
            for r in [
                {"query_id": "q1", "text": "a", "relevant": {"d": 1}, "tags": ["x"]},
                {"query_id": "q2", "text": "b"},
                {"query_id": "q3", "text": "b", "relevant": {"d": 2, "e": 1}},
            ]
        )
    )
    b = parse_dataset(
        "\n".join(
            json.dumps(r)
            for r in [
                {"query_id": "q1", "text": "a", "relevant": {"d": 2}, "tags": ["x"]},
                {"query_id": "q3", "text": "b", "relevant": {"d": 2, "e": 1}},
                {"query_id": "q4", "text": "c"},
            ]
        )
    )
    diff = diff_datasets(a, b)
    assert diff["added"] == ["q4"]
    assert diff["removed"] == ["q2"]
    assert diff["changed"] == {"q1": ["relevant"]}
    assert not diff["same"]
    assert diff_datasets(a, a)["same"]

    stats = dataset_stats(a, cohort_minimum=5)
    assert stats["n_labelled"] == 2
    assert stats["relevant_per_query"] == {"min": 1, "mean": 1.5, "max": 2}
    assert stats["grades"] == {"1.0": 2, "2.0": 1}
    assert stats["tags"] == {"x": 1}
    text = " ".join(stats["warnings"])
    assert "Only 2 labelled queries" in text
    assert "1 queries have no labels" in text
    assert "below the policy minimum of 5" in text
    assert "appear more than once" in text
    assert "dataset embed" in text


def test_cli_dataset_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(release_cli, "_embedder", lambda model, normalize: fake_embed)
    (tmp_path / "q.csv").write_text(CSV)
    ds_path = tmp_path / "retrieval" / "dataset.jsonl"
    result = runner.invoke(
        app,
        [
            "dataset",
            "import",
            str(tmp_path / "q.csv"),
            "--out",
            str(ds_path),
            "--tags-column",
            "segment",
            "--filter",
            "lang",
            "--filter",
            "year:int",
            "--name",
            "support",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 3 queries (2 labelled)" in result.output
    v1 = load_dataset(ds_path)

    embedded = tmp_path / "embedded.jsonl"
    result = runner.invoke(
        app, ["dataset", "embed", str(ds_path), "--model", "fake-3d", "--out", str(embedded)]
    )
    assert result.exit_code == 0, result.output
    assert "dimension 3" in result.output
    assert load_dataset(embedded).meta["embedding_model"] == "fake-3d"

    result = runner.invoke(app, ["dataset", "diff", str(ds_path), str(embedded)])
    assert result.exit_code == 0
    assert "changed 3" in result.output
    assert "Header changed: embedding_model, normalized" in result.output
    result = runner.invoke(app, ["dataset", "diff", str(ds_path), str(ds_path)])
    assert f"Identical: version {v1.sha256[:12]}" in result.output
    result = runner.invoke(app, ["dataset", "diff", str(ds_path), str(embedded), "--json"])
    assert json.loads(result.output)["meta_changed"] == ["embedding_model", "normalized"]

    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"metrics": {"ndcg": {"max_drop": 0.02}}, "cohorts": {"min_queries": 3}})
    )
    result = runner.invoke(app, ["dataset", "stats", str(embedded), "--policy", str(policy)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Cohorts: account (1), billing (2)" in flat
    assert "below the policy minimum of 3" in flat
    stats = json.loads(runner.invoke(app, ["dataset", "stats", str(embedded), "--json"]).output)
    assert stats["meta"]["name"] == "support"


def test_cli_import_errors(tmp_path) -> None:
    (tmp_path / "q.csv").write_text("id,text\nq1,hi\n")
    result = runner.invoke(
        app, ["dataset", "import", str(tmp_path / "q.csv"), "--out", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code == 1
    assert "missing column 'query_id'" in result.output
    assert not Path(tmp_path / "o.jsonl").exists()
