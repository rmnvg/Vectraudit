"""The quickstart works as written: build script, scaffold, import, embed, record, check.

The real guide embeds SciFact with all-MiniLM-L6-v2; here a stand-in encoder maps each text
to a fixed random vector, and each question's labels are its exact nearest documents, so
probing every IVF list passes and probing one fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app

pytest.importorskip("faiss")

ROOT = Path(__file__).parents[1]
DIM, DOCS, QUERIES, NLIST = 32, 800, 60, 16


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def beir(tmp_path, monkeypatch):
    """A small BEIR folder with integer corpus ids, and an encoder that knows its texts."""
    rng = np.random.default_rng(7)
    docs, queries = _unit(rng, DOCS), _unit(rng, QUERIES)
    folder = tmp_path / "scifact"
    (folder / "qrels").mkdir(parents=True)
    vectors: dict[str, np.ndarray] = {}
    with (folder / "corpus.jsonl").open("w") as f:
        for i, v in enumerate(docs):
            doc = {"_id": str(1000 + i), "title": f"Title {i}", "text": f"abstract {i}"}
            f.write(json.dumps(doc) + "\n")
            vectors[f"Title {i} abstract {i}"] = v
    nearest = np.argsort(-(queries @ docs.T), axis=1)[:, :3]
    with (folder / "queries.jsonl").open("w") as f, (folder / "qrels" / "test.tsv").open("w") as q:
        q.write("query-id\tcorpus-id\tscore\n")
        for j, v in enumerate(queries):
            f.write(json.dumps({"_id": f"q{j}", "text": f"claim {j}"}) + "\n")
            vectors[f"claim {j}"] = v
            for i in nearest[j]:
                q.write(f"q{j}\t{1000 + i}\t1\n")

    class SentenceTransformer:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **kwargs):
            single = isinstance(texts, str)
            out = np.stack([vectors[t] for t in ([texts] if single else texts)])
            return out[0] if single else out

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.chdir(tmp_path)
    return folder


def build_index(args: list[str], monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "quickstart_build_index", ROOT / "examples" / "quickstart" / "build_index.py"
    )
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["build_index.py", *args])
    script.main()


def cli(*args: str) -> int:
    result = CliRunner().invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    return result.exit_code


def test_quickstart_catches_a_lower_nprobe_and_passes_a_full_probe(beir, monkeypatch):
    build_index(["scifact", "--out", "retrieval/index", "--nlist", str(NLIST)], monkeypatch)
    assert {p.name for p in Path("retrieval/index").iterdir()} == {
        "index.faiss",
        "vectors.npy",
        "ids.npy",
    }

    assert cli("init", "--backend", "faiss") == 0
    Path("retrieval/connection.json").write_text(
        json.dumps(
            {
                "backend": "faiss",
                "name": "scifact ivf",
                "index": "index/index.faiss",
                "vectors": "index/vectors.npy",
                "ids": "index/ids.npy",
                "params": {"nprobe": NLIST},
            }
        )
    )
    dataset, connection = "retrieval/dataset.jsonl", "retrieval/connection.json"
    assert cli("dataset", "import", "scifact", "--format", "beir", "--split", "test",
               "--out", dataset) == 0  # fmt: skip
    assert cli("dataset", "embed", dataset, "--model", "stand-in", "--normalize") == 0

    record = ("record", "--connection", connection, "--dataset", dataset)
    assert cli(*record, "--out", "baseline.json", "--label", "main") == 0
    assert cli(*record, "--out", "low.json", "--label", "low", "--param", "nprobe=1") == 0
    assert cli(*record, "--out", "same.json", "--label", "same") == 0

    policy = ("--policy", "retrieval/policy.json")
    assert cli("check", "baseline.json", "low.json", *policy, "--html", "report.html") == 3
    assert "Fail" in Path("report.html").read_text()
    assert cli("check", "baseline.json", "same.json", *policy) == 0


def test_build_script_refuses_ids_faiss_cannot_key_by(beir, monkeypatch):
    corpus = beir / "corpus.jsonl"
    corpus.write_text(json.dumps({"_id": "MED-10", "title": "t", "text": "x"}) + "\n")
    with pytest.raises(SystemExit, match="not an integer"):
        build_index(["scifact"], monkeypatch)
