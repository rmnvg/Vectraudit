"""Run with an isolated Python containing the built wheel, FAISS and httpx."""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient

import vectraudit
from vectraudit.pipelines import PipelineResult, execute_pipeline
from vectraudit.pipelines.web import create_workbench
from vectraudit.server.app import STATIC_DIR, create_app
from vectraudit.session import Session

assert Path(vectraudit.__file__).is_relative_to(Path(sys.prefix)), vectraudit.__file__
assert (STATIC_DIR / "index.html").is_file()
x = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
index = faiss.IndexFlatL2(2)
index.add(x)
with TestClient(create_app(Session(index, vectors=x, disk_cache=False))) as client:
    assert client.get("/api/health").json()["status"] == "ok"
    assert "<html" in client.get("/").text.lower()
    response = client.post("/api/search", json={"query": {"vector": [0, 0]}, "k": 2})
    assert response.status_code == 200, response.text
    assert response.json()["recall"] == 1.0
subprocess.run([sys.executable, "-I", "-m", "vectraudit", "--version"], check=True)


# The actual shipped wheel must support onboarding and local annotation without text/LLM extras.
with tempfile.TemporaryDirectory(prefix="vectraudit-wheel-smoke-") as directory:
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.evaluation.dataset_tools import embed_dataset
    from vectraudit.experiments.corpus import Document, make_corpus
    from vectraudit.experiments.embedders import HashingEmbedder
    from vectraudit.labelling.pooling import pool
    from vectraudit.labelling.task import save_task
    from vectraudit.labelling.web import create_app as create_label_app

    root = Path(directory)
    questions = parse_dataset('{"query_id":"q0","text":"retrieval relevance"}')
    corpus = make_corpus(
        [Document("0", "retrieval relevance labels"), Document("1", "cooking recipes")]
    )
    task = pool(questions, corpus, k=2)
    task_path = root / "task.json"
    save_task(task, task_path)
    with TestClient(create_label_app(task_path), base_url="http://127.0.0.1") as client:
        response = client.get("/?annotator=Wheel")
        fields = dict(
            re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', response.text)
        )
        # Synthetic simulated human input, only in this packaging test.
        assert client.post("/judge", data={**fields, "grade": "2"}).status_code == 200
        exported = client.get("/dataset.jsonl")
        assert exported.status_code == 200
        assert parse_dataset(exported.text).n_labelled == 1
    model = HashingEmbedder(8)
    embedded = embed_dataset(
        parse_dataset(exported.text), model.embed_queries, model="hashing-8", normalized=True
    )
    dataset_path = root / "dataset.jsonl"
    dataset_path.write_text(embedded, encoding="utf-8")
    vectors = model.embed_documents([d.text for d in corpus.documents])
    index = faiss.IndexFlatIP(8)
    index.add(vectors)
    faiss.write_index(index, str(root / "index.faiss"))
    (root / "connection.json").write_text(json.dumps({"backend": "faiss", "index": "index.faiss"}))
    cmd = [sys.executable, "-I", "-m", "vectraudit"]
    diagnosed = subprocess.run(
        [
            *cmd,
            "doctor",
            "--connection",
            str(root / "connection.json"),
            "--dataset",
            str(dataset_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(diagnosed.stdout)["status"] != "blocked"
    run_path = root / "run.json"
    subprocess.run(
        [
            *cmd,
            "record",
            "--connection",
            str(root / "connection.json"),
            "--dataset",
            str(dataset_path),
            "--out",
            str(run_path),
        ],
        check=True,
    )
    subprocess.run([*cmd, "check", str(run_path), str(run_path), "--no-github-summary"], check=True)

# Full-pipeline support and the standalone workbench must also ship in the wheel.

no_answer = parse_dataset('{"query_id":"none","text":"out of scope","no_answer":true}')
record = execute_pipeline(no_answer, lambda request: PipelineResult(abstained=True))
assert record["summary"]["no_answer"]["accuracy"] == 1
assert list((STATIC_DIR / "assets").glob("Workbench-*.js"))
with (
    tempfile.TemporaryDirectory() as directory,
    TestClient(create_workbench(Path(directory))) as client,
):
    assert client.get("/api/info").json()["workbench"]
    assert client.get("/api/workbench").json()["artifacts"] == []

# Exercise the guided API and exported CLI with the installed package, never the checkout.
# The demo is an explicit input script; -I prevents its directory from supplying imports.
with tempfile.TemporaryDirectory(prefix="vectraudit-wheel-release-") as directory:
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve().parents[1] / "examples" / "pilot_release_demo.py"),
            "--out",
            str(Path(directory) / "demo"),
        ],
        cwd=directory,
        check=True,
    )
