"""A synthetic 20,000-document corpus with one group no benchmark question is close to."""

import json
from pathlib import Path

import numpy as np

from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines.atlas_build import build_from_files, write_bundle

out = Path("e2e/.data-atlas")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
topics = ["billing invoices", "password reset", "shipping delivery", "refund policy",
          "api tokens", "tax receipts", "team seats", "data export"]  # fmt: skip
sizes = [3000, 3000, 3000, 2400, 2000, 2000, 2000, 2600]  # "refund policy" has no questions
D = 16
centres = rng.standard_normal((len(topics), D)).astype("float32") * 14
rows, doc_topic = [], []
for g, (topic, n) in enumerate(zip(topics, sizes, strict=True)):
    for i in range(n):
        v = centres[g] + rng.standard_normal(D).astype("float32")
        rows.append({"doc_id": f"d{g}-{i}", "text": f"{topic} guide {i}", "vector": v.tolist()})
        doc_topic.append(g)
(out / "corpus.jsonl").write_text("\n".join(map(json.dumps, rows)))
queries, entries = [], []
for g in range(len(topics)):
    if topics[g] == "refund policy":
        continue
    for j in range(8):
        qid = f"q{g}-{j}"
        v = centres[g] + rng.standard_normal(D).astype("float32")
        relevant = f"d{g}-{j}"
        queries.append({"query_id": qid, "vector": v.tolist(), "relevant": {relevant: 1}})
        found = [relevant] if j % 4 else ["d0-0"]  # every fourth question fails
        entries.append({"query_id": qid, "ids": found, "error": None})
queries.append({"query_id": "heldout-1", "vector": centres[3].tolist(), "relevant": {"d3-0": 1},
                "split": "heldout"})  # fmt: skip
(out / "queries.jsonl").write_text("\n".join(map(json.dumps, queries)))
run = {"run_id": "run-e2e", "label": "candidate", "queries": entries}
atlas = build_from_files(out / "corpus.jsonl", load_dataset(out / "queries.jsonl"), selected=run,
                         name="e2e")  # fmt: skip
write_bundle(atlas, out, "e2e")
