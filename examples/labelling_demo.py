"""Prepare a blind three-candidate labelling demo, then evaluate human labels.

    uv run python examples/labelling_demo.py prepare --out /tmp/labels --rag
    uv run vectraudit label serve /tmp/labels/task.json
    uv run python examples/labelling_demo.py finish --out /tmp/labels

--rag explicitly permits the existing example's public RAG dataset download (cached on
later runs). Without it, use synthetic inputs entirely offline. No labels are fabricated:
finish requires the user to label in the browser first.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from experiment_demo import real_inputs, synthetic_inputs

from vectraudit.experiments.corpus import load_corpus
from vectraudit.labelling.pooling import load_questions, pool
from vectraudit.labelling.task import save_task


def main() -> None:
    """Prepare source files and manifest, or export and run the standard CLI workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finish"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rag", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "finish":
        commands = [
            ["label", "export", str(out / "task.json"), "--out", str(out / "dataset.jsonl")],
            ["dataset", "validate", str(out / "dataset.jsonl")],
            ["experiment", "run", str(out / "experiment.json"), "--out", str(out / "results")],
            [
                "check",
                str(out / "results/runs/words-80.json"),
                str(out / "results/runs/words-80.json"),
                "--html",
                str(out / "check.html"),
            ],
        ]
        for command in commands:
            subprocess.run([sys.executable, "-m", "vectraudit", *command], check=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Use an empty output directory to preserve existing labels")
    docs, queries = real_inputs(500, 40, 0) if args.rag else synthetic_inputs(40, 0)
    (out / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    with (out / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "text"])
        writer.writeheader()
        writer.writerows({"query_id": q["query_id"], "text": q["text"]} for q in queries)
    candidates = [
        {
            "name": f"words-{size}",
            "embedder": {"type": "hashing"},
            "chunking": {"method": "words", "size": size, "overlap": 10},
        }
        for size in (80, 150, 300)
    ]
    manifest = {
        "name": "human-labelled-rag",
        "corpus": "corpus.jsonl",
        "queries": "dataset.jsonl",
        "k": 5,
        "candidates": candidates,
    }
    path = out / "experiment.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    task = pool(
        load_questions(out / "questions.csv"),
        load_corpus(out / "corpus.jsonl"),
        manifest=path,
        lexical=False,
        k=5,
        name="rag-human-labels",
    )
    save_task(task, out / "task.json")
    print(f"Pooled three candidates for {len(task.queries)} questions.")
    print(f"Next: vectraudit label serve {out / 'task.json'}")


if __name__ == "__main__":
    main()
