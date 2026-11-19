"""Write an experiment's candidates as MTEB-style retrieval results.

A team choosing an embedding model reads public leaderboards (MTEB) and wants to put its own
numbers, measured on its own labelled questions, beside them. MTEB stores a result as one JSON
file per model and task::

    {"dataset_revision": ..., "evaluation_time": ..., "kg_co2_emissions": ..., "mteb_version": ...,
     "scores": {"test": [{"hf_subset": "default", "languages": [...], "main_score": ...,
                          "ndcg_at_10": ..., "map_at_10": ..., "mrr_at_10": ...,
                          "precision_at_10": ..., "recall_at_10": ...}]},
     "task_name": ...}

(layout read from a published result file in ``embeddings-benchmark/results``). This writes the
same shape from a finished experiment, one file per candidate: ``<out>/<candidate>/<task>.json``.

**The numbers are not like for like with the leaderboard.** They are computed by Vectraudit on the
user's corpus and labels (a different task from any public one), averaged with equal weight
over question families, at the experiment's ``k`` (MTEB's retrieval default is 10), with
Vectraudit's definitions, which match trec_eval and ranx. The files say so under ``vectraudit``,
and ``mteb_version`` is ``null``: MTEB did not produce them. ``main_score`` is nDCG at the
experiment's k (MTEB's default main score for retrieval is nDCG@10);
``vectraudit.main_score_metric`` names the exact key. Languages are unknown to Vectraudit and left
empty.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vectraudit import __version__

TASK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
KEYS = {"ndcg": "ndcg", "map": "map", "mrr": "mrr", "precision": "precision", "recall": "recall"}
NOT_LIKE_FOR_LIKE = (
    "Measured by Vectraudit on the user's own corpus and labelled questions, which is not a "
    "public MTEB task: do not rank this against leaderboard scores as if it were. Averaged with "
    "equal weight over question families, at the experiment's k."
)


class MtebExportError(ValueError):
    """The experiment cannot be exported; the message says why."""


def candidate_result(
    experiment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    task_name: str,
    split: str = "test",
) -> dict[str, Any]:
    """One candidate's MTEB-style result."""
    if not TASK.match(task_name):
        raise MtebExportError("The task name may use letters, digits, '.', '_' and '-'.")
    metrics = (candidate.get("summary") or {}).get("metrics")
    if not metrics:
        raise MtebExportError(
            f"Candidate {candidate['name']!r} has no label metrics: score it on labelled questions."
        )
    k = int(experiment["k"])
    summary = candidate["summary"]
    if summary.get("n_errors"):
        raise MtebExportError(
            f"Candidate {candidate['name']!r} had {summary['n_errors']} failed questions; an "
            "incomplete result is not exported as if it were whole."
        )
    score: dict[str, Any] = {"hf_subset": "default", "languages": []}
    score["main_score"] = metrics["ndcg"]
    for ours, mteb in KEYS.items():
        if ours in metrics:
            score[f"{mteb}_at_{k}"] = metrics[ours]
    stats = candidate.get("stats") or {}
    seconds = sum(
        stats[key]
        for key in ("embed_documents_s", "embed_queries_s", "build_index_s")
        if key in stats
    )
    return {
        "dataset_revision": experiment["dataset"]["sha256"],
        "evaluation_time": seconds,
        "kg_co2_emissions": None,
        "mteb_version": None,
        "scores": {split: [score]},
        "task_name": task_name,
        "vectraudit": {
            "generator": f"vectraudit {__version__}",
            "not_like_for_like": NOT_LIKE_FOR_LIKE,
            "main_score_metric": f"ndcg_at_{k}",
            "k": k,
            "corpus_sha256": (experiment.get("corpus") or {}).get("sha256"),
            "documents": (experiment.get("corpus") or {}).get("documents"),
            "n_queries": experiment["dataset"]["n_queries"],
            "n_families": experiment["dataset"].get("n_families"),
            "candidate": {
                "name": candidate["name"],
                "embedder": candidate.get("embedder"),
                "chunking": candidate.get("chunking"),
                "index": candidate.get("index"),
            },
            "evaluation_time_means": "embedding and index time measured by Vectraudit",
        },
    }


def export_experiment(
    result: Mapping[str, Any], out: Path, *, task_name: str, split: str = "test"
) -> list[Path]:
    """Write ``<out>/<candidate>/<task_name>.json`` for every candidate of ``result``."""
    if result.get("format") != "vectraudit.experiment":
        raise MtebExportError("This is not a vectraudit experiment result (result.json).")
    written = []
    for candidate in result["candidates"]:
        body = candidate_result(result, candidate, task_name=task_name, split=split)
        path = out / candidate["name"] / f"{task_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)
    return written
