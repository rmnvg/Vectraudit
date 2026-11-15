"""Ragas evaluation results as answer judgments for ``attach-answers``.

Teams that already score answers with Ragas keep doing so; this reads the table
``evaluate(...).to_pandas()`` produces, saved as CSV (``to_csv``), JSON Lines
(``to_json(orient="records", lines=True)``) or JSON (a list of records, or pandas' default
column layout), and writes the judgment lines :mod:`vectraudit.evaluation.answers` reads.
Each row holds the sample's fields (``user_input``, ``response``, ``reference``,
``retrieved_contexts`` ...) followed by one column per metric, named as Ragas names it
(``faithfulness``, ``exact_match``, ``rouge_score(mode=fmeasure)`` ...).

- **Which metrics are answer judgments is the user's choice** (``metrics``). Ragas
  metrics measure different things, and a faithful answer can still be wrong, so none is
  assumed to mean "correct". Metrics that judge the retrieved context (every metric with
  ``context`` in its name: context precision, recall, entity recall, utilization and
  relevance, LLM-based or not) are refused: Vectraudit measures retrieval against the
  benchmark's labels itself.
- **Ragas gives scores, not decisions.** Each judgment carries only the score; the
  threshold that makes it a verdict is declared when attaching (``attach-answers
  --threshold ragas.METRIC=VALUE``), never assumed here. A metric that failed (Ragas
  writes an empty cell or ``null`` when the judge errored and ``raise_exceptions=False``)
  is skipped and counted. Judgments are ``model`` judgments named ``ragas.METRIC``.
- **Matching a row to a question:** a ``query_id`` column, else a ``trace_id`` column
  (both added by the user before saving; Ragas has neither), else ``user_input`` when it
  matches exactly one benchmark question (case and spacing ignored). Multi-turn samples
  have no single question and match only by ID. Ambiguous and unmatched rows are counted.
- **Nothing private is copied:** inputs, responses, references and contexts are never
  written.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vectraudit.evaluation.deepeval_import import _norm

PREFIX = "ragas."
RETRIEVAL_MARKER = "context"
"""Ragas metrics with this in their name judge the retrieved context, not the answer."""
SAMPLE_COLUMNS = frozenset(
    {
        # ragas.dataset_schema.SingleTurnSample / MultiTurnSample
        "user_input", "retrieved_contexts", "reference_contexts", "retrieved_context_ids",
        "reference_context_ids", "response", "multi_responses", "reference", "rubrics",
        "persona_name", "query_style", "query_length", "reference_tool_calls",
        "reference_topics",
        # Testset.to_pandas, and the IDs a user adds for matching
        "synthesizer_name", "query_id", "trace_id",
    }
)  # fmt: skip
MAX_BYTES = 200 * 2**20


class RagasImportError(ValueError):
    """The Ragas file or settings cannot be used; the message says why."""


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """The rows of a saved Ragas result table (.csv, .jsonl or .json)."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise RagasImportError(f"{path} is larger than {MAX_BYTES // 2**20} MiB.")
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RagasImportError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows: Any = list(csv.DictReader(text.splitlines()))
    else:
        try:
            if suffix == ".jsonl":
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                rows = _records(json.loads(text))
        except ValueError as exc:
            raise RagasImportError(f"{path} is not JSON.") from exc
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(r, dict) for r in rows)
        or not any("user_input" in r or "query_id" in r for r in rows)
    ):
        raise RagasImportError(
            f"{path} is not a Ragas result table: expected rows with user_input and metric "
            "columns (from evaluate(...).to_pandas())."
        )
    return rows


def _records(data: Any) -> Any:
    """Records from a JSON list, or from pandas' default ``{column: {row: value}}``."""
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        index = sorted({k for col in data.values() for k in col}, key=_index_key)
        return [{c: data[c].get(i) for c in data} for i in index]
    return data


def _index_key(key: str) -> tuple[int, int | str]:
    return (0, int(key)) if key.isdigit() else (1, key)


def _number(value: Any) -> float | str | None:
    """A finite score, ``None`` when missing or failed, or ``"text"`` when not a number."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "text"
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return "text"
        return number if math.isfinite(number) else None
    return "text"


def metric_names(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Metric columns (numeric, not sample fields) and how many rows have a score in each."""
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(c for c in row if c not in SAMPLE_COLUMNS))
    found: Counter[str] = Counter()
    for column in columns:
        values = [_number(row.get(column)) for row in rows]
        if "text" in values:
            continue
        found[column] = sum(v is not None for v in values)
    return found


def is_retrieval_metric(name: str) -> bool:
    """Whether a Ragas metric judges the retrieved context rather than the answer."""
    return RETRIEVAL_MARKER in name.lower()


def to_judgments(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    questions: Mapping[str, str | None],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Score-only judgment lines for the chosen metrics, and counts of what was skipped.

    ``questions`` maps benchmark question IDs to their text.
    """
    if not metrics:
        raise RagasImportError("Choose the metrics that judge answers with --metric.")
    names = [m.removeprefix(PREFIX) for m in metrics]
    refused = sorted({m for m in names if is_retrieval_metric(m)})
    if refused:
        raise RagasImportError(
            f"{', '.join(refused)} judge the retrieved context, not the answer; Vectraudit "
            "measures retrieval against the benchmark's labels itself."
        )
    available = metric_names(rows)
    missing = [m for m in names if m not in available]
    if missing:
        listing = ", ".join(sorted(available)) or "none"
        raise RagasImportError(
            f"No numeric metric column {', '.join(missing)} in this file. Metrics: {listing}."
        )
    by_text: dict[str, list[str]] = defaultdict(list)
    for qid, text in questions.items():
        if text:
            by_text[_norm(text)].append(qid)
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        key = _match(row, questions, by_text, counts)
        if key is None:
            continue
        for metric in dict.fromkeys(names):
            score = _number(row.get(metric))
            if not isinstance(score, float):
                counts["not_scored"] += 1
                continue
            out.append(
                {key[0]: key[1], "evaluator": PREFIX + metric, "kind": "model", "score": score}
            )
            counts["judgments"] += 1
    return out, counts


def _match(
    row: Mapping[str, Any],
    questions: Mapping[str, str | None],
    by_text: Mapping[str, list[str]],
    counts: Counter[str],
) -> tuple[str, str] | None:
    query_id = row.get("query_id")
    if (
        isinstance(query_id, (str, int))
        and not isinstance(query_id, bool)
        and str(query_id) in questions
    ):
        counts["matched_by_query_id"] += 1
        return ("query_id", str(query_id))
    trace_id = row.get("trace_id")
    if isinstance(trace_id, str) and trace_id:
        counts["matched_by_trace_id"] += 1
        return ("trace_id", trace_id)
    text = row.get("user_input")
    if isinstance(text, list) or (isinstance(text, str) and text.lstrip().startswith("[{")):
        counts["multi_turn"] += 1
        return None
    found = by_text.get(_norm(text)) if isinstance(text, str) else None
    if found and len(found) == 1:
        counts["matched_by_text"] += 1
        return ("query_id", found[0])
    counts["ambiguous_text" if found else "unmatched"] += 1
    return None


def render(counts: Mapping[str, int], written: int, out: str, metrics: Sequence[str]) -> str:
    """What the import did, and the thresholds attach-answers needs, for the terminal."""
    labels = {
        "matched_by_query_id": "matched by query_id",
        "matched_by_trace_id": "matched by trace_id",
        "matched_by_text": "matched by user_input",
        "ambiguous_text": "with user_input matching several questions",
        "unmatched": "matching no question",
        "multi_turn": "multi-turn, without a query_id",
    }
    lines = [f"Wrote {written} score judgments to {out}."]
    for key, label in labels.items():
        if counts.get(key):
            lines.append(f"  {counts[key]} rows {label}")
    if counts.get("not_scored"):
        lines.append(f"  {counts['not_scored']} scores missing or failed (skipped)")
    names = dict.fromkeys(PREFIX + m.removeprefix(PREFIX) for m in metrics)
    lines.append(
        "Ragas gives scores, not verdicts; declare the pass mark when attaching: "
        + " ".join(f"--threshold {n}=VALUE" for n in names)
    )
    return "\n".join(lines)
