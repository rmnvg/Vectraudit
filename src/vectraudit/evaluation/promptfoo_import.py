"""promptfoo eval results as answer judgments for ``attach-answers``.

Teams that grade answers with promptfoo keep doing so; this reads ``promptfoo eval -o
results.json`` and writes the judgment lines :mod:`vectraudit.evaluation.answers` reads.
The format follows promptfoo's ``OutputFile`` and ``EvaluateResult`` types: ``results.results[]``
with ``testCase`` (``vars``, ``metadata``), ``provider``, ``prompt``, ``success``, ``score``,
``error`` and ``gradingResult.componentResults[]`` (``pass``, ``score``, ``assertion``).

- **One variant.** A promptfoo eval usually compares prompts or models. Judgments of
  different variants describe different systems and must never be attached to one run, so
  when the file holds more than one provider/prompt pair, one is chosen with ``provider``
  and ``prompt``.
- **Chosen assertions.** ``assertions`` name what judges answers: an assertion's ``metric``
  label, else its ``type``, or ``overall`` for the test's own pass/fail. ``context-recall``
  and ``context-relevance`` judge the retrieved context and are refused (Vectraudit measures
  retrieval against labels itself), and ``overall`` is not used for a test that includes
  them, since its pass/fail would then mix retrieval into the answer verdict.
- **Verdicts are promptfoo's decisions** (``pass``), with the score kept. Results that
  errored (``failureReason`` 2) are skipped and counted; ``error`` alone is not used,
  because promptfoo also sets it to the message of an ordinary assertion failure. Each
  repeat is its own judgment, so repeats that disagree are reported as disputed by
  ``attach-answers``.
- **Matching:** ``query_id`` or ``trace_id`` in the test's metadata or vars, else the
  question variable (``question``, ``query``, ``input``, or the one named) when it matches
  exactly one benchmark question.
- Reasons, prompts, outputs and other variables are never written.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vectraudit.evaluation.deepeval_import import _norm

RETRIEVAL_ASSERTIONS = ("context-recall", "context-relevance")
OVERALL = "overall"
QUESTION_VARS = ("question", "query", "input")
FAILURE_ERROR = 2  # promptfoo ResultFailureReason: 0 NONE, 1 ASSERT, 2 ERROR
MAX_BYTES = 200 * 2**20


class PromptfooImportError(ValueError):
    """The promptfoo file or settings cannot be used; the message says why."""


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """The evaluation results in a ``promptfoo eval -o`` JSON file."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise PromptfooImportError(f"{path} is larger than {MAX_BYTES // 2**20} MiB.")
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PromptfooImportError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise PromptfooImportError(f"{path} is not JSON.") from exc
    summary = data.get("results") if isinstance(data, dict) else None
    results = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(results, list):
        raise PromptfooImportError(
            f"{path} is not promptfoo eval output: expected results.results (from "
            "promptfoo eval -o results.json)."
        )
    return [r for r in results if isinstance(r, dict)]


def _variant(result: Mapping[str, Any]) -> tuple[str, str]:
    provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
    prompt = result.get("prompt") if isinstance(result.get("prompt"), dict) else {}
    assert isinstance(provider, dict)
    assert isinstance(prompt, dict)
    name = provider.get("label") or provider.get("id") or "?"
    label = prompt.get("label") or result.get("promptId") or "?"
    return str(name), str(label)


def variants(results: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    """How many results each (provider, prompt) pair has."""
    return Counter(_variant(r) for r in results)


def _assertion_name(component: Mapping[str, Any]) -> str | None:
    assertion = component.get("assertion")
    if not isinstance(assertion, dict):
        return None
    name = assertion.get("metric") or assertion.get("type")
    return str(name) if isinstance(name, str) and name else None


def assertion_names(results: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """How many results each assertion (metric label, else type) was graded on."""
    found: Counter[str] = Counter()
    for result in results:
        for component in _components(result):
            name = _assertion_name(component)
            if name:
                found[name] += 1
    return found


def _components(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    grading = result.get("gradingResult")
    components = grading.get("componentResults") if isinstance(grading, dict) else None
    return [c for c in components or [] if isinstance(c, dict)]


def to_judgments(
    results: Sequence[Mapping[str, Any]],
    assertions: Sequence[str],
    questions: Mapping[str, str | None],
    *,
    provider: str | None = None,
    prompt: str | None = None,
    question_var: str | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for the chosen variant and assertions, and what was matched or skipped."""
    if not assertions:
        raise PromptfooImportError("Choose the assertions that judge answers with --assertion.")
    refused = sorted(set(assertions) & set(RETRIEVAL_ASSERTIONS))
    if refused:
        raise PromptfooImportError(
            f"{', '.join(refused)} judge the retrieved context, not the answer; Vectraudit "
            "measures retrieval against the benchmark's labels itself."
        )
    chosen = [
        r
        for r in results
        if (provider is None or provider in _provider_names(r))
        and (prompt is None or prompt in _prompt_names(r))
    ]
    found = variants(chosen)
    if len(found) > 1:
        listing = "; ".join(f"provider {p!r}, prompt {q!r} ({n})" for (p, q), n in found.items())
        raise PromptfooImportError(
            "The results compare several variants; judgments of different systems cannot be "
            f"attached to one run. Choose one with --provider and --prompt: {listing}."
        )
    if not chosen:
        raise PromptfooImportError("No results match the chosen --provider and --prompt.")
    by_text: dict[str, list[str]] = defaultdict(list)
    for qid, text in questions.items():
        if text:
            by_text[_norm(text)].append(qid)
    wanted = set(assertions)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for result in chosen:
        # promptfoo sets ``error`` to the assertion message on ordinary failures too; only
        # failureReason 2 (ERROR, as opposed to 1, ASSERT) means the test did not run.
        if result.get("failureReason") == FAILURE_ERROR or not isinstance(
            result.get("gradingResult"), dict
        ):
            counts["errored"] += 1
            continue
        key = _match(result, questions, by_text, counts, question_var)
        if key is None:
            continue
        for name, passed, score in _graded(result, wanted, counts):
            row: dict[str, Any] = {
                key[0]: key[1],
                "evaluator": name[:200],
                "kind": "model",
                "verdict": "correct" if passed else "incorrect",
            }
            if score is not None:
                row["score"] = score
            rows.append(row)
            counts["judgments"] += 1
    return rows, counts


def _provider_names(result: Mapping[str, Any]) -> set[str]:
    provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
    assert isinstance(provider, dict)
    return {str(v) for v in (provider.get("id"), provider.get("label")) if v}


def _prompt_names(result: Mapping[str, Any]) -> set[str]:
    prompt = result.get("prompt") if isinstance(result.get("prompt"), dict) else {}
    assert isinstance(prompt, dict)
    return {str(v) for v in (prompt.get("label"), result.get("promptId")) if v}


def _graded(
    result: Mapping[str, Any], wanted: set[str], counts: Counter[str]
) -> list[tuple[str, bool, float | None]]:
    out = []
    components = _components(result)
    if OVERALL in wanted:
        if any(_assertion_type(c) in RETRIEVAL_ASSERTIONS for c in components):
            counts["overall_includes_retrieval"] += 1
        elif isinstance(result.get("success"), bool):
            out.append(("promptfoo", bool(result["success"]), _score(result.get("score"))))
    for component in components:
        name = _assertion_name(component)
        if name in wanted and isinstance(component.get("pass"), bool):
            out.append((name, bool(component["pass"]), _score(component.get("score"))))
    return out


def _assertion_type(component: Mapping[str, Any]) -> str | None:
    assertion = component.get("assertion")
    return assertion.get("type") if isinstance(assertion, dict) else None


def _score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _match(
    result: Mapping[str, Any],
    questions: Mapping[str, str | None],
    by_text: Mapping[str, list[str]],
    counts: Counter[str],
    question_var: str | None,
) -> tuple[str, str] | None:
    test = result.get("testCase") if isinstance(result.get("testCase"), dict) else {}
    assert isinstance(test, dict)
    metadata = test.get("metadata") if isinstance(test.get("metadata"), dict) else {}
    variables = result.get("vars") or test.get("vars")
    variables = variables if isinstance(variables, dict) else {}
    assert isinstance(metadata, dict)
    for source in (metadata, variables):
        query_id = source.get("query_id")
        if isinstance(query_id, str) and query_id in questions:
            counts["matched_by_query_id"] += 1
            return ("query_id", query_id)
        trace_id = source.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            counts["matched_by_trace_id"] += 1
            return ("trace_id", trace_id)
    names = (question_var,) if question_var else QUESTION_VARS
    text = next((variables[n] for n in names if isinstance(variables.get(n), str)), None)
    found = by_text.get(_norm(text)) if text else None
    if found and len(found) == 1:
        counts["matched_by_text"] += 1
        return ("query_id", found[0])
    counts["ambiguous_text" if found else "unmatched"] += 1
    return None


def render(counts: Mapping[str, int], written: int, out: str) -> str:
    """What the import did, for the terminal."""
    labels = {
        "matched_by_query_id": "results matched by query_id",
        "matched_by_trace_id": "results matched by trace_id",
        "matched_by_text": "results matched by question text",
        "ambiguous_text": "results with text matching several questions (skipped)",
        "unmatched": "results matching no question (skipped)",
        "errored": "results that errored (skipped)",
        "overall_includes_retrieval": (
            "results whose overall pass includes a retrieval assertion (overall skipped)"
        ),
    }
    lines = [f"Wrote {written} judgments to {out}."]
    lines += [f"  {counts[k]} {label}" for k, label in labels.items() if counts.get(k)]
    return "\n".join(lines)
