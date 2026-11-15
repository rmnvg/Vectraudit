"""promptfoo red-team results for the RAG plugins, as suite-level counts and safety records.

``promptfoo redteam run`` writes the same output file as ``promptfoo eval`` (see
:mod:`vectraudit.evaluation.promptfoo_import`) with, in each test's metadata, the plugin that
generated it (``pluginId``) and the strategy that disguised it (``strategyId``). A test
**passes** when the system resisted the attack and **fails** when it did not; promptfoo
decides that, with a model as the grader. This keeps the RAG-related plugins:

- ``rag-poisoning``, ``rag-document-exfiltration``, ``indirect-prompt-injection``,
  ``rag-source-attribution`` and ``prompt-extraction``.

What is written, and what is not:

- **Suite-level counts** (a ``vectraudit.promptfoo_redteam`` report): per plugin and strategy,
  how many tests passed, failed or errored, with a Wilson interval on the failure rate. Red-team
  prompts are generated, so most match no benchmark question and are only counted here.
- **Per-question safety records** only for a test whose metadata or variables carry a
  ``query_id`` of the benchmark or a ``trace_id``: a ``safety`` record
  (``promptfoo.redteam.PLUGIN``) holding the plugin, the strategy and whether it passed,
  kept apart from answer verdicts.
- Never the reason, the prompt, the output or any variable. Results that errored are skipped
  and counted; other plugins are counted by name and skipped.

Everything is model-graded, so the evidence level is ``supported``, never ``verified``, and a
red-team result is exploratory: it approves nothing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.promptfoo_import import FAILURE_ERROR, load_results
from vectraudit.evaluation.significance import wilson_interval

__all__ = ["PLUGINS", "REPORT_FORMAT", "import_redteam", "load_results", "plugin_counts", "render"]

REPORT_FORMAT = "vectraudit.promptfoo_redteam"
REPORT_FORMAT_VERSION = 1
PLUGINS = (
    "rag-poisoning",
    "rag-document-exfiltration",
    "indirect-prompt-injection",
    "rag-source-attribution",
    "prompt-extraction",
)
EVALUATOR_PREFIX = "promptfoo.redteam."
VERSION = "promptfoo red team; pass means the system resisted"


def _metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    test = result.get("testCase") if isinstance(result.get("testCase"), dict) else {}
    for source in (
        test.get("metadata") if isinstance(test, dict) else None,
        result.get("metadata"),
    ):
        if isinstance(source, dict) and isinstance(source.get("pluginId"), str):
            return source
    return {}


def plugin_counts(results: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Plugin IDs present in the results and how many tests each has."""
    return Counter(m["pluginId"] for r in results if (m := _metadata(r)))


def _passed(result: Mapping[str, Any]) -> bool | None:
    """Whether the system resisted; ``None`` when the test errored or has no decision."""
    if result.get("failureReason") == FAILURE_ERROR:
        return None
    grading = result.get("gradingResult")
    for value in (
        result.get("success"),
        grading.get("pass") if isinstance(grading, dict) else None,
    ):
        if isinstance(value, bool):
            return value
    return None


def _key(result: Mapping[str, Any], questions: Mapping[str, str | None]) -> tuple[str, str] | None:
    """The benchmark question or trace this test names: by ``query_id`` or ``trace_id`` only."""
    test = result.get("testCase") if isinstance(result.get("testCase"), dict) else {}
    variables = result.get("vars") or (test.get("vars") if isinstance(test, dict) else None)
    metadata = test.get("metadata") if isinstance(test, dict) else None
    for source in (metadata, variables):
        if not isinstance(source, dict):
            continue
        query_id = source.get("query_id")
        if isinstance(query_id, str) and query_id in questions:
            return "query_id", query_id
        trace_id = source.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            return "trace_id", trace_id
    return None


def import_redteam(
    results: Sequence[Mapping[str, Any]], questions: Mapping[str, str | None]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-question safety records for the matched tests, and the suite-level report."""
    records: list[dict[str, Any]] = []
    tests: dict[str, Counter[str]] = defaultdict(Counter)
    strategies: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    other: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for result in results:
        metadata = _metadata(result)
        plugin = metadata.get("pluginId")
        if not isinstance(plugin, str):
            other["(no plugin)"] += 1
            continue
        if plugin not in PLUGINS:
            other[plugin] += 1
            continue
        strategy = metadata.get("strategyId")
        strategy = strategy if isinstance(strategy, str) and strategy else None
        passed = _passed(result)
        if passed is None:
            tests[plugin]["errored"] += 1
            continue
        tests[plugin]["tests"] += 1
        tests[plugin]["failed"] += not passed
        row = strategies[plugin][strategy or "none"]
        row["tests"] += 1
        row["failed"] += not passed
        key = _key(result, questions)
        if key is None:
            matched["unmatched"] += 1
            continue
        matched[f"by_{key[0]}"] += 1
        tests[plugin]["matched_to_questions"] += 1
        records.append(
            {
                key[0]: key[1],
                "evaluator": EVALUATOR_PREFIX + plugin,
                "kind": "safety",
                "evaluator_version": VERSION,
                "redteam": {"plugin": plugin, "strategy": strategy, "passed": passed},
            }
        )
    if not tests:
        raise ValueError(
            "No result is from a RAG red-team plugin ("
            + ", ".join(PLUGINS)
            + "). Plugins in this file: "
            + (", ".join(f"{p} ({n})" for p, n in sorted(other.items())) or "none")
            + "."
        )
    plugins = {}
    for plugin in sorted(tests):
        c = tests[plugin]
        low, high = wilson_interval(c["failed"], c["tests"])
        plugins[plugin] = {
            "tests": c["tests"],
            "passed": c["tests"] - c["failed"],
            "failed": c["failed"],
            "errored": c["errored"],
            "failure_rate": (c["failed"] / c["tests"]) if c["tests"] else None,
            "ci_low": low,
            "ci_high": high,
            "matched_to_questions": c["matched_to_questions"],
            "by_strategy": {
                s: {"tests": row["tests"], "failed": row["failed"]}
                for s, row in sorted(strategies[plugin].items())
            },
        }
    report = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_FORMAT_VERSION,
        "evidence": "supported",
        "note": (
            "promptfoo's grader is a model, so these counts are model-judged and exploratory; "
            "they approve nothing. Red-team prompts are generated, so most match no benchmark "
            "question and are counted here only. No reason, prompt or output is recorded."
        ),
        "summary": {
            "results": len(results),
            "kept": sum(c["tests"] + c["errored"] for c in tests.values()),
            "errored": sum(c["errored"] for c in tests.values()),
            "skipped_other_plugins": dict(sorted(other.items())),
            "matched_by_query_id": matched["by_query_id"],
            "matched_by_trace_id": matched["by_trace_id"],
            "unmatched": matched["unmatched"],
        },
        "plugins": plugins,
    }
    return records, report


def render(report: Mapping[str, Any], written: int, out: str) -> str:
    """What the import did, for the terminal."""
    s = report["summary"]
    lines = [
        f"Kept {s['kept']} of {s['results']} red-team results; wrote {written} per-question "
        f"safety records to {out} (the rest are counted in the report only)."
    ]
    for plugin, p in report["plugins"].items():
        rate = "n/a" if p["failure_rate"] is None else f"{p['failure_rate']:.0%}"
        lines.append(
            f"  {plugin}: {p['failed']} of {p['tests']} failed ({rate}, 95% interval "
            f"{p['ci_low']:.0%} to {p['ci_high']:.0%})"
            + (f", {p['errored']} errored" if p["errored"] else "")
        )
    if s["skipped_other_plugins"]:
        skipped = ", ".join(f"{k} ({n})" for k, n in s["skipped_other_plugins"].items())
        lines.append(f"  skipped other plugins: {skipped}")
    lines.append("  Model-graded by promptfoo: exploratory evidence, not a release approval.")
    return "\n".join(lines)
