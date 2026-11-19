"""Synthetic demonstration of the "permission and freshness reliability" entry offer.

Three short stories, each a planted defect found and its correction verified, all through
product code:

1. **A stale answer.** The refund policy changed, the old revision stayed in the index, and
   the assistant packed it into the model context next to the new one.
2. **A revocation that never arrived.** A user lost access to the payroll document; the
   assistant's own access index was never updated, so the document kept reaching them.
3. **A cache shared between users.** Answers cached by question text alone served one
   user's payroll document to another user asking the same question.

Run it, then open the page it writes, or serve the folder and open the Checks tab:

    uv run python examples/reliability_story_demo.py --out /tmp/vectraudit-reliability
    uv run vectraudit workbench serve /tmp/vectraudit-reliability

Everything here is synthetic: a tiny assistant, documents, users and change logs written
for the demo. It shows how Vectraudit reasons, not results on anyone's data.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vectraudit.evaluation.access import apply_access_changes, check_access, read_access_changes
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.freshness import check_freshness, read_changes
from vectraudit.evaluation.lineage import read_manifest
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines.cache_probe import cache_probe
from vectraudit.pipelines.models import (
    ContextItem,
    Hit,
    ModelContext,
    PipelineRequest,
    PipelineResult,
)
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

NOW = datetime.now(timezone.utc).replace(microsecond=0)
CHANGED = (NOW - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 0.05}}, "min_queries": 1})
OPTIONS = PipelineOptions(k=2, warmup=0, repeats=2)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- 1. a stale answer ---------------------------------------------------------------------

REFUND = parse_dataset(
    json.dumps(
        {
            "query_id": "refund-window",
            "text": "How long do I have to request a refund?",
            "relevant": {"refund-policy#r9": 1},
        }
    )
)


class RefundAssistant:
    """Retrieves every indexed refund-policy chunk, oldest first, and packs them all."""

    def __init__(self, indexed: list[str]) -> None:
        self.indexed = indexed

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        chunks = self.indexed[: request.k]
        return PipelineResult(
            hits=[Hit(id=c) for c in chunks],
            context=ModelContext(items=[ContextItem(id=c, truncated=False) for c in chunks]),
        )


def _lineage(indexed: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": "refund-policy",
            "revision": "r9",
            "chunks": [
                {"chunk_id": c, "start": 0, "end": 400, "indexed_revision": c.split("#")[1]}
                for c in indexed
            ],
        }
    ]


def stale_answer(root: Path) -> dict[str, Any]:
    changes = read_changes(
        _jsonl(
            root / "content-changes.jsonl",
            [
                {
                    "source_id": "refund-policy",
                    "change": "update",
                    "to_revision": "r9",
                    "changed_at": CHANGED,
                    "window_minutes": 60,
                }
            ],
        )
    )
    reports = {}
    for name, indexed in (
        ("before", ["refund-policy#r8", "refund-policy#r9"]),
        ("after", ["refund-policy#r9"]),
    ):
        run = execute_pipeline(REFUND, RefundAssistant(indexed), OPTIONS)
        sources = read_manifest(_jsonl(root / f"lineage-{name}.jsonl", _lineage(indexed)))
        report = check_freshness(changes, sources, as_of=NOW, run=run)
        _write(root / f"freshness-{name}.json", report)
        reports[name] = report
    return reports


# --- 2. a revocation that never arrived, and 3. a cache shared between users ------------

PAYROLL = [
    {
        "query_id": f"payroll-{user}",
        "family_id": "payroll",
        "text": "When is payroll processed this month?",
        "relevant": {"payroll-2026": 2, "handbook": 1},
        "context": {"user": user, "groups": ["finance", "staff"]},
    }
    for user in ("u-17", "u-2")
]
REVOKE = {
    "change": "revoke",
    "principal": {"user": "u-17"},
    "doc_ids": ["payroll-2026"],
    "changed_at": CHANGED,
    "window_minutes": 15,
}
SYNCED = {"payroll-2026": {"u-2"}, "handbook": {"u-17", "u-2"}}
UNSYNCED = {"payroll-2026": {"u-17", "u-2"}, "handbook": {"u-17", "u-2"}}


class PayrollAssistant:
    """Serves what its own access index allows, behind a cache keyed by ``key``."""

    def __init__(self, acl: dict[str, set[str]], key: tuple[str, ...] = ("text", "user")):
        self.acl, self.key = acl, key
        self.cache: dict[tuple[Any, ...], list[str]] = {}

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        user = request.context["user"]
        parts = {"text": request.text, "user": user}
        cache_key = tuple(parts[k] for k in self.key)
        if cache_key not in self.cache:
            self.cache[cache_key] = [d for d in ("payroll-2026", "handbook") if user in self.acl[d]]
        docs = self.cache[cache_key][: request.k]
        return PipelineResult(
            hits=[Hit(id=d) for d in docs],
            context=ModelContext(items=[ContextItem(id=d, truncated=False) for d in docs]),
        )


def revocation(root: Path) -> dict[str, Any]:
    original = parse_dataset("\n".join(json.dumps(r) for r in PAYROLL))
    log = read_access_changes(_jsonl(root / "access-changes.jsonl", [REVOKE]))
    text, applied = apply_access_changes(original, log, as_of=NOW)
    (root / "questions.access.jsonl").write_text(text, encoding="utf-8")
    expected = parse_dataset(text)
    synced = execute_pipeline(expected, PayrollAssistant(SYNCED), OPTIONS)
    unsynced = execute_pipeline(expected, PayrollAssistant(UNSYNCED), OPTIONS)
    before = compare_runs(synced, unsynced, POLICY)
    after = compare_runs(
        synced, execute_pipeline(expected, PayrollAssistant(SYNCED), OPTIONS), POLICY
    )
    report = {
        "format": "vectraudit.access",
        "format_version": 1,
        "applied": applied,
        "check": check_access(log, unsynced, original, as_of=NOW),
    }
    _write(root / "access.json", report)
    _write(root / "release-unsynced.json", before)
    _write(root / "release-synced.json", after)
    return {"report": report, "before": before, "after": after, "expected": expected}


def cache_leak(root: Path, expected: Any) -> dict[str, Any]:
    leaky = cache_probe(expected, PayrollAssistant(SYNCED, key=("text",)), k=2)
    isolated = cache_probe(expected, PayrollAssistant(SYNCED), k=2)
    _write(root / "cache-probe-before.json", leaky)
    _write(root / "cache-probe-after.json", isolated)
    return {"before": leaky, "after": isolated}


def rehearse(root: Path) -> dict[str, Any]:
    """Run the three stories; returns the facts the page and tests read."""
    root.mkdir(parents=True, exist_ok=False)
    fresh = stale_answer(root)
    access = revocation(root)
    cache = cache_leak(root, access["expected"])
    stale = fresh["before"]["served"]["stale"][0]
    violation = access["report"]["check"]["violations"][0]
    leak = cache["before"]["leaks"][0]
    failed = [r["message"] for r in access["before"]["reasons"] if r["verdict"] == "fail"]
    return {
        "stale": {
            "status": fresh["before"]["changes"][0]["status"],
            "chunk": stale["chunk_id"],
            "where": stale["where"],
            "conflicts": len(fresh["before"]["served"]["conflicts"]),
            "after": fresh["after"]["changes"][0]["status"],
            "after_failing": fresh["after"]["failing"],
        },
        "revocation": {
            "doc": violation["doc_id"],
            "reached": violation["reached"],
            "label_review": access["report"]["applied"]["labels_to_review"],
            "before": access["before"]["verdict"],
            "reason": failed[0] if failed else "",
            "after": access["after"]["verdict"],
        },
        "cache": {
            "doc": leak["doc_id"],
            "observed": leak["observed"],
            "before": cache["before"]["failing"],
            "after": cache["after"]["failing"],
        },
    }


def _report_style() -> str:
    spec = importlib.util.spec_from_file_location(
        "sample_evidence_report", Path(__file__).with_name("sample_evidence_report.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    style: str = module.STYLE
    return style


def page(facts: dict[str, Any]) -> str:
    """One self-contained page telling the three stories from the demo's own results."""
    e = html.escape
    s, r, c = facts["stale"], facts["revocation"], facts["cache"]
    stories = [
        (
            "The policy changed. The old answer kept reaching the model.",
            f"Three hours after the refund policy moved to revision r9, the change was "
            f'<span class="fail">{e(s["status"])}</span>: the old chunk '
            f"<code>{e(s['chunk'])}</code> was still indexed and reached the "
            f"<b>{e(s['where'])}</b>, alongside the new one ({s['conflicts']} conflicting "
            "revision pair delivered together). After re-indexing, the change is "
            f'<span class="pass">{e(s["after"])}</span> and nothing stale is served.',
        ),
        (
            "Access was revoked. The assistant never heard.",
            f"The identity system revoked <code>{e(r['doc'])}</code> for user u-17. Vectraudit "
            "took the expected access from that change log, not from the assistant, and removed "
            "the now-forbidden label for review. The assistant's own access index was never "
            f"synced, so the document reached the {e(' and the '.join(r['reached']))}. "
            "Release check: "
            f'<span class="fail">{e(r["before"])}</span>: {e(r["reason"])} After the index '
            f'sync: <span class="pass">{e(r["after"])}</span>.',
        ),
        (
            "Two users asked the same question. They got the same answer.",
            "A cache keyed by question text alone served the payroll answer for u-2 to u-17. "
            f"The cache probe asked as u-2 first, then as u-17: <code>{e(c['doc'])}</code> "
            f"leaked, and u-17's answer was identical to u-2's ({c['before']} leak). With the "
            f'user in the cache key: <span class="pass">{c["after"]} leaks</span>.',
        ),
    ]
    body = "".join(
        f'<div class="step"><b class="n">{i}.</b><b>{e(title)}</b><p>{text}</p></div>'
        for i, (title, text) in enumerate(stories, 1)
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Vectraudit: content and access that changed</title><style>{_report_style()}"
        "</style></head><body><main>"
        '<p class="banner"><b>Synthetic demonstration.</b> A tiny assistant, documents, users and '
        "change logs written for this demo. It shows how Vectraudit reasons, not results on "
        "anyone's data.</p>"
        "<h1>Your content and your access rules change. Does your assistant keep up?</h1>"
        '<p class="muted">Three planted defects, each found from your own change logs and '
        "exports, and each correction verified.</p>"
        f'<div class="steps">{body}</div>'
        '<h2>On your own assistant</h2><p class="muted">The same checks with your '
        "content system's change log, your identity system's access changes, your ingestion "
        "export and your release check. Expected access always comes from your systems of "
        "record, never from the assistant's own answers. A passing check is evidence about the "
        "cases tested, not proof that authorization is correct.</p>"
        "</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="New folder for the results.")
    args = parser.parse_args(argv)
    facts = rehearse(args.out)
    (args.out / "reliability-story.html").write_text(page(facts), encoding="utf-8")
    s, r, c = facts["stale"], facts["revocation"], facts["cache"]
    print(f"Stale answer: {s['status']} ({s['chunk']} in the {s['where']}); after: {s['after']}")
    print(
        f"Revocation: release {r['before']} ({r['doc']} reached the "
        f"{' and the '.join(r['reached'])}); "
        f"after: {r['after']}"
    )
    print(f"Cache: {c['before']} leak; after: {c['after']}")
    print(f"Wrote {args.out / 'reliability-story.html'}")
    ok = (
        s["status"] == "lingering"
        and s["after"] == "propagated"
        and (r["before"], r["after"]) == ("fail", "pass")
        and (c["before"], c["after"]) == (1, 0)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
