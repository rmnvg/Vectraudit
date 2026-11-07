"""The workbench's check-report fixtures match what the checks write today.

``frontend/src/lib/fixtures/*.json`` are real reports on synthetic test data; the Checks tab
is tested against them. If a report's shape changes, this fails: regenerate the fixture
from the same calls below and update ``frontend/src/lib/checks.ts`` if a field it reads
moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "lib" / "fixtures"


def lineage(tmp_path: Path) -> dict[str, Any]:
    from tests.test_lineage import dataset, source, write
    from vectraudit.evaluation.lineage import check_lineage, read_manifest

    return check_lineage(dataset(("policy", 450, 550)), read_manifest(write(tmp_path, source())))


def freshness(tmp_path: Path) -> dict[str, Any]:
    from tests.test_freshness import UPDATE, check, chunk, run_with

    sources = [
        {
            "source_id": "policy",
            "revision": "r9",
            "chunks": [chunk("policy#0@r8", "r8"), chunk("policy#0@r9", "r9", 100, 200)],
        }
    ]
    run = run_with(["policy#0@r8", "policy#0@r9"], context_ids=["policy#0@r8"])
    return check(tmp_path, [UPDATE], sources, run=run)


def access(tmp_path: Path) -> dict[str, Any]:
    from tests.test_access_changes import LATER, REVOKE, STALE, Assistant, changes, dataset
    from vectraudit.evaluation.access import apply_access_changes, check_access
    from vectraudit.pipelines import PipelineOptions, execute_pipeline

    log = changes(tmp_path, REVOKE)
    _, applied = apply_access_changes(dataset(), log, as_of=LATER)
    run = execute_pipeline(dataset(), Assistant(STALE), PipelineOptions(k=2, warmup=0))
    return {
        "format": "vectraudit.access",
        "format_version": 1,
        "applied": applied,
        "check": check_access(log, run, dataset(), as_of=LATER),
    }


def cache(tmp_path: Path) -> dict[str, Any]:
    from tests.test_cache_probe import Assistant, dataset
    from vectraudit.pipelines.cache_probe import cache_probe

    return cache_probe(dataset(), Assistant(key=("text",)), k=2)


def migration(tmp_path: Path) -> dict[str, Any]:
    from tests.test_migration_report import System, report, run

    return report(run(System("ignores_filter"), "qdrant"))


def intake(tmp_path: Path) -> dict[str, Any]:
    from tests.test_intake import first_batch, run

    return run(tmp_path, first_batch())


def fusion_replay(tmp_path: Path) -> dict[str, Any]:
    from tests.test_fusion_replay import _dataset, _run
    from vectraudit.pipelines.fusion_replay import Config, next_step, replay

    dataset, run = _run(_dataset(extra={0: {"split": "heldout"}}))
    configs = [Config("rrf"), Config("wsum", norm="none")]
    report = replay(run, dataset, configs)
    report["next_step"] = next_step(report, configs)
    return report


def corpus_scan(tmp_path: Path) -> dict[str, Any]:
    from tests.test_corpus_scan import CLEAN, POISONED
    from vectraudit.evaluation.corpus_scan import scan_corpus

    keep = ("instruction_override", "chat_template_token", "html_comment", "exfiltration_beacon")
    docs = [("billing-faq", CLEAN["plain"])]
    docs += [(f"page-{name}", POISONED[name]) for name in keep]
    return scan_corpus(docs)


def poisoning(tmp_path: Path) -> dict[str, Any]:
    from tests.test_poisoning import policy, runs_for
    from vectraudit.pipelines.poisoning import analyse

    loaded, dataset, make = runs_for(tmp_path)
    hijack, inject = "poison:hijack:q-refund:1", "poison:inject:q-refund:1"
    clean = make("clean-run", {"q-refund": ["refund-policy"], "q-reset": ["reset-guide"]})
    poisoned = make(
        "poisoned-run",
        {"q-refund": [hijack, inject], "q-reset": ["reset-guide"], "q-open": [hijack]},
    )
    limits = {"max_retrieved_rate": {"hijack": 0.0}, "max_displacement_rate": 0.05}
    rules = policy(robustness=limits)
    return analyse(clean, poisoned, dataset, loaded, rules, controls=1, seed=0)


def promptfoo_redteam(tmp_path: Path) -> dict[str, Any]:
    from tests.test_promptfoo_redteam import SPECS, run_import

    return run_import(tmp_path, SPECS)[1]


@pytest.mark.parametrize(
    "build",
    [
        lineage,
        freshness,
        access,
        cache,
        migration,
        intake,
        fusion_replay,
        corpus_scan,
        poisoning,
        promptfoo_redteam,
    ],
)
def test_fixture_has_the_shape_the_check_writes(tmp_path, build):
    current = build(tmp_path)
    fixture = json.loads((FIXTURES / f"{build.__name__}.json").read_text())
    assert fixture["format"] == current["format"]
    # The migration fixture omits the embedded release check to stay small.
    assert set(fixture) == set(current) - ({"check"} if build is migration else set())
