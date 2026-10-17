"""The permission and freshness reliability demo runs end to end through product code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_demo():
    spec = importlib.util.spec_from_file_location(
        "reliability_story_demo", ROOT / "examples" / "reliability_story_demo.py"
    )
    assert spec is not None
    assert spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def test_three_planted_defects_are_found_and_their_corrections_verified(tmp_path):
    demo = load_demo()
    assert demo.main(["--out", str(tmp_path / "story")]) == 0
    page = (tmp_path / "story" / "reliability-story.html").read_text()
    assert "Synthetic demonstration" in page
    assert "<script" not in page
    facts = demo.rehearse(tmp_path / "again")
    assert facts["stale"] == {
        "status": "lingering",
        "chunk": "refund-policy#r8",
        "where": "context",
        "conflicts": 1,
        "after": "propagated",
        "after_failing": 0,
    }
    assert (facts["revocation"]["before"], facts["revocation"]["after"]) == ("fail", "pass")
    assert facts["revocation"]["reached"] == ["results", "model context"]
    assert facts["revocation"]["label_review"] == [
        {"query_id": "payroll-u-17", "removed_labels": ["payroll-2026"]}
    ]
    assert facts["cache"] == {
        "doc": "payroll-2026",
        "observed": "same_as_primer",
        "before": 1,
        "after": 0,
    }


def test_the_demo_folder_opens_in_the_checks_tab(tmp_path):
    load_demo().rehearse(tmp_path / "story")
    formats = {json.loads(p.read_text()).get("format") for p in (tmp_path / "story").glob("*.json")}
    assert {
        "vectraudit.knowledge_freshness",
        "vectraudit.access",
        "vectraudit.cache_probe",
    } <= formats
