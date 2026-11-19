"""The sample evidence page: built from demo reports, self-contained, labelled synthetic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_policy_exceptions import failing_pair, policy
from vectraudit.evaluation.compare import compare_runs

ROOT = Path(__file__).parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "sample_evidence_report", ROOT / "examples" / "sample_evidence_report.py"
    )
    assert spec is not None
    assert spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def demo_folder(tmp_path):
    base, cand = failing_pair()
    rules = policy()
    for name, result in (
        ("candidate", compare_runs(base, cand, rules)),
        ("fixed", compare_runs(base, base, rules)),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.json").write_text(json.dumps(result))
    return tmp_path


def test_page_tells_the_story_and_stays_self_contained(tmp_path):
    page = module().build(demo_folder(tmp_path))
    assert "Synthetic demonstration." in page
    assert '<span class="verdict fail">FAIL</span>' in page
    assert '<span class="verdict pass">PASS</span>' in page
    assert "lost on 1 of 1 executions" in page
    assert "<script" not in page.lower()
    assert "http://" not in page
    assert "https://" not in page
    assert page.count("<iframe") == 2
    assert str(tmp_path) not in page


def test_cli_writes_the_page_and_explains_a_missing_demo(tmp_path, capsys):
    sample = module()
    out = tmp_path / "page.html"
    assert sample.main([str(demo_folder(tmp_path)), "--out", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")
    with pytest.raises(SystemExit, match=r"run examples/release_demo\.py first"):
        sample.main([str(tmp_path / "missing"), "--out", str(out)])
