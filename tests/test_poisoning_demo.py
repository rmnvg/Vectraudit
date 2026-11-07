"""Phase 29's acceptance, checked on the demo's real output and files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vectraudit.evaluation.corpus_scan import DETECTORS, read_corpus, scan_corpus

ROOT = Path(__file__).parents[1]


def load_demo():
    spec = importlib.util.spec_from_file_location(
        "poisoning_demo", ROOT / "examples" / "poisoning_demo.py"
    )
    assert spec is not None
    assert spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


@pytest.fixture(scope="module")
def story(tmp_path_factory):
    out = tmp_path_factory.mktemp("poisoning-demo")
    return load_demo().run(out), out


def test_the_scan_finds_every_synthetic_injection_and_nothing_in_the_clean_corpus(story):
    result, _ = story
    scan = result["scan"]
    assert scan["clean_flagged"] == 0
    assert scan["injected_detectors"] == sorted(DETECTORS)  # every detector found its pattern
    assert scan["injected_flagged"] == len(DETECTORS)
    for name in ("workbench", "reference_search"):
        report = scan_corpus(read_corpus(ROOT / "examples" / name / "corpus.jsonl"))
        assert report["summary"]["flagged"] == 0


def test_the_scan_cannot_see_the_poison_suite_but_the_poison_run_can(story):
    result, _ = story
    assert result["scan"]["suite_documents"] == 6
    assert result["scan"]["suite_flagged"] == 0  # a hijack reads like an ordinary document
    assert result["scenarios"]["no_tenant_filter"]["hijack_retrieved"] >= 1


def test_the_hijack_is_retrieved_when_poisoned_and_never_in_the_clean_run(story):
    result, _ = story
    for name, scenario in result["scenarios"].items():
        assert scenario["clean_retrieved_any_poison"] is False, name
    assert result["scenarios"]["no_tenant_filter"]["hijack_retrieved"] == 2
    assert result["scenarios"]["no_tenant_filter"]["verdict"] == "fail"


def test_a_tenant_filter_blocks_the_hijack_and_a_tenant_mimicking_one_gets_past_it(story):
    result, _ = story
    blocked = result["scenarios"]["tenant_filter"]
    assert blocked["hijack_retrieved"] == 0
    assert blocked["verdict"] == "pass"
    assert set(blocked["stages"].values()) == {"fusion"}  # seen, then dropped by the filter
    mimicked = result["scenarios"]["tenant_mimicked"]
    assert mimicked["hijack_retrieved"] == 2
    assert mimicked["verdict"] == "fail"


def test_an_injection_displaces_required_evidence(story):
    result, _ = story
    injection = result["scenarios"]["injection"]
    assert injection["inject_retrieved"] == 2
    assert injection["displaced"] == 1
    assert injection["lost_relevant"] == {"q-refund": ["refund-faq"]}
    assert injection["verdict"] == "fail"


def test_a_poisoned_document_in_the_model_context_fails_the_release_check_with_a_safety_reason(
    story,
):
    result, _ = story
    check = result["check"]
    assert check["verdict"] == "fail"
    assert check["in_model_context"] is True
    reasons = " ".join(check["reasons"])
    assert "Forbidden documents reached the model context" in reasons
    assert "Forbidden documents delivered" in reasons


def test_no_poisoned_or_corpus_text_appears_in_any_run_file_or_report(story):
    _, out = story
    texts = []
    for suite in out.glob("suite-*"):
        texts += [
            json.loads(line)["text"]
            for line in (suite / "poison-corpus.jsonl").read_text().splitlines()
        ]
    texts += [
        json.loads(line)["text"]
        for line in (out / "corpus.jsonl").read_text().splitlines()
        if "doc_id" in line
    ]
    assert len(texts) > 12
    files = [*(out / "runs").glob("*.run.json"), *out.glob("poisoning-*.json")]
    assert len(files) == 8 + 4
    for path in files:
        written = path.read_text()
        assert not [t for t in texts if t in written], path.name
    for path in (out / "runs").glob("*poisoned.run.json"):
        assert "canary-" not in path.read_text()  # not even the sentinel token


def test_the_demo_is_ascii_and_prints_its_story(story, capsys, tmp_path):
    source = (ROOT / "examples" / "poisoning_demo.py").read_text()
    assert source.isascii()
    demo = load_demo()
    import sys

    argv = sys.argv
    sys.argv = ["poisoning_demo.py", "--out", str(tmp_path / "again")]
    try:
        demo.main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "1. Corpus scan" in text
    assert "clean corpus: 0 documents flagged" in text
    assert "tenant_filter: hijack retrieved for 0 of 2 questions" in text
    assert "3. Release check with the poison in the model context: fail" in text
