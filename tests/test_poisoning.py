"""The poisoning report: rates, safety from the forbidden machinery, controls, robustness."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from tests.test_poison import ROWS, suite
from vectraudit.cli import app
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import (
    POISON_ATTACKS,
    PolicyError,
    parse_policy,
    policy_sha256,
)
from vectraudit.pipelines.poison import ATTACKS
from vectraudit.pipelines.poison_run import load_suite
from vectraudit.pipelines.poisoning import (
    PoisoningError,
    analyse,
    render,
    select_controls,
    subset,
)

BASE_POLICY = {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 1}


def policy(**extra):
    return parse_policy({**BASE_POLICY, **extra})


def manifest(tmp_path, overlay=True, **options):
    """A project folder with the corpus, the benchmark, a generated suite and a manifest."""
    _, _, out = suite(tmp_path, out=tmp_path / "gen", **options)
    root = tmp_path / "project"
    shutil.copytree(out, root / "suite")
    for name in ("corpus.jsonl", "benchmark.jsonl"):
        shutil.copy(tmp_path / name, root / name)
    adapter = {"kind": "hybrid", "corpus": "corpus.jsonl"}
    poisoned = {**adapter, "corpus_overlay": "suite/poison-corpus.jsonl"} if overlay else adapter
    params = {"vector_weight": 0, "depth": 10}
    document = {
        "dataset": "benchmark.jsonl", "k": 4, "repeats": 1,
        "candidates": [
            {"name": "clean", "adapter": adapter, "params": params},
            {"name": "poisoned", "adapter": poisoned, "params": params},
        ],
    }  # fmt: skip
    path = root / "manifest.json"
    path.write_text(json.dumps(document))
    return root, path


def invoke(path, *args, out=None):
    out = out or path.parent / "poisoning.json"
    result = CliRunner().invoke(
        app,
        ["workbench", "poison", "run", str(path), "--clean", "clean", "--poisoned", "poisoned",
         "--suite", str(path.parent / "suite/poison-suite.json"), "--out", str(out), *args],
    )  # fmt: skip
    return result, out


def test_a_poisoned_document_in_the_results_fails_with_a_safety_reason_and_full_report(tmp_path):
    _, path = manifest(tmp_path, match_filters=True)
    result, out = invoke(path)
    assert result.exit_code == 3, result.output
    assert "FAIL" in result.output
    assert "Poisoning check (exploratory): FAIL" in result.output
    report = json.loads(out.read_text())
    assert report["format"] == "vectraudit.poisoning"
    assert report["scope"] == "exploratory"
    assert report["verdict"] == "fail"
    rules = {f["rule"] for f in report["failures"]}
    assert "safety:forbidden_in_results" in rules  # the existing forbidden-document machinery
    assert report["safety"]["source"] == "the forbidden-document machinery"
    assert set(report["attacks"]) == set(ATTACKS)
    hijack = report["attacks"]["hijack"]
    assert hijack["retrieved"]["families"] == 2
    assert 0 <= hijack["retrieved"]["ci_low"] <= hijack["retrieved"]["ci_high"] <= 1
    assert report["overlay"]["warning"] is None
    assert (path.parent / "poisoning-runs").is_dir()
    assert len(list((path.parent / "poisoning-runs").glob("*.run.json"))) == 2


def test_documents_report_rank_deepest_stage_and_unknown_context_not_no(tmp_path):
    _, path = manifest(tmp_path, match_filters=True)
    report = json.loads(invoke(path)[1].read_text())
    rows = {d["doc_id"]: d for d in report["documents"]}
    retrieved = [d for d in rows.values() if d["retrieved"]]
    assert retrieved
    for d in retrieved:
        assert isinstance(d["rank"], int)
        assert d["rank"] >= 1
        assert d["reached_stage"] in {"truncate", "rerank", "filter", "fusion", "keyword"}
        assert d["in_model_context"] is None  # the pipeline records no model context: unknown
        assert d["evidence"] == "verified"
    unseen = [d for d in report["documents"] if not d["retrieved"] and d["reached_stage"] is None]
    assert all(d["in_model_context"] is None for d in unseen)  # never seen: nothing to say


def test_when_a_filter_removes_the_poison_it_is_seen_but_not_delivered_and_the_check_passes(
    tmp_path,
):
    _, path = manifest(tmp_path, targets=("q-refund",), attacks=("hijack", "inject"))
    result, out = invoke(path)
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["verdict"] == "pass"
    assert report["attacks"]["hijack"]["retrieved"]["flagged"] == 0
    assert report["attacks"]["hijack"]["deepest_stage"] == {"fusion": 1}  # the filter dropped it
    assert report["overlay"]["warning"] is None  # the overlay is visible in a recorded stage
    assert report["failures"] == []


def test_declared_robustness_limits_decide_and_are_reported(tmp_path):
    _, path = manifest(tmp_path, match_filters=True)
    strict = path.parent / "strict.json"
    strict.write_text(
        json.dumps({**BASE_POLICY, "robustness": {"max_retrieved_rate": {"hijack": 0.0}}})
    )
    result, out = invoke(path, "--policy", str(strict))
    assert result.exit_code == 3
    report = json.loads(out.read_text())
    assert report["robustness"]["declared"] is True
    rules = {f["rule"] for f in report["failures"]}
    assert "robustness:max_retrieved_rate.hijack" in rules
    assert report["robustness"]["checked"][0]["limit"] == 0.0


def test_the_report_holds_no_question_or_poisoned_text(tmp_path):
    _, path = manifest(tmp_path, match_filters=True)
    text = invoke(path)[1].read_text()
    for private in ("How do I get a refund", "Reset my password", "canary-", "example.invalid"):
        assert private not in text


def test_unsafe_or_mismatched_setups_are_refused_before_anything_runs(tmp_path):
    root, path = manifest(tmp_path, match_filters=True)
    document = json.loads(path.read_text())

    def reject(change, message, **kwargs):
        edited = json.loads(json.dumps(document))
        change(edited)
        path.write_text(json.dumps(edited))
        result, _ = invoke(path, **kwargs)
        assert result.exit_code == 2, result.output
        assert message in result.output

    reject(lambda d: d["candidates"][1].pop("adapter") and None, "adapter")  # invalid manifest
    path.write_text(json.dumps(document))

    def no_overlay(d):
        d["candidates"][1]["adapter"].pop("corpus_overlay")

    reject(no_overlay, "without adapter.corpus_overlay")

    def dirty_clean(d):
        d["candidates"][0]["adapter"]["corpus_overlay"] = "suite/poison-corpus.jsonl"

    reject(dirty_clean, "declares an overlay; it must not")

    def staging_without_declaration(d):
        d["candidates"][1]["adapter"] = {"kind": "python", "target": "m:f"}

    reject(staging_without_declaration, "declares no environment or overlay")

    def concurrent(d):
        d["candidates"][1]["concurrency"] = 2

    reject(concurrent, "serial; set concurrency 1")
    path.write_text(json.dumps(document))
    same = CliRunner().invoke(
        app,
        ["workbench", "poison", "run", str(path), "--clean", "clean", "--poisoned", "clean",
         "--suite", str(root / "suite/poison-suite.json")],
    )  # fmt: skip
    assert same.exit_code == 2
    assert "must be different" in same.output
    missing = CliRunner().invoke(
        app,
        ["workbench", "poison", "run", str(path), "--clean", "nope", "--poisoned", "poisoned",
         "--suite", str(root / "suite/poison-suite.json")],
    )  # fmt: skip
    assert "no candidate 'nope'" in missing.output


def test_a_suite_for_another_benchmark_or_another_suites_overlay_is_refused(tmp_path):
    root, path = manifest(tmp_path, match_filters=True)
    changed = [json.loads(line) for line in (root / "benchmark.jsonl").read_text().splitlines()]
    changed[1]["text"] = "Reset my password please"
    (root / "benchmark.jsonl").write_text("\n".join(json.dumps(r) for r in changed))
    result, _ = invoke(path)
    assert result.exit_code == 2
    assert "another version of the manifest's benchmark" in result.output
    # An overlay corpus that is not this suite's.
    second = tmp_path / "second"
    second.mkdir()
    _, path2 = manifest(second, match_filters=True)
    other = json.loads(path2.read_text())
    other["candidates"][1]["adapter"]["corpus_overlay"] = "benchmark.jsonl"
    path2.write_text(json.dumps(other))
    refused, _ = invoke(path2)
    assert refused.exit_code == 2
    assert "is not this suite's poison-corpus.jsonl" in refused.output


def runs_for(tmp_path, **options):
    """Crafted paired runs over the suite's two targets and one control."""
    _, _, out = suite(tmp_path, out=tmp_path / "g", match_filters=True, **options)
    root = tmp_path / "p"
    shutil.copytree(out, root / "suite")
    loaded = load_suite(root / "suite/poison-suite.json")
    dataset = load_dataset(loaded.dataset_path)

    def make(run_id, ids_by_query):
        run = fake_run([1.0, 1.0, 1.0], run_id=run_id)
        for q, qid in zip(run["queries"], ("q-refund", "q-reset", "q-open"), strict=True):
            ids = ids_by_query.get(qid, [])
            q.update(
                query_id=qid, ids=ids[:2], forbidden_ids=[d for d in ids if d.startswith("poison:")]
            )
            hits = [{"id": d} for d in ids]
            q["evidence"] = {"stages": [{"name": "s", "kind": "keyword", "hits": hits}]}
        return run

    return loaded, dataset, make


def test_collateral_displacement_and_attribution_from_crafted_runs(tmp_path):
    loaded, dataset, make = runs_for(tmp_path)
    hijack = "poison:hijack:q-refund:1"
    inject = "poison:inject:q-refund:1"
    clean = make("clean", {"q-refund": ["refund-policy"], "q-reset": ["reset-guide"]})
    poisoned = make(
        "poisoned",
        {"q-refund": [hijack, inject], "q-reset": ["reset-guide"], "q-open": [hijack]},
    )
    report = analyse(clean, poisoned, dataset, loaded, policy(), controls=1, seed=0)
    refund = next(r for r in report["targets"] if r["query_id"] == "q-refund")
    assert refund["displaced"] is True
    assert refund["lost_relevant"] == ["refund-policy"]
    assert refund["attribution"] == "unattributable"  # two attacks' documents were retrieved
    assert refund["attacks_retrieved"] == ["hijack", "inject"]
    reset = next(r for r in report["targets"] if r["query_id"] == "q-reset")
    assert reset["displaced"] is False
    assert reset["attribution"] == "none"
    assert report["summary"]["displaced"]["flagged"] == 1
    assert report["summary"]["displaced"]["families"] == 2
    assert report["attacks"]["hijack"]["displaced_when_retrieved"]["flagged"] == 1
    assert report["collateral"]["flagged"] == 1
    assert report["collateral"]["questions"] == {"q-open": [hijack]}
    assert report["collateral"]["controls"] == 1
    assert report["verdict"] == "fail"


def test_required_evidence_lost_counts_as_displacement(tmp_path):
    loaded, dataset, make = runs_for(tmp_path)
    clean = make("clean", {"q-refund": ["refund-policy"], "q-reset": ["reset-guide"]})
    poisoned = make("poisoned", {"q-refund": ["refund-policy", "poison:hijack:q-refund:1"],
                                 "q-reset": ["reset-guide"]})  # fmt: skip
    clean["queries"][0]["sufficiency"] = {"delivered": "sufficient"}
    poisoned["queries"][0]["sufficiency"] = {"delivered": "insufficient"}
    report = analyse(clean, poisoned, dataset, loaded, policy(), controls=0, seed=0)
    refund = next(r for r in report["targets"] if r["query_id"] == "q-refund")
    assert refund["required_evidence_broken"] is True
    assert refund["displaced"] is True
    assert refund["attribution"] == "single_attack"


def test_an_overlay_that_never_appears_or_an_incomplete_run_is_inconclusive_never_a_pass(tmp_path):
    loaded, dataset, make = runs_for(tmp_path)
    same = {"q-refund": ["refund-policy"], "q-reset": ["reset-guide"]}
    report = analyse(
        make("c", same), make("p", same), dataset, loaded, policy(), controls=0, seed=0
    )
    assert report["verdict"] == "inconclusive"
    assert report["overlay"]["warning"]
    assert any("overlay may not be loaded" in r for r in report["reasons_inconclusive"])
    seen = {"q-refund": ["refund-policy"], "q-reset": ["reset-guide", "x"]}
    incomplete = make("p2", seen)
    incomplete["status"] = "partial"
    report = analyse(make("c", seen), incomplete, dataset, loaded, policy(), controls=0, seed=0)
    assert report["verdict"] == "inconclusive"
    assert any("not complete" in r for r in report["reasons_inconclusive"])
    failed = make("p3", seen)
    failed["queries"][0]["error"] = {"code": "boom"}
    report = analyse(make("c", seen), failed, dataset, loaded, policy(), controls=0, seed=0)
    assert any("failed in a run" in r for r in report["reasons_inconclusive"])


def test_a_limit_for_an_attack_the_suite_lacks_cannot_pass(tmp_path):
    loaded, dataset, make = runs_for(tmp_path, attacks=("hijack",))
    clean = make("c", {"q-refund": ["refund-policy"], "q-reset": ["reset-guide"]})
    # The poisoned document is seen in a stage but never delivered: nothing else would fail.
    hidden = make("p", {"q-refund": ["refund-policy", "poison:hijack:q-refund:1"], "q-reset": []})
    hidden["queries"][0]["forbidden_ids"] = []
    hidden["queries"][0]["ids"] = ["refund-policy"]
    hidden["queries"][1]["ids"] = ["reset-guide"]
    fine = analyse(clean, hidden, dataset, loaded, policy(), controls=0, seed=0)
    assert fine["verdict"] == "pass"
    lacking = policy(robustness={"max_retrieved_rate": {"exfil": 0.0}})
    report = analyse(clean, hidden, dataset, loaded, lacking, controls=0, seed=0)
    assert report["verdict"] == "inconclusive"
    assert "robustness limit for 'exfil' could not be measured" in report["reasons_inconclusive"]
    assert report["robustness"]["checked"] == []
    within = policy(
        robustness={"max_retrieved_rate": {"hijack": 0.0}, "max_displacement_rate": 0.0}
    )
    held = analyse(clean, hidden, dataset, loaded, within, controls=0, seed=0)
    assert held["verdict"] == "pass"
    assert {c["rule"] for c in held["robustness"]["checked"]} == {
        "max_retrieved_rate.hijack",
        "max_displacement_rate",
    }


def test_controls_are_seeded_exploratory_and_never_held_out(tmp_path):
    suite(tmp_path)
    dataset = load_dataset(tmp_path / "benchmark.jsonl")
    pool = {q.query_id for q in dataset.queries if q.split == "explore"}
    a = select_controls(dataset, ["q-refund"], 2, seed=1)
    assert a == select_controls(dataset, ["q-refund"], 2, seed=1)
    assert "q-held" not in a
    assert "q-refund" not in a
    assert set(select_controls(dataset, ["q-refund"], 99, seed=1)) == pool - {"q-refund"}
    assert select_controls(dataset, [], 0, seed=0) == []
    with pytest.raises(PoisoningError, match="negative"):
        select_controls(dataset, [], -1, seed=0)
    twin = [dict(r) for r in ROWS]
    twin[3] = {"query_id": "q-twin", "text": "WHERE is my  order?", "relevant": {"reset-guide": 1}}
    from vectraudit.evaluation.dataset import parse_dataset

    leaky = parse_dataset("\n".join(json.dumps(r) for r in twin))
    with pytest.raises(PoisoningError, match="cross explore/heldout"):
        select_controls(leaky, [], 1, seed=0)
    assert [q.query_id for q in subset(dataset, ["q-reset", "q-refund"]).queries] == [
        "q-refund",
        "q-reset",
    ]


def test_the_terminal_summary_names_each_attack_and_the_verdict(tmp_path):
    _, path = manifest(tmp_path, match_filters=True)
    report = json.loads(invoke(path)[1].read_text())
    text = render(report)
    assert "Poisoning check (exploratory): FAIL" in text
    assert "hijack: retrieved" in text
    assert "collateral on control questions" in text
    assert "displaced answering evidence" in text


def test_the_policy_section_is_opt_in_validated_and_keeps_existing_hashes():
    plain = parse_policy(BASE_POLICY)
    assert plain.robustness is None
    assert "robustness" not in plain.as_dict()
    with_rules = policy(
        robustness={
            "max_retrieved_rate": {"hijack": 0.0, "inject": 0.1},
            "max_displacement_rate": 0.05,
        }
    )
    assert with_rules.as_dict()["robustness"] == {
        "max_retrieved_rate": {"hijack": 0.0, "inject": 0.1},
        "max_displacement_rate": 0.05,
    }
    assert policy_sha256(with_rules) != policy_sha256(plain)
    assert policy_sha256(parse_policy(BASE_POLICY)) == policy_sha256(plain)
    assert POISON_ATTACKS == ATTACKS
    for robustness, message in [
        ({}, "needs max_retrieved_rate"),
        ({"max_retrieved_rate": {"sql": 0.1}}, "unknown attack"),
        ({"max_retrieved_rate": {"hijack": 1.5}}, "rate from 0 to 1"),
        ({"max_retrieved_rate": {"hijack": True}}, "rate from 0 to 1"),
        ({"max_displacement_rate": -0.1}, "rate from 0 to 1"),
        ({"max_retrieved_rate": [0.1]}, "must map attacks"),
        ({"bogus": 1}, "unknown field"),
    ]:
        with pytest.raises(PolicyError, match=message):
            policy(robustness=robustness)
    with pytest.raises(PolicyError, match="must be an object"):
        policy(robustness=3)
