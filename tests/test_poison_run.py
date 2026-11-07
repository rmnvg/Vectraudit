"""Running a poison suite: overlays, staging declarations, verified suites, visibility."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.test_poison import suite
from vectraudit.evaluation.dataset import load_dataset
from vectraudit.pipelines.manifest import (
    Adapter,
    Candidate,
    candidate_config,
    open_adapter,
)
from vectraudit.pipelines.poison_run import (
    PoisonRunError,
    load_suite,
    overlay_visibility,
    poisoned_side_problem,
)
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline


def project(tmp_path, **options):
    """A manifest folder holding the corpus and a generated suite (targets both questions)."""
    _, _, out = suite(tmp_path, out=tmp_path / "gen", **options)
    root = tmp_path / "project"
    shutil.copytree(out, root / "suite")
    shutil.copy(tmp_path / "corpus.jsonl", root / "corpus.jsonl")
    return root, load_suite(root / "suite" / "poison-suite.json")


def run(root, suite_, overlay: bool, k=4):
    adapter = {"kind": "hybrid", "corpus": "corpus.jsonl"}
    if overlay:
        adapter["corpus_overlay"] = "suite/poison-corpus.jsonl"
    pipeline = open_adapter(Adapter.model_validate(adapter), root)
    options = PipelineOptions(k=k, warmup=0, params={"vector_weight": 0, "depth": 10})
    return execute_pipeline(load_dataset(suite_.dataset_path), pipeline, options)


def ids(run_, query):
    return next(q for q in run_["queries"] if q["query_id"] == query)["ids"]


def test_the_hybrid_baseline_loads_the_overlay_beside_its_corpus_in_memory(tmp_path):
    root, suite_ = project(tmp_path, match_filters=True)
    before = (root / "corpus.jsonl").read_bytes()
    clean, poisoned = run(root, suite_, False), run(root, suite_, True)
    assert not [d for d in ids(clean, "q-refund") if d.startswith("poison:")]
    hijacked = [d for d in ids(poisoned, "q-refund") if d.startswith("poison:")]
    assert hijacked
    assert (root / "corpus.jsonl").read_bytes() == before  # nothing is written back


def test_without_matched_filters_the_tenant_filter_removes_the_poisoned_documents(tmp_path):
    root, suite_ = project(tmp_path)  # metadata is empty, and q-refund filters on tenant
    poisoned = run(root, suite_, True)
    assert not [d for d in ids(poisoned, "q-refund") if d.startswith("poison:")]
    stages = next(q for q in poisoned["queries"] if q["query_id"] == "q-refund")["evidence"][
        "stages"
    ]
    reached = {s["kind"] for s in stages for h in s["hits"] if h["id"].startswith("poison:")}
    assert "keyword" in reached
    assert "filter" not in reached  # they were retrieved, then the filter dropped them


def test_a_poisoned_document_in_the_results_is_a_permission_violation_by_construction(tmp_path):
    root, suite_ = project(tmp_path, match_filters=True)
    poisoned = run(root, suite_, True)
    refund = next(q for q in poisoned["queries"] if q["query_id"] == "q-refund")
    assert refund["forbidden_ids"]  # the existing forbidden-document machinery sees it
    assert all(d.startswith("poison:") for d in refund["forbidden_ids"])
    assert poisoned["summary"]["permissions"]["violations"] >= 1


def test_overlay_ids_must_be_new(tmp_path):
    root, _ = project(tmp_path)
    clash = [
        json.loads(line) for line in (root / "suite/poison-corpus.jsonl").read_text().splitlines()
    ]
    clash[0]["doc_id"] = "refund-policy"
    (root / "clash.jsonl").write_text("\n".join(json.dumps(r) for r in clash))
    adapter = Adapter(kind="hybrid", corpus="corpus.jsonl", corpus_overlay="clash.jsonl")
    with pytest.raises(ValueError, match=r"repeats 1 document IDs.*'refund-policy'"):
        open_adapter(adapter, root)


def test_the_visibility_check_warns_when_no_poisoned_document_appears_anywhere(tmp_path):
    root, suite_ = project(tmp_path, match_filters=True)
    seen = overlay_visibility(run(root, suite_, True), suite_)
    assert seen["warning"] is None
    assert seen["targets_where_a_poisoned_document_appeared"] >= 1
    assert seen["documents_seen"]
    unloaded = overlay_visibility(run(root, suite_, False), suite_)
    assert unloaded["targets_where_a_poisoned_document_appeared"] == 0
    assert "overlay may not be loaded" in unloaded["warning"]
    assert "a warning, not proof" in unloaded["warning"]
    assert unloaded["targets_with_stage_traces"] == unloaded["targets_in_run"] == 2


def test_with_no_stage_traces_only_delivered_results_are_read_and_it_says_so(tmp_path):
    root, suite_ = project(tmp_path, match_filters=True)
    recorded = run(root, suite_, False)
    for q in recorded["queries"]:
        q.pop("evidence", None)
    result = overlay_visibility(recorded, suite_)
    assert result["targets_with_stage_traces"] == 0
    assert "No stage traces were recorded" in result["warning"]
    failed = run(root, suite_, True)
    for q in failed["queries"]:
        q["error"] = {"code": "x"}
    assert overlay_visibility(failed, suite_) == {
        "targets_in_run": 0, "targets_with_stage_traces": 0,
        "targets_where_a_poisoned_document_appeared": 0, "documents_seen": [], "warning": None,
    }  # fmt: skip


def test_a_suite_is_verified_against_its_files_before_use(tmp_path):
    root, suite_ = project(tmp_path)
    assert suite_.targets == ["q-refund", "q-reset"]
    assert suite_.documents_for("q-reset")
    assert all(d.startswith("poison:") for d in suite_.documents)
    path = root / "suite" / "poison-suite.json"

    def edit(name, change):
        file = root / "suite" / name
        original = file.read_text()
        file.write_text(change(original))
        return lambda: file.write_text(original)

    for name, change, message in [
        ("poison-corpus.jsonl", lambda t: t.replace("refund", "reimburse", 1), "was changed after"),
        ("poison-dataset.jsonl", lambda t: t + "\n", "was changed after"),
    ]:
        restore = edit(name, change)
        with pytest.raises(PoisonRunError, match=message):
            load_suite(path)
        restore()
    (root / "suite" / "poison-corpus.jsonl").rename(root / "suite" / "gone.jsonl")
    with pytest.raises(PoisonRunError, match="is missing beside"):
        load_suite(path)
    (root / "suite" / "gone.jsonl").rename(root / "suite" / "poison-corpus.jsonl")
    restore = edit("poison-suite.json", lambda t: t.replace(suite_.suite_hash, "0" * 12))
    with pytest.raises(PoisonRunError, match="suite hash does not match"):
        load_suite(path)
    restore()
    restore = edit(
        "poison-suite.json",
        lambda t: t.replace('"format": "vectraudit.poison_suite"', '"format": "x"'),
    )
    with pytest.raises(PoisonRunError, match="not a poison suite"):
        load_suite(path)
    restore()
    with pytest.raises(PoisonRunError, match="Cannot read"):
        load_suite(root / "nope.json")
    path.write_text("nope")
    with pytest.raises(PoisonRunError, match="is not JSON"):
        load_suite(path)


def test_a_dataset_from_another_suite_is_refused(tmp_path):
    root, _ = project(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_root, other = project(other_dir, seed=5)
    shutil.copy(
        other_root / "suite" / "poison-dataset.jsonl", root / "suite" / "poison-dataset.jsonl"
    )
    descriptor = json.loads((root / "suite/poison-suite.json").read_text())
    descriptor["outputs"]["poison-dataset.jsonl"]["sha256"] = json.loads(
        (other_root / "suite/poison-suite.json").read_text()
    )["outputs"]["poison-dataset.jsonl"]["sha256"]
    (root / "suite/poison-suite.json").write_text(json.dumps(descriptor))
    with pytest.raises(PoisonRunError, match="was not generated by this suite"):
        load_suite(root / "suite/poison-suite.json")
    assert other.suite_hash != descriptor["suite_hash"]


def candidate(**fields):
    return Candidate.model_validate(
        {"name": "c", "adapter": {"kind": "python", "target": "m:f"}, **fields}
    )


def test_a_poisoned_side_needs_an_overlay_and_a_staging_declaration():
    hybrid = {"kind": "hybrid", "corpus": "corpus.jsonl"}
    plain = Candidate.model_validate({"name": "h", "adapter": hybrid})
    assert "without adapter.corpus_overlay" in poisoned_side_problem(plain)
    overlaid = Candidate.model_validate(
        {"name": "h", "adapter": {**hybrid, "corpus_overlay": "suite/poison-corpus.jsonl"}}
    )
    assert poisoned_side_problem(overlaid) is None
    assert "declares no environment or overlay" in poisoned_side_problem(candidate())
    assert "declares no overlay" in poisoned_side_problem(candidate(environment="staging"))
    both = candidate(environment="staging", overlay="suite/poison-suite.json")
    assert poisoned_side_problem(both) is None
    assert "never writes to your backend" in poisoned_side_problem(candidate())


def test_declarations_are_validated_when_the_manifest_is_read():
    with pytest.raises(ValueError, match='needs "environment": "staging"'):
        candidate(overlay="suite/poison-suite.json")
    with pytest.raises(ValueError, match="corpus_overlay is for the hybrid baseline"):
        Candidate.model_validate(
            {
                "name": "x",
                "adapter": {"kind": "python", "target": "m:f", "corpus_overlay": "o.jsonl"},
            }
        )
    with pytest.raises(ValueError, match="staging"):
        candidate(environment="production")


def test_overlay_hashes_join_the_candidate_identity_only_when_declared(tmp_path):
    root, _ = project(tmp_path, match_filters=True)
    base = {"name": "h", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
    plain = candidate_config(Candidate.model_validate(base), root)
    with_overlay = candidate_config(
        Candidate.model_validate(
            {**base, "adapter": {**base["adapter"], "corpus_overlay": "suite/poison-corpus.jsonl"}}
        ),
        root,
    )
    assert "corpus_overlay_sha256" not in plain
    assert with_overlay["corpus_overlay_sha256"]
    assert with_overlay["sha256"] != plain["sha256"]
    staged = candidate_config(
        candidate(environment="staging", overlay="suite/poison-suite.json"), root
    )
    assert staged["overlay_sha256"]
    assert "overlay_sha256" not in candidate_config(candidate(), root)
