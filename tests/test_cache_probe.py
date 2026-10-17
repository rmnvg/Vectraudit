"""Cache isolation: one context's question must not warm another context's answer."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines import (
    ContextItem,
    Hit,
    ModelContext,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    execute_pipeline,
)
from vectraudit.pipelines.cache_probe import cache_probe, probe_pairs
from vectraudit.pipelines.errors import PipelineCallError

ROWS = [
    {
        "query_id": "payroll-u2",
        "family_id": "payroll",
        "text": "payroll schedule",
        "relevant": {"payroll-2026": 1},
        "context": {"user": "u-2"},
    },
    {
        "query_id": "payroll-u17",
        "family_id": "payroll",
        "text": "payroll schedule",
        "relevant": {"handbook": 1},
        "forbidden": ["payroll-2026"],
        "context": {"user": "u-17"},
    },
    {"query_id": "other", "text": "holidays", "relevant": {"handbook": 1}},
]
ACL = {"payroll-2026": {"u-2"}, "handbook": {"u-2", "u-17"}}


def dataset(rows=ROWS):
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


class Assistant:
    """Answers from an access-controlled search, behind a cache keyed by ``key``."""

    def __init__(self, key=("text", "user"), acl=ACL):
        self.key, self.acl, self.cache = key, acl, {}

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        parts = {"text": request.text, "user": request.context.get("user")}
        cache_key = tuple(parts[k] for k in self.key)
        if cache_key not in self.cache:
            user = request.context.get("user")
            docs = [d for d in ("payroll-2026", "handbook") if user in self.acl[d]]
            self.cache[cache_key] = docs
        docs = self.cache[cache_key]
        return PipelineResult(
            hits=[Hit(id=d) for d in docs],
            context=ModelContext(items=[ContextItem(id=d) for d in docs]),
        )


def test_pairs_are_the_same_question_under_another_context():
    pairs = [(t.query_id, p.query_id) for t, p in probe_pairs(dataset())]
    assert pairs == [("payroll-u17", "payroll-u2")]  # only targets with forbidden documents


def test_a_cache_keyed_on_text_alone_leaks_and_the_corrected_key_passes():
    """The strategy's 2.2 acceptance: a planted cache defect caught, then corrected."""
    leaky = cache_probe(dataset(), Assistant(key=("text",)), k=2)
    assert leaky["leaks"] == [
        {
            "target": "payroll-u17",
            "primer": "payroll-u2",
            "doc_id": "payroll-2026",
            "reached": ["results", "model context"],
            "observed": "same_as_primer",
        }
    ]
    fixed = cache_probe(dataset(), Assistant(), k=2)
    assert (fixed["failing"], fixed["complete"], fixed["calls"]) == (0, True, 2)


def test_asking_the_target_first_would_hide_the_defect():
    """Why the primer goes first: the target's own clean answer fills a text-only cache."""
    cache = Assistant(key=("text",))
    rows = {q.query_id: q for q in dataset().queries}
    request = PipelineRequest(query_id="t", text="payroll schedule", k=2, context={"user": "u-17"})
    cache(request)
    cache(request.model_copy(update={"context": {"user": "u-2"}}))
    assert [h.id for h in cache(request).hits] == ["handbook"]
    assert rows["payroll-u17"].forbidden == ("payroll-2026",)


def test_a_leak_unlike_the_primers_answer_points_at_the_access_check():
    class Broken(Assistant):
        def __call__(self, request):
            result = super().__call__(request)
            if request.context["user"] == "u-17":  # access check drops the user's own rule
                result.hits = [Hit(id="payroll-2026")]
            return result

    report = cache_probe(dataset(), Broken(), k=2)
    assert [x["observed"] for x in report["leaks"]] == ["differs_from_primer"]


def test_a_normal_run_in_benchmark_order_already_leaks_through_a_text_cache():
    """Why a normal run is no clean reference: it primes the cache in benchmark order."""
    run = execute_pipeline(dataset(), Assistant(key=("text",)), PipelineOptions(k=2, warmup=0))
    assert run["queries"][1]["forbidden_ids"] == ["payroll-2026"]


def test_failed_calls_make_the_probe_incomplete():
    def down(request: PipelineRequest) -> PipelineResult:
        raise PipelineCallError("timeout", "no answer")

    report = cache_probe(dataset(), down, k=2)
    assert (report["complete"], report["errors"][0]["error"]["code"]) == (False, "timeout")


def test_questions_without_a_family_pair_by_text_and_large_probes_are_refused():
    rows = [{k: v for k, v in r.items() if k != "family_id"} for r in ROWS]
    assert len(probe_pairs(dataset(rows))) == 1
    many = [
        {"query_id": f"q{i}", "text": "same", "context": {"user": f"u{i}"}, "forbidden": ["x"]}
        for i in range(30)
    ]
    with pytest.raises(ValueError, match="exceed the limit"):
        cache_probe(dataset(many), Assistant(), max_pairs=100)


APP = """
from vectraudit.pipelines import Hit, PipelineResult

CACHE = {}

def pipeline(request):
    if request.text not in CACHE:  # the defect: the cache key ignores the user
        allowed = ["payroll-2026", "handbook"] if request.context["user"] == "u-2" else ["handbook"]
        CACHE[request.text] = allowed
    return PipelineResult(hits=[Hit(id=d) for d in CACHE[request.text]])
"""


def test_cli_probes_the_manifest_pipeline_and_can_fail_ci(tmp_path):
    (tmp_path / "cache_probe_app.py").write_text(APP)
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "k": 2,
                "candidates": [
                    {
                        "name": "current",
                        "adapter": {"kind": "python", "target": "cache_probe_app:pipeline"},
                        "pipeline_version": "1",
                    }
                ],
            }
        )
    )
    out = tmp_path / "probe.json"
    result = CliRunner().invoke(
        app, ["workbench", "cache-probe", str(manifest), "--json", str(out), "--fail-on-leak"]
    )
    assert result.exit_code == 3, result.output
    assert "Plan: 1 target/primer pair(s), 2 calls" in result.output
    assert "payroll-u17 after payroll-u2: payroll-2026 reached results" in result.output
    assert "identical to the primer's answer" in result.output
    assert json.loads(out.read_text())["leaks"][0]["doc_id"] == "payroll-2026"
