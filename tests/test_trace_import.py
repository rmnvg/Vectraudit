"""OpenInference trace import: runs scored like live runs, with gaps kept as gaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import check_run
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines.traces import (
    Settings,
    TraceImportError,
    _epoch,
    _nanos,
    import_traces,
    read_spans,
)

SECRET = "PLANTED-PRIVATE-PASSAGE"
QUESTIONS = {
    "q1": ("How do I reset my password?", {"d1": 1}),
    "q2": ("Where is the invoice?", {"d2": 1}),
    "q3": ("Cancel my plan", {"d3": 1}),
}


def dataset(questions: dict[str, tuple[str, dict[str, int]]] = QUESTIONS):
    rows = [{"query_id": q, "text": t, "relevant": r} for q, (t, r) in questions.items()]
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def docs(prefix: str, ids: list[str | None], *, scores: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, doc in enumerate(ids):
        if doc is not None:
            out[f"{prefix}.{i}.document.id"] = doc
        out[f"{prefix}.{i}.document.content"] = f"{SECRET} {doc}"
        if scores:
            out[f"{prefix}.{i}.document.score"] = 1.0 - i / 10
    return out


def trace(
    trace_id: str,
    query: str,
    *,
    retrieved: list[str | None] | None,
    reranked: list[str] | None = None,
    rerank_input: list[str] | None = None,
    start: int = 0,
    extra: list[dict[str, Any]] | None = None,
    root_attrs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Spans of one RAG request: chain → retriever (→ reranker) → LLM, times in ms."""
    t = start * 1_000_000
    spans = [
        span(
            trace_id,
            "root",
            None,
            "rag",
            "CHAIN",
            t,
            t + 900,
            {"input.value": query, **(root_attrs or {})},
        ),
    ]
    retriever = {"input.value": query}
    if retrieved is not None:
        retriever.update(docs("retrieval.documents", retrieved))
    spans.append(span(trace_id, "ret", "root", "retrieve", "RETRIEVER", t + 10, t + 50, retriever))
    if reranked is not None:
        attrs = {"reranker.query": query}
        attrs.update(docs("reranker.input_documents", rerank_input or retrieved or []))
        attrs.update(docs("reranker.output_documents", reranked))
        spans.append(span(trace_id, "rr", "root", "rerank", "RERANKER", t + 60, t + 80, attrs))
    spans.append(
        span(
            trace_id,
            "llm",
            "root",
            "generate",
            "LLM",
            t + 90,
            t + 800,
            {"llm.input_messages.0.message.content": f"{SECRET} context"},
        )
    )
    return spans + (extra or [])


def span(trace_id, span_id, parent, name, kind, start_ms, end_ms, attrs):
    return {
        "trace_id": trace_id,
        "span_id": f"{trace_id}-{span_id}",
        "parent": None if parent is None else f"{trace_id}-{parent}",
        "name": name,
        "start": start_ms * 1_000_000 if start_ms < 10**12 else start_ms,
        "end": end_ms * 1_000_000 if end_ms < 10**12 else end_ms,
        "attributes": {"openinference.span.kind": kind, **attrs},
    }


def otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def write_otlp(path: Path, spans: list[dict[str, Any]]) -> Path:
    """One ExportTraceServiceRequest per line, as the OpenTelemetry file exporter writes."""
    lines = []
    for s in spans:
        raw = {
            "traceId": s["trace_id"],
            "spanId": s["span_id"],
            "name": s["name"],
            "startTimeUnixNano": str(s["start"]),
            "endTimeUnixNano": str(s["end"]),
            "attributes": [{"key": k, "value": otlp_value(v)} for k, v in s["attributes"].items()],
        }
        if s["parent"]:
            raw["parentSpanId"] = s["parent"]
        lines.append(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [raw]}]}]}))
    path.write_text("\n".join(lines))
    return path


def write_records(path: Path, spans: list[dict[str, Any]]) -> Path:
    """Phoenix-style span records: nested context, ISO times, kind as a column."""

    def iso(ns: int) -> str:
        seconds, frac = divmod(ns, 1_000_000_000)
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{stamp}.{frac:09d}Z"

    records = []
    for s in spans:
        attrs = dict(s["attributes"])
        kind = attrs.pop("openinference.span.kind")
        records.append(
            {
                "name": s["name"],
                "context": {"trace_id": s["trace_id"], "span_id": s["span_id"]},
                "parent_id": s["parent"],
                "span_kind": kind,
                "start_time": iso(s["start"] + 1_700_000_000 * 10**9),
                "end_time": iso(s["end"] + 1_700_000_000 * 10**9),
                "attributes": attrs,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def good_traces() -> list[dict[str, Any]]:
    return (
        trace(
            "t1", "How do I reset my password?", retrieved=["d9", "d1", "d4"], reranked=["d1", "d9"]
        )
        + trace(
            "t2",
            "where is  the INVOICE?",
            retrieved=["d2", "d5", "d6"],
            reranked=["d5", "d6"],
            start=1000,
        )
        + trace("t3", "Cancel my plan", retrieved=["d3", "d7"], reranked=["d3", "d7"], start=2000)
    )


def test_otlp_retriever_and_reranker_become_a_scored_run(tmp_path):
    run, report = import_traces(dataset(), write_otlp(tmp_path / "t.jsonl", good_traces()))
    check_run(run)
    assert run["status"] == "complete"
    assert run["search"]["k"] == 2  # the longest delivered ranking
    by_id = {q["query_id"]: q for q in run["queries"]}
    assert by_id["q1"]["ids"] == ["d1", "d9"]
    assert by_id["q1"]["metrics"]["mrr"] == 1.0
    assert by_id["q2"]["metrics"]["recall"] == 0.0  # matched despite case and spacing
    stages = by_id["q2"]["evidence"]["stages"]
    assert [(s["name"], s["kind"], s["parents"]) for s in stages] == [
        ("retrieve", "custom", []),
        ("rerank", "rerank", ["retrieve"]),
    ]
    assert all(s["source"] == "reported" for s in stages)
    # The reranker's recorded input is the retriever's output: the drop is attributable.
    [loss] = by_id["q2"]["evidence"]["failures"]
    assert (loss["doc_id"], loss["stage"], loss["parents"]) == ("d2", "rerank", ["retrieve"])
    assert by_id["q1"]["latency_ms"] == pytest.approx(70.0)  # 10 ms → 80 ms, from spans
    assert run["timing"]["source"] == "trace"
    assert run["backend"]["config"]["adapter"] == "trace_import"
    caps = report["capabilities"]
    assert caps["stable_document_ids"]["status"] == "observed"
    assert caps["document_flow"]["status"] == "observed"
    assert caps["model_context"]["status"] == "recorded_not_imported"
    assert caps["replay"]["status"] == "not_available"
    assert report["questions"] == {
        "benchmark": 3,
        "matched": 3,
        "measured": 3,
        "by": "query text",
        "several_traces": 0,
    }
    assert report["gaps"] == []


def test_span_records_with_nested_attributes_and_metadata_ids(tmp_path):
    spans = []
    for n, (qid, chunks) in enumerate([("q1", ["d1", "d1", "d8"]), ("q2", ["d2"])]):
        root = {"metadata": json.dumps({"query_id": qid, "tenant": "x"})}
        spans += trace(
            f"t{n}",
            "unmatched text",
            retrieved=[None] * len(chunks),
            start=n * 1000,
            root_attrs=root,
        )
        retriever = spans[-2]["attributes"]
        retriever["retrieval"] = {
            "documents": [
                {"document": {"content": SECRET, "metadata": json.dumps({"doc_id": c})}}
                for c in chunks
            ]
        }
        for key in [k for k in retriever if k.startswith("retrieval.documents.")]:
            del retriever[key]
    path = write_records(tmp_path / "spans.jsonl", spans)
    settings = Settings(query_attribute="metadata.query_id", id_field="doc_id")
    run, report = import_traces(dataset(), path, settings, k=5)
    by_id = {q["query_id"]: q for q in run["queries"]}
    assert by_id["q1"]["ids"] == ["d1", "d8"]  # two chunks of d1 keep the best rank
    assert by_id["q2"]["ids"] == ["d2"]
    assert by_id["q3"]["error"]["code"] == "trace_missing"
    assert run["status"] == "partial"
    assert "tenant" in report["capabilities"]["filters_and_tenant_context"]["detail"]
    assert any("repeated document IDs" in gap for gap in report["gaps"])
    assert by_id["q1"]["latency_ms"] == pytest.approx(40.0)  # nanosecond ISO timestamps


def test_gaps_fail_with_their_cause_and_never_pass(tmp_path):
    spans = (
        trace("a", "How do I reset my password?", retrieved=None)  # documents not recorded
        + trace("b", "Where is the invoice?", retrieved=["d2", None])  # an ID is missing
        + trace(
            "c",
            "Cancel my plan",
            retrieved=["d3"],
            start=10,
            extra=[
                span(
                    "c",
                    "ret2",
                    "c-root",
                    "bm25",
                    "RETRIEVER",
                    20,
                    30,
                    docs("retrieval.documents", ["d7"]),
                )
            ],
        )
        + trace("z", "a question nobody benchmarked", retrieved=["d1"])
    )
    run, report = import_traces(dataset(), write_otlp(tmp_path / "t.jsonl", spans), k=3)
    codes = {q["query_id"]: (q["error"] or {}).get("code") for q in run["queries"]}
    assert codes == {
        "q1": "trace_documents_unrecorded",
        "q2": "document_id_missing",
        "q3": "trace_final_ambiguous",  # two retrievers, no reranker: which was delivered?
    }
    check_run(run)
    assert run["status"] == "partial"
    assert (
        compare_runs(run, run, parse_policy({"metrics": {"ndcg": {"max_drop": 0.01}}}))["verdict"]
        == "error"
    )
    assert "1 traces matched no benchmark question" in report["gaps"][0]
    assert report["capabilities"]["stable_document_ids"]["status"] == "partial"

    named, _ = import_traces(dataset(), tmp_path / "t.jsonl", Settings(final_span="bm25"), k=3)
    q3 = next(q for q in named["queries"] if q["query_id"] == "q3")
    assert q3["ids"] == ["d7"]


def test_several_traces_for_one_question_need_a_decision(tmp_path):
    spans = trace("old", "Cancel my plan", retrieved=["d9"]) + trace(
        "new", "Cancel my plan", retrieved=["d3"], start=5000
    )
    path = write_otlp(tmp_path / "t.jsonl", spans)
    run, report = import_traces(dataset({"q3": QUESTIONS["q3"]}), path)
    assert run["queries"][0]["error"]["code"] == "trace_ambiguous"
    assert report["questions"]["several_traces"] == 1
    latest, _ = import_traces(dataset({"q3": QUESTIONS["q3"]}), path, Settings(duplicates="latest"))
    assert latest["queries"][0]["ids"] == ["d3"]


def test_reranker_input_cut_from_retrieval_is_its_own_stage(tmp_path):
    spans = trace(
        "t",
        "Where is the invoice?",
        retrieved=["d5", "d6", "d2"],
        reranked=["d6"],
        rerank_input=["d5", "d6"],
    )  # the application passed only the top two on
    run, report = import_traces(dataset({"q2": QUESTIONS["q2"]}), write_otlp(tmp_path / "t", spans))
    q = run["queries"][0]
    assert [(s["name"], s["parents"], s["latency_ms"]) for s in q["evidence"]["stages"]] == [
        ("retrieve", [], 40.0),
        ("rerank input", ["retrieve"], 0.0),
        ("rerank", ["rerank input"], 20.0),
    ]
    [loss] = q["evidence"]["failures"]  # cut before reranking, not by the reranker
    assert (loss["doc_id"], loss["stage"]) == ("d2", "rerank input")
    assert q["observations"][0]["coverage"]["stage_trace"] == "observed"
    assert report["capabilities"]["document_flow"]["status"] == "observed"


def test_reranker_input_of_unknown_origin_is_not_attributed(tmp_path):
    spans = trace(
        "t",
        "Where is the invoice?",
        retrieved=["d2", "d5"],
        reranked=["d5"],
        rerank_input=["d5", "d6"],
    )  # d6 came from somewhere unrecorded
    path = write_otlp(tmp_path / "t", spans)
    run, report = import_traces(dataset({"q2": QUESTIONS["q2"]}), path, k=2)
    # Retriever and reranker are both unconnected outputs: which was delivered is unknown.
    assert run["queries"][0]["error"]["code"] == "trace_final_ambiguous"
    assert report["capabilities"]["document_flow"]["status"] == "partial"
    run, _ = import_traces(dataset({"q2": QUESTIONS["q2"]}), path, Settings(final_span="rerank"))
    q = run["queries"][0]
    assert q["ids"] == ["d5"]
    assert q["evidence"]["stages"][1]["parents"] == []
    assert q["observations"][0]["coverage"]["stage_trace"] == "partial"
    assert q["evidence"]["failures"] == []  # d2 was retrieved; where it was lost is unknown


def test_private_text_stays_in_the_trace_file(tmp_path):
    run, report = import_traces(dataset(), write_otlp(tmp_path / "t.jsonl", good_traces()))
    for artifact in (run, report, project_run(run)):
        text = json.dumps(artifact)
        assert SECRET not in text
        assert "reset my password" not in text


def test_traced_latency_is_never_judged_against_measured_latency(tmp_path):
    run, _ = import_traces(dataset(), write_otlp(tmp_path / "t.jsonl", good_traces()))
    policy = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.01}},
            "min_queries": 1,
            "latency": {"p95_max_ms": 0.5, "p50_max_increase_pct": 1},
        }
    )
    result = compare_runs(run, run, policy)
    assert result["verdict"] == "pass"
    assert result["latency"]["compared"] is False
    assert any("application traces" in w for w in result["warnings"])
    assert project_run(run)["timing"]["source"] == "trace"
    measured = fake_run([1.0] * 3)
    assert compare_runs(measured, measured, policy)["verdict"] != "pass"  # limit still applies


def test_unusable_files_are_refused(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(TraceImportError, match="no spans"):
        read_spans(empty)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"resourceSpans": []}\nnot json\n')
    with pytest.raises(TraceImportError, match="not JSON"):
        read_spans(bad)
    with pytest.raises(TraceImportError, match="Unknown stage kind"):
        Settings(stage_kinds={"bm25": "magic"})


def test_timestamps():
    assert _nanos("2026-10-11T00:00:00.123456789Z") % 10**9 == 123456789
    assert _nanos("1700000000000000001") == 1700000000000000001
    assert _nanos("yesterday") is None
    second = 1_760_140_800
    for value in (second, second * 1000, second * 10**6, second * 10**9, float(second) + 0.5):
        assert _epoch(value) // 10**9 == second


def test_dataframe_export_with_flattened_columns_and_epoch_milliseconds(tmp_path):
    rows = []
    for span_id, parent, kind, start, end, attrs in [
        ("r", None, "CHAIN", 0, 900, {"attributes.input.value": "Cancel my plan"}),
        (
            "x",
            "r",
            "RETRIEVER",
            10,
            35,
            {
                "attributes.retrieval.documents": [
                    {"document.id": "d3", "document.score": 0.9, "document.content": SECRET},
                    {"document.id": "d7", "document.score": 0.4, "document.content": SECRET},
                ],
            },
        ),
    ]:
        rows.append(
            {
                "name": span_id,
                "span_kind": kind,
                "parent_id": parent,
                "context.trace_id": "tr",
                "context.span_id": span_id,
                "start_time": 1_760_140_800_000 + start,
                "end_time": 1_760_140_800_000 + end,
                **attrs,
            }
        )
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    run, _ = import_traces(dataset({"q3": QUESTIONS["q3"]}), path)
    assert run["queries"][0]["ids"] == ["d3", "d7"]
    assert run["queries"][0]["latency_ms"] == pytest.approx(25.0)


def test_cli_writes_run_and_report(tmp_path):
    traces = write_otlp(tmp_path / "t.jsonl", good_traces())
    data = tmp_path / "d.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q, "text": t, "relevant": r}) for q, (t, r) in QUESTIONS.items()
        )
    )
    out, rep = tmp_path / "run.json", tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "import-traces",
            str(data),
            str(traces),
            "--out",
            str(out),
            "--report",
            str(rep),
            "--stage-kind",
            "retrieve=vector",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3 of 3 questions matched" in result.output
    assert "replay: not available" in result.output
    run = json.loads(out.read_text())
    assert run["queries"][0]["evidence"]["stages"][0]["kind"] == "vector"
    assert json.loads(rep.read_text())["run_id"] == run["run_id"]
    bad = CliRunner().invoke(
        app, ["workbench", "import-traces", str(data), str(traces), "--stage-kind", "x"]
    )
    assert bad.exit_code == 2
