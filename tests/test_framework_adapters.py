"""LangChain and LlamaIndex retrievers as workbench pipelines, tested on the real classes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import Field

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.integrations.retrievers import langchain_pipeline, llamaindex_pipeline
from vectraudit.pipelines import PipelineOptions, PipelineRequest, execute_pipeline

lc_docs = pytest.importorskip("langchain_core.documents")
lc_retrievers = pytest.importorskip("langchain_core.retrievers")
lc_compressor = pytest.importorskip("langchain_core.documents.compressor")

Document = lc_docs.Document

CORPUS = {
    "refunds": "Refunds are issued within 14 days.",
    "returns": "Returns need a receipt and the original box.",
    "shipping": "Shipping takes three to five days.",
    "password": "Reset your password from Settings.",
}


def _rank(query: str) -> list[str]:
    """A deterministic keyword ranking: shared words, then id."""

    def words(text: str) -> set[str]:
        return set(text.lower().replace(".", "").split())

    return sorted(CORPUS, key=lambda d: (-len(words(query) & words(CORPUS[d])), d))


class KeywordRetriever(lc_retrievers.BaseRetriever):  # type: ignore[misc,name-defined]
    k: int = 3
    seen_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: Any, **kwargs: Any) -> list:
        self.seen_kwargs.append(kwargs)
        k = kwargs.get("k", self.k)
        ids = [d for d in _rank(query) if d != kwargs.get("filter", {}).get("exclude")]
        return [
            Document(page_content=CORPUS[d], id=f"chunk-{d}", metadata={"doc_id": d})
            for d in ids[:k]
        ]


class DropShipping(lc_compressor.BaseDocumentCompressor):  # type: ignore[misc,name-defined]
    """A reranker that demotes nothing but drops one document."""

    def compress_documents(self, documents: Sequence[Any], query: str, callbacks: Any = None):
        return [d for d in documents if d.metadata["doc_id"] != "shipping"]


class CompressionRetriever(lc_retrievers.BaseRetriever):  # type: ignore[misc,name-defined]
    """Mirrors langchain's ContextualCompressionRetriever: base retrieval, then compress."""

    base_retriever: Any
    base_compressor: Any

    def _get_relevant_documents(self, query: str, *, run_manager: Any, **kwargs: Any) -> list:
        docs = self.base_retriever.invoke(query, **kwargs)
        return list(self.base_compressor.compress_documents(docs, query))


def request(text: str | None = "shipping takes days refunds", k: int = 2, **extra: Any):
    return PipelineRequest(query_id="q1", text=text, k=k, **extra)


def test_langchain_single_stage_and_truncation() -> None:
    pipeline = langchain_pipeline(KeywordRetriever(), id_key="doc_id")
    result = pipeline(request())
    assert [h.id for h in result.hits] == ["shipping", "refunds"]
    assert [(s.name, s.kind, s.source) for s in result.stages] == [
        ("retrieve:KeywordRetriever", "custom", "measured"),
        ("truncate", "truncate", "measured"),
    ]
    assert [h.id for h in result.stages[0].hits] == ["shipping", "refunds", "password"]
    assert result.stages[1].parents == ["retrieve:KeywordRetriever"]
    # Document.id by default.
    assert langchain_pipeline(KeywordRetriever())(request(k=1)).hits[0].id == "chunk-shipping"


def test_langchain_compression_is_recorded_as_two_stages() -> None:
    retriever = CompressionRetriever(
        base_retriever=KeywordRetriever(), base_compressor=DropShipping()
    )
    result = langchain_pipeline(retriever, id_key="doc_id")(request(k=3))
    assert [s.name for s in result.stages] == [
        "retrieve:KeywordRetriever",
        "compress:DropShipping",
    ]
    assert [h.id for h in result.stages[0].hits] == ["shipping", "refunds", "password"]
    assert [h.id for h in result.hits] == ["refunds", "password"]
    # The decomposed run returns what the retriever itself returns.
    assert [d.metadata["doc_id"] for d in retriever.invoke("shipping takes days refunds")] == [
        "refunds",
        "password",
    ]
    opaque = langchain_pipeline(retriever, id_key="doc_id", decompose=False)(request(k=3))
    assert [s.name for s in opaque.stages] == ["retrieve:CompressionRetriever"]
    assert opaque.hits == result.hits


def test_langchain_forwards_filters_only_when_asked() -> None:
    filtered = request(filters={"exclude": "shipping"})
    with pytest.raises(ValueError, match="pass invoke_kwargs"):
        langchain_pipeline(KeywordRetriever(), id_key="doc_id")(filtered)
    with pytest.raises(ValueError, match="pass invoke_kwargs"):
        langchain_pipeline(KeywordRetriever(), id_key="doc_id")(request(context={"tenant": "a"}))
    retriever = KeywordRetriever()
    pipeline = langchain_pipeline(
        retriever, id_key="doc_id", invoke_kwargs=lambda r: {"k": r.k, "filter": r.filters}
    )
    result = pipeline(filtered)
    assert [h.id for h in result.hits] == ["refunds", "password"]
    assert retriever.seen_kwargs == [{"k": 2, "filter": {"exclude": "shipping"}}]
    assert [s.name for s in result.stages] == ["retrieve:KeywordRetriever"]


def test_langchain_rejects_unusable_input() -> None:
    with pytest.raises(TypeError, match="invoke"):
        langchain_pipeline(object())
    with pytest.raises(ValueError, match="no text"):
        langchain_pipeline(KeywordRetriever())(request(text=None, vector=[0.1]))

    class NoIds(KeywordRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager: Any, **kwargs: Any):
            return [Document(page_content="x")]

    with pytest.raises(ValueError, match="pass id_key"):
        langchain_pipeline(NoIds())(request())
    with pytest.raises(ValueError, match="has no id"):
        langchain_pipeline(NoIds(), id_key="doc_id")(request())


def test_langchain_run_attributes_losses_and_feeds_release_checks() -> None:
    from vectraudit.evaluation.compare import Verdict, compare_runs
    from vectraudit.evaluation.policy import DEFAULT_POLICY

    ds = parse_dataset(
        json.dumps({"query_id": "q1", "text": "shipping days", "relevant": {"shipping": 1}})
        + "\n"
        + json.dumps({"query_id": "q2", "text": "refunds days", "relevant": {"refunds": 1}})
        + "\n"
    )
    options = PipelineOptions(k=2, repeats=1, warmup=0)
    baseline = execute_pipeline(
        ds, langchain_pipeline(KeywordRetriever(), id_key="doc_id"), options
    )
    retriever = CompressionRetriever(
        base_retriever=KeywordRetriever(), base_compressor=DropShipping()
    )
    candidate = execute_pipeline(ds, langchain_pipeline(retriever, id_key="doc_id"), options)
    assert baseline["status"] == candidate["status"] == "complete"
    lost = candidate["queries"][0]["evidence"]["failures"]
    assert lost == [
        {
            "doc_id": "shipping",
            "stage": "compress:DropShipping",
            "kind": "rerank",
            "parents": ["retrieve:KeywordRetriever"],
            "evidence": "measured",
            "reason": "present in parent output; absent from this stage",
        }
    ]
    assert compare_runs(baseline, candidate, DEFAULT_POLICY)["verdict"] == Verdict.FAIL.value


# --- LlamaIndex ----------------------------------------------------------------------------

# Stand-ins with the llama-index-core interface the adapter uses (checked against 0.14.25):
# BaseRetriever.retrieve(str) -> list[NodeWithScore]; NodeWithScore.node.node_id /
# .ref_doc_id / .metadata and .score; BaseNodePostprocessor.postprocess_nodes(nodes,
# query_bundle=None, query_str=None). llama-index-core is not a test dependency because its
# nltk dependency carries an advisory the security scan would report.


@dataclass
class TextNode:
    node_id: str
    metadata: dict[str, Any]
    ref_doc_id: str | None = None


@dataclass
class NodeWithScore:
    node: TextNode
    score: float | None = None


class LlamaKeyword:
    def __init__(self, top_k: int = 3, exclude: str | None = None) -> None:
        self.top_k, self.exclude = top_k, exclude

    def retrieve(self, str_or_query_bundle: str) -> list[NodeWithScore]:
        ids = [d for d in _rank(str_or_query_bundle) if d != self.exclude][: self.top_k]
        return [
            NodeWithScore(
                TextNode(f"node-{d}-{part}", {"doc_id": d}, ref_doc_id=f"src-{d}"),
                score=1.0 - rank / 10,
            )
            for rank, d in enumerate(ids)
            for part in (1, 2)  # two chunks per document
        ]


class SimilarityPostprocessor:
    def __init__(self, similarity_cutoff: float) -> None:
        self.similarity_cutoff = similarity_cutoff

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Any = None, query_str: str | None = None
    ) -> list[NodeWithScore]:
        assert query_str
        return [n for n in nodes if (n.score or 0) >= self.similarity_cutoff]


def test_llamaindex_document_level_ids_and_postprocessors() -> None:
    cutoff = SimilarityPostprocessor(similarity_cutoff=0.85)
    pipeline = llamaindex_pipeline(LlamaKeyword(), id_key="doc_id", postprocessors=[cutoff])
    result = pipeline(request(k=3))
    assert [s.name for s in result.stages] == [
        "retrieve:LlamaKeyword",
        "postprocess1:SimilarityPostprocessor",
    ]
    # Two chunks per document collapse to one hit at the best rank, with its score.
    assert [(h.id, h.score) for h in result.stages[0].hits] == [
        ("shipping", 1.0),
        ("refunds", 0.9),
        ("password", 0.8),
    ]
    assert [h.id for h in result.hits] == ["shipping", "refunds"]
    by_node = llamaindex_pipeline(LlamaKeyword())(request(k=2))
    assert [h.id for h in by_node.hits] == ["node-shipping-1", "node-shipping-2"]
    by_source = llamaindex_pipeline(LlamaKeyword(), id_key="ref_doc_id")(request(k=1))
    assert [h.id for h in by_source.hits] == ["src-shipping"]

    twice = llamaindex_pipeline(LlamaKeyword(), id_key="doc_id", postprocessors=[cutoff, cutoff])
    assert [s.name for s in twice(request(k=3)).stages][1:] == [
        "postprocess1:SimilarityPostprocessor",
        "postprocess2:SimilarityPostprocessor",
    ]


def test_llamaindex_filters_need_a_retriever_factory() -> None:
    filtered = request(filters={"exclude": "shipping"})
    with pytest.raises(ValueError, match="builds a filtered retriever"):
        llamaindex_pipeline(LlamaKeyword(), id_key="doc_id")(filtered)
    pipeline = llamaindex_pipeline(
        lambda r: LlamaKeyword(top_k=r.k, exclude=r.filters.get("exclude")), id_key="doc_id"
    )
    assert [h.id for h in pipeline(filtered).hits] == ["refunds", "password"]
    with pytest.raises(TypeError, match="retrieve"):
        llamaindex_pipeline(object())
    with pytest.raises(TypeError, match="postprocess_nodes"):
        llamaindex_pipeline(LlamaKeyword(), postprocessors=[object()])


def test_workbench_imports_an_adapter_beside_the_manifest(tmp_path) -> None:
    import sys

    from typer.testing import CliRunner

    from vectraudit.cli import app

    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"query_id": "q1", "text": "shipping days", "relevant": {"shipping": 1}}) + "\n"
    )
    (tmp_path / "team_retrieval_adapter.py").write_text(
        "from langchain_core.documents import Document\n"
        "from langchain_core.retrievers import BaseRetriever\n"
        "from vectraudit.integrations.retrievers import langchain_pipeline\n\n"
        "class Fixed(BaseRetriever):\n"
        "    def _get_relevant_documents(self, query, *, run_manager):\n"
        "        return [Document(page_content='x', id='shipping')]\n\n"
        "pipeline = langchain_pipeline(Fixed())\n"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "repeats": 1,
                "candidates": [
                    {
                        "name": "team",
                        "adapter": {"kind": "python", "target": "team_retrieval_adapter:pipeline"},
                    }
                ],
            }
        )
    )
    before = list(sys.path)
    result = CliRunner().invoke(
        app, ["workbench", "run", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "runs")]
    )
    assert result.exit_code == 0, result.output
    assert "team: complete" in result.output
    assert sys.path == before
    run = json.loads(next((tmp_path / "runs").glob("*.run.json")).read_text())
    assert run["queries"][0]["ids"] == ["shipping"]
