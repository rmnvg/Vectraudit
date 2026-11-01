"""A tiny offline RAG pipeline (Chroma, fake embeddings, fake chat model) for real traces.

Not imported by the test suite: it needs the instrumentation libraries listed in the README.
"""

import base64
import json
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DOCS = [
    ("d1", "To reset your password, open Settings and choose Security, then Reset password."),
    ("d2", "Invoices are listed under Billing; each invoice can be downloaded as a PDF."),
    ("d3", "To cancel your plan, go to Billing and choose Cancel subscription."),
    ("d4", "Our support team answers within one business day."),
    ("d5", "Two-factor authentication can be enabled in Security settings."),
]
QUESTIONS = ["How do I reset my password?", "Where is the invoice?", "Cancel my plan"]


def _metadata(doc_id):
    # OpenLIT records a document's ID only from metadata "id" or "source".
    return {"doc_id": doc_id, **({"id": doc_id} if os.environ.get("WITH_ID") else {})}


def _join(docs):
    return "\n\n".join(d.page_content for d in docs)


def build():
    """The pipeline: retrieve three documents, put them in the prompt, call the model."""
    store = Chroma(
        collection_name="support-kb", embedding_function=DeterministicFakeEmbedding(size=32)
    )
    store.add_documents(
        [Document(page_content=text, metadata=_metadata(d)) for d, text in DOCS],
        ids=[d for d, _ in DOCS],
    )
    retriever = store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages(
        [("user", "Answer using:\n{context}\n\nQuestion: {question}")]
    )
    model = FakeListChatModel(responses=["An answer."] * 20)
    inputs = {"context": retriever | _join, "question": RunnablePassthrough()}
    return inputs | prompt | model | StrOutputParser()


def _hex(value):
    return base64.b64decode(value).hex() if value else value


def write_otlp(spans, path):
    """OTLP JSON as the Collector's file exporter writes it: one request per line, hex IDs."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    request = MessageToDict(encode_spans(spans))
    for resource in request.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    if key in span:
                        span[key] = _hex(span[key])
    with open(path, "w") as out:
        out.write(json.dumps(request) + "\n")
