"""Send the offline RAG pipeline's OpenInference spans to a running Phoenix server."""

import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from rag_common import QUESTIONS, build

endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006")
project = os.environ.get("PHOENIX_PROJECT", "rag-demo")
tracer_provider = TracerProvider(resource=Resource({"openinference.project.name": project}))
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(f"{endpoint}/v1/traces")))
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
chain = build()
for question in QUESTIONS:
    chain.invoke(question)
tracer_provider.shutdown()
print("sent", len(QUESTIONS), "traces to", project)
