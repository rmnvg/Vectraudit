"""Run the pipeline under a tracer provider that keeps spans in memory, then write them."""

import sys

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rag_common import QUESTIONS, build, write_otlp


def provider():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


def run_and_write(exporter):
    chain = build()
    exporter.clear()  # drop spans from building the vector store
    for question in QUESTIONS:
        chain.invoke(question)
    spans = exporter.get_finished_spans()
    print(len(spans), "spans:", sorted({s.name for s in spans}))
    write_otlp(spans, sys.argv[1])
