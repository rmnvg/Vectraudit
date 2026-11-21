"""Agent traces from OpenLLMetry (opentelemetry-instrumentation-langchain and -chromadb)."""

import sys

from _capture import provider
from agent_common import build, run
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from rag_common import write_otlp

tracer_provider, exporter = provider()
LangchainInstrumentor().instrument(tracer_provider=tracer_provider)
ChromaInstrumentor().instrument(tracer_provider=tracer_provider)
agent = build()
exporter.clear()
run(agent)
spans = exporter.get_finished_spans()
print(len(spans), "spans:", sorted({s.name for s in spans}))
write_otlp(spans, sys.argv[1])
