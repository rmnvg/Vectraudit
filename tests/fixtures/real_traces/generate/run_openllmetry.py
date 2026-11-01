"""Traces from OpenLLMetry's LangChain and Chroma instrumentations."""

from _capture import provider, run_and_write
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

tracer_provider, exporter = provider()
LangchainInstrumentor().instrument(tracer_provider=tracer_provider)
ChromaInstrumentor().instrument(tracer_provider=tracer_provider)
run_and_write(exporter)
