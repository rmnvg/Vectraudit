"""Traces from openinference-instrumentation-langchain."""

from _capture import provider, run_and_write
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider, exporter = provider()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
run_and_write(exporter)
