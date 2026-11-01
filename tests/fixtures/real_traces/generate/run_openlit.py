"""Traces from OpenLIT, which reuses a global tracer provider set before openlit.init()."""

import openlit
from _capture import provider, run_and_write
from opentelemetry import trace

tracer_provider, exporter = provider()
trace.set_tracer_provider(tracer_provider)
openlit.init(disable_metrics=True, capture_message_content=True, application_name="rag-demo")
run_and_write(exporter)
