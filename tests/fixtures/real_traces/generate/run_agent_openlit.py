"""Agent traces from OpenLIT, which reuses a global tracer provider set before openlit.init()."""

import sys

import openlit
from _capture import provider
from agent_common import build, run
from opentelemetry import trace
from rag_common import write_otlp

tracer_provider, exporter = provider()
trace.set_tracer_provider(tracer_provider)
openlit.init(disable_metrics=True, capture_message_content=True, application_name="agent-demo")
agent = build()
exporter.clear()
run(agent)
spans = exporter.get_finished_spans()
print(len(spans), "spans:", sorted({s.name for s in spans}))
write_otlp(spans, sys.argv[1])
