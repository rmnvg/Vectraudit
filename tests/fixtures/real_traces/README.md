# Traces from real instrumentation

Exports produced by the instrumentation libraries themselves, not written by hand, so the
importers are tested against what those libraries actually emit. A tiny offline RAG
pipeline (`generate/rag_common.py`: Chroma with deterministic fake embeddings, a LangChain
retriever, prompt and fake chat model) answers the three questions in `questions.jsonl`,
and each library's spans are written as OTLP JSON the way the OpenTelemetry Collector's
file exporter writes it (one request per line, hex IDs). All text is synthetic.

| File | Instrumentation | Versions |
|---|---|---|
| `openinference.json` | `openinference-instrumentation-langchain` | 0.1.81 |
| `openllmetry.json` | `opentelemetry-instrumentation-langchain` and `-chromadb` (OpenLLMetry) | 0.62.4 |
| `openlit.json` | `openlit` (documents carry `metadata.doc_id` only) | 1.45.0 |
| `openlit_ids.json` | `openlit` with `metadata.id` set (`WITH_ID=1`) | 1.45.0 |

Common: `langchain-core` 1.7.3, `chromadb` 1.5.9, `opentelemetry-sdk` 1.45.2, Python 3.12.

To regenerate (in a separate environment; none of these are Vectraudit dependencies):

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install langchain-core langchain-chroma chromadb \
  opentelemetry-sdk opentelemetry-exporter-otlp-proto-common \
  openinference-instrumentation-langchain opentelemetry-instrumentation-langchain \
  opentelemetry-instrumentation-chromadb
.venv/bin/python generate/run_openinference.py openinference.json
.venv/bin/python generate/run_openllmetry.py openllmetry.json
# OpenLIT instruments everything installed, so give it its own environment:
uv venv .venv-openlit && VIRTUAL_ENV=.venv-openlit uv pip install openlit langchain-core \
  langchain-chroma chromadb opentelemetry-exporter-otlp-proto-common
.venv-openlit/bin/python generate/run_openlit.py openlit.json
WITH_ID=1 .venv-openlit/bin/python generate/run_openlit.py openlit_ids.json
```
