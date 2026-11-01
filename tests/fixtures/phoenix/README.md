# Phoenix API responses from a real server

Every response a running Phoenix server gave to `phoenix-pull`, recorded by
`generate/record.py` and replayed in `tests/test_phoenix_api.py`, so the pull is tested
against what Phoenix actually returns. All text is synthetic.

| File | Recorded with |
|---|---|
| `responses.json` | normal page sizes (4 requests) |
| `responses_paged.json` | `small`: pages of 4 spans and 2 annotations, 2 IDs per request (26 requests), so cursor paging and ID batching are real |

Server: `arize-phoenix` 21.0.0 (`phoenix serve`, local, no auth). Traces: the offline
LangChain + Chroma pipeline of `../real_traces/generate/rag_common.py`, instrumented with
`openinference-instrumentation-langchain` 0.1.81 and sent over OTLP/HTTP by
`generate/run_phoenix.py` to project `rag-demo`.

Annotations were then added through the REST API:

| Trace | Annotation |
|---|---|
| password (q1) | `correctness` HUMAN `correct` (score 1, with an explanation); a note |
| invoice (q2) | `correctness` HUMAN `Incorrect`; `qa_correctness` LLM `incorrect` on the root span |
| cancel (q3) | `relevance` CODE score 0.4; `tone` HUMAN `friendly`; `qa_correctness` LLM `correct` on the root span |

Observed: Phoenix's `/spans` flattens document metadata into
`retrieval.documents.N.document.metadata.doc_id` (the OTLP export keeps one JSON string);
the span kind is a top-level `span_kind`; notes are stored as HUMAN annotations named
`note` with an identifier starting `px-trace-note:`.

To regenerate (in a separate environment; Phoenix is not a Vectraudit dependency):

```bash
uv venv .venv-phoenix && VIRTUAL_ENV=.venv-phoenix uv pip install arize-phoenix
PHOENIX_WORKING_DIR=$PWD/phx-data .venv-phoenix/bin/phoenix serve &
# in the real_traces environment, plus opentelemetry-exporter-otlp-proto-http:
python generate/run_phoenix.py      # with rag_common.py from ../real_traces/generate
# add the annotations above with POST /v1/trace_annotations and /v1/span_annotations, then:
uv run python generate/record.py http://127.0.0.1:6006 rag-demo SINCE UNTIL responses.json
uv run python generate/record.py http://127.0.0.1:6006 rag-demo SINCE UNTIL responses_paged.json small
```

The trace IDs and times in the tests are the ones this recording produced.
