# Retrieval engineering workbench

Test the deployed pipeline, inspect recorded stage losses, compare measured candidates,
and guard the next release. Everything runs locally unless you explicitly configure an
HTTP endpoint or invoke the existing cloud upload commands.

## Try the complete local workflow

From a source checkout with the frontend built:

```sh
vectraudit workbench run examples/workbench/manifest.json --out workbench-runs
vectraudit workbench optimize examples/workbench/manifest.json --out workbench-runs/optimization.json
vectraudit workbench benchmark examples/workbench/queries.jsonl --runs workbench-runs --out workbench-runs/benchmark.json
vectraudit workbench serve workbench-runs
```

Open `http://127.0.0.1:8766`. The workbench is also a navigation item in the index debugger.
Import artifacts directly from your computer or select files from the configured local
artifact directory. Inspect a query, click a document ID, and follow it through each stage.
The example is deliberately tiny: it demonstrates behavior, not production effectiveness.

The browser presents runs, stage evidence, experiments, benchmark review queues, and
monitoring history. Optimization, monitoring and bulk labelling still use the CLI.

## Guided workflow

The **Guided workflow** tab (the default in `vectraudit workbench serve`) takes you from
a pipeline to a measured fix without editing JSON files:

1. **Connect pipeline.** Choose the built-in local BM25/hybrid baseline and paste or
   upload a JSONL corpus (`doc_id`, `text`, optional `metadata` and `vector`). You can
   also choose an adapter registered at startup with
   `vectraudit workbench serve DIR --manifest pipelines.json`. **Use starter example**
   loads a four-document corpus and benchmark.
2. **Build benchmark.** Add or import questions. For each one, set relevance grades,
   an explicit *no relevant document* judgment, tenant context, filters, forbidden
   document IDs, cohort tags, and the exploratory or held-out split. Fields containing
   invalid JSON block saving. Saved benchmarks are versioned in `DIR/.workbench/drafts/`.
   A save based on an older revision is rejected rather than overwriting newer edits.
3. **Evaluate & inspect.** Runs execute in a single local worker with progress and
   cancellation. Cancelled runs are not saved. Completed runs are written to `DIR` as
   ordinary `*.run.json` files, so **Inspect failures** opens them in the detailed trace view.
4. **Compare a fix.** Choose a baseline and a candidate, run the release comparison
   (saved as `*.check.json`), and work through the ranked findings.

### Findings

Every comparison is triaged per query, most severe first:

| Severity | Finding |
|---|---|
| Blocking | Forbidden documents delivered (with the stages that carried them); a no-answer query that returned results |
| Error | The query failed in either run, so it has no score |
| Regressed | A relevant document the baseline delivered and the candidate lost; otherwise a lower score from re-ranking |
| Missing in both | Relevant documents neither run delivered |
| Improved | Relevant documents recovered, or a higher score |

A lost document is located in one of two ways. The candidate's own trace can show the
stage that dropped it from its parent's output. Otherwise, the first stage that held it in
the baseline but not in the candidate shows where the runs diverge. Without stage traces a
finding says the ranking regressed and lists the dropped results, without guessing a cause.
Each finding names the `*.run.json` and `*.check.json` files it came from. Selecting it
highlights the document in both traces and marks the stage where it was dropped. **Open in
candidate run** opens the same query and document in the detailed trace view. The triage is
`vectraudit.evaluation.triage.triage(baseline, candidate)` and works on any two run files.

Exploratory runs never include held-out queries. A held-out run requires an explicit
confirmation. A comparison between different splits is reported as not comparable.

The browser cannot register Python code or new endpoints. Pipelines come only from the
startup manifest. A registered HTTP adapter runs only after you confirm that query text,
vectors, filters and user context will be sent to its displayed destination. Write
requests need a per-process token and a same-origin request, and request bodies are
limited to 5 MiB.

## Pipeline adapters

A manifest contains `dataset`, `k`, `repeats`, `max_candidates`, and named `candidates`.
Paths are relative to the manifest. Each candidate has an `adapter` and `params`.
Use environment variable names for HTTP credentials; never put secrets in parameters.

```json
{
  "dataset": "queries.jsonl",
  "k": 10,
  "repeats": 3,
  "corpus_version": "catalog-2026-10-05",
  "candidates": [{
    "name": "production",
    "adapter": {"kind": "http", "target": "https://retrieval.example/search", "token_env": "RETRIEVAL_TOKEN", "timeout": 30},
    "params": {"depth": 100, "reranker": "production-v2"}
  }]
}
```

An HTTP adapter POSTs a `PipelineRequest` containing `query_id`, `text`, `vector`, `k`,
`filters`, `context`, and `params`. It expects a `PipelineResult` JSON object:

```json
{"hits":[{"id":"doc-1","score":0.9}],"abstained":false,"stages":[
  {"name":"retrieve","kind":"vector","parents":[],"hits":[{"id":"doc-1","score":0.9}],"latency_ms":3.5}
]}
```

Stages are optional for opaque endpoints. Without them, internal failure causes remain
unknown. Every trace includes complete recorded output IDs, optional scores, input-stage
names, and latency. Stage names must be unique and parents must appear earlier. The final
stage must match delivered results. Duplicate IDs, non-finite measurements, contradictory
abstention and more than `k` delivered hits fail the query. Responses are limited to 8 MiB,
64 stages and 10,000 hits per stage. Redirects and ambient proxies are disabled. Private
endpoints are intentionally supported; configuration is trusted local input.

For custom Python, use `{"kind":"python","target":"my_package.retrieval:search"}`.
The callable accepts `PipelineRequest` and returns `PipelineResult`. The package must be
importable in the runner environment. Only trusted CLI configuration loads code.

```python
from vectraudit.pipelines import Hit, PythonPipeline, Stage

search = PythonPipeline(
    [
        Stage("keyword", "keyword", lambda request, parents: [Hit(id="doc-1")]),
        Stage(
            "final",
            "truncate",
            lambda request, parents: parents["keyword"][: request.k],
            parents=("keyword",),
        ),
    ]
)
```

Instrument real keyword/vector/filter/fusion/rerank/truncation functions with `Stage`.
Explicit parent edges preserve parallel retrieval branches. A recorded disappearance
proves a transition, not why an opaque ranker made its decision. Python stages are timed
by Vectraudit; endpoint stage timing is marked `reported`. End-to-end latency is always
measured by the client. Repeats produce a median per query; p95 is across those medians,
not a production traffic p95. Repeated safety failures are counted even if the first call
was safe. Raw exceptions are redacted from artifacts to avoid leaking query text/secrets.

The built-in `hybrid` adapter reads JSONL `{doc_id,text,metadata?,vector?}` documents.
It provides BM25, optional cosine vector search, weighted reciprocal-rank fusion,
metadata equality filters, a transparent `phrase` baseline reranker and explicit final
truncation. Parameters: `depth`, `vector_weight` in [0,1], `reranker` = `none` or `phrase`.
Nonzero vector weight requires query and corpus vectors from the same embedding space.
Its tenant filter is a reference implementation; use the customer's actual adapter to
verify their authorization system. The phrase reranker is not a semantic cross-encoder.

## No-answer and permissions

```json
{"query_id":"outside-scope","text":"Tomorrow's weather?","no_answer":true}
{"query_id":"tenant-a","text":"Invoice export","context":{"tenant":"a"},"relevant":{"public-guide":1},"forbidden":["tenant-b-invoice"],"tags":["permission"]}
```

`no_answer:true` is an explicit human judgment. Empty `relevant:{}` remains unannotated.
A no-answer assertion passes only if every measured repeat delivers no documents.
Abstention must also deliver no documents. No-answer questions do not receive invented
nDCG scores. Any forbidden delivered ID is an absolute release failure, including when
the baseline violated the same rule. Positive labels separately measure allowed recall.
No-answer-only suites can be checked using the default policy; an explicit nDCG policy
still requires measured nDCG. Legacy vector connectors reject nonempty permission context
rather than silently ignore it. Dataset hashes include new fields only when used, preserving
historical hashes. Native import/export preserves them.

Use cohort tags for exact identifiers, typos, paraphrases, multilingual, ambiguous, outdated,
no-answer and permission cases. Benchmark auditing suggests categories, finds normalized
text duplicates, and prioritizes unjudged pooled documents where candidates disagree.
Suggestions require human review and do not synthesize positive labels. The existing
annotation UI continues to manage positive document judgments; explicit no-answer and
permission assertions are authored in dataset JSONL in this release.

## Controlled optimization

Mark final evaluation rows `"split":"heldout"`; other rows default to `explore`.
`workbench optimize` evaluates a bounded candidate list only on exploratory rows, selects
the highest nDCG candidate meeting latency/cost/memory and safety constraints, then evaluates
only that candidate on held-out rows. Duplicate normalized questions across splits are
rejected. Semantic near-duplicates still need human review. It never automatically tunes
against a held-out failure or deploys a configuration. Repeated human optimization against
the same held-out set also contaminates it: rotate benchmark versions deliberately.

Set `max_candidates` (default 20, maximum 100), `max_p95_ms` (default 150), optionally
`max_cost_usd` (mean per call) and `max_memory_bytes` (maximum reported per call). Unknown
cost/memory measurements cannot satisfy those constraints. These two measurements come
from adapters; they are not estimated or fabricated. Python calls are trusted code and
must implement their own cancellation/timeouts; HTTP calls have bounded timeouts.

## Scheduled monitoring

Create a manifest with exactly one deployed candidate and retain a complete baseline run:

```sh
vectraudit workbench monitor monitor.json baseline.run.json --history monitor-history --policy policy.json
vectraudit workbench serve monitor-history
```

The baseline dataset hash is frozen. Changed benchmark content is a configuration error;
monitoring never silently re-baselines. Each execution writes a unique run and monitor
artifact atomically. Exit 0 means pass, 3 means a check alert, 2 means configuration/execution
setup failed. Failed query runs never pass. Connect these exit codes to your scheduler's
alerting. A self-hosted GitHub Actions schedule template lives in `deploy/monitor/`;
cron can execute the same command. No external alert messages are sent by Vectraudit.

Import several monitor artifacts to review history. Reports separately identify corpus
version changes, configuration changes, and added/removed result IDs. Without a supplied
corpus version, corpus drift is **unknown**. Result coverage is measured; whole-corpus
inventory coverage is not inferred from a small query sample. Built-in corpus files are
content-hashed automatically. HTTP/Python deployments must supply immutable corpus versions
and version their adapter configuration when deployed code changes.

## Validation and deployment boundaries

`tests/test_pipelines.py`, `tests/test_workbench.py`, `tests/test_guided_workbench.py`,
`scripts/workbench_smoke.mjs` and `scripts/guided_smoke.mjs` exercise adapters, stage
evidence, safety assertions, split isolation, guided persistence and browser flows.
The existing release-check run format is retained. See
[enterprise deployment](enterprise-deployment.md) for governance, SSO and recovery checks.
Real-provider SSO certification, representative load tests and PostgreSQL recovery rehearsals
are deployment acceptance work; passing local fixtures is not a claim of market readiness.
