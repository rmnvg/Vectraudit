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
   **Test pipeline** sends one sample question to the selected pipeline before you build
   a benchmark (see [Test a pipeline connection](#test-a-pipeline-connection)).
2. **Build benchmark.** Add or import questions. For each one, set relevance grades,
   an explicit *no relevant document* judgment, tenant context, filters, forbidden
   document IDs, cohort tags, and the exploratory or held-out split. Fields containing
   invalid JSON block saving. Saved benchmarks are versioned in `DIR/.workbench/drafts/`.
   A save based on an older revision is rejected rather than overwriting newer edits.
3. **Evaluate & inspect.** Runs execute in a single local worker with progress and
   cancellation. Query progress survives cancellation and server restarts in
   `DIR/.workbench/evaluations/`; **Saved progress** resumes the original settings.
   Cancelled evaluations remain checkpoints rather than release artifacts. Completed runs are written to `DIR` as
   ordinary `*.run.json` files, so **Inspect failures** opens them in the detailed trace view.
4. **Compare a fix.** Choose a baseline and a candidate, run the release comparison
   (saved as `*.check.json`), and work through the ranked findings.

### Resume an evaluation

Every guided evaluation and `workbench run` writes a local SQLite checkpoint before
and after each query. Resume skips successful queries and leaves recorded failures
unchanged unless you explicitly select **Retry failed queries too** or pass
`--retry-errors`. Previous attempt results, sanitized errors, elapsed time and attempt
counts remain in the exported query evidence, including the rankings a failed attempt
returned before it failed; critical-question checks judge those rankings as well as the
retry's. An interrupted attempt has a start time but no fabricated duration or result.

Failed queries retain known adapter error codes (for example `http_auth`,
`http_rate_limit`, `timeout`, `connection_failed` and `invalid_response`) with a safe
message and next action. Unclassified exceptions and unknown adapter codes remain
`pipeline_error`; exception text is never saved. Run JSON includes
`summary.errors_by_code`, counting each currently failed query once. Historical failed
attempts remain in the checkpoint evidence but leave this count after a successful retry.
A failed query still makes the run partial and prevents a passing release check.

Transient failures retry automatically within a declared budget: `timeout`,
`connection_failed`, `http_rate_limit` (HTTP 429) and `http_unavailable` (HTTP 502, 503 or
504). Each pipeline call (warm-up or repeat) gets up to three tries, waiting 1 s, then 2 s
(doubling, capped at 30 s), or as long as the endpoint's `Retry-After` asks. A `Retry-After`
longer than the cap is not waited for. At most 100 retries happen per run invocation, so an
endpoint that is down fails quickly instead of stalling CI. Authentication, 404, invalid
responses and unclassified exceptions are never retried. Python adapters opt in by raising
`PipelineCallError` with one of these codes.

Each retry is recorded on its query (`retries`: call, code and delay) and counted in
`summary.retries_by_code`; the policy is saved in `timing.retry`. Latency samples measure
only the successful call. A query whose calls still fail keeps its cause and leaves the run
partial; retries never turn missing evidence into a pass. Cancelling during a backoff wait
stops the run promptly. On the command line, `--max-attempts 1` disables automatic retries
and `--retry-budget N` changes the per-run limit; both are frozen into the checkpoint, so a
resume must use the same values. Guided evaluations, release exports and optimization use
the defaults; plain `RunOptions` in the Python API do not retry.

The CLI prints a checkpoint directory. Keep that directory and resume the same manifest:

```sh
vectraudit workbench run manifest.json --out workbench-runs
# After interruption, use the exact directory printed by the first command:
vectraudit workbench run manifest.json --out workbench-runs --resume workbench-runs/.checkpoints/BATCH_ID
# Explicitly retry recorded failures as well:
vectraudit workbench run manifest.json --out workbench-runs --resume workbench-runs/.checkpoints/BATCH_ID --retry-errors
```

In the browser, reopen **Evaluate & inspect → Saved progress → Resume evaluation**.
You do not need to rebuild the editor state. The saved draft revision and evaluation
settings are revalidated, and remote transfers and held-out evaluation require fresh
confirmation. Changing the original draft requires a new evaluation. Reloading while
a job is still active also offers **View progress**.

Resume rejects changed dataset content, candidate configuration, corpus version,
query execution options, local adapter/evaluator source, Python/runtime details or
installed package versions. For Python and HTTP candidates, declare an immutable
`pipeline_version` on the **candidate**, and `corpus_version` on the **manifest**, before
starting the evaluation:

```json
{
  "dataset": "queries.jsonl",
  "corpus_version": "corpus-snapshot-2026-10-07",
  "candidates": [{
    "name": "production",
    "pipeline_version": "deployment-42-model-v3",
    "adapter": {"kind": "python", "target": "my_pipeline:retrieve"}
  }]
}
```

The pipeline version must cover external deployment, model, prompts, dependencies and
other behavior not captured in the local adapter file. The corpus version must identify
a fixed snapshot, not a mutable alias. Vectraudit cannot detect remote changes behind
an unchanged version string. Built-in hybrid adapters derive corpus identity from the
actual documents. Unversioned custom adapters can still run, but cannot resume safely.

The Python API accepts `checkpoint=Path(...)`, `resume=True`, and `retry_errors=True`
on `execute_pipeline`; custom callers supply `config={"pipeline_version": "..."}`
and `corpus_version="..."`. Retry requires resume. Checkpoints are protected against
concurrent writers; existing checkpoints cannot be accidentally replaced by a new run.

A crash can occur after an adapter received a query but before its result was committed.
That one unfinished query, including its warmup/repeats, may execute again. This is
query-level recovery, not exactly-once remote execution. Successful query samples retain
their original timings; `timing.wall_s` covers the latest invocation. A resumed evaluation
keeps its run ID. Incomplete or failed query results never become a passing release check.

Keep checkpoints on a local filesystem alongside the artifacts. The journal contains
retrieval evidence; browser recovery metadata also retains the supplied built-in corpus.
A missing/corrupt started journal fails closed: restore it or start a new evaluation.
The guided editor retains its 500-question limit; the CLI supports larger datasets.
Optimization, scheduled monitoring, and connector-only evaluation do not yet use this
checkpoint workflow.

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

### Test a pipeline connection

**Test pipeline** on the *Connect pipeline* step calls the selected pipeline once with a
sample question (prefilled from the first exploratory benchmark question) and optional
filters, user/tenant context, forbidden document IDs, or a no-answer expectation. It
saves no run. The report lists one result per check:

| Check | What it shows |
|---|---|
| Response | The call returned a valid `PipelineResult` within the time limit. |
| Ranking | How many documents came back; IDs are unique and at most `k`. |
| Document lookup | Which returned IDs map to passages in the local corpus (unmapped IDs are listed). |
| Stage evidence | Whether stage traces were returned and how many use adapter-reported timings. |
| Corpus version | Whether a corpus version is configured. This is not an inventory of a remote corpus. |
| Filters / context | Marked *Not verified* when supplied. Reaching the adapter does not prove it applied them. Returned documents are checked against filter values in the local corpus metadata. |
| Forbidden / no-answer | The supplied assertions, for this one call only. |

Status is *Passed*, *Warning*, *Failed*, *Not verified* or *Not tested*. A failed
call reports a fixed category with a next step instead of the exception text:

- missing credential variable
- import or dependency failure
- adapter configuration
- HTTP 401/403, 404, 429 or another HTTP error
- connection failure or timeout
- oversized or invalid response
- too many hits
- missing document ID
- framework retriever cannot forward filters/context
- adapter exception

The test runs in a separate Python process with its output discarded, so adapter logs and
exception messages (which may contain credentials) never reach the browser. It is killed
at its time limit (default 60 s) or on **Cancel test**, and only one test runs at a time.
HTTP adapters still require consent to send the sample. The API is
`POST /api/guided/diagnostics` and `POST /api/guided/diagnostics/cancel`.

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
The callable accepts `PipelineRequest` and returns `PipelineResult`. A module in the
manifest's folder is importable directly; otherwise the package must be installed in the
runner environment. Only trusted CLI configuration loads code.

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

### Concurrency and rate limits

By default a candidate runs one query at a time. Set `concurrency` (1–32 queries in flight)
and optionally `max_rps` (calls per second, retries and warm-ups included) on a candidate to
run large suites faster without overloading the endpoint:

```json
{"name": "production", "adapter": {"kind": "http", "target": "https://retrieval.example/search"},
 "concurrency": 8, "max_rps": 20}
```

Results are identical to a serial run apart from timing, and the run file lists queries in
dataset order. All workers share one automatic-retry budget and one rate limit. Checkpoint
writes stay on one thread, so cancel and resume work as before: queries that finished before
cancellation are kept, and calls already running finish and are saved. HTTP adapters and
the built-in hybrid baseline are thread-safe. A Python adapter that is not must use
`concurrency` 1, or set `thread_safe = False` on the callable so higher values are refused.

These settings change how fast queries run, not what the pipeline returns, so they are not
part of the candidate's configuration identity. They are part of the checkpoint identity:
resume with the same values. Runs record `timing.concurrency` and `timing.max_rps`. Latency
measured with several queries in flight is latency under load, so release checks compare
latency only between runs with the same concurrency and otherwise say why they did not.
Guided evaluations use the registered candidate's settings. An exported release check copies
the concurrency and rate limit its candidate run was measured with into `release.json`
(`concurrency`, `max_rps`; omitted when serial, so earlier versions can still read the
file). In CI the approved release's values run both sides, like `k` and `repeats`: a pull
request that proposes different values gets a warning and they apply only after merge.
`vectraudit release validate` warns when a Python adapter would run concurrently.
Optimization still runs serially.

### LangChain and LlamaIndex retrievers

Evaluate the retriever your application already builds, without rewriting it. Put a small
module beside the manifest:

```python
# my_rag.py
from vectraudit.integrations.retrievers import langchain_pipeline, llamaindex_pipeline
from my_app.retrieval import build_retriever, build_index, build_reranker

# LangChain: anything with invoke(query). id_key names the metadata field your labels use
# (default: Document.id). invoke_kwargs forwards k and the dataset's filters.
dense = langchain_pipeline(
    build_retriever(), id_key="doc_id", invoke_kwargs=lambda r: {"k": r.k, "filter": r.filters}
)

# LlamaIndex: anything with retrieve(query), or a function that builds one per request.
# id_key="ref_doc_id" evaluates source documents rather than chunks.
index = build_index()
reranked = llamaindex_pipeline(
    lambda r: index.as_retriever(similarity_top_k=50),
    id_key="ref_doc_id",
    postprocessors=[build_reranker(top_n=10)],
)
```

```json
{"name": "dense", "adapter": {"kind": "python", "target": "my_rag:dense"}}
```

Stages are recorded from real calls. A LangChain retriever exposing `base_retriever` and
`base_compressor` (such as `ContextualCompressionRetriever`) runs as those two steps, just
as it runs itself, so a document the compressor or reranker drops is attributed to it;
pass `decompose=False` to treat it as one opaque stage. Each LlamaIndex postprocessor is its
own stage. When the last stage returns more than `k` documents, a `truncate` stage records
the cut. Chunks that resolve to the same id count once, at their best rank.

A query with filters or tenant context fails rather than running unfiltered unless the
adapter forwards them (`invoke_kwargs` for LangChain, a retriever-building function for
LlamaIndex). Framework filter syntax differs per vector store, so translate the dataset's
equality filters there. Neither framework is a Vectraudit dependency: the adapters call
`invoke`, `compress_documents`, `retrieve` and `postprocess_nodes` on the objects you pass.
Run files are redacted, so to see an adapter's own error, call it once directly:
`dense(PipelineRequest(query_id="t", text="a question", k=10))`, importing `PipelineRequest`
from `vectraudit.pipelines`.

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

## Passage review and benchmark maintenance

The guided **Build benchmark** step now includes a passage browser. Attach a JSONL corpus
(`doc_id`, `text`, optional `metadata.title`) and read passages while choosing **Irrelevant**,
**Relevant**, **Highly relevant**, or **Unjudged**. This works with registered pipelines too:
lookup uses the attached local corpus by default. A registered document resolver can fetch
pooled passages on demand as described below. Missing document text is reported explicitly. Passage order is corpus order, not retrieval rank; use the
search field and pagination to inspect other documents. Existing standalone labeling tools
remain available for pooling, independent second passes and reviewer resolution.

After evaluating two pipelines, choose an exploratory question and select **Review baseline**
and **Review candidate**, then **Build passage review queue**. The queue pools delivered
documents from those runs, with unjudged disagreements first. Positive, zero-grade, span,
separate irrelevant and forbidden judgments are excluded. Held-out questions never enter
this queue. Both runs must cover the exact saved exploratory dataset; cancelled or incomplete
runs are rejected. A failed question in a finished partial run contributes no ranking and is
explicitly reported as unknown, not treated as an empty result or a disagreement.

Read and grade the queued passages using your attached local corpus. Missing text remains
visible and cannot be graded until supplied. Retrieval source and rank are hidden in the
grader until a judgment is entered; **Show retrieval evidence** then reveals them. This is
not an independent blind second pass: previous inspection or document IDs can reveal the
source. **Browse all passages** lets you inspect documents outside the retrieved pool.

You can review multiple questions before saving. A changed question, tenant context, filter
or split invalidates its active pool. Saving new relevance labels changes the dataset version;
remeasure both pipelines before building another pool or making a release decision. Separate
irrelevant judgments persist across reloads and are excluded from later queues. Comparing
runs does not modify their evidence or the labels. From a comparison finding, **Review passages
for this question** returns to that exploratory question in the open benchmark with the
comparison's baseline/candidate selected. Version compatibility is checked when building.

An irrelevant judgment is saved separately from the evaluation dataset when needed; it
never implies that the question has no answer. Mark no-answer only after reviewing coverage.
Assigning a positive passage grade clears no-answer and replaces span judgments for that
same document; other spans, contexts, filters and held-out splits remain intact. The advanced
JSON grade editor remains available. Draft saves preserve the local corpus and irrelevant
judgments across restarts. A JSONL dataset export contains evaluation labels; frozen versions
also retain the separate irrelevant judgments.

### Look up passages without importing the whole corpus

Register a trusted Python resolver in the startup manifest alongside your pipelines:

```json
{
  "dataset": "queries.jsonl",
  "corpus_version": "catalog-v7",
  "candidates": [
    {"name": "baseline", "adapter": {"kind": "python", "target": "retriever:search"}}
  ],
  "document_lookup": {"target": "document_source:lookup", "timeout_s": 30}
}
```

Place `document_source.py` beside the manifest or install it in the workbench environment.
Start with `vectraudit workbench serve workbench-runs --manifest pipelines.json`.
The resolver is registered only at startup: browser requests cannot supply Python code,
module paths, destination URLs, credentials, tenant context or filters.

After building a passage review queue, choose **Passage source** (baseline or candidate
corpus version), confirm sending IDs and saved context, then **Fetch visible passages**.
At most ten IDs are requested per click. Grade available passages using the existing
controls; missing, inaccessible and unavailable-version documents remain ungradable.
Switching the source or question clears fetched content. **Attached local corpus** and
**Browse all passages** retain the existing local workflow.

The resolver accepts a `LookupRequest` and returns a `LookupResult` or its JSON-compatible
dictionary. Import these contracts from `vectraudit.pipelines.document_lookup`:

- Request: `query_id`, `doc_ids`, saved `context`, saved `filters`, selected `run_id`,
  and selected run's `corpus_version` (or snapshot ID). Query text and vectors are not sent.
- Result: `corpus_version` and exactly one entry per requested ID, without duplicates or
  additional IDs. Each entry has `doc_id` and a `status`.
- `found`: include nonempty `text` (at most 100,000 characters), `version` (document content
  version), and optional `title`. Other statuses are `missing`, `forbidden`, and
  `version_unavailable`; those entries must not contain text, titles or versions.

Example result:

```json
{
  "corpus_version": "catalog-v7",
  "documents": [
    {"doc_id": "guide", "status": "found", "text": "Invoice export instructions.",
     "version": "guide-v3", "title": "Export invoices"},
    {"doc_id": "old-guide", "status": "missing"}
  ]
}
```

The resolver must enforce the source's authorization and filter semantics, and read the
requested snapshot rather than merely echo its identifier. A returned corpus version that
differs from the known run version is rejected before any text reaches the browser. If the
run has no version, the UI explicitly reports that correspondence is unknown. Versions are
adapter-reported; the workbench also calculates a SHA-256 of each returned passage.

See `examples/workbench/document_source.py` for a read-only SQLite implementation that
looks up IDs within a tenant and snapshot and rejects unsupported filters. Adapt it to your
customer-controlled source; credentials remain in the local environment. A Python resolver
can use an existing database or HTTP client, so a new backend connector is not required.

Lookups run in fresh processes, one at a time, with a configured 1–120 second timeout.
Adapter stdout/stderr and raw exceptions are suppressed. Python code is trusted local code,
not sandboxed; it must use read-only access and enforce authorization itself. A browser
navigation discards the response; an already-running lookup may finish or reach its timeout.

Fetched text and document versions are held only in the review UI; they are not written
into drafts, frozen corpus inventories, run files or exported release bundles. Saving a
grade saves the judgment, not a remote content snapshot. Preserve source snapshots externally
when reproducible annotation or later label-freshness checks are required. Held-out questions
and IDs outside the exact saved exploratory review pool cannot be fetched through this API.

**Import questions or production incidents** previews CSV, JSON arrays (including
`examples`/`items`/`data` wrappers), and JSONL. Map question, ID and relevance columns; nested
JSON question/ID paths are supported. Missing IDs are derived from question/context/filter
content. The relevance column accepts the existing `doc-id:grade; other-id` CSV syntax.
Native filters, vectors, span labels and headers are preserved for ordinary imports.
Select preview rows before adding them; conflicting existing IDs are rejected, never replaced.
The older native JSONL import still replaces the current editor contents.

In incident mode, relevance/no-answer/forbidden judgments and held-out status are discarded.
Exact duplicate question/context/filter combinations are combined with source references.
Feedback and retrieved documents are not ground truth. Selected incidents are exploratory,
unannotated questions that need human passage review. Include ordinary traffic for coverage;
this is a local export workflow, not a live monitoring collector. Import accepts up to 5000
source rows and 500 distinct questions; select smaller source files for larger exports.

**Build permission scenarios** clones an exploratory question into named context/filter
cases with expected relevant IDs, forbidden IDs and explicit no-answer assertions. Scenarios
must specify context and an expectation. The form's filters are strings; use the existing
query editor or API for numeric/boolean values. Contradictory relevant/forbidden expectations
are rejected. The resulting queries use the existing real pipeline runner, so registered
adapters must forward context and filters. Passing sampled cases is not an access-control
certification. Held-out questions cannot seed new scenarios.

### Freeze and check label freshness

Save the benchmark, then choose **Freeze saved benchmark**. Each immutable local snapshot
contains the dataset, its hash, the draft revision, irrelevant judgments and an optional
inventory of document text/metadata hashes. Corpus text remains in the local draft. Changing
or deleting a draft does not rewrite frozen snapshots.

Choose a frozen version and **Check current corpus** after attaching a newer corpus. The
review list identifies labels (including spans, forbidden and irrelevant IDs) referring to
changed/deleted documents. No-answer judgments are flagged when new/changed content might
answer them. Missing inventory is reported as **unknown**. A changed hash is a review signal,
not proof that a label is wrong or the cause of a retrieval regression.

**Restore as new draft** restores the frozen dataset and irrelevant judgments, keeping the
currently attached corpus. Review affected passages, save/freeze the revised benchmark, then
remeasure both baseline and candidate. Existing run files and frozen versions stay unchanged.
These files live under `.workbench/benchmarks/`; draft corpora live under `.workbench/drafts/`.
All are customer-controlled local files and are not uploaded by these workflows.

## Release policies and exported release checks

A comparison means little until the team agrees what "acceptable" is. **Compare a fix**
therefore starts with a **Release policy** selector:

- **Exploratory defaults** (when nothing else is chosen) judge whichever metrics both runs
  measured, from a single question upward. Use them to explore. Verdicts are labelled
  *exploratory, not a release decision*, and they cannot be exported.
- **Saved policies** are created in the policy editor: required metrics with allowed drops
  or minimums, the minimum number of measured questions, a limit on individually regressed
  questions, cohort rules and latency limits, each explained in plain terms. You can also
  paste an existing `policy.json` and download the result. The editor uses the same parser
  as `vectraudit check --policy`. Saving under an existing name creates the next version;
  versions are immutable files under `.workbench/policies/`.
- **Your CI policy**: `vectraudit workbench serve DIR --policy retrieval/policy.json` offers
  that exact file (read-only) and selects it by default.

Every comparison, tracked case and verification records the policy's id, version and rule
hash (`policy_sha256`, the same digest the hosted service uses). On the same run files the
browser, `vectraudit check --policy`, case verification and the exported CI check give the
same verdict, because all of them call one comparison with the same parsed rules.

### Export release check

After a **passing** comparison under a saved policy, **Export release check** creates a
versioned release configuration (`.workbench/releases/`). Exporting again under the same name
creates the next version. The configuration binds:

- the frozen query split that was measured (the benchmark is frozen automatically if it
  has not changed since the runs; otherwise the export is refused),
- the candidate pipeline and its parameters, k, the CI repeat count, and the concurrency and
  rate limit the candidate was measured with,
- the policy and its rule hash, the corpus version, the Vectraudit version and the evidence
  (both run IDs and a saved check file).

Download the files as a zip or write them into a checkout for review; nothing is pushed:

```sh
vectraudit workbench export RELEASE_ID --workspace DIR --out path/to/repo
```

| File | Purpose |
|---|---|
| `retrieval/release/release.json` | The configuration: benchmark/policy hashes, pipeline, k, repeats |
| `retrieval/release/benchmark.jsonl` | The exact frozen split (exploratory or held-out, never mixed) |
| `retrieval/release/policy.json` | The policy rules |
| `retrieval/release/README.md` | What it checks, prerequisites, secrets and local commands |
| `retrieval/release/corpus.jsonl` | Only when *Include the measured corpus* is checked |
| `.github/workflows/retrieval-release-check.yml` | Runs the check on every pull request |

Before export the workbench lists what the customer must provide: the corpus (when not
included), an importable Python adapter and its dependencies, and repository secrets for
HTTP deployments and tokens. Secrets and deployment URLs are referenced by environment
variable only. Local endpoints that a CI runner cannot reach are flagged, and so are held-out
exports.

**How the CI check decides.** The target branch's `release.json` is the *approved*
configuration. The workflow runs the target branch's pipeline (baseline) and the pull
request's pipeline (candidate) on the approved benchmark and k, and judges them with the
approved policy. A pull request that edits the policy or benchmark cannot loosen its own
gate: the proposal is reported in the summary and applies after merge. For Python adapters,
each side imports its own checkout (`python_path`, default the repository root), and the run
records a hash of the adapter module that actually executed. For HTTP pipelines, set
`RETRIEVAL_BASELINE_URL` and `RETRIEVAL_CANDIDATE_URL` to the two deployments being
compared. Until the target branch has a release configuration, the workflow validates the
proposed one, runs the candidate and reports that the comparison starts after merge.

```sh
vectraudit release validate retrieval/release/release.json   # files, hashes, prerequisites
vectraudit release run retrieval/release/release.json --out runs/candidate.run.json \
  [--approved base/retrieval/release/release.json]
vectraudit release check base/retrieval/release/release.json \
  runs/baseline.run.json runs/candidate.run.json --proposed retrieval/release/release.json
vectraudit release update retrieval/release/release.json      # re-approve reviewed edits
```

`release check` uses the exit codes of `vectraudit check` and the same reports and GitHub
publishing options. Its summary names the configuration, policy version and rule hash. If
`policy.json` or `benchmark.jsonl` is edited without `release update`, `validate`, `run` and
`check` fail and name the changed file; `update` records the change as a new configuration
version and marks the earlier comparison evidence as superseded.

Reproduce the browser journey with `node scripts/release_smoke.mjs` after building the UI.
It creates a policy, compares, exports, runs `vectraudit release` on the exported files,
and shows that a stricter policy version changes both the comparison and case verification.

## Persistent regression cases

Findings and regression cases show question text beside IDs when the exact saved local
benchmark is available. The join checks the recorded draft revision, full dataset hash
and evaluated split hash; baseline and candidate text are resolved separately. A changed,
missing or unreadable draft produces an explicit unavailable message. Vector-only queries
and runs without a saved draft reference keep their IDs without invented question text.
Only queries recorded in that run receive display text.

This is a local display join: run/check artifacts, stored cases, cloud uploads and downloaded
issue summaries do not gain question text. The workbench does not reconstruct older draft
revisions or resolve text from unrelated benchmarks that happen to reuse the same IDs.

In **Compare a fix**, select a finding and choose **Track finding**. A case preserves the
original baseline/candidate IDs, benchmark hash and recorded finding. Assign an owner,
record investigation notes, and move it through open/investigating/fix proposed. Owners are
local attribution, not authenticated identities. Cases survive browser and server restarts
under `.workbench/cases/`; **Download issue summary** produces Markdown for your issue tracker.

Record a new candidate and choose **Verify under <policy>**. Verification reruns the release
comparison under the selected release policy (see below) against the case's original
baseline and requires:

- A complete, comparable run of the same query set, dataset version and k.
- A passing release verdict under that policy across that comparison suite.
- No unresolved finding for the tracked query.

Only this measured operation can assign **verified**; manual updates cannot. The original
baseline or failing candidate cannot be used as verification evidence. Errors and
non-comparable runs do not change case status. A later explicit verification that fails
reopens the case and appends evidence to its history. Each verification also saves the full
release-check artifact (including the policy) and links it from the history, together
with the policy's name, version and rule hash. A later policy version never rewrites earlier
verification evidence. Verification covers the baseline's
recorded split, not any held-out queries that were never run. Case status never overrides
release verdicts, and does not automatically update merely because a new run was recorded.

The guided run selector lists the newest 100 local runs as summaries (without per-question
records, cached by file size and modification time); a run's questions load only when you
inspect it. Guided benchmarks hold up to 1,000 questions. Pipeline run files are written as
compact JSON and may be up to 64 MiB: a 1,000-question run with a six-stage trace at
retrieval depth 100 is about 39 MiB (104 MiB if indented). In Chromium, comparing two such
runs rendered in about 2 seconds. The run browser shows 500 questions at a time; search by
ID or cohort to reach the rest. Snapshot/case lists
show the newest 200 records. Local writes use version checks and file locks; stale edits
must reload before retrying. Existing token, same-origin and request-size protections apply
to all new guided APIs.

Reproduce the full browser journey with `node scripts/market_fit_smoke.mjs` after building
the UI. It tests grading/reload, snapshots, findings, incident review, permission creation
and freshness at desktop/mobile sizes against a real local server.

`node scripts/passage_review_smoke.mjs` exercises registered Python adapters, measured runs,
pooled grading, missing text, multi-question review, stale-version rejection, persistence,
and a fresh browser comparison. Both scripts run in CI.

Run `VECTRAUDIT_LOOKUP_SMOKE=1 node scripts/passage_review_smoke.mjs` for the same journey
without an attached corpus, using a registered resolver. CI exercises both modes.
