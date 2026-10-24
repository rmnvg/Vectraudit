# Release checks

Catch retrieval regressions before you deploy. A release check measures a **baseline**
retrieval setup and a **candidate** on the same labelled queries, then decides under your
policy:

| Verdict | Exit code | Meaning |
|---|---|---|
| pass | 0 | Every policy rule holds. |
| error | 1 | A run is incomplete (failed, partial or cancelled), or an input file is invalid. Never a pass. |
| fail | 3 | A policy rule is broken: quality dropped, latency rose, too many queries regressed. |
| not comparable | 4 | Different dataset version, `k`, metric contract or query set; too few queries; or a metric the policy needs wasn't measured in both runs. |
| fail, accepted | 0 | Only against a hosted baseline (`check cloud:<name>`): every failing rule is covered by an active, audited exception. Reports say "fail, accepted", never pass; the GitHub check is neutral. See [Audited exceptions](#audited-exceptions). |

Everything runs on your machine or CI runner. Credentials stay in your environment, and
run files hold ids, metrics and timings, not documents or vectors.

## Workflow

```bash
vectraudit dataset validate retrieval/dataset.jsonl
vectraudit record --connection base.json --dataset retrieval/dataset.jsonl --out baseline.json --label main
vectraudit record --connection cand.json --dataset retrieval/dataset.jsonl --out candidate.json --label pr-42
vectraudit check baseline.json candidate.json --policy retrieval/policy.json --html report.html
```

For GitHub Actions, start from
[examples/github/retrieval-check.yml](../examples/github/retrieval-check.yml): one
`retrieval-check` step (written by `vectraudit init`) runs the commands above on the same
runner, with the target branch's connection as the baseline. `check`
appends a Markdown summary to the job summary automatically, and with the job's
`GITHUB_TOKEN`:

- `--github-check` publishes a check run on the pull request's head commit (permission
  `checks: write`). Pass maps to success; fail, error and not comparable all map to failure,
  so a required check never lets an unverified change through. `--check-name` keeps one
  check per backend apart.
- `--github-comment` creates one pull-request comment and updates it on later pushes
  (permission `pull-requests: write`).

Publishing never changes the exit code. Pull requests from forks get a read-only token, so
publishing is skipped with a warning and the job summary still carries the result. Never
use `pull_request_target` to work around this: it would expose your secrets to fork code.

## Dataset (JSONL)

One query per line:

```json
{"query_id": "refund-1", "vector": [0.01, -0.2, ...], "relevant": {"doc-42": 2, "doc-7": 1}, "filter": {"lang": "en"}, "tags": ["billing"]}
{"query_id": "refund-2", "text": "how do I get a refund?", "relevant": {"doc-42": 1}}
```

| Field | Required | Notes |
|---|---|---|
| `query_id` | yes | Unique, stable string. |
| `family_id` | no | Shared independent question family. Omit to use `query_id`. Permission variants inherit their source's family; a family cannot cross exploratory/held-out splits. |
| `vector` | one of `vector` / `text` | Query embedding from the same model as the corpus. |
| `text` | one of `vector` / `text` | Embedded at run time with `--embedder MODEL` (sentence-transformers). |
| `relevant` | no | Document id → grade (> 0 is relevant). Without it, a query only contributes ANN recall and latency. |
| `filter` | no | Portable filter: field → string / number / boolean (equality), a non-empty list (any of; up to 256 values) or a numeric range such as `{"gte": 2024, "lt": 2027}`. All fields are AND-ed. Lists and ranges need a full-pipeline adapter; vector connectors accept only equality. |
| `context` | no | Request context such as `{"tenant": "acme", "groups": ["support", "billing"]}`: scalars or lists of scalars, at most 64 fields and 32 KiB. Needs a full-pipeline adapter. |
| `history` | no | Earlier conversation turns, oldest first: `[{"role": "user", "content": "..."}]` (roles `system`, `user`, `assistant`; up to 64 turns and 64 KiB). The current question stays in `text`. Part of the dataset version; needs a full-pipeline adapter. |
| `tags` | no | Cohorts, judged separately when the policy has cohort rules. |

An optional first line `{"dataset": {"name": ..., "description": ..., "embedding_model": ...,
"normalized": ..., "cohort_fields": [...]}}` holds metadata; it is part of the version.

**Cohorts from context.** `"cohort_fields": ["tenant", "plan"]` turns those context fields
into cohorts: a question with `"context": {"tenant": "acme"}` also belongs to cohort
`tenant=acme` (booleans read `true`/`false`), judged by the policy's cohort rules exactly
like a tag. A change that helps large tenants and quietly hurts a small one then fails that
tenant's cohort instead of disappearing into the average. Declare up to 10 fields that hold
one value per question; a list-valued field such as `groups` is refused (tag those groups
explicitly). Derived cohorts are not written back into the benchmark. They are recorded in
run files and hosted uploads like tags, so declare only fields whose values you are willing
to share as cohort names. `vectraudit dataset stats` counts them with the other cohorts.

Building one:

```bash
# From a spreadsheet export: query_id, text, relevant ("doc-1:2; doc-7"), tags, filters
vectraudit dataset import questions.csv --out retrieval/dataset.jsonl \
  --tags-column segment --filter lang --filter year:int --name support-faq
# Add vectors with the corpus's embedding model (recorded in the header)
vectraudit dataset embed retrieval/dataset.jsonl --model BAAI/bge-small-en-v1.5 --normalize
vectraudit dataset stats retrieval/dataset.jsonl --policy retrieval/policy.json
vectraudit dataset diff old.jsonl retrieval/dataset.jsonl   # review label changes
```

### Bringing labels you already have

If your benchmark already lives in another tool's format, import it as is. The importers
only translate your labels; they never invent questions or judgments.

| Source | Command | What becomes the label |
|---|---|---|
| BEIR folder | `dataset import ./fiqa --format beir [--split test]` | `qrels/<split>.tsv` grades by corpus id |
| TREC | `dataset import topics.tsv --format trec --qrels qrels.txt` | `qid iter docno grade` lines; the query file is `qid<TAB>text` or JSONL |
| Ragas test set | `dataset import testset.jsonl --format ragas --corpus chunks.jsonl` | `reference_context_ids`, or `reference_contexts` located in the corpus |
| DeepEval goldens | `dataset import goldens.json --format deepeval --corpus chunks.jsonl` | `context` located in the corpus (`retrieval_context` is never a label) |
| LangSmith / Langfuse export | `dataset import export.json --id-column id --text-column inputs.question --relevant-column outputs.doc_ids` | Any list or `"id:grade; id"` field, by dotted path |

Ragas and DeepEval usually name reference passages by their text. `--corpus` is a JSONL,
JSON or CSV file of the documents your backend returns (`_id`/`doc_id`/`id` and
`text`/`content`/`page_content`, or `--corpus-id-field` / `--corpus-text-field`); each
passage is matched, ignoring whitespace differences, to the documents that contain it. A
passage that is not found, appears in more than `--max-matches` documents (default 5) or is
under 20 characters is not used, and the import reports how many. Queries with no located
passage are skipped, and a query with some unlocated passages keeps only the located ones,
so its labels may be incomplete. Use the corpus at the chunking the test set was built from.

Negative TREC grades are dropped, judged queries without a positive grade are skipped, and
the dataset header records the source format, file hashes and skip counts as `provenance`.

`stats` warns about too few labelled queries, unlabelled queries, cohorts below the
policy minimum, duplicate texts and missing vectors. `record` adds the dataset's embedding
model to the run's provenance and warns if `--embedder` differs from it.

The dataset **version** is the SHA-256 of its content (line order and formatting don't
matter). Runs on different versions are never compared. Label documents by stable ids that
survive re-chunking; if chunk ids change between baseline and candidate, map the labels to
each chunk version first, or the comparison is meaningless.

## Connections (JSON)

```json
{"backend": "faiss", "index": "ivf_pq.index", "vectors": "vectors.npy", "params": {"nprobe": 16}}
```

Relative paths resolve against the connection file. `params` are the default search
parameters (`--param key=value` overrides them). **Secrets never go in the file:** write
`"api_key_env": "QDRANT_API_KEY"` and export the variable; a literal `api_key`, `dsn`,
`password` or `token` is rejected.

| Backend | Fields | Tuning params | Exact reference |
|---|---|---|---|
| `faiss` | `index`, optional `vectors` (+ `ids`), `reconstructed_reference` | `nprobe` (IVF), `ef_search` (HNSW) | Raw `vectors`, or a flat index itself. `reconstructed_reference` decodes vectors from the index and is marked as a weaker reference. |
| `qdrant` (`vectraudit[qdrant]`) | `url` (or `path` for embedded local mode), `collection`, optional `api_key_env`, `vector_name` (required when the collection has several named vectors), `timeout_s` | `hnsw_ef` | Qdrant's exact search over the same collection. |
| `pgvector` (`vectraudit[pgvector]`) | `dsn_env`, `table` (`schema.table`), `id_column` (default `id`), `vector_column` (default `embedding`), `distance` (`cosine`, `l2`, `ip`: match your index's operator class), `filter_columns`, `statement_timeout_ms`, `connect_timeout_s` | `hnsw.ef_search`, `ivfflat.probes` | A sequential scan with index scans disabled for that query. |
| `pinecone` (no extra) | `index`, `api_key_env`, optional `namespace`, `host` (private endpoint), `control_plane_url`, `timeout_s` | none | None: Pinecone is measured as a black box (no exact mode, no snapshot). |
| `elasticsearch` or `opensearch` (no extra) | `url`, `index` (or alias), `vector_field`, optional `id_field` (default `_id`), `api_key_env` (an Elasticsearch API key) or `basic_auth_env` (a variable holding `user:password`), `timeout_s` | `num_candidates` (Elasticsearch), `ef_search` (OpenSearch 2.16+) | A brute-force `script_score` over the same documents, filters and distance. |
| `weaviate` (`vectraudit[weaviate]`) | `url`, `collection`, optional `api_key_env` (required for Weaviate Cloud), `id_property` (default: object UUID), `vector_name`, `tenant`, `grpc_url` (self-hosted), `timeout_s` | none (ef is collection configuration) | Only collections with a flat index. |

Qdrant filters map to `must` conditions (`match` for strings, integers and booleans; an equal-bounds `range` for floats). Qdrant has no immutable snapshot id, so reports warn that concurrent updates can't be ruled out: freeze the collection, or point both runs at a versioned collection or alias.

pgvector sessions are read-only (`default_transaction_read_only`) with a `statement_timeout`; identifiers are validated and quoted and every value is a bound parameter. Filters work only on columns listed in `filter_columns`. The evaluation role needs `SELECT` on the table and `USAGE` on its schema, nothing more; the connector checks this before searching. `describe` lists the vector indexes and whether one matches the configured distance (a mismatch means PostgreSQL scans every row).

Pinecone is reached over its REST API with a pinned API version and portable filters become `$eq` conditions; Weaviate filters become property equality. Both send API keys only over HTTPS.

Elasticsearch 8 and OpenSearch 2 are reached over their REST APIs, so there's no client
library to install. Which engine you have is detected from the cluster (`GET /`), whichever
name the connection uses, and recorded in the run.

- **Search:** Elasticsearch uses the `knn` search option on a `dense_vector` field
  (`num_candidates` is never set below k). OpenSearch uses the `knn` query on a
  `knn_vector` field.
- **Filters:** become `term` clauses, so use them on `keyword`, numeric or boolean fields.
- **Exact reference:** scores every matching document with the field's own similarity
  (`cosine`, `dot_product`, `max_inner_product`, `l2_norm`; OpenSearch `cosinesimil`,
  `innerproduct`, `l2`). An Elasticsearch field mapped with `index: false` supports only
  the exact search.
- **Credentials:** sent only over HTTPS (or to a local cluster).
- **No snapshot ID:** freeze the index, or point both runs at a versioned index or alias.

Each connector declares its capabilities (filters, exact reference, configuration,
tuning parameters). An unsupported request, such as a filter on FAISS, is a per-query error,
never an empty result.

## Metrics

- **Semantic** (needs `relevant` labels): recall@k, precision@k, MRR@k, nDCG@k with linear
  gains (grade / max grade of the query). Unjudged results count as non-relevant; duplicate
  results get no extra credit.
- **ANN recall@k** (needs an exact reference): overlap with an exact search over the same
  corpus and distance. It measures approximation loss, **not** whether the results answer the
  query, so it can't choose between embedding models.
- **Latency**: per-query median of `--repeats` timed searches after `--warmup` untimed ones,
  measured around the connector call on the runner. p50/p95 are compared only between runs
  from matching runner environments (OS, CPU architecture and count, Python), unless the
  policy says otherwise. Embedding time is recorded separately.

Changes are paired per query, averaged within each declared family, then averaged with equal
weight across families. Seeded 95% bootstrap intervals resample these family means. With fewer
than 30 families the report warns about limited evidence. One family's interval cannot estimate
between-family uncertainty, even if it has many scenario rows.

## Policy (JSON)

```json
{
  "version": 1,
  "metrics": {"ndcg": {"max_drop": 0.02}, "recall": {"max_drop": 0.02}, "ann_recall": {"min": 0.9}},
  "latency": {"p95_max_increase_pct": 25, "p95_max_ms": 80},
  "min_queries": 50,
  "max_regressed_queries": 10,
  "primary_metric": "ndcg",
  "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.05}}},
  "require_significance": false
}
```

- `max_drop` is absolute (0.02 = two points); `min` is a floor for the candidate.
- Every metric a policy names must be measured in both runs, or the check is not comparable.
- `latency.min_increase_ms` (default 2): percentage latency rules only fail when the p50 /
  p95 rise is also larger than this in absolute terms, so scheduler jitter on shared CI
  runners (about 0.1 ms on local indexes, 1.5 ms against networked backends) is not a
  regression. Lower it, or set 0 to judge every rise, on dedicated hardware.
- `require_significance: true` fails a `max_drop` rule only when the whole 95% interval of
  the change is beyond the limit; useful for small or noisy query sets. Not detecting harm
  this way is not evidence that the change is within tolerance.
- `non_inferiority: true` is the opposite stance: a suite-level `max_drop` rule passes only
  when the whole 95% interval of the change stays above `-max_drop` (shown within
  tolerance). There are three outcomes:
  - **within tolerance:** passes.
  - **inconclusive:** the interval reaches past the allowed drop; the check **fails**, and
    the reason says so ("not shown to be within tolerance (inconclusive)"). More query
    families narrow the interval.
  - **beyond tolerance:** the whole interval lies past the allowed drop.

  Both failures use the same rule key (`metric:<m>.max_drop`), so an audited exception can
  accept an inconclusive result deliberately. Reports add a "Non-inferiority" line, each
  metric gains `tolerance.decision`, and the lead summary says "shown within tolerance"
  instead of "not proof of safety". Cohort rules keep their own semantics (see below). The
  optimizer does not exclude a candidate whose exploratory screen is only inconclusive;
  the held-out check decides. `vectraudit calibrate` sizes the margin from the A/A
  interval's lower bound under this option.

  `require_significance` and `non_inferiority` cannot both be set. Policies without
  `non_inferiority` keep their hashes and meaning. In the workbench, choose "Judge allowed
  drops by: shown within tolerance".
- **Many cohorts.** Each tag with enough families is a cohort, judged on its own. With many
  cohorts, some fail by chance even when nothing changed: in a 20-cohort A/A test, every
  comparison raised at least one chance cohort failure.
  - **What each check reports:** how many cohorts and cohort rules it judged and how many
    failed (the "Cohorts:" line of reports, `cohort_judging` in check JSON, and the hosted
    decision's summary). From 5 cohort rules judged without control it warns.
  - **`cohorts.control: "bonferroni"`:** a cohort `max_drop` fails only when its interval,
    widened to 1 − 0.05/m for the m cohort `max_drop` rules judged, lies entirely beyond the
    allowed drop. The interval is a normal approximation from the family-level standard
    error, slightly narrow for very small cohorts.
  - **Effect:** chance failures fell from every A/A comparison to at most about one in 20,
    while a real 0.4 drop in one cohort still failed.
  - **Unchanged by the control:** `min` floors, the suite-level metrics and the rule keys
    that audited exceptions name.
  - **Choosing:** run `vectraudit calibrate` with and without the control to see the
    difference on your own runner. Leaving the control out keeps existing policy hashes.
  - **In the workbench:** the policy editor's "Correct for many cohorts (Bonferroni)"
    checkbox sets the control.
- Without `--policy`, a built-in policy allows drops of 0.01 in nDCG, recall and ANN recall
  (0.02 in MRR) and a 50% p95 rise, judged on whatever both runs measured. The workbench
  labels this mode *exploratory*; it is not a substitute for an agreed release policy.
- The workbench edits, versions and applies the same policy files
  (`vectraudit workbench serve DIR --policy retrieval/policy.json`). From a passing
  comparison it exports a release configuration that `vectraudit release validate | run |
  check | update` runs in CI under the target branch's approved policy. See
  [Release policies and exported release checks](retrieval-workbench.md#release-policies-and-exported-release-checks).

## Calibrate the policy (A/A)

A threshold tighter than your system's normal run-to-run noise fails releases that changed
nothing, and a gate that cries wolf gets switched off. Measure the noise instead of guessing:
record the **same, unchanged** setup several times (5 or more) on the runner and endpoint
CI uses, then calibrate the policy against those recordings:

```bash
for i in 1 2 3 4 5; do
  vectraudit record --connection main.json --dataset dataset.jsonl --out aa/run-$i.json --label aa-$i
done
vectraudit calibrate aa/run-*.json --policy policy.json --draft policy.calibrated.json --json calibration.json
```

Every recording is compared with every other in both directions (5 runs give 20
comparisons). Nothing changed between them, so each failure is a false alarm, reported by its
rule key. The draft policy only raises noise-driven thresholds to the worst observed
variation times `--margin` (default 1.5), rounded up: metric and cohort `max_drop`,
`latency.min_increase_ms` (for percentage latency rules) and `max_regressed_queries`. The
report shows false alarms under the current policy and under the draft.

Calibration never loosens a requirement. If an unchanged system misses a critical question,
delivers a forbidden document, fails a no-answer assertion, or breaks an absolute floor
(`min`, `p95_max_ms`), it is listed as **not noise**: retrieval is intermittent or already
out of budget, so investigate it. The draft is a proposal: review it, then adopt it like any
policy change. `calibrate` never overwrites a file. Few runs underestimate rare noise, and
comparisons reuse the same runs, so a low false-alarm count is evidence for that setup, not
a guarantee; re-run after infrastructure changes. Recording costs the same as `record`:
runs × queries × repeats calls to your backend.

If the recordings differ in configuration (search parameters, backend settings, pipeline
version), the report lists them in `config_differences` and says it is not an A/A
comparison. The false alarms may then be real changes rather than noise.

**In the workbench.** Under the release policy picker, open **Calibrate this policy against
unchanged runs (A/A)** and choose 2–10 complete recordings of the same benchmark.
- **The result:** false alarms by rule, the failures that are not noise, each proposed
  change with the noise it was measured from, and the same caveats as the CLI.
- **Different configurations:** a set mixing configurations is called out.
- **Saving:** measuring writes nothing. **Save draft as a new version** saves the draft
  through the normal policy versioning, with a note recording the before-and-after false
  alarms, and selects it. The previous version stays available.

## Audited exceptions

Sometimes a team deliberately ships a known regression. Instead of loosening the policy or
bypassing the required check, an approver accepts the specific failing rule, with a reason,
an owner who follows up, and an expiry. The check then reads **fail, accepted**, never pass.

Every failing reason carries a stable rule key: `metric:ndcg.max_drop`, `metric:recall.min`,
`cohort:billing:ndcg.max_drop`, `regressed_queries`, `latency.p95_max_ms`,
`latency.p95_increase`, `critical:<query>:<document>`, `safety.forbidden:<query>` and
`safety.no_answer:<query>`. An exception names one key and covers exactly that failure; any
other failure in the same check still blocks. Errors and incomparable runs are never
accepted.

The policy decides what can be accepted. Without an `exceptions` block, nothing can:

```json
"exceptions": {"allow": ["metric", "cohort", "latency"], "approver": "engineer",
               "safety": "never", "max_days": 30}
```

- `allow`: rule kinds open to exceptions (`metric`, `cohort`, `regressed_queries`,
  `latency`, `critical`).
- `approver`: least role that may accept them (`engineer` or `owner`; default `owner`).
- `safety`: `never` (default) or `owner`, for forbidden-document and no-answer failures.
  An engineer can never accept a safety failure.
- `max_days`: longest allowed expiry (default 30, at most 365).

Adding the block does not change the hash of policies without one. Exceptions are granted
and stored in the hosted service, where they are audited; see
[cloud operations](cloud-operations.md#audited-exceptions). `check cloud:<name>` applies
the project's active exceptions, so its exit code matches the hosted check. Local file
checks and the workbench do not apply exceptions. The workbench shows them: each failing
rule of a comparison says who may accept it under that policy, and on request the
connected hosted project's exceptions are listed against those rules (see
[the workbench guide](retrieval-workbench.md#failing-rules-and-audited-exceptions)).

## Evidence and diagnosis

`record` asks each backend for per-query evidence (turn it off with `--no-evidence`):

- **FAISS IVF**: for every exact neighbour the search missed, whether its inverted list was
  not probed or it was ranked out by compressed distances, and the nprobe that would find
  them all. Replayed from the coarse quantizer's probe order, and labelled as a replay.
- **pgvector**: the planner's plan (scan types, index used, cost) from `EXPLAIN` without
  `ANALYZE`, so nothing extra executes.

`check` turns the comparison and the evidence into findings, each with a level:

| Level | Meaning |
|---|---|
| verified | Measured in these runs: a metric change, evidence counts, plan changes. |
| supported | Consistent with a measured change, e.g. a configuration difference that coincides with the effect. A likely cause, not proof. |
| hypothesis | Worth checking; nothing in the runs confirms it. |

Examples: approximation loss and whether raising nprobe would recover it; relevance falling
while ANN recall holds (an embedding, chunking or corpus change, which index tuning can't
fix); plans that stopped using the vector index; filtered queries returning fewer than k
results; regressions concentrated in one cohort; latency moving with search parameters;
nondeterministic results.

## Reports

`--html` writes a self-contained report (no scripts or external assets): verdict and
reasons, both runs' configuration and provenance, quality with intervals, latency,
regressed queries with the ids they lost and gained, cohorts and the configuration diff.
`--json` writes the same result as data; `--markdown` a compact summary.

## Run files

`record` writes a JSON run file even when the run fails, so `check` can report the failure.
It contains the dataset version, backend description and capabilities, search parameters,
timing protocol, metric contract, provenance (commit SHA, CI run URL, runner environment,
`--tag key=value` labels such as embedding model or chunking version) and per-query ranked
ids, latency, metrics and errors.


Policy `min_queries` counts independent query families, both for the suite and for the paired
measurements of each judged quality metric. Forty permission scenarios from one question count
as one family, not forty; a large suite with only two labelled families cannot satisfy a minimum
of 30 for nDCG. `check` returns `not_comparable` and reports both rows and families. Cohort
thresholds also count measured families within each tag; undersized cohorts remain unjudged.

Per-query relevance metrics are unchanged. Run-level relevance, ANN recall and no-answer
accuracy average measured rows within each family, then weight families equally. Latency,
execution/error counts and the explicit `max_regressed_queries` cap remain per-row measures.
Permission, no-answer and critical-document assertions still inspect individual cases and
executions; a passing family average cannot hide one of these failures.

`family_id` is included in the dataset hash and retained in runs, checkpoints and label exports.
Generated permission scenarios inherit `family_id` from their source (or the source query ID).
A family cannot cross exploratory and held-out splits. Comparisons reject changed family
assignments. Old datasets keep their hashes and old run files load normally; missing family IDs
mean independent rows. Previously generated permission rows cannot be retrospectively grouped
from their opaque IDs: regenerate them, or explicitly annotate their source families and rerun
both sides. Saved run artifacts are never silently rewritten.
Use the family-aware checker for both new runs; older checker binaries do not understand
the optional family metadata and may still weight rows independently.

This grouping does not correct multiple-cohort false alarms or prove that declared families
are independent. A/A calibration and multiplicity control remain separate work.

## Critical-question assertions

Use `critical_queries` in a release policy to require a named document within an inclusive
rank for a particular question. For example, this policy blocks a release unless the
candidate retrieves `current-guide` in the top three results for `invoice`:

```json
{
  "version": 1,
  "metrics": {"ndcg": {"max_drop": 0.02}},
  "critical_queries": [
    {"query_id": "invoice", "doc_id": "current-guide", "max_rank": 3}
  ]
}
```

The workbench's release policy editor provides **Add critical assertion** and preserves
these rules through JSON import, policy versions and release-check export. The browser,
CLI, hosted checks and CI use the same comparison engine. JSON results include each rule's
verdict, the worst observed rank, the number of executions checked, and up to ten failing
executions with their repeat (and attempt, for retried queries) and observed ranking. HTML and
Markdown reports include passed and failed assertions and a table of failing executions.

### Required evidence must reach the model

A document can rank first and still never reach the model, or reach it with the clause that
mattered trimmed away. `critical_context` protects a question against that:

```json
{
  "version": 1,
  "metrics": {"ndcg": {"max_drop": 0.02}},
  "critical_context": [{"query_id": "account-recovery"}]
}
```

The question's `required_evidence` (declared in the benchmark) must reach the model in full,
as recorded by the pipeline's model-context evidence, on **every** execution. A demonstrated
loss on any execution fails (rule `critical_context:<question>`, which exceptions treat as a
critical rule). Context that was not recorded, or whose completeness is unknown, is not
comparable, never a pass; so is a question without `required_evidence`, or a run recorded
before per-execution sufficiency existed. Uploads carry only each execution's sufficiency
states, so hosted checks reach the same verdict. Policies without the rule keep their hashes.

The workbench's policy editor edits these assertions (**Add context assertion**) and keeps
them through every save.

### Where forbidden documents travelled

A permission check that inspects only the final results misses two leaks: a forbidden
document handed to the model without being shown, and one sent to a stage that should never
see it, such as a third-party reranker running before access control. For every question
with `forbidden` documents, runs record a `permission_path`: each forbidden document seen in
any execution, the stages it reached (as a stage's output, or as its input through a parent
stage), and whether it reached the results or the model context. Forbidden documents match by
result ID or by the hit's source ID, so a forbidden source is caught through its chunks.

- **In the model context:** always fails (rule `safety.forbidden:<question>`, like a
  delivered forbidden document), even when the results are clean. Context that was not
  recorded in full is reported as unknown, not as clean.
- **Upstream stages:** a policy names the stage kinds a forbidden document must never reach:

  ```json
  {"version": 1, "metrics": {"ndcg": {"max_drop": 0.02}}, "forbidden_stages": ["rerank"]}
  ```

  Kinds are those of stage traces: `keyword`, `vector`, `filter`, `fusion`, `rerank`,
  `truncate` and `custom`. Reaching one fails with the same safety rule. A run whose trace
  does not show stages for a permission-checked question is not comparable under such a
  policy, never a pass. Policies without it keep their hashes, and the workbench editor keeps
  it through every save.

Uploads carry document IDs, stage kinds and the two flags (not stage names), so hosted checks
reach the same verdict. As with every permission check, a passing suite is evidence about the
scenarios tested, not proof that authorization is correct. To test access that changes
over time (revocations, moves, archives) see [permission lifecycle](permission-lifecycle.md).

### Every repeat counts

Runs record one ranking per execution (`repeat_ids`), and a critical document must meet its
rank on **every** repeat, not just the first. A document ranked #2 on the first execution
that disappears on the second fails the check. For checkpointed pipeline evaluations,
rankings returned before a query failed are kept in its attempt history, and a retried query
is judged on those rankings as well as the retry's, so a retry cannot hide a bad observation.

Runs recorded before this check existed carry no per-repeat rankings, so a critical
assertion against them is not comparable rather than passing; re-run the candidate. The same
applies when fewer rankings are recorded than the run's `timing.repeats`, or when the first
per-repeat ranking does not match the recorded ranking.

These are absolute requirements on the candidate, even if the baseline also fails, average
quality improves or `require_significance` is enabled. A missing document or rank beyond the
limit fails. A question absent from the measured split, or `max_rank` greater than the run's
`k`, makes the check not comparable. Incomplete runs remain errors. Select a policy whose
question IDs belong to the exact exploratory or held-out split being checked.

An assertion-only policy may omit `metrics`; it needs no relevance grades but still obeys
`min_queries` and the existing run-comparability checks. This does not establish quality
for other questions. Each query/document pair must be unique; a policy supports at most
1,000 assertions, with maximum ranks from 1 to 10,000.

Assertions evaluate the **recorded candidate ranking**. For repeated pipeline calls this is
the first result, matching current relevance metrics; they do not promise that every repeat
retrieves the document. Existing nondeterminism warnings still apply. Policies without
assertions retain their existing content hashes.
