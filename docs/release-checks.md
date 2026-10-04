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
[examples/github/retrieval-check.yml](../examples/github/retrieval-check.yml). `check`
appends a Markdown summary to the job summary automatically.

## Dataset (JSONL)

One query per line:

```json
{"query_id": "refund-1", "vector": [0.01, -0.2, ...], "relevant": {"doc-42": 2, "doc-7": 1}, "filter": {"lang": "en"}, "tags": ["billing"]}
{"query_id": "refund-2", "text": "how do I get a refund?", "relevant": {"doc-42": 1}}
```

| Field | Required | Notes |
|---|---|---|
| `query_id` | yes | Unique, stable string. |
| `vector` | one of `vector` / `text` | Query embedding from the same model as the corpus. |
| `text` | one of `vector` / `text` | Embedded at run time with `--embedder MODEL` (sentence-transformers). |
| `relevant` | no | Document id → grade (> 0 is relevant). Without it, a query only contributes ANN recall and latency. |
| `filter` | no | Portable filter: field → string / number / boolean, all AND-ed. |
| `tags` | no | Cohorts, judged separately when the policy has cohort rules. |

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

Qdrant filters map to `must` conditions (`match` for strings, integers and booleans; an equal-bounds `range` for floats). Qdrant has no immutable snapshot id, so reports warn that concurrent updates can't be ruled out: freeze the collection, or point both runs at a versioned collection or alias.

pgvector sessions are read-only (`default_transaction_read_only`) with a `statement_timeout`; identifiers are validated and quoted and every value is a bound parameter. Filters work only on columns listed in `filter_columns`. The evaluation role needs `SELECT` on the table and `USAGE` on its schema, nothing more; the connector checks this before searching. `describe` lists the vector indexes and whether one matches the configured distance (a mismatch means PostgreSQL scans every row).

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

Changes are paired per query with seeded bootstrap 95% intervals. With fewer than 30 queries
the report warns that one query moves the mean noticeably.

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
- `require_significance: true` fails a `max_drop` rule only when the whole 95% interval of
  the change is beyond the limit; useful for small or noisy query sets.
- Without `--policy`, a built-in policy allows drops of 0.01 in nDCG, recall and ANN recall
  (0.02 in MRR) and a 50% p95 rise, judged on whatever both runs measured.

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
