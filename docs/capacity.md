# Capacity curve

`vectraudit capacity` shows how a backend behaves as concurrent load grows: throughput,
p50/p95/p99 latency, the error rate and the recall of a sample of answers, per concurrency
level. It is a probe of a **staging** target to answer "does this still hold at 16 concurrent
requests?", not a load-testing tool: every limit errs towards sending less.

```bash
vectraudit capacity --connection retrieval/connection.json --dataset dataset.jsonl \
  --concurrency 1,4,16 --duration 30 --max-rps 50 \
  --confirm-target my-staging-collection --out capacity.json
```

## What it sends, and when it refuses

It sends read-only searches taken from your dataset (with their filters, if any), repeated.
Before sending anything it prints the plan and the most it can send: the serial queries, each
concurrency level for `--duration` seconds, and the exact-reference lookups for the recall
sample, all under one global `--max-rps` ceiling.

| Limit | Default | Bound |
|---|---|---|
| `--duration` per level | 30 s | 1 to 600 s |
| `--max-rps`, all workers together | 50 | up to 1000 |
| `--concurrency` levels | `1,4,16` | 1 to 8 levels, each 1 to 32 |

It will not start when:

- `--confirm-target` is missing or differs from the connection's target (the collection for
  Qdrant and Weaviate and Chroma, the index for FAISS, Pinecone and Elasticsearch, the table
  for pgvector). A command copied from another project cannot hit a collection it was not
  written for.
- the target or the connection's `name` contains `prod`, unless you pass `--allow-production`.
- the dataset holds permission contexts, conversation history, or list or range filters (those
  need a full pipeline).

An authentication or configuration error stops the run at the first rejected request. Other
errors (timeouts, query errors) are counted, not fatal: a rising error rate is the finding.
Ctrl-C ends the run and keeps the levels measured so far (exit code 130).

## What the report says

`vectraudit.capacity` v1 holds the plan, the target you confirmed, the runner environment, the
serial baseline and one row per level:

- `achieved_qps`: successful searches per second; `attempts_per_s` includes failures.
- `p50_ms`, `p95_ms`, `p99_ms`: over successful searches; `null` when none succeeded.
- `error_rate` and `errors` by connector error code.
- `recall`: the answers to a fixed sample of queries are kept during the level and compared,
  afterwards and not under load, with the exact reference. `null` where the backend has no
  exact reference. It shows whether quality holds up under load, not general recall.
- `at_ceiling`: the level reached 90% of `--max-rps`. The backend kept up, so its limit is
  *at least* that high, not exactly that. Raise `--max-rps` deliberately to look further.

Latency is measured by this process around the connector call, so it includes the network and
the client library and depends on the machine. Backends whose connector is not thread safe
(pgvector, Qdrant, Weaviate, Pinecone, Chroma) get one connection per worker; FAISS and
Elasticsearch share one.

## Comparing two reports

`--baseline earlier.json` adds a `comparison`. Two reports compare only when both completed and
they match on runner environment (`system`, `machine`, `cpu_count`, `python`: the rule the
release comparison applies to latency), dataset, backend, `k`, search parameters and
`max_rps`. Otherwise `comparable` is false and the reasons are listed; no deltas are given and
nothing is judged. There is no pass or fail threshold: this report describes a curve, and a
regression limit would need a capacity target you set.

The Checks tab reads the report; so does `vectraudit.selectivity_sweep`, the JSON of
`vectraudit sweep --filter ... --json` ([filtered search](filtered-search.md)).
