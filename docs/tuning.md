# Tuning a search setting on any backend

Most vector backends have a setting that trades recall for speed at query time. `vectraudit tune`
runs the same queries at each value of one such setting, measures ANN recall against one exact
reference and client-side latency, and recommends the smallest value that reaches your recall
target. It is the FAISS debugger's recall-against-latency sweep, for the other backends.

```bash
vectraudit tune --connection connection.json --dataset queries.jsonl \
  --param hnsw_ef --values 16,32,64,128,256 --confirm-target docs \
  --target 0.95 --max-p95-ms 50 --out tune.json
```

## What can be tuned

| Backend | `--param` | Exact reference |
|---|---|---|
| Qdrant | `hnsw_ef` (per request) | its own exact search (`exact: true`) |
| pgvector | `hnsw.ef_search`, `ivfflat.probes` (set for one transaction) | an exact scan |
| Elasticsearch | `num_candidates` (per shard, at most 10,000, at least `k`) | exact kNN |
| OpenSearch | `ef_search` (`method_parameters`; `faiss` and `lucene` engines) | exact kNN |
| FAISS | `nprobe`, `efSearch` | raw vectors, or the index itself |
| Weaviate | none: `ef` is part of the collection's configuration | flat indexes only |
| Chroma | none: `ef_search` is changed by modifying the collection | yes |
| Pinecone | none | none: use `--reference` |

A backend with no parameter here is not tuned: changing a collection's configuration is a
write, and Vectraudit's connectors only read. Measure recall at its current setting with
`vectraudit record`. The parameters were checked against each backend's documentation on
2026-11-16; see `docs/DECISIONS.md`.

## What it measures, and what it will not say

- **Recall** is ANN recall at `--k` against one stated exact reference: the backend's own exact
  search, run once per query, or an external reference (`--reference`, for example from
  `dataset import-annb`). Without either the sweep refuses. Each value gets a 95% interval,
  the per-query distribution and its worst queries.
- **Latency** is the client's wall clock for one search call, network round trip included, one
  query at a time. It is comparable between the values of one sweep. It is not comparable with
  another backend's numbers or with the FAISS debugger's in-process timings.
- **The recommendation** is the smallest swept value whose mean recall meets `--target`, within
  `--max-p95-ms` when given. If none does, the report says whether recall or the latency
  budget was the limit, and the command exits 2.
- **Missed neighbours.** For the lowest-recall queries at the recommended value, the report
  lists each true neighbour that was missed and the smallest swept value that returns it.
  `--query QUERY_ID` prints one question at every value. **No cause is given**: a database
  does not expose why its index skipped a vector. For causes (cell not probed, quantization),
  the [FAISS debugger](faiss-debugger.md) reads the index itself.

## Filters, batch throughput and comparing two setups

```bash
vectraudit tune --connection connection.json --dataset queries.jsonl --param hnsw_ef \
  --values 16,64,256 --confirm-target docs --filter lang=de --filter 'tier=pro;public=true' \
  --batch --out tune.json
vectraudit tune-compare old.tune.json new.tune.json --json comparison.json
```

- **`--filter`** (repeatable) repeats the sweep with that filter on every query. Each filter
  gets its own curve, recommendation and exact neighbours (computed among the documents the
  filter allows), and its pass rate, counted by the backend with the same condition. A backend
  that cannot count reports the pass rate as unknown, never a guess. Filters need the
  backend's own exact search, so they cannot be combined with `--reference`. Conditions are
  scalar `key=value` pairs. The FAISS debugger's synthetic `--pass-rates` filters are not
  offered here: they would need a field written into the collection.
- **`--batch`** also sends every query in one batch call per value (FAISS and Qdrant's batch
  query API; other backends are refused) and reports queries per second, best of the repeats.
  It is a throughput, kept in its own field and never compared with the latency columns. A
  batch is a burst: the rate ceiling paces single searches, not the inside of a batch.
- **`tune-compare`** puts two sweeps side by side: two collections built differently, two
  index setups, or two backends. Recall is compared at matched values only when both sweeps
  used the same questions, the same `k` and the same kind of exact reference; otherwise the
  result is "not comparable" (exit 4) with the reasons. Latency differences are shown only
  when both sides are the same backend. It also lists the questions that miss neighbours in
  one sweep and not the other, among each sweep's worst questions.

Comparing two collections is also how to judge build-time settings (`m`, `ef_construction`,
number of lists): build both yourself, sweep each, compare. Vectraudit never rebuilds an index.

## Guard the recommendation in release checks

`--write-connection` writes a copy of the connection with the recommended value as a default
search parameter. The connection file you swept is never changed, so it stays the baseline:

```bash
vectraudit tune --connection retrieval/connection.json --dataset queries.jsonl \
  --param hnsw_ef --values 16,32,64,128 --confirm-target docs --out tune.json \
  --write-connection retrieval/tuned.json
vectraudit record --connection retrieval/connection.json --dataset queries.jsonl --out base.json
vectraudit record --connection retrieval/tuned.json --dataset queries.jsonl --out tuned.json
vectraudit check base.json tuned.json --policy retrieval/policy.json
```

The check judges the tuned setting under your release policy (for example an `ann_recall`
floor) and reports its latency change beside the verdict. Relative file paths in the copy
are rewritten so they still name the same files from its new folder. Nothing is written when
the sweep recommends no value, and the command refuses a `--write-connection` path that is
the connection file itself.

## In the workbench

Open a `tune.json` report, or a `tune-compare --json` result, from the workbench's artifact
list: the **Checks** tab shows the recommendation, every swept value with its recall interval
and latency, the questions that missed neighbours, and each filtered curve. A sweep that
reached no value, was cancelled or failed is never shown as a pass. A comparison is flagged
when the second setup loses recall at a shared value. There is no web tuner for backends: run
the sweep from the command line, where it is bounded and confirmed.

## Bounds

A sweep only sends searches, and it is bounded before the first one:

- it prints the number of searches and the time they will take;
- `--confirm-target` must repeat the collection, index or table name, so a copied command
  cannot search another one, and a target named like production needs `--allow-production`;
- `--max-searches` (default 100,000) refuses a larger sweep;
- `--max-rps` (default 50) paces every search; the wait is not counted as latency.

An interrupted or failed sweep writes a report marked `cancelled` or `failed` and recommends
nothing.
