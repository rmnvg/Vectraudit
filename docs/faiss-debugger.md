# FAISS index debugger reference

The FAISS debugger (`vectraudit serve`, `vectraudit.launch`) inspects one index in depth:
missed neighbours against exact ground truth, quantization loss, list balance, HNSW search
traces and recall-versus-latency tuning. The [README](../README.md#faiss-index-debugger)
gives the overview; this page holds the inputs, command-line options and measurement details.

## Inputs

| Input | Required | Formats | Used for |
|---|---|---|---|
| FAISS index | yes | index file or in-memory `faiss.Index` | everything |
| Raw vectors | recommended | `.npy` float32, n × d (row *i* = id *i*, or use `--ids`) | exact ground truth, quantization error |
| Ids | optional | `.npy` int64 | when row order isn't the id |
| Metadata | optional | `.jsonl` / `.csv` / `.parquet` with an `id` column (plus e.g. `text`, `title`) | reading results as text |
| Embedder | optional | sentence-transformers model name, or a Python callable | text queries |
| Query vectors | optional | `.npy` | sweeps on your own queries |

Without raw vectors, vectraudit reconstructs them from the index; the UI then warns that
ground truth is computed on reconstructed vectors, so PQ/SQ error isn't measured.

## Labelled relevance and reranking

Use held-out query vectors and human relevance judgements to measure whether retrieved
chunks answer the question. Create a JSONL file with one entry per query row:

```json
{"row": 0, "relevant": {"42": 2, "77": 1}}
{"row": 1, "relevant": {"105": 1}}
```

Chunk ids are decimal strings; grades are finite non-negative numbers. Every query needs
at least one positive judgement, and every labelled chunk must exist in the corpus.
Unjudged chunks count as non-relevant. Recall@k uses all positively labelled chunks as its
denominator; MRR@k uses the first relevant hit; nDCG@k uses **linear graded gains**.
The summary is the arithmetic mean across queries, so each query has equal weight.
Incomplete judgements can underestimate relevance. These metrics are separate from ANN recall.

```bash
vectraudit evaluate index.faiss --vectors vectors.npy --queries queries.npy \
  --labels judgements.jsonl --k 10 --candidates 100 --mmap --json > relevance.json
```

`--candidates` additionally retrieves a larger candidate pool and reranks it using exact
raw-vector distances. JSON includes both sets of metrics and per-query results; ids are
strings to preserve int64 precision. Timings are single-query observations, not benchmark
estimates. The Python equivalents are `core.load_judgements`, `core.evaluate_relevance`,
and `core.relevance_metrics`.

In **Query explorer**, set **Rerank candidates** to at least k to compare the original
search with candidate retrieval plus exact reranking. Raw vectors are required. The
experiment shows both result tables, recall and latency; settings survive shared links,
and results can be exported. Reranking cannot recover neighbours outside the candidate set.
Python users can call `viewer.session.query(id=42, k=10, candidates=100)`.

The same evaluation runs in the UI: start with `--vectors` and `--queries`, open
**Relevance**, and load (or paste) the judgements JSONL. It shows recall, MRR and nDCG@k
before and after optional exact reranking, every query's scores (worst first, or most
helped by reranking) with a link that explains the query in the Query Explorer, and CSV or
JSON exports. The web view takes up to 10,000 queries and 5 MB of judgements; use
`vectraudit evaluate` for more.

**Run history** keeps the newest 100 finished sweeps, index comparisons and relevance
evaluations in `~/.cache/vectraudit/history/` (or `$VECTRAUDIT_CACHE_DIR/history/`), so they
survive restarts. Rename, download or delete them, or compare two saved sweeps with the same
query-set and corpus checks as `vectraudit runs diff`. Saved sweeps record the decision for a
0.95 recall target; use **Save run** in the Tuner for your own target and latency budget.
`vectraudit cache clear` leaves history alone. It is off in `--demo-mode` and for sessions
created with `disk_cache=False`.

The Overview's **Diagnose my index** guide connects input checks, representative queries,
tuning, and baseline verification.

## Command line

```text
vectraudit serve INDEX [--vectors v.npy] [--ids ids.npy] [--meta chunks.jsonl]
                      [--embedder all-MiniLM-L6-v2] [--queries q.npy]
                      [--host 127.0.0.1] [--port 8765] [--no-browser] [--max-points 50000]
                      [--compare other.index ...] [--mmap]
vectraudit info INDEX                          # kind, wrappers, parameters
vectraudit sweep INDEX --vectors v.npy [--param nprobe] [--target 0.95]
                      [--max-p95-ms 10] [--repeats 3] [--seed 0] [--json]
                      [--save run.json] [--baseline earlier.json]
vectraudit runs diff EARLIER.json LATER.json   # regressions between saved runs
vectraudit compare LEFT RIGHT --vectors v.npy [--queries q.npy] [--ids ids.npy]
                      [--left-nprobe 4] [--right-ef-search 64] [--json]
vectraudit evaluate INDEX --vectors v.npy --queries q.npy --labels judgements.jsonl
                      [--k 10] [--candidates 100] [--json]   # labelled relevance
vectraudit demo [--index ivf_pq|ivf_flat|hnsw]
vectraudit cache info | clear [--older-than-days 30] [--max-mb 500]
```

`vectraudit sweep` exits with code 2 when no value reaches `--target` (within the p95 budget
of `--max-p95-ms`, if given), 3 when `--baseline` shows a regression, and 4 when `--baseline`
couldn't be compared at all (different query sets, no overlapping values, ...) &mdash; that is
not a pass, and CI should treat it as a failure, not silently skip the check. `vectraudit runs
diff` uses the same three failure codes.

### Saved runs and baselines

`vectraudit sweep --save run.json` (or **Save run** in the Tuner) keeps a sweep as JSON: the
index's sha1, parameters and distance metric, the query set's fingerprint, a fingerprint of
the raw `--vectors` corpus ground truth was computed on, the settings, the environment,
every measurement, the recommendation for your target, and the worst queries. Compare a
later sweep with it using `--baseline run.json`, the Tuner's **Compare with a saved run**, or
`vectraudit runs diff`. A setting regresses when its recall drops by more than
`--max-recall-drop` (0.01), or when its p95 grows by more than `--max-p95-increase` (20%)
*and* by more than `--min-p95-increase-ms` (0.05 ms): sub-millisecond timings vary between
runs, and the floor keeps that noise from failing CI.

Recall is compared only when both runs used the same k, parameter, query set, metric and
ground-truth corpus; latency only when both were measured with the same FAISS version and
machine. Otherwise the comparison says why, and reports itself as not comparable rather than
reporting a false "no regressions". A saved run is validated on load and before it's written:
every recall and latency number must be finite and in range, point values must be unique, and
the identifying fingerprints must be present, so a corrupted or hand-edited file is rejected
rather than silently compared against.

```bash
vectraudit sweep index.faiss --vectors v.npy --queries eval.npy --save baseline.json
# ... rebuild or retrain the index ...
vectraudit sweep new.faiss --vectors v.npy --queries eval.npy --baseline baseline.json
```

## Reproducible tuning and index comparison

Sweeps warm up each setting, then time each query three times on one FAISS thread.
Use `--repeats` and `--seed` (also available in the Tuner) to control the measurement.
JSON exports include the query-set SHA-256, sampling seed, timing seed, repetition count,
and Python/NumPy/FAISS and platform versions. Supplied `--queries` use the first
`--n-queries` rows; the seed controls sampling only when queries are drawn from stored vectors.

Each setting also reports an approximate 95% interval for mean recall, the exact per-query
recall distribution (so "how many queries fall below the target" is known, not just the
mean) and its lowest-recall queries. The recommendation is the *smallest* value meeting the
target; when a larger value happens to measure faster, it is reported separately as the
fastest measured setting, since that gap is usually timing noise. A p95 latency budget
(`--max-p95-ms`, or the Tuner's budget field) adds a second constraint. When the target is
reachable but not within the budget, the result says so, with the setting recall needs and
the best recall the budget allows, instead of reporting "not reached".

### Suggested next steps

The Tuner, `vectraudit sweep` and `GET /api/sweep/{job_id}/advice?target=0.95` turn a sweep
into next steps, each with the measurements behind it. For IVF sweeps, every setting also
reports the share of true neighbours in the probed lists. That share caps recall, so the gap
between them tells the two failure modes apart:

- **Neighbours in lists that weren't probed**: raise nprobe. The suggestion names the
  nprobe at which the target share of neighbours is within reach and offers that sweep.
- **Neighbours in probed lists but ranked out**: more probing won't help. PQ/SQ codes, a
  PCA transform, or (for IVF-Flat) vectors that differ from what was indexed are named as
  the cause, with a fix such as re-ranking with exact distances.

HNSW sweeps suggest larger efSearch values while recall still rises fast enough to reach the
target, and a denser graph when it has levelled off. At the recommended setting, a tail of
queries finding fewer than half their neighbours, and uneven IVF lists, are flagged too.

Sampled stored vectors make convenient queries, but they are not your users' queries. Pass
held-out queries with `--queries` before trusting a recommendation; the Tuner and Compare
views show a "How far to trust this" checklist. Recall is agreement with exact search on
the same embeddings. It is not answer quality: if the exact neighbours are poor answers, the
embedding model or chunking is at fault, not the index.

```bash
vectraudit compare exact.index compressed.index --vectors vectors.npy \
  --queries queries.npy --right-nprobe 16 --repeats 5 --seed 42 --json > comparison.json
```

Comparison uses the same raw ground truth and queries for both indexes, reports recall,
mean/p95 latency and serialized index size, and lists neighbours unique to either index
for each query. Serialized size is not process RAM. Both indexes must use the same metric,
input dimension and user IDs. Python users can call `vectraudit.core.compare_indexes` with
loaded indexes, a `VectorSource`, and a `QuerySet`.

To compare in the UI, pass the other indexes to `serve` (or `launch(..., compare=[...])`);
they must be built from the same `--vectors`:

```bash
vectraudit serve ivf_flat.index --vectors vectors.npy --compare ivf_pq.index --compare hnsw.index
```

The **Compare** view runs the same paired measurement as `vectraudit compare`, with each
index's own nprobe/efSearch, and links changed queries to the Query Explorer.

Raw-vector IDs are checked against index IDs before analysis. Supply `--ids` for custom
IDs. The API uses numeric IDs through `2**53 - 1` and decimal strings above that limit,
preserving the full non-negative int64 range in browser inputs, links and exports.

Background work uses at most two workers and eight queued jobs per session; a full queue
returns `429 JOB_CAPACITY`. The Tuner's **Cancel sweep** button (or
`DELETE /api/sweep/{job_id}`) stops queued work immediately and running work at its next
checkpoint. A native FAISS call already running must finish first. Sessions retain up to
32 completed jobs for one hour, with expired entries removed on the next access; query and
ground-truth caches are limited to four entries and 64 MiB. An evicted sweep can be rerun.

Projections are cached on disk under `~/.cache/vectraudit/<index sha1>/` (or
`$VECTRAUDIT_CACHE_DIR`). The cache only saves time: if it can't be written (a read-only home
or a full disk), the projection is still shown, with a warning in the Cluster map and the
server log. `vectraudit cache info` shows usage per index. `vectraudit cache clear` removes
everything, or only entries unused for `--older-than-days`, or least recently used entries
beyond `--max-mb`. It never touches the demo data. `serve --no-cache` skips the disk cache.

## How it compares

- **[Feder](https://github.com/zilliztech/feder)** (Zilliz) visualises FAISS IVF-Flat and
  hnswlib indexes in JavaScript and is great for learning how they search. vectraudit does
  all FAISS work in Python, supports PQ/SQ, HNSW and wrapper indexes, and focuses on
  debugging: exact ground truth, per-neighbour miss reasons, recall/latency tuning and
  chunk text.
- **Embedding projectors** (TensorFlow Projector, Nomic Atlas, Renumics Spotlight, Arize
  Phoenix) visualise embeddings. vectraudit visualises the *index*: what it probes, stores and
  loses.

## Limitations

- CPU FAISS only; convert GPU indexes first.
- The HNSW trace is reconstructed in Python. It matches FAISS's own results in our tests
  (100% overlap across efSearch 4–256, L2 and IP), but FAISS internals can change.
- Memory is roughly index size + n × d × 4 bytes (about 0.6 GB for 1M × 64, measured);
  see [performance and memory](performance.md).
- The debugger is a local, single-user tool with no authentication. Use `--demo-mode`
  for a read-only shared deployment ([Hugging Face Space setup](../deploy/hf-space/README.md)).
