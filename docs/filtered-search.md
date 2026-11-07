# Filtered search

A metadata filter ("language is English", "tenant is acme") changes what a vector index can
retrieve. Vectraudit applies it the way a filtering index does, **inside the search**, and
measures recall against the exact neighbours **among the allowed ids**, never against the
unfiltered truth.

## How it works

- `core.filtering.allowed_from_metadata(metadata, filters, ids)` evaluates the filter once
  with the same portable matcher the release checks use (equality, list membership, numeric
  ranges) and returns an `AllowedSet`: the sorted user-facing ids that satisfy it. A document
  with no metadata row, or without the field, never satisfies a condition.
- `search(li, query, k, allowed=...)` hands FAISS an `IDSelector` through the search
  parameters. The index never returns a disallowed id. Fewer than `k` results come back
  (padded with `-1`) when the probed cells or visited nodes hold too few allowed vectors.
  That shortfall is the effect being measured, so it is not hidden.
- `GroundTruth.search(..., allowed=...)` runs exact k-NN over the allowed subset. The subset
  is cached by the set's fingerprint (the ids, not how the filter was written), two at a time.
- `explain_query(..., allowed=...)` explains a filtered IVF query with measured facts only:
  which true neighbours sat in unprobed cells, and how many allowed vectors the probed cells
  held (`allowed_in_probed`) against how many exist (`allowed_total`).

## Pass rate

`pass_rate` is the fraction of the corpus that **satisfies** the filter. Benchmarks such as
VectorDBBench quote a *filter rate*, the fraction **removed**. A 99% filter rate is a 1% pass
rate; convert explicitly when importing.

## Filtered recall in release checks

A dataset query's `filter` goes to the backend with the search **and** with the exact
reference, so ANN recall is measured against the exact neighbours *among the documents that
pass*. Each filtered query in a run record also carries `pass_rate`, the fraction of the
corpus its filter keeps, **counted** by the backend with the condition its search applies:

| Backend | Count |
|---|---|
| FAISS (with `metadata`) | ids whose metadata satisfies the filter |
| Qdrant | the count API with `exact=true` |
| pgvector | `SELECT count(*)` with the search's own `WHERE` (not the planner's estimate) |
| Elasticsearch, OpenSearch | the `_count` API with the search's filter |
| Weaviate | the aggregate `total_count` with the search's filter |
| Pinecone, Chroma | not counted: no documented exact count for a metadata filter, so `pass_rate` is `null` |

An unknown pass rate is never read as high or low: the query is in no selectivity cohort, and
`search.pass_rates` in the run says how many filtered queries were counted. See
`selectivity_buckets` in `docs/release-checks.md` for cohorts by pass rate.

## Notes

- With a refine stage (`IVF..,RFlat`) FAISS honours the selector only inside the base
  parameters; the search code sets it there.
- A flat index under any filter has recall 1.0 by construction, which makes it a check on the
  ground truth itself.

## Selectivity sweep: recall by pass rate

```bash
vectraudit sweep INDEX --vectors x.npy --meta meta.jsonl \
  --filter 'lang=de' --filter 'lang=de;year={"gte":2020}' \
  --pass-rates 0.001,0.01,0.1,0.5 --target 0.95
```

The sweep runs once unfiltered and once per filter. Each filter has its own exact truth over
the documents it allows, so every row is recall against the right neighbours.

- `--filter` is a metadata filter (needs `--meta`). `;` ANDs conditions; a value that is valid
  JSON is read as JSON (numbers, lists, `{"gte": 2020}` ranges), anything else is a string.
  Repeat the option for several filters.
- `--pass-rates` adds synthetic id filters, like VectorDBBench's integer filter: the highest
  `round(n * rate)` ids pass, so the rate is exact whatever the corpus. The filter is
  deliberately unrelated to the vectors, which isolates selectivity from correlation; a
  metadata filter measures your real correlation.
- A filter that allows fewer documents than `k` is refused (its recall would be computed over
  fewer than `k` neighbours and compare with nothing).
- Exit code 2 when any filter, including the unfiltered run, never reaches `--target`.
  `--json` prints `vectraudit.selectivity_sweep` v1 (no document text). `--save` and
  `--baseline` are not available for filtered sweeps.

The Tuner has the same sweep as a card below the results: pass rates, and metadata filters when
the server was started with `--meta`, a recall-versus-parameter line per pass rate, a table of
the same numbers, and the advice below.

### What the advice says

Only measured statements. For a pass rate that never reaches the target: "At a 1% pass rate,
efSearch 512 reaches recall 0.93; the 0.95 target was not reached in the sweep range." When the
unfiltered recommendation loses recall under a filter: the recall at each pass rate at that
setting. Brute-force search over the allowed ids and partitioning the index by the filter
field are listed as *not measured*: they may help very selective filters, but this sweep did
not try them.
