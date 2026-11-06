# Experiments: choose an embedding model, chunking and index

Release checks answer "did this change make retrieval worse?". Experiments answer the
question before it: "which of these setups should we use?" They compare several candidates
on one labelled query set, fairly, in one report, and hand the winner to release checks.

```bash
vectraudit experiment run experiment.json --out results/
open results/report.html
```

## Inputs

**Corpus** (JSONL): your source documents before chunking.

```json
{"doc_id": "kb-142", "text": "Refunds for annual plans are issued within five days...", "lang": "en"}
```

**Queries** (a normal dataset, see [release-checks.md](release-checks.md)) with **text**
(each candidate embeds it with its own model) and labels on **source documents**, never on
chunk ids:

```json
{"query_id": "refund-1", "text": "how long does a refund take", "relevant": {"kb-142": 2}}
```

For finer evaluation, label the exact passage with character offsets in the source document:

```json
{"query_id": "refund-1", "text": "...", "relevant_spans": [{"doc_id": "kb-142", "start": 0, "end": 61, "grade": 2}]}
```

**Manifest** (JSON):

```json
{
  "name": "embedding-bakeoff",
  "corpus": "corpus.jsonl",
  "queries": "queries.jsonl",
  "k": 10,
  "candidates": [
    {"name": "lexical", "embedder": {"type": "hashing"}, "chunking": {"method": "words", "size": 150}},
    {"name": "minilm-150", "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
     "chunking": {"method": "words", "size": 150, "overlap": 30}},
    {"name": "bge-small", "embedder": {"model": "BAAI/bge-small-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: "},
     "chunking": {"method": "paragraphs", "size": 200}},
    {"name": "minilm-hnsw", "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
     "chunking": {"method": "words", "size": 150, "overlap": 30},
     "index": {"factory": "HNSW32", "params": {"ef_search": 64}}}
  ]
}
```

| Part | Options |
|---|---|
| `embedder` | `type`: `sentence-transformers` (default; any model name), `hashing` (lexical baseline, no download, `dim`), `custom` (from Python). `normalize` (default true), `query_prefix`, `document_prefix` (e.g. E5's `query: ` / `passage: `), `batch_size`. |
| `chunking` | `method`: `words` (`size`, `overlap`), `paragraphs` (merged up to `size` words), `document`. |
| `index` | `factory`: any FAISS factory string (default `Flat`: exact); `metric`: `ip` (default) or `l2`; `params`: `nprobe`, `ef_search`. |

## How candidates are compared fairly

- **Document level.** Retrieved chunks map to their source documents (first occurrence
  wins); search continues until k distinct documents are found. A candidate with small
  chunks is not rewarded for returning five chunks of the same document.
- **Same labels for every chunking.** Labels on documents and spans are mapped onto each
  chunking; chunk ids are never reused across chunkings (the PRD's warning).
- **Each vector space gets its own exact reference.** For approximate indexes, ANN recall
  compares against exact search over that candidate's own embeddings.
- **Paired statistics.** Every candidate is compared with the leader on the same queries,
  with a bootstrap 95% interval. If the interval includes zero, the report says *tied*:
  the queries can't separate them, so pick on cost or add queries.
- **Many candidates at once.** With three or more candidates, every pairwise interval is
  another chance to see a winner that is not there. The report also runs a randomized Tukey
  HSD (Carterette, 2012): system labels are permuted within each family and the largest gap
  between any two candidates is compared with each observed gap, so the chance of *any*
  false winner stays at 5%. Each candidate gets `hsd_group` letters (candidates that share a
  letter cannot be told apart) and `p_vs_leader_adjusted`, and the report says which
  candidates are "not distinguishable from the leader (randomized Tukey HSD, α = 0.05)". It
  is paired, because all candidates answer the same questions. The bootstrap tie above is
  unchanged. `workbench optimize` adds the same annotation to its exploratory ranking
  (`significance` in the report, and extra columns in the workbench's Experiments tab); the
  choice of candidate and the held-out check do not read it.
- **Costs alongside quality.** Chunks, dimensions, index size, embedding time and search
  latency (query embedding is timed separately from search).

Document embeddings are cached in `~/.cache/vectraudit/embeddings`, keyed by model
settings, corpus content and chunking, so re-running after adding a candidate only embeds
the new one.

## From winner to release checks

Every candidate's results are a normal run file in `results/runs/`. Make the winner your
baseline:

```bash
vectraudit cloud push results/runs/bge-small.json --project acme/search
vectraudit cloud pin <run-id> --note "bge-small chosen in embedding-bakeoff"
```

or keep the file and use it with `vectraudit check`. From then on, release checks guard it.

## From Python

```python
from vectraudit.experiments.manifest import load_manifest
from vectraudit.experiments.report import rank
from vectraudit.experiments.runner import run_experiment

summary, results = run_experiment(
    load_manifest("experiment.json"), embedders={"my-model": MyEmbedder()}
)  # type: custom
ranked = rank(summary, results)
```

A custom embedder has `embed_documents(texts)` and `embed_queries(texts)` returning float32
arrays.
