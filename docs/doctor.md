# Diagnose a release-check setup offline

Use `vectraudit doctor` before the first run, or when moving a setup to CI. It checks the
installation, packaged FAISS UI, connection fields, optional dependencies, secret presence,
local files, dataset validity, label coverage and policy minimums. It never opens a backend,
executes a query, loads/downloads an embedding model or uploads customer data.

```bash
vectraudit doctor
vectraudit doctor --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --policy retrieval/policy.json
# Text-only dataset: declare the model you will also pass to record.
vectraudit doctor --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --embedder YOUR-MODEL
# Machine-readable output, warnings block CI as well as errors.
vectraudit doctor --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --policy retrieval/policy.json --json --strict
```

Each supplied file is optional; with no files, the report scope is installation only.
`ready` means the requested offline checks passed, `needs_attention` means warnings, and
`blocked` means errors. Exit 1 on errors, or on warnings with `--strict`; otherwise exit 0.
A stable diagnostic `code` accompanies every check. Invalid configuration reports point
to local validation commands rather than echoing input contents. JSON reports omit question
text, vectors, label IDs, tags, paths and secret values; share them voluntarily with support.
A dataset hash is still an identifier of customer data, so review reports before sharing.

For FAISS raw vectors, doctor checks a memory-mapped floating-point `.npy` matrix without
pickle loading and compares its dimension to pre-embedded query vectors. It does not open
the index or verify row/ID alignment. Those checks happen when `record` opens the backend.
FAISS datasets with filters and Pinecone policies requiring exact ANN recall fail early.
Other exact-reference capabilities require an actual connector; a Weaviate HNSW collection
cannot supply the exact reference that a flat collection can.

Required relevance metrics need at least the policy minimum of positively labelled queries.
Labels on source spans count as document labels. Unlabelled questions contribute ANN recall
and latency only, not semantic quality. Small labelled sets, duplicate questions and small
cohorts trigger warnings. Use `label pool`, `label serve` and `label export` to build labels.

Text-only queries need an embedder for `record`. A supplied `--embedder` checks the text
extra is installed, but model files, output dimension and normalization are still unverified.
An offline machine must already have the required model cached; model downloading is part
of the usual embedding workflow, never doctor.

Backend reachability, permissions, snapshots, timeouts, TLS and successful model execution
are **not tested** by doctor. Follow with a small `record` run, inspect its terminal status,
and compare that run against itself before involving real release candidates. Run the
same frozen labelled dataset on both sides and use the real candidate check for acceptance.
