# Your first release check in 15 minutes

You need: Python 3.10+, the Vectraudit wheel, read-only access to your vector backend, and
30 or more real questions with the ids of the documents that answer them.

## 1. Install and scaffold (2 minutes)

```bash
pip install "./vectraudit-0.1.0-py3-none-any.whl[qdrant]"   # or [pgvector] / [faiss-cpu]
vectraudit init --backend qdrant
```

This creates `retrieval/connection.json`, `retrieval/policy.json`, `retrieval/queries.csv`,
`retrieval/README.md`, `.github/workflows/retrieval-check.yml` and the two actions it uses
under `.github/actions/`. Existing files are kept unless you pass `--force`.

## 2. Connect (3 minutes)

Edit `retrieval/connection.json`: the URL and collection (Qdrant), the table and columns
(pgvector) or the index file (FAISS). Put credentials in an environment variable and name
it in the file (`api_key_env`, `dsn_env`); a credential written into the file is rejected.
Use a read-only key or role.

## 3. Label queries (5 minutes for a first pass)

Fill `retrieval/queries.csv` with real user questions: one row per question, the ids of the
documents that answer it (`doc-42:2; doc-7` gives grades 2 and 1), and optional tags for
cohorts. Then:

```bash
vectraudit dataset import retrieval/queries.csv --out retrieval/dataset.jsonl --tags-column tags
vectraudit dataset embed retrieval/dataset.jsonl --model YOUR-EMBEDDING-MODEL [--normalize]
vectraudit dataset stats retrieval/dataset.jsonl --policy retrieval/policy.json
```

Use the same embedding model as your corpus (normalised if your index uses cosine or inner
product). `stats` tells you if the set is too small to trust; 30 labelled queries detect
large regressions, 100+ detect subtle ones and make cohorts meaningful.

### Starting without labels

Use the local assistant with a CSV of questions (`text`, optional `query_id`) and a JSONL
source corpus (`doc_id`, `text`):

```bash
vectraudit label pool questions.csv corpus.jsonl --k 5 --out task.json
vectraudit label serve task.json
# Label at http://127.0.0.1:8766, then stop the server.
vectraudit label export task.json --out retrieval/dataset.jsonl
```

Allow about 30 minutes for a small first pass, more for long passages. Continue with the
embedding and record commands above/below. See [labelling.md](labelling.md) for pooled
runs, second-pass review, privacy and versioning. Labels use source-document IDs; the
experiment runner maps chunks to sources automatically.

### Check the setup before recording

```bash
vectraudit doctor --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --policy retrieval/policy.json
```

For text-only questions, add `--embedder YOUR-EMBEDDING-MODEL`. Doctor stays offline:
it checks installed dependencies and input readiness, while `record` verifies actual
backend access and model execution. Fix errors before running. Warnings are advisory;
`--strict` makes them fail CI. See [doctor.md](doctor.md).

## 4. Record and check (3 minutes)

```bash
vectraudit record --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --out baseline.json --label current
# change one thing: a search parameter, an index, an embedding model, the chunking...
vectraudit record --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --out candidate.json --label candidate --param hnsw_ef=32
vectraudit check baseline.json candidate.json --policy retrieval/policy.json --html report.html
```

Open `report.html`: the verdict, why, the diagnosis (each finding marked verified,
supported or hypothesis), quality with confidence intervals, latency, and the queries that
got worse with the documents they lost.

## 5. Put it in CI (2 minutes)

Commit `retrieval/`, `.github/actions/` and the workflow. The workflow is a single
`retrieval-check` step that installs the pinned release, records the target branch and the
pull request on the same runner, checks them and keeps the report. It downloads the wheel
from the Vectraudit GitHub release; commit the release version and the wheel's SHA-256
(from the release's `SHA256SUMS`) in the workflow, and add your backend credential as a
repository secret. See [installation](RELEASING.md#install-and-upgrade-in-ci). Every pull
request that touches retrieval now gets a check run and a summary comment. Fork pull
requests get no secrets by design, so they need a trusted CI path before a check against a
private backend can run.

To upgrade, install the newer release locally and run
`vectraudit init --backend qdrant --ci-only --force`: it rewrites only the files under
`.github/`. Then update the version and SHA-256 pins together.

## Tune the policy

Start from the scaffolded `policy.json` (nDCG and recall may drop by 0.02, p95 latency may
rise 30%, at most 10 queries may regress, cohorts of 15+ queries may drop by 0.05). Tighten
it once you have seen a few real checks. See [release-checks.md](release-checks.md) for
every option.
