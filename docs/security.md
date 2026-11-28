# Security and data handling

What Vectraudit reads, what it writes, what leaves your machines and when, and where each
control is tested. For reporting a vulnerability, see [SECURITY.md](../SECURITY.md); for the
self-hosted control plane's threats and controls in full, the [threat model](threat-model.md).

## In one paragraph

Vectraudit runs on your machines and in your CI. It reads your vector store with credentials
taken from environment variables, never writes to it, and never sends those credentials
anywhere. It has no telemetry. Release-check run files hold IDs, metrics and timings, not
document text, question text or vectors. Nothing is uploaded unless you run the optional
control plane yourself and push to it, and then only an allowlisted projection that you can
preview first.

## What it reads

| Input | How it is read |
|---|---|
| Your vector store | Read-only connectors: searches only, never writes or schema changes. pgvector runs in read-only transactions with validated identifiers and bound parameters. A per-request setting (such as `ef_search`) never outlives the request. |
| Credentials | Only from environment variables named in the connection file (`api_key_env`, `dsn_env`); a literal secret in a connection file is rejected. Pinecone and Weaviate keys are never sent over plain HTTP except to a local emulator. |
| Labelled questions | A local JSONL file in your repository, identified in runs by its SHA-256. |
| Index files | FAISS indexes are loaded with FAISS's own reader: open only index files you trust. LangChain's `index.pkl` is read by an unpickler that refuses every class except LangChain's docstore and documents, and LangChain is never imported. |
| Benchmarks and imports | BEIR, TREC, MTCB, HDF5, Parquet and trace exports are read from local files, never downloaded, and never modified. `.npy` files are read with `allow_pickle=False`. |

## What it writes

| Output | Contents |
|---|---|
| Release-check run files (`record`) | Query and document IDs, ranks, metrics, timings, backend configuration and the dataset hash. No document text, question text or vectors. |
| Workbench run files | Can also hold question text and stage evidence from your pipeline. They stay in the local workbench directory unless you upload them. |
| Reports (`check --html`, job summaries, pull-request comments) | Verdicts, metrics, IDs and the reasons for each finding. Document titles appear only if your dataset or metadata supplies them. |
| Evidence bundles (`check --bundle`) | IDs, numbers and SHA-256 hashes; runs only with `--bundle-runs`, through the same projection as uploads. |

IDs, tags, backend names and commit SHAs can themselves be sensitive. You choose them; review
them before sharing a report.

## What leaves your machine, and when

Vectraudit makes a network call only when you run a command that needs one:

| Call | When |
|---|---|
| Your vector store or pipeline | `record`, `tune`, `capacity`, workbench runs: the endpoints in your connection or manifest. |
| GitHub API | `check --github-check` or `--github-comment`, with the token you pass. |
| Your control plane | `record --upload`, `check ... --upload`, `cloud push`: an allowlisted projection of the run (below). |
| Observability tools | `*-pull` commands read from Langfuse, LangSmith, Phoenix, Opik or OpenObserve; `langfuse-push` and `phoenix-push` write a diagnosis back only with `--send`, and preview otherwise. |
| Model downloads | `dataset embed`, `--embedder` and the labelling helper load sentence-transformers or Transformers models, which those libraries fetch from the Hugging Face Hub the first time. `vectraudit demo` downloads its public demo corpus. |
| Release wheel | The CI install action downloads the pinned wheel from the GitHub release and checks its SHA-256 before installing. |

### Uploads to a control plane you host

An upload is a versioned projection (`release-metadata-v1`), never the raw run file. It keeps
query, document and family IDs, tags, run labels, metrics, timings, safety assertions,
numeric and boolean search parameters, text parameters as SHA-256 fingerprints, and pipeline
identity. It excludes request context and filters, vectors, endpoint hosts, module paths,
raw stage evidence, error messages and unknown fields. Query text is excluded unless you pass
`--share-query-text`. See exactly what would be sent, without credentials or a network call:

```bash
vectraudit cloud preview runs/candidate.run.json --out review/candidate.projection.json
```

## Local servers

`vectraudit serve` (the FAISS debugger), `vectraudit workbench serve` and `vectraudit label
serve` are single-user and unauthenticated, and bind to `127.0.0.1`. Do not expose them on a
network; the FAISS debugger's `--demo-mode` is read-only and caps expensive operations for a
public demo.

## In CI

- The generated workflow runs on `pull_request`, never `pull_request_target`, so pull
  requests from forks get no secrets and only a job summary.
- Each side's secrets reach only that side's `record` command; the GitHub token reaches only
  `check`; the installer sees neither.
- The installer downloads exactly the pinned wheel, verifies its bytes and version metadata,
  removes runner tokens from pip's environment, and only then installs.

## Checks that touch a backend harder

- **Poisoning:** suites are generated locally from fixed templates with harmless sentinel
  claims and run in memory or against a staging index you loaded; a candidate that adds
  documents must declare `"environment": "staging"`.
- **Capacity and tuning:** searches only, bounded by a duration, a requests-per-second
  ceiling and a search budget printed before starting; they refuse to start unless
  `--confirm-target` names the collection, and refuse a target named like `prod` without
  `--allow-production`.

## Dependencies and reporting

CI audits the locked Python (`pip-audit`) and frontend (`npm audit`) dependencies on every
pull request, on `main` and weekly. New dependencies must have permissive licences. Report a
vulnerability privately through
[GitHub's private vulnerability reporting](https://github.com/rmnvg/Vectraudit/security/advisories/new);
see [SECURITY.md](../SECURITY.md).
