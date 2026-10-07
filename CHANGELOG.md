# Changelog

All notable changes to Vectraudit (formerly faissight) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Registered document lookup for passage review: fetch bounded, context-scoped passages
  without importing a full corpus, inspect content versions, and reject mismatched snapshots.

- Critical-question release assertions: require a document within a specified rank, edit and
  export rules from the workbench, and inspect assertion evidence in JSON/HTML/Markdown reports.
  Missing queries and insufficient retrieval depth cannot silently pass.

- Connected passage-review queues in the guided benchmark editor: pool actual baseline and
  candidate results, prioritize unjudged disagreements, grade local passages, and return from
  comparison findings. Queues enforce exact exploratory versions, preserve irrelevant/span
  judgments, report missing text and failed rankings, and reveal source/rank after grading.
- Guided passage grading with durable local corpora and separate irrelevant judgments;
  CSV/JSON import previews and deduplicated incident review without inferred labels.
- Permission scenario builder for tenant contexts, allowed/forbidden documents and no-answer cases.
- Immutable benchmark snapshots and content/metadata inventory comparisons that queue stale
  labels for human review without rewriting existing versions.
- Persistent regression cases with owners, notes, evidence history, Markdown issue summaries,
  measured fix verification and explicit recurrence checks. Guided runs reload after restart.

- LangChain and LlamaIndex retrievers as workbench pipelines
  (`vectraudit.integrations.retrievers`): compression retrievers and postprocessors are
  recorded as real stages, so losses are attributed to the reranker that caused them. A
  Python adapter module beside the manifest is importable without installing it.

- `vectraudit dataset import --format beir|trec|ragas|deepeval` imports labelled
  benchmarks teams already keep. Ragas/DeepEval reference passages given as text are
  located in a `--corpus` file; unlocated and ambiguous passages are reported, never
  guessed. JSON exports (LangSmith, Langfuse) import through dotted column paths.

- Retrieval workbench: Python/HTTP pipeline adapters, DAG stage evidence, local BM25/vector
  fusion and phrase reranking, full-pipeline run artifacts and a standalone browser UI.
- Explicit no-answer and tenant-context dataset assertions, forbidden-document release
  gates, benchmark coverage/judgment queues and bounded held-out candidate optimization.
- Frozen scheduled checks with separate corpus/configuration/result drift and local history.
- Project access maps, configurable bulk-export/retention governance, schema migration 0005,
  and enterprise authorization-code/PKCE sign-in via a configured verified-userinfo provider.
- Runnable workbench example, browser regression script and deployment/recovery guidance.

- Offline `vectraudit doctor` for installation, connection, secret presence, dataset and
  policy readiness, with JSON output and `--strict` for CI; no network or model loading.
- Local source-document labelling with pooling, human grades and spans, optional confirmed
  suggestions, annotator agreement, browser review/resume/status and dataset download.
  Task transactions protect concurrent writers and stale tabs; frozen exports are preserved
  unless `--force` is explicit.
- Installed-wheel smoke coverage from annotation through `record` and `check`, and a repeatable
  Chrome labelling regression in CI. Paid-pilot launch evidence in `docs/launch-readiness.md`.

- **Release checks:** `vectraudit dataset validate`, `vectraudit record` and
  `vectraudit check` compare a baseline and a candidate retrieval setup on a versioned,
  labelled dataset and return pass (0), error (1), fail (3) or not comparable (4), with HTML,
  JSON and Markdown reports and a GitHub job summary. Connectors share one contract with
  declared capabilities and typed errors. See `docs/release-checks.md`.
- **Pinecone and Weaviate connectors:** Pinecone over its REST API (pinned version, no extra
  dependency; black box: no exact reference) and Weaviate (`vectraudit[weaviate]`: named
  vectors, tenants, id property, flat-index exact reference). Both pass the release-check
  acceptance suite; `vectraudit init --backend pinecone|weaviate`.
- **Experiments** (`vectraudit experiment run`): compare embedding models, chunkings and
  indexes on one labelled set at document level (labels on documents or text spans survive
  re-chunking), with paired intervals against the leader, ties called out, cohort winners
  and costs; every candidate is a normal run file for release checks. See
  `docs/experiments.md`.
- **Security hardening:** route audit test for tenant isolation, threat model, rate limits on
  failed authentication, workspace export and deletion, user deletion, member management and
  single-use invite links in the dashboard, personal token settings, dependency scanning in
  CI, incident runbook with `revoke-access` and `audit` commands.
- **Vectraudit Cloud** (`vectraudit[cloud]`, `vectraudit-cloud`): the hosted control plane.
  Workspaces with owner, engineer and viewer roles; project-scoped runner tokens; run upload
  (idempotent); pinned baselines and project policies; server-side checks with the CLI's
  verdict; run and check history; private reports and optional signed share links; a
  dashboard with GitHub or token sign-in; plans, quotas, retention and usage export; Docker
  Compose deployment with backup and restore scripts. See `docs/cloud-operations.md`.
- **Investigation:** per-query backend evidence (FAISS IVF miss reasons, pgvector query
  plans) and diagnoses with evidence levels (verified, supported, hypothesis) in every report.
- **Datasets:** `dataset import` (CSV/JSONL with column mapping and typed filters),
  `dataset embed` (records the model in a dataset header), `dataset diff` and `dataset stats`
  (label coverage, cohort sizes and readiness warnings).
- **Pilot kit:** `vectraudit init --backend faiss|qdrant|pgvector` scaffolds `retrieval/`
  and a pull-request workflow; a 15-minute guide; a reproducible before/after case study on a
  real RAG corpus (`examples/release_check_demo.py`).
- **GitHub:** `check --github-check` publishes a check run and `--github-comment` keeps one
  pull-request comment up to date, using the job's `GITHUB_TOKEN`.
- **Connectors:** FAISS (index files; raw vectors or a flat index as exact reference),
  Qdrant (`vectraudit[qdrant]`: named vectors, payload filters, `hnsw_ef`, exact reference)
  and PostgreSQL + pgvector (`vectraudit[pgvector]`: read-only sessions, statement timeouts,
  filter columns, `hnsw.ef_search` / `ivfflat.probes`, exact reference, query plans).

### Changed

- Release-policy minimums now count measured pairs for each quality metric. Sparse cohort
  metrics are reported without judging, and bootstrap memory is bounded for large suites.
- Dataset statistics count positive span labels; embedding retains spans and diff reports
  span-only changes. Keyboard grading targets grade buttons, avoiding hidden question fields.

- Renamed the project from faissight to Vectraudit. The Python package, CLI command,
  environment variables (`VECTRAUDIT_*`) and cache directory (`~/.cache/vectraudit/`) all
  use the new name. An existing `~/.cache/faissight/` (or `$XDG_CACHE_HOME/faissight/`) is
  moved to the new location on first use, keeping projections and run history.
- Vectraudit is now distributed privately; it is no longer published to PyPI.
- Licence changed from MIT to proprietary (all rights reserved). The wheel carries the
  `Private :: Do Not Upload` classifier and ships `THIRD_PARTY_NOTICES.txt` for the open-source
  packages bundled in the UI.

## [0.1.0] - 2026-10-03

First public release.

### Added

- **Getting started:** `faissight serve INDEX` opens the web UI; `faissight demo` builds a
  RAG demo (12,000 Apache-2.0 passages, MiniLM embeddings, IVF-Flat, IVF-PQ and HNSW) and
  opens it; `faissight.launch(...)` does the same from Python and Jupyter (JupyterLab,
  VS Code, Colab and JupyterHub proxies). Python 3.10 to 3.13.
- **Index support:** Flat, IVF-Flat, IVF-PQ, IVF-SQ and HNSW (Flat/SQ/PQ storage), through
  IDMap/IDMap2, PreTransform (PCA, OPQ) and Refine wrappers. Binary, GPU and FastScan
  indexes load with basic stats. User ids are preserved across the full int64 range in the
  API, links and exports.
- **Query explorer:** search by text, stored id, held-out evaluation query or vector against
  exact ground truth, with a reason for every missed neighbour (cell not probed,
  quantization, transform, ranked out), the smallest nprobe that finds them all, a
  probe-order chart, an optional exact-rerank experiment, shareable URLs and CSV/JSON export.
- **Tuner** and `faissight sweep`: recall@k and latency across nprobe or efSearch, with 95%
  recall intervals, per-query recall and the worst queries (one click to explain each), the
  cheapest setting for a target recall and optional p95 latency budget, suggested next steps
  backed by the measurements (including IVF probe coverage), a "How far to trust this"
  checklist, a copyable snippet, and cancellation. Exit codes gate CI on the target.
- **Saved runs and baselines:** `sweep --save`, `sweep --baseline`, `faissight runs diff`
  and the Tuner's **Save run** / **Compare with a saved run** keep a sweep as validated JSON
  and flag recall drops and p95 growth per setting, or say plainly when two runs aren't
  comparable (different queries, metric or corpus).
- **Compare** view and `faissight compare`: indexes over the same vectors measured on
  identical queries and ground truth: recall with intervals, mean/p95 latency, serialized
  size and per-query neighbour changes.
- **Relevance** view and `faissight evaluate`: labelled judgements (JSONL) give recall, MRR
  and nDCG@k before and after exact reranking, per query, with CSV/JSON export.
- **Run history:** finished sweeps, comparisons and evaluations are kept in
  `~/.cache/faissight/history/` (newest 100) across restarts; rename, download or diff them.
- **HNSW graph:** a layered 3D view and an animated, reconstructed search trace that matches
  FAISS's own results.
- **Quantization:** reconstruction error, compression ratio and near-pair distance fidelity
  for PQ/SQ codes.
- **Overview** (list sizes, imbalance verdict, largest lists, a "Diagnose my index" guide)
  and **Cluster map** (PCA/UMAP in 2D/3D, up to 50,000 stratified points, click a list to
  read its chunks).
- **Large indexes:** `--mmap` memory-maps raw vectors; the Overview reports memory costs.
  HNSW inspection doesn't copy the graph or vectors, background work is bounded and
  cancellable, and projections are cached on disk (`faissight cache info | clear`).
- **Public demos:** a read-only `--demo-mode` with capped workloads, and a Hugging Face
  Space image (`deploy/hf-space/`).
