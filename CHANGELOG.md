# Changelog

All notable changes to Vectraudit (formerly faissight) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Fixed

- A forbidden document served in a failed attempt was hidden by a clean retry: failed
  attempts now keep the forbidden documents they served (results and model context), and
  the release check fails on them, locally and in hosted checks.
- A candidate that stopped recording the model context skipped the context-leak check its
  baseline could be judged by; such a comparison is now not comparable.
- Knowledge freshness counted old content served while its change was still inside the
  propagation window as a failure; it is now listed as pending.
- A forbidden source document returned as a chunk (a result whose source reference names it)
  was not caught by permission checks; it now is.
- The workbench policy editor dropped `critical_context` rules when an existing policy was
  edited and saved; it now edits and keeps them.
- A run file marked complete but containing failed or missing questions is now rejected; it
  could crash a comparison or hide failures.
- A regression could pass if the regressed questions had no scores on the candidate side
  only; such comparisons are now not comparable.
- Comparisons validate every run (not only newer-format ones), so duplicate rows and
  non-finite metrics can no longer reach a verdict.

### Added

- `vectraudit.core.fusion`, a fusion library: RRF, ISR, Borda, weighted sum, CombSUM,
  CombMNZ and CombMAX, with min-max, max, sum, z-score, rank and Borda normalisation,
  checked against ranx 0.3.21 (`docs/fusion.md`).
- Metric conformance tests: recall, precision, MRR, nDCG, MAP and Hit Rate at k agree with
  ranx 0.3.21 to within 1e-9 on 200 seeded queries (graded labels, ties, short lists).
- `workbench import-traces` and `workbench intake` read LangSmith run exports and Langfuse
  observations (an API page or a list) as well as OpenInference. Retrievers are the runs or
  observations the export marks as retrievers, or the span names given with `--stage-kind`;
  the run, gaps, model context and capability report are those of an OpenInference import.

- Elasticsearch 8 and OpenSearch 2 connector (`"backend": "elasticsearch"` or
  `"opensearch"`, no extra package): approximate kNN search with filters, an exact
  `script_score` reference for ANN recall, `num_candidates` / `ef_search` tuning,
  `vectraudit init --backend elasticsearch|opensearch`, and CI service containers for both.

- Email notifications (PRD FR13): a project owner can add `mailto:` destinations with
  `style: "email"`. Release decisions and monitor alerts are emailed once per event, with
  the same headline, action and link as Slack and the same retries, through the operator's
  mail server (`VECTRAUDIT_CLOUD_SMTP_*`).

- `examples/reliability_story_demo.py`: a synthetic demonstration of the permission and
  freshness reliability offer: a stale answer, a revocation the assistant never received and
  a cache shared between users, each found and its correction verified; it writes a page for
  prospects and the reports the Checks tab reads.

- A **Checks** tab in the workbench reads the JSON written by lineage, freshness, access,
  cache-probe, migration and intake checks as cards with a verdict, headline and findings.
  `workbench access --json` now writes a top-level format (`vectraudit.access`).

- Production intake (`vectraudit workbench intake`): takes new OpenInference traces since a
  cursor, reports likely recurrences of benchmark questions (linked to regression cases),
  and keeps a de-duplicated review queue prioritised by feedback and frequency, with an
  independent sample and reviewer routing; feedback never becomes labels.

- Migration report (`vectraudit workbench migration`): the current and proposed systems on
  one benchmark under the release policy, plus corpus equivalence and ID mapping from each
  system's inventory, filter semantics against document metadata, environment differences,
  and measured versus declared costs; `--fail-unless-ready` gates the migration.

- Cache isolation probe (`vectraudit workbench cache-probe`): asks each permission-scenario
  question after the same question under another context, serially, and reports forbidden
  documents that crossed over, with whether the answer was identical to the primer's.

- Permission lifecycle (`vectraudit workbench access`): an access change log (revoke, grant,
  move, archive, with propagation windows) becomes each question's expected access in a
  written benchmark, with revoked labels listed for review; a run recorded after the changes
  is checked for revoked documents still reaching a stage, the results or the model context.

- Permission checks follow forbidden documents upstream: runs record a `permission_path`
  (stages reached, results, model context). A forbidden document in the model context fails
  even when the results are clean, and a policy's `forbidden_stages` (for example `rerank`)
  fails one that reaches those stage kinds. Findings and investigations name context leaks.

- Investigations use a content change log (`workbench serve --changes`, `investigate
  --changes`, with `--lineage`): stale content a question was served and conflicting
  versions become losses with a re-index caution, the panel lists changes to the sources the
  question touched, and labelled content not yet indexed gets no retrieval fix.
- Investigations use a lineage export (`workbench serve --lineage`, `investigate --lineage`):
  passages lost during ingestion are named in the panel, and retrieval fixes are no longer
  suggested for them.
- Knowledge freshness (`vectraudit workbench freshness`): checks a content change log against
  a lineage export and an optional run: changes propagated, pending, overdue or lingering,
  stale content served in results or context, and conflicting versions; questions tagged
  `historical` are exempt.
- Source-to-context lineage (`vectraudit workbench lineage`): follows each labelled passage
  through a customer-exported ingestion manifest and an optional run, names the first stage
  that lost it (extraction, chunk boundary, index, stale revision, retrieval, delivery or
  context), and lists empty, stale and duplicate chunks; `--fail-on-ingestion` gates CI.
- Shared investigations: investigation records can be shared with the hosted project
  (states, counts, times, outcome and protecting policy versions; no names or free text), and
  hosted recurring problems link to them.
- **Protect this question**: a fix verified in an investigation adds context and rank rules
  for its question to a saved release policy as a new version, recorded on the
  investigation.
- Investigation records: accept or dispute a diagnosis, request missing evidence, link fix
  trials and validations, and close with an outcome; `vectraudit workbench investigations`
  reports diagnoses, verified and reproduced fixes and median times for pilot measurement.
- `critical_context` release rules: a question's required evidence must reach the model in
  full on every execution; unknown or unrecorded context is never a pass, and hosted checks
  reach the same verdict.
- `examples/context_story_demo.py`: the retrieved-but-never-seen story, from wrong answer to
  protected release, through product code; writes a synthetic one-page summary.
- Investigation panel in the guided workbench: from a finding, **Investigate this question**
  shows the loss, unestablished causes, cautions and the pipeline's declared fixes, with
  **Try this fix** and held-out **Validate** buttons running as workspace jobs.
- Investigate and try a fix: `vectraudit workbench investigate` explains one failed
  question (losses, hypotheses, cautions, uncertainty) and lists the fixes its pipeline
  declares; `vectraudit workbench try-fix` runs current and fixed configurations
  interleaved on the failing and sampled exploratory questions, reporting recoveries, moved
  losses and regressions, and `--validate` judges the fix under the release policy on
  held-out questions. Candidates declare supported fixes in the manifest (`fixes`).
- Answer judgments and evidence sufficiency: benchmarks can declare `required_evidence`
  groups ("A and B together, or C"), runs record whether a complete group was delivered and
  reached the model, and `vectraudit workbench attach-answers` imports human or tool
  judgments (Ragas, DeepEval, Langfuse, Phoenix) to diagnose each question's first
  observable loss, separating context losses from wrong answers. Model judges' agreement
  with human review is reported; disagreements are kept as disputes.
- Model-context evidence: pipelines can report which documents actually reached the model
  (`PipelineResult.context`), and trace imports infer it from recorded prompts. Runs show
  relevant documents that were retrieved but omitted from, or cut short in, the prompt, and
  comparisons flag new context losses. IDs and flags only; never uploaded.
- `vectraudit workbench import-traces`: OpenInference traces (OTLP JSON or Phoenix span
  exports) become a run judged like a live one, with a capability report of what the traces
  record and what they cannot show. Unrecorded documents, ambiguous delivered rankings and
  untraced questions fail with a stated cause; traced latency is never judged.
- Launch scorecard (`docs/launch-scorecard.md`): every launch gate from the strategy mapped
  to the evidence that exists, what is still needed and who must act. No gate is marked met
  without customer evidence.
- Paired release checks: `vectraudit release run-paired` records the approved and proposed
  pipelines interleaved per question, each in its own worker process from its own
  checkout; set `paired: true` on the CI step to use it. Each side's variables reach only
  its own worker.
- `examples/sample_evidence_report.py`: one self-contained, clearly synthetic page from the
  pilot demo (fail, explain, fix, protect) to send to prospects before a call.
- Discovery ledger template (`docs/discovery-ledger.md`, blank CSV in `docs/templates/`) for
  recording qualified conversations and pilot sessions against the strategy's experiments.
  The filled-in ledger is kept outside the repository.
- Paired runs: `workbench run --paired` runs two candidates interleaved per question,
  alternating which goes first, so endpoint drift and cache warm-up affect both alike.
  Comparisons and reports say when latency came from a paired recording, and warn when
  the two runs used different warm-up.
- Cohorts from context: a benchmark header can declare `"cohort_fields": ["tenant"]`, and
  each question then also belongs to cohort `tenant=<value>`, judged by the policy's cohort
  rules like a tag. A change that hurts one tenant now fails that tenant's cohort. Derived
  cohorts are recorded and uploaded like tags, so fields are opted in one by one.
- Two evaluations at once in the workbench, each with its own progress and cancel. A
  registered pipeline runs one evaluation at a time so its declared concurrency and rate
  limit hold; the built-in baseline may overlap. Runs record `timing.overlapped_runs`, and
  comparisons warn when latency was measured beside another evaluation.
- Concurrent optimization: `workbench optimize` runs each candidate with its manifest
  `concurrency` and `max_rps` (tuning and held-out runs alike) and accepts
  `--max-attempts` and `--retry-budget`. When candidates were measured at different
  concurrency, the result says their p95 latencies do not compare like for like.
- Actual versus planned spend: `workbench run`, `optimize` and `monitor` now report the
  calls actually made (warm-ups, retries and failed calls included) and the cost adapters
  reported, beside the preview's plan. Run files carry a `usage` block per question and in
  total. Failed and unpriced calls are counted as unknown, never free. Usage stays local.
- Monitor staleness alerts: declare how often a scheduled monitor run is expected
  (dashboard **Monitor schedule**, `vectraudit cloud monitor --every 24h --grace 1h`, or
  `PUT .../monitor/schedule`). When no run arrives in time, the project page shows the
  monitor as overdue and webhooks subscribed to monitor events receive one
  `monitor.stale_opened` event; the next run sends `monitor.stale_resolved`. The
  `deliver-webhooks` service checks schedules on every pass; `vectraudit-cloud
  check-monitors` does it once, for cron. Migration 0011.
- Failing rules in the workbench: under a failing comparison, each rule says who may
  accept it under the comparison's policy (engineer or owner, and for how long) or why
  nobody can. **Check accepted regressions in the hosted project** lists the connected
  project's audited exceptions against those rules: covered, expired, used up or revoked
  (blocking again), and what recording the comparison there would show ("fail, accepted
  by exception", never pass) when both judge under the same policy. Read-only: the local
  verdict never changes, and exceptions are still granted only in the hosted project.
- One reusable CI step: `vectraudit init` and workbench release exports now write
  `.github/actions/retrieval-check/`, which installs the pinned release, records the
  target branch and the pull request on the same runner, checks them and uploads the
  report. Both generated workflows call it as a single step. Each side's secrets
  (`baseline-env`, `candidate-env`) reach only that side's record command, and the GitHub
  token only `check`. `vectraudit init --ci-only --force` refreshes only the `.github/`
  files when upgrading.
- Draft pilot offer (`docs/pilot-offer.md`) and trust package (`docs/trust-package.md`)
  for prospective design partners and their security reviewers. They are drawn from what
  the product does today, with commercial and deployment decisions left marked for the
  owner and no compliance claims.
- A/A calibration in the workbench: under the release policy picker, choose 2-10
  recordings of the unchanged setup to see false alarms by rule, failures that are not
  noise, and each proposed threshold with the noise behind it. Then save the draft as a new
  policy version; nothing is applied silently. Calibration (CLI and workbench) now reports
  configuration differences between the recordings, and says when they are not A/A.
- Readable document titles: findings and regression cases show document titles beside IDs,
  from a corpus document's `title` or `metadata.title`. Titles join from the run's own
  saved benchmark corpus; they are withheld when the run used a different corpus and
  marked unverified when the run recorded none. They are never written to runs, checks or
  cases. A shared case carries its cited documents' titles only with explicit text sharing,
  and the hosted case page shows them, or says why it can't. The starter example now has
  titles.
- Share from the workbench: a **Share with hosted project…** panel on regression cases,
  frozen benchmarks and saved policies previews the destination, what is sent, what stays
  local and the size. Question text is shared only when opted in, and the panel shows it
  first. The upload happens only after an explicit confirmation, uses the same projection as
  `workbench share` and links to the hosted record. The hosted credentials stay in the
  server's environment.
- Recurring problems on the hosted project: questions that failed in several checks on the
  current benchmark (regressed, critical or safety), with "failed in N of M", whether and
  for how long they still fail, and links to their shared regression cases. Question text
  appears only from cases shared with text. Shown on the project page, at
  `/app/WORKSPACE/PROJECT/recurring` and from `GET .../recurring`. It reads the decision
  ledger, whose summaries (version 6) now keep the failing question IDs.
- Opt-in non-inferiority policy: with `"non_inferiority": true`, a suite-level `max_drop`
  rule passes only when the whole 95% interval of the change stays above the allowed drop.
  An interval reaching past it fails as inconclusive with its own wording, and a clear drop
  as beyond tolerance. Reports, ledger summaries (version 5) and lead summaries state the
  decision. Calibration sizes the margin from the A/A lower bound, the optimizer does not
  exclude on an inconclusive screen, and the workbench editor offers it as one of three
  ways to judge allowed drops. Existing policies keep their hashes.
- Cohort false-alarm control. Each check reports how many cohorts and cohort rules it
  judged and how many failed (`cohort_judging`, a "Cohorts:" report line, the hosted
  decision summary), and warns from 5 rules judged without control. A new policy option
  `cohorts.control: "bonferroni"` fails a cohort `max_drop` only when its
  multiplicity-adjusted interval lies beyond the limit. Policies without it keep their
  hashes. Lead summaries flag an uncontrolled cohort failure as possibly chance, while the
  failure still stands.
- `vectraudit workbench preview MANIFEST` shows the pipeline calls and spend a run,
  optimization or monitor would make, without opening any adapter. Cost per call is measured
  from an earlier run's adapter-reported `cost_usd`, declared with the new manifest field
  `cost_per_call_usd` (outside the candidate identity), or reported as unknown, never as $0.
  `workbench run`, `optimize` and `monitor` take `--max-calls` and `--max-spend` and refuse
  to start over budget.
- Project webhooks: owners send each release decision, and each scheduled-monitor alert
  and its resolution, once to a signed JSON endpoint or a Slack incoming webhook. The
  payload has a headline, verdict and dashboard link only. Retries back off over about 8
  hours. Webhooks accept https public endpoints only, never follow redirects, and can be
  tested, rotated and removed; each change is audited. `vectraudit-cloud deliver-webhooks`
  (a new compose service) sends them. Migration 0010 adds `webhooks` and
  `webhook_deliveries`.
- Scheduled monitor runs in hosted history: `vectraudit workbench monitor MANIFEST cloud:main
  --upload` compares with the pinned hosted baseline under the project policy and records a
  check marked `monitor`, with corpus, configuration and result drift. Monitor runs appear
  labelled in the timeline, trends and summaries. A derived monitor status
  (`GET .../monitor` and the project page) reports when an alert opened and which run
  resolved it, with one stable alert ID per failing streak. Migration 0009 adds the
  `trigger` column to checks and decisions.
- A release view for the engineering lead, read from the decision ledger so it outlives run
  retention. Each decision gets a plain-language summary (what changed, why it matters,
  what is uncertain, what to do), shown on the project page and on its own page, with a
  Markdown download. The trends page charts quality and p95 latency against the baseline
  for each benchmark version and marks policy changes. Both are also in the API
  (`.../decisions/CHECK_ID/summary`, `.../trends`), and `vectraudit cloud summary CHECK_ID`
  prints the Markdown, for example into `$GITHUB_STEP_SUMMARY`.
- `vectraudit workbench share WORKSPACE KIND ID` uploads one saved regression case,
  benchmark snapshot or policy revision to a hosted project as an allowlisted projection
  (`--preview` shows it offline; question text only with `--share-query-text`). Revisions
  are immutable; the dashboard lists them under **Shared cases** for a second engineer or
  lead; they survive run retention, are exported and go with their project.
- Quality metrics now average every execution within each query, retaining per-repeat
  scores and rankings. The workbench can inspect individual repeats and changed pipeline
  traces. Metric contract v2 requires re-recording both sides of older comparisons;
  family bootstrap sampling and all-execution safety checks remain in effect.
- `vectraudit workbench optimize --policy POLICY --baseline NAME` screens candidates against
  the baseline under the release policy and approves a candidate only if the policy passes on
  held-out questions; results are labelled approved, not approved, baseline remains best or
  exploratory only, and a quality tie keeps the baseline.
- A/A calibration: `vectraudit calibrate RUN... --policy policy.json` compares recordings
  of an unchanged system, counts false alarms by rule, and drafts a policy that only raises
  noise-driven thresholds; critical, safety and absolute-floor failures are reported, never
  loosened.
- Audited exceptions: an approver can accept one named failing rule of a hosted check with a
  reason, owner and expiry, as the policy's new `exceptions` block allows. Checks whose
  failures are all covered read "fail, accepted" (exit code 0, neutral GitHub check), never
  pass. Safety failures need the owner or cannot be accepted. `vectraudit cloud accept`,
  `cloud exceptions` and `cloud revoke-exception`; the dashboard shows and grants them.
- Hosted decision ledger: each release check records a compact decision that is kept after
  retention deletes its runs and check. The project page shows a release timeline of
  decisions and baseline pins, also available from `GET .../decisions`; the purge records
  missing decisions first, and `vectraudit-cloud backfill-decisions` records older checks.
- Request contract: list-valued `context` (such as user groups), list and numeric-range
  `filter` values, and conversation `history` in datasets, imports, permission scenarios,
  connection tests and document lookup. Existing dataset versions are unchanged.
- Optional source/chunk identity on hits and declared evidence coverage on results; an
  incompletely observed ranking fails the query instead of passing.
- Hosted uploads send an allowlisted projection of each run (`vectraudit cloud preview`
  shows it without credentials or network access); query text and history need
  `--share-query-text`. The service re-projects every upload, and `check cloud:<name>`
  compares projections on both sides.
- Private distribution foundations: wheel-only release staging with SHA-256 manifests,
  optional publishing to a separate private repository, and one shared installer in
  `init` and workbench exports. CI pins version and checksum, rejects the source repository
  as a download target, and verifies package identity before pip runs. Channel provisioning
  and customer credential/revocation acceptance remain required before outside installs.
- Exported release checks carry `concurrency` and `max_rps`; in CI the approved values run
  both sides so latency stays comparable. Serial exports are unchanged byte for byte.
- Guided benchmarks hold up to 1,000 questions. Pipeline run files are written as compact
  JSON (about a third of the size), the per-run limit is 64 MiB, and the run list loads
  summaries, fetching a run's questions only when inspected.
- Concurrent pipeline runs: per-candidate `concurrency` (up to 32 queries in flight) and
  `max_rps` rate limits in the workbench manifest, with results identical to serial runs,
  a shared retry budget, and cancel/resume preserved. Release checks compare latency only
  between runs with equal concurrency.

- Automatic retries for transient pipeline failures (timeouts, dropped connections, HTTP 429
  and the new `http_unavailable` for 502/503/504), with exponential backoff, `Retry-After`
  support and a per-run budget. Every retry is recorded in the run; a query that still fails
  keeps its cause and blocks the release check. `workbench run --max-attempts` and
  `--retry-budget` adjust or disable it.
- Independent query families: optional `family_id` groups related rows (permission
  scenarios inherit their source question's family). Quality metrics weight families
  equally, intervals resample families, and policy minimums count families. Reports,
  preflight and the guided comparison show rows and families; individual safety and
  critical assertions are unchanged.

- Reference support-search case study with actual BM25 ranking, measured stage latency,
  tenant/current-document filters, independent baseline/candidate/fixed source revisions,
  and guided/exported CLI verdict parity. Versioned fictional documents and varied questions
  include proposed judgments explicitly awaiting human review. Installed-wheel CI preserves
  the generated reports, source revisions, input hashes and reproduction records.
- Reference-search browser coverage verifies failing repeat evidence, stage attribution,
  export gating and the downloaded fixed release, with policy/metric parity against CLI
  reports. CI retains desktop/mobile screenshots and a Playwright trace.

- Pilot release rehearsal (`examples/pilot_release_demo.py`): synthetic improving-average
  regression, a critical miss on repeat 2, and a passing fix. The installed-wheel smoke
  now exercises guided comparison, export, independent checkout execution and JSON/HTML/
  Markdown reports, checking policy and verdict parity and missing-benchmark rejection.
- Interrupted workbench restart/resume browser coverage in the main and tagged-release CI.
- Durable evaluation checkpoints with browser/CLI resume, explicit failed-query retry,
  and preserved attempt history. Critical-question gates inspect every recorded repeat
  and prior attempt; missing repeat evidence cannot pass.
- Versioned release policies shared by guided comparison, case verification and exported
  CLI/CI checks; exports bind a frozen benchmark and approved policy to pipeline execution.
- Explicit pipeline connection diagnostics with bounded live execution and actionable
  capability/error results in the workbench.

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

### Fixed

- Editing a policy in the workbench no longer drops its `exceptions` block.
- `vectraudit workbench run` works when the output folder sits below a symlinked system
  folder, such as macOS `/tmp` or `/var`. Links inside folders Vectraudit manages are still
  refused.
- Cancelling a run during a retry backoff no longer records the in-flight query as a
  `pipeline_error` failure; the attempt stays interrupted and is retried on resume.

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
