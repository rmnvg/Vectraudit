# Startup strategy implementation progress

Source: [October 8 strategy](startup-strategy-2026-10-08.md).
Steps 1–12 are merged into `main` (through PR #45, `f7b0cf9`). Step 13 is committed as
`979c29c` on `feat/optimizer-policy`. Step 14 continues on that same branch and remains
local and uncommitted for the maintainer to review and commit.
Each step below is a separate reviewable increment, not a commitment to implement every
strategy item before customer learning.

## Step 1: preserve pipeline failure causes

Implemented the small error-reporting increment from Phase 0 / item 1.5:

- Known `PipelineCallError` categories survive in each failed query, with static messages
  and remedies shared with connection diagnostics.
- Arbitrary exception messages and unknown error codes remain redacted behind
  `pipeline_error`.
- `summary.errors_by_code` counts current failed queries by cause, including connector
  runs that use the shared summary function. Successful runs produce an empty mapping.
- Resume preserves typed errors. Explicit successful retries clear current failure counts
  while retaining the original cause in attempt history.
- Failed queries still make a run partial; release checks cannot approve it.

The summary field is additive. Existing run files remain loadable without it. This step
does not add automatic retries, backoff or concurrency, nor classify every arbitrary Python
exception. Those remain separate work.

Validation covers known and unknown error categories, redaction, mixed success/failure
counts, explicit retry history, connector summaries and release gating. Run the pipeline,
checkpoint, diagnostic, framework-adapter, evaluation-runner and comparison test modules,
plus Ruff and mypy. HTTP fixture tests need permission to bind localhost.

October 8 validation: all 149 selected tests passed (141 in the sandbox and the eight
HTTP tests rerun with localhost access). Ruff lint/format checks and mypy across 148 source
files passed. The full suite, frontend and deployed services were not rerun for this step.

## Step 2: readable local questions (1.6)

Implemented the question-text increment without requiring external service setup:

- Findings, query selectors and regression cases show question text alongside stable IDs.
- The local response joins text only after checking draft revision, full dataset hash and
  evaluated split hash. Baseline and candidate displays resolve independently.
- Missing, changed, corrupt or mismatched drafts leave an explicit unavailable message.
  Legacy runs and vector-only questions remain usable with IDs.
- Case creation, updates, verification and reload resolve text from the original baseline.
  Display fields are never written into run files, checks or case records. Existing cloud
  payloads and issue-summary downloads retain their existing content.

This increment does not add document titles, historical draft reconstruction, hosted record
sharing or text-bearing exports. Distribution remains pending channel selection and is
still required before outside installs.

Validation: 34 Python tests across local display, guided workflow and case lifecycle;
81 frontend unit tests; production frontend build; Ruff, Oxlint and TypeScript checks;
mypy across 149 source files; and the guided Chromium browser journey passed. The browser
journey covers text labels, ID-based selection, case updates/reload, changed-draft fallback
and mobile overflow. The full Python suite and hosted services were not rerun.

## Step 3: independent query-family statistics (1.4)

Implemented the family-counting increment:

- Optional `family_id` survives native import/export, permission generation, pipeline and
  connector runs, experiments, checkpoints and label review/export. Permission variants
  inherit their source's original family, including variants of variants.
- Family assignments affect dataset identity; a family cannot cross explore/heldout splits.
  Comparisons reject different family assignments between paired runs.
- Quality summaries weight measured families equally. Comparisons pair rows first, average
  within families, and bootstrap family means. Duplicating a family's identical scenarios
  does not narrow the confidence interval or increase its contribution to the mean.
- Suite, metric and cohort minimums count families; preflight uses the same counts.
  Run/check JSON, HTML/Markdown release reports and the local comparison show family counts
  alongside row counts.
- Permission, no-answer and critical checks remain absolute individual assertions. Latency,
  execution/error counts and the explicit regressed-query cap remain row-based.
- Old datasets retain their hashes and old run/task files still load. Existing permission
  rows without family provenance need explicit grouping or regeneration and fresh runs;
  the tool cannot recover their parent from an opaque ID. Use the updated checker.

This is not the whole of item 1.4. Repeat-quality measurement, optimizer-policy integration,
cohort multiplicity control, non-inferiority policy and A/A calibration remain separate work.

October 8 validation: full Python suite 1,289 passed, 22 skipped (pgvector, Pinecone and
Weaviate live services not configured), 1 deselected. Ruff lint/format, mypy on Python 3.12
and 3.10 (150 source files), API type generation (no drift), TypeScript, Oxlint, 81 frontend
unit tests, the production build and the market-fit Chromium journey (4 query rows / 3
families after adding a permission scenario) passed. Hosted services were not exercised.

## Step 4: automatic retries for transient failures (1.5)

Implemented the retry half of the 1.5 error work:

- Retryable codes: `timeout`, `connection_failed`, `http_rate_limit` and a new
  `http_unavailable` for HTTP 502/503/504 (previously the generic `http_error`). The HTTP
  adapter parses `Retry-After` as seconds or an HTTP date.
- `RetryPolicy` on `PipelineOptions`: 3 tries per call, backoff 1 s doubling to 30 s,
  `Retry-After` honoured up to the cap (longer requests fail the call), and at most 100
  retries per run invocation.
- Retries wrap each warm-up or repeat call, never the whole query, so a ranking already
  observed cannot be discarded. Query-level retry of recorded failures stays explicit
  through resume.
- Each retry is recorded on its query and in `summary.retries_by_code`; `timing.retry`
  stores the policy. Latency samples measure the successful call only.
- Backoff waits notice cancellation within 0.2 s. A query that still fails keeps its typed
  cause and leaves the run partial, so release checks still cannot pass it.
- `workbench run --max-attempts` (1 disables) and `--retry-budget` are frozen into the
  batch and checkpoint identity. Guided, export and optimize paths use the defaults; plain
  `RunOptions` never retry.

Not included: concurrency, per-pipeline rate limits, manifest-level retry settings, the
1,000-question guided cap, or retries for framework adapters that do not raise typed codes.

October 8 validation: full Python suite 1,309 passed, 22 skipped (pgvector, Pinecone and
Weaviate live services not configured), 1 deselected; 20 new tests cover each retryable
code, non-retryable codes, exhausted attempts, exponential cap, `Retry-After` (honoured and
refused), the run budget, warm-up/repeat calls, cancellation during backoff, checkpoint
resume, policy validation, HTTP-date parsing, a live local HTTP 503 with `Retry-After`,
and CLI options frozen for resume. Ruff lint/format, mypy on Python 3.12 and 3.10 and API
type generation (no drift) passed. No frontend files changed.

## Step 5: concurrent execution and rate limits (1.5)

Implemented bounded parallel execution for pipeline runs:

- Per-candidate manifest fields `concurrency` (1–32 queries in flight) and `max_rps`
  (calls per second, retries and warm-ups included). The `workbench run` CLI and guided
  evaluations of registered candidates use them; the built-in guided baseline stays serial.
- Workers only call the pipeline and build query records. Checkpoint writes stay on the
  calling thread (SQLite connections are single-threaded); records keep dataset order.
- The retry budget and rate limit are shared, lock-protected state across workers.
- Cancellation stops waits immediately, lets running calls finish and journals them, and
  leaves unstarted queries for resume. Resume skips everything already saved.
- Fixed a Step 4 bug: a cancellation raised during a retry backoff was caught as a
  `pipeline_error` failure. The attempt now stays interrupted and reruns on resume.
- Adapters can declare `thread_safe = False` to refuse concurrency. HTTP adapters and the
  built-in hybrid baseline are thread-safe (read-only closures).
- Execution settings stay out of the candidate configuration hash, so existing identities
  and drift checks are unchanged, but they are part of checkpoint identity. Runs record
  `timing.concurrency` and `timing.max_rps`; release checks compare latency only between
  runs with equal concurrency and warn otherwise.

Not included: concurrency for release exports and optimization (needs a release-format
change), CLI overrides, two concurrent workspace runs, or the 1,000-question guided cap
with matching UI bounds.

Validation: full Python suite 1,323 passed, 22 skipped (pgvector, Pinecone and Weaviate live
services not configured), 1 deselected. 14 new tests cover identical serial/concurrent
results in dataset order, the in-flight limit, typed failures, thread-unsafe adapters, rate
limits with 1 and 4 workers, a shared retry budget, cancellation and resume without repeated
calls, rejected resume with changed concurrency, the cancellation fix, latency comparability,
option validation, configuration identity and a concurrent CLI manifest run. The
concurrency and retry modules passed 8 consecutive repetitions. Ruff, mypy on Python 3.12
and 3.10, API type generation (no drift) and the guided and resume Chromium journeys passed.

## Step 6: release-check concurrency and 1,000-question suites (rest of 1.5)

- **Release files:** top-level `concurrency` and `max_rps` beside `repeats`. Exports copy the
  values recorded on the measured candidate run. In CI the approved release's values run
  both sides, so latency stays comparable and a pull request cannot change its own
  measurement protocol; a differing proposal is reported as a warning. `release validate`
  warns when a Python adapter would run concurrently, and the generated README states the
  protocol.
- **Compatibility:** the fields are excluded from the pipeline fingerprint and written only
  when non-default, so serial exports are byte-identical to before (verified) and older
  versions, which reject unknown fields, can still read them.
- **1,000 questions:** the guided editor, import and server accept 1,000 questions.
- **Artifact bounds, found by measuring:** a 1,000-question run from the built-in six-stage
  pipeline at depth 100 is 39 MiB compact and 104 MiB with the previous indentation. The
  old 20 MiB per-run limit would have refused it, and the old 20 MiB combined listing
  budget would have hidden every run after the first, so a baseline and candidate could not
  even be selected. Pipeline run files are now compact, the per-run limit is 64 MiB, the
  run list returns cached summaries without per-question records, and "Inspect failures"
  fetches one run on demand. The run browser notes when it shows 500 of more questions.

Measured in Chromium with two 39 MiB runs: run list 1.0 s, comparison rendered 2.1 s
(133 MiB JS heap), inspecting a run 0.8 s, no page errors.

Not included: optimization concurrency, two concurrent workspace runs, a calls/spend
preview, or trimming stage evidence in comparison responses (each comparison still sends
both full runs to the browser).

**Found while validating (pre-existing, not fixed here):** checkpoint paths refuse any
symlinked ancestor. On macOS `/var` and `/tmp` are symlinks, so `workbench_smoke.mjs` fails
under the default temp directory and `workbench run --out /tmp/...` is refused on a Mac.
The script passes with `TMPDIR=/private/tmp`, and Linux CI is unaffected. Resolving
system-level symlinks before the check would keep the protection against attacker-placed
links inside the workspace.

Validation: full Python suite 1,329 passed, 22 skipped (pgvector, Pinecone and Weaviate
live services not configured), 1 deselected; 6 new tests cover byte-compatible serial
exports, exported and approved protocols running both sides with a proposal warning,
validation of concurrent Python adapters, summary listings with cache refresh and on-demand
loading, run-size limits, and the 1,000-question limits for drafts and imports. Ruff, mypy on
Python 3.12 and 3.10, API type generation (no drift), TypeScript, Oxlint, 81 frontend unit
tests and the production build passed. Browser journeys passed: guided, market-fit,
resume, release, passage review, reference search, and workbench (with a non-symlinked
temp directory, see above).

## Step 7: checkpoint paths below symlinked system folders

Fixed the pre-existing macOS problem found in step 6. `validate_path` now takes the folder
the caller chose as a trusted root and refuses symlinks only at the journal, its SQLite and
lock companions, and directories strictly below that root:

- The guided workspace passes its resolved root, so `.workbench/...` protection is unchanged.
- `workbench run` passes `--out` (or `--resume`), so a linked `.checkpoints/<batch>` inside
  the output folder is still refused, before any file is written there.
- Direct `execute_pipeline` calls check the checkpoint file and its companions.

Result: `workbench run --out /tmp/...` works on macOS, where `/tmp` and `/var` are links.

Validation: full Python suite 1,340 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 11 new tests cover runs and resumes below a
symlinked folder, links at the journal and each companion, links inside managed folders
(absolute and relative `--out`), the CLI with a linked output parent and with a linked
`.checkpoints`, and the guided workspace. Four of them fail against the previous code.
Ruff, mypy on Python 3.12 and 3.10 (in a fresh environment; the `/tmp` one was incomplete)
passed. The workbench and resume Chromium journeys now pass under the default macOS temp
directory.

## Step 8: private distribution foundations (1.9)

October 9 code audit against `0e29a51` confirmed steps 1–7 are in the current checkout.
This increment was committed by the user as `9788547` on `feat/private-distribution-foundations`.

- Both `init` and workbench release exports include the same local composite installation
  action and standalone standard-library Python installer. Generated workflows no longer
  request customer access to `rmnvg/Vectraudit`.
- Installation requires an exact version and reviewed SHA-256. It downloads one exact
  wheel, checks its bytes and package/version metadata before pip runs, and removes the
  download credentials from pip's environment. It reinstalls even an already-installed
  version so a different artifact with the same version cannot silently satisfy the pin.
  Offline pre-downloaded wheels use the same verifier. Transitive dependencies are not
  locked by the Vectraudit wheel checksum.
- Release automation builds the wheel, runs the existing installed-wheel smoke, and stages
  exactly that wheel plus `SHA256SUMS` in a fresh directory. Source archives and unrelated
  files are excluded. The optional customer publication job checks that the configured
  destination is a different, private, non-fork repository and uses its own publish token.
  An unset destination skips customer publication; the maintainer release remains private.
- Setup, checksum handoff, upgrade/rollback and independent-customer revocation acceptance
  are documented in [RELEASING.md](RELEASING.md). This does not provision a repository,
  issue tokens, publish a release, or claim that download revocation invalidates copied wheels.
  Python modules in the wheel remain readable; the boundary is source-repository access.

The separate private GitHub repository is the prepared channel, not a provisioned service.
Item 1.9 remains partial: live source-independent install/revocation, the broader reusable
record/check step, owner-approved support/compatibility commitments and the buyer trust
package still remain. No customer/commercial gate is marked complete.

Validation: 19 distribution tests plus the existing scaffold/release-export tests (39
selected tests total); full suite 1,359 passed, 22 live-backend skips, 1 deselected. The
44 tests requiring local server binding passed when rerun outside the socket sandbox.
Ruff lint/format, mypy over 151 source files and API generation checks passed. The new
standalone scripts also typecheck targeting Python 3.10; the full Python 3.10 environment
was incomplete, so no full 3.10 validation is claimed. The wheel builds with the UI and
installer templates; an isolated installed-wheel smoke passes. Frontend code was unchanged.
Actual GitHub publication, customer credentials and hosted services were not exercised.

## Step 9: evidence and sharing contract (1.0)

Started in another session, which stopped partway; reviewed, corrected and completed here.
Committed as `1ccf3a7` and merged into `main` as PR #42.

- **Request contract:** `context` accepts lists of scalars (group membership); `filters`
  accept lists (any of) and numeric ranges; `history` carries earlier conversation turns.
  All are bounded. They flow through datasets, imports, permission scenarios, connection
  diagnostics and document lookup, and reach Python and HTTP pipelines. Paths that cannot use
  them (vector connectors, experiments, annotation pooling, the built-in baseline's history,
  framework adapters without forwarding) reject them explicitly. Existing dataset hashes are
  unchanged; `history` changes identity only when present and is omitted from HTTP payloads
  when empty.
- **Evidence coverage:** optional source/chunk identity on hits and declared coverage on
  results. An incompletely observed final ranking fails with `incomplete_evidence`; a
  partially observed trace is not used to attribute a loss.
- **Sharing boundary:** uploads send a versioned allowlisted projection with source and
  payload digests; the raw run is never changed; the server re-projects every upload;
  `vectraudit cloud preview` shows the exact payload without credentials or network access;
  query text and history need `--share-query-text`.

Corrections made while completing it:

- The projection originally dropped all search parameters and configuration. Comparing a
  raw candidate with a projected hosted baseline then reported false changes
  (`backend.config.adapter`, `search.evidence`, `search.params.vector_weight`), and hosted
  checks could not show any change. The projection now keeps numeric/boolean parameters as
  values, text parameters as SHA-256 fingerprints (the suite's planted-secret test showed no
  content rule can separate a reranker name from a prompt or token), and pipeline identity
  (adapter kind, name, configuration hash, versions) without hosts or module paths.
- `check cloud:<name>` now projects the candidate too. Measured on two real runs, the local
  file check, `check cloud:` and the hosted check now report the same verdict, reasons and
  changed settings (`backend.config.name`, `search.params.depth`, `search.params.reranker`).
- Re-validating a projection fingerprinted its fingerprints again, a digest mismatch that
  would have rejected every client upload on the server. Projected text parameters now pass
  through only if they are already fingerprints; raw text is rejected.
- Fixed a new test that resumed a custom adapter without declared versions, and lint errors.
- Wrote the documentation the first session had not: dataset fields, request values,
  coverage and source identity, and what uploads contain.

Not included: trace import (1.1), context/answer/citation records (1.2), and sharing
settings in the workbench UI (CLI only). Earlier hosted runs keep their stored content.

Validation: full Python suite 1,386 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 27 tests cover the contract and sharing boundary:
26 in the two new modules, including parity across surfaces for passing and failing
comparisons, plus a CLI `check cloud:main` test that fails without the symmetric
projection. Ruff, mypy on Python 3.12 and 3.10, API type generation (no drift),
TypeScript, Oxlint, 81 frontend unit tests, the production build and eight Chromium
journeys (guided, market-fit, resume, release, workbench, passage review, reference search,
labelling) passed. A manual preview of a run
with planted private text produced a projection with none of it.

## Step 10: decision ledger and release timeline (1.6)

Retention used to delete checks with their runs, so a pilot project lost its release
history after 30 days. Now:

- **Ledger:** a `decisions` table (migration 0006) keeps one compact row per check: verdict,
  policy digest, baseline name, run IDs and digests, candidate label and commit, and a
  versioned summary (per-metric changes and intervals, latency, critical-question counts,
  first 10 reasons, changed setting names). No query text, document text or raw evidence.
- **Survives retention:** the row has no foreign key to its check. The purge records any
  missing decision before deleting a check; rows go only with their project or workspace
  and appear in workspace exports.
- **Timeline:** the hosted project page shows decisions and baseline pins (from the existing
  audit log), with a report link while the check exists and "Expired under retention" after.
  `GET .../projects/{project}/decisions` returns the same timeline, paginated.
- **Existing deployments:** `vectraudit-cloud backfill-decisions` records decisions for checks
  made before the ledger; repeated runs change nothing.

Not included: shared regression cases and benchmark identities, trend charts, monitor
uploads, plain-language summaries for leads, and audited exceptions (1.10), which will be
recorded in this ledger.

Validation: full Python suite 1,393 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 7 new tests cover one decision per check
(idempotent), the summary's allowed fields and absence of private text, survival through a
purge with the timeline marking expired evidence, purge back-fill (fails if the back-fill is
removed), timeline paging, the dashboard after expiry, export, and the back-fill command.
The tenant route audit covers the new endpoint, and the model/migration drift check passes.
The cloud and pgvector tests also passed against PostgreSQL in a temporary
`pgvector/pgvector:pg17` container (142 passed, 4 Pinecone/Weaviate skips). Ruff, mypy on
Python 3.12 and 3.10 and API type generation (no drift) passed. No frontend files changed.

## Step 11: audited exceptions (1.10, first half)

A team can now deliberately accept a known regression instead of loosening the policy,
re-pinning or bypassing the required check:

- **Stable rule keys:** every failing reason names its rule (`metric:ndcg.max_drop`,
  `cohort:billing:ndcg.max_drop`, `regressed_queries`, `latency.p95_max_ms`,
  `critical:q7:doc-42`, `safety.forbidden:q12`, …). An exception names one key, so it
  covers exactly that failure and any other failure still blocks.
- **Verdict `accepted`:** only when every failing reason is covered. Reports say "fail,
  accepted by exception", never pass; CI exits 0 and the GitHub check is neutral. Errors and
  not-comparable results are never accepted.
- **Policy decides:** an optional `exceptions` block opens rule kinds, sets the approver
  role (engineer or owner) and the longest expiry. Safety failures are `never` or
  owner-only. Without the block nothing can be accepted, and existing policy hashes are
  unchanged (pinned by a test against values from `main`). Changing the block needs the
  owner role, so an engineer cannot open exceptions to themselves.
- **Hosted and audited:** `policy_exceptions` (migration 0007) stores rule, reason, owner,
  origin check, expiry, optional check allowance and revocation. Grants and revocations
  are audit events and timeline entries. Granting re-evaluates the origin check and its
  decision; later checks apply active exceptions; expired, used-up or revoked ones stop
  applying while earlier decisions keep their record. Decisions record the verdict before
  exceptions and which exceptions applied (ledger summary version 2).
- **Every surface:** API (grant, list, revoke), `vectraudit cloud accept | exceptions |
  revoke-exception`, the dashboard's failing-rules table with an accept form for allowed
  roles and an "Accepted regressions" list with revoke, and `check cloud:` applying active
  exceptions so its exit code matches the hosted verdict (a failure stands if they cannot be
  read).

Not included: A/A calibration (the second half of 1.10); exceptions in local file checks and
the workbench (they have no audited approver); reopening regression cases on expiry, since
cases are still local-only (1.6).

Validation: full Python suite 1,408 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 15 new tests: 6 for rule keys, policy gates,
partial versus full coverage, never-accepted errors and incomparable runs, and pinned policy
hashes; 9 hosted tests for grant permissions and validation, origin-check re-evaluation and
ledger update, later checks with a new failure still blocking, use limits, expiry and
revocation, safety rules, owner-only policy changes, the CI exit code via `check cloud:`,
CLI commands, and the timeline and dashboard. The tenant route audit now exercises the
grant and revoke routes with real victim data. Cloud tests passed on PostgreSQL in a
temporary `pgvector/pgvector:pg17` container (116 passed). Ruff, mypy on Python 3.12 and
3.10, API type generation (no drift) and the guided, release, reference-search and
market-fit Chromium journeys passed.

## Step 12: A/A calibration (1.10, second half)

`vectraudit calibrate RUN... --policy policy.json [--draft out.json] [--json report.json]`
measures how often a policy fails an unchanged system and drafts a policy that does not:

- Recordings of the same setup are compared pairwise in both directions (5 runs give 20
  comparisons). Every failure is a false alarm, counted by its stable rule key.
- The draft only raises noise-driven thresholds to the worst observed variation times a
  margin (default 1.5): metric and cohort `max_drop`, `latency.min_increase_ms` and
  `max_regressed_queries`. It re-evaluates itself and reports its own false alarms.
- Critical, permission and no-answer failures and absolute floors (`min`, `p95_max_ms`) on
  an unchanged system are reported as not noise, never absorbed.
- Inputs must be complete recordings of one dataset version and k; the report states how
  many runs and comparisons it rests on and makes no statistical guarantee; nothing is
  applied or overwritten.

Real-run check: five recordings of the bundled example pipeline under a strict latency rule
(5% p95 rise, no jitter floor) gave 7 false alarms in 20 comparisons, all
`latency.p95_increase` from about 0.04 ms of timer jitter. The draft raised only
`latency.min_increase_ms` to 0.5 ms, leaving the 5% rule in place, and had 0 false alarms.
Quality was identical across runs, so no metric threshold changed. (The shipped default
floor of 2 ms already absorbs this jitter.)

Not included: a workbench screen for calibration, and recording runs inside the command (it
calibrates existing recordings, so any backend or pipeline works).

Validation: full Python suite 1,418 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 10 new tests cover measured false alarms by rule,
drafts that absorb quality, cohort, regressed-query and latency-jitter noise without
touching other settings or the input policy, a policy already wider than the noise, an
absolute latency budget and an intermittent critical miss reported as not noise, input
validation, margin bounds, and a CLI that writes only new files. Ruff, mypy on Python 3.12
and 3.10 and API type generation (no drift) passed.

## Step 13: the optimizer under the release policy (1.4)

`vectraudit workbench optimize manifest.json --policy policy.json --baseline NAME` makes the
optimizer use the same release policy as CI:

- Candidates that fail the policy against the named baseline on exploratory questions are
  excluded, with failing rule keys; inconclusive screens do not exclude.
- The chosen candidate and the baseline run on held-out questions and the full policy
  decides. Results read "approved under the release policy", "not approved" (with verdict
  and reasons), "the baseline remains the best candidate" or, without a policy, "best
  exploratory candidate; not checked under a release policy". Output format version 2.
- Critical-question rules are applied in the split that holds their question; the report
  lists which were screened and which were checked on held-out questions.
- A quality tie keeps the baseline. The first real run on the bundled example chose an
  equal-quality candidate on sub-millisecond latency jitter and approved a change with no
  gain; it now reports that the baseline remains best.
- The workbench's Experiments tab shows the status, held-out check and per-candidate screen.
  Exit codes with a policy: 0 when approved or the baseline remains best, 3 otherwise.

Not included: a browser form to start optimization (it reads the CLI artifact, as before),
and concurrency for optimization runs.

Validation: full Python suite 1,427 passed, 22 skipped (live pgvector, Pinecone and Weaviate
services not configured), 1 deselected. 9 new tests cover exploratory-only labelling, approval with only the chosen
candidate and baseline seeing held-out questions, exclusion by a critical miss despite a
higher score, a held-out regression, insufficient held-out evidence, required arguments, the
quality tie, and deterministic CLI exit codes for approved and overfit candidates. Ruff,
mypy on Python 3.12 and 3.10, API type generation (no drift), TypeScript, Oxlint, 81 frontend
unit tests, the production build and the workbench and guided Chromium journeys passed.

## Step 14: quality across repeated executions (1.4)

Pipeline and connector runners now score every measured repeat and average within each
query before family aggregation. Run JSON retains `repeat_metrics`, `repeat_ann_recall`
and `repeat_ids`; single-execution experiments use the same contract. Release checks and
optimizer scores therefore include intermittent misses even when repeat 1 looks healthy.

Metric contract v2 declares `mean_per_query`. Old v1 recordings remain readable and
comparable with other v1 recordings, but upgrading a comparison requires re-recording both
sides. New comparisons require equal repeat counts and validate sample counts, numeric
coverage, aggregate means and ANN scores against their reference rankings. Repeats remain
one query for family bootstrap sampling. Critical and safety checks retain their existing
all-execution behavior; failed queries cannot contribute a partial quality average.

The workbench run and finding views select an execution, show its delivered IDs and nDCG,
and show the observed range alongside mean quality. Changed pipeline stage traces and losses
are retained per repeat; unchanged outputs reuse the first trace without claiming its stage
timings. Hosted transfers preserve numeric repeat measurements but strip raw repeated traces.
Legacy/connector/hosted records with no later stage trace display that absence explicitly.
Markdown and HTML reports explain that their default ID lists and primary traces are repeat 1.

Not included: execution-seed pairing or provider-level uncertainty estimates, cohort
multiplicity/non-inferiority semantics, per-repeat connector diagnostic calls, or a new
concurrency protocol. The rest of 1.4 remains open.

Validation: 11 new Python tests cover intermittent failure, changed/sparse traces, hosted
projection, ANN averages, failed later executions, corrupted evidence and contract migration;
4 new frontend tests cover repeat selection and absent traces. Full Python suite: 1,438
passed, 22 skipped (live services not configured), 1 deselected. The final triage wording
change also passed the 16 targeted repeat/triage tests. Ruff, mypy on Python 3.12 and 3.10,
API type generation (no drift), TypeScript, Oxlint, 85 frontend tests, production build,
and workbench/guided Chromium journeys passed, including selecting a later repeat.

## Step 15: shared cases, benchmarks and policies (1.6)

`vectraudit workbench share WORKSPACE KIND ID` uploads one saved regression case, benchmark
snapshot or policy revision to a hosted project, so a second engineer or lead can read the
evidence behind a release without the first engineer's machine. `--preview` shows the exact
projection offline.

The upload is a strict allowlist (`workbench-metadata-v1`) validated on both ends: cases keep
status, findings (kinds, document IDs), policy identity and verification history; snapshots
keep dataset identity and judgments; policies keep rules, hash and lineage, and the server
re-checks that the rules match their hash. Owners, notes, passages and context are never
sent; question text only with `--share-query-text`, flagged on the record and page. Local
revisions are immutable once uploaded (identical retry is idempotent, different content is
409). Records are listed and paginated per project in the API and dashboard, audited on
upload, kept after run retention, exported, and deleted with their project.

Not included: synchronization back to the workbench, hosted editing of cases, showing shared
records on checks or timelines, workbench UI for sharing, storage quotas for records, trend
charts, readable document titles, monitor uploads and lead summaries. Lead/second-engineer
acceptance with real users remains open.

Validation: 11 new tests cover projection allowlisting, opt-in text, nested injection,
identity mismatches, offline preview, revision conflicts and pagination, tenant isolation,
roles, digests, server rejection without echo, retention, export, project deletion and
second-engineer dashboard reading; the tenant route audit now covers the records routes with a
real record. Full Python suite: 1,449 passed, 22 skipped (live services not configured),
1 deselected; the 250 cloud tests also pass on PostgreSQL 17. Ruff and mypy on Python 3.12
and 3.10 passed.

## Step 16: release view for the engineering lead (1.6)

Each hosted decision now has a plain-language summary built from its decision-ledger row:
- a headline and the action it implies;
- **what changed:** metrics in words, p95 latency and the names of changed settings;
- **why it matters:** failing rules, accepted exceptions, critical questions and questions
  that scored lower;
- **what is uncertain:** intervals that include zero, fewer than 30 independent families,
  warnings, and an expired report.

Uncertainty never softens a verdict. A failure within the noise still stands, a pass within
the noise is "not proof of safety", and an accepted failure never reads as pass. The
project page leads with the latest decision; each decision has its own page with a Markdown
download. The same summary is in `GET .../decisions/CHECK_ID/summary` and in
`vectraudit cloud summary CHECK_ID`, which can append to `$GITHUB_STEP_SUMMARY`.
`check --upload` prints that command.

A trends page (and `GET .../trends`) charts each check's candidate value against its
baseline, plus p95 latency, coloured by verdict. Each benchmark version is its own series,
and policy changes are marked. Charts are inline SVG, so the dashboard stays script-free.
Everything is derived on read from existing ledger rows: there is no migration, older
decisions gain the view, and nothing is lost to run retention.

Not included: readable document titles (the ledger holds IDs only), monitor uploads into
history, recurring-finding views, notifications, cohort trends, and validation with a real
engineering lead. The wording is a hypothesis to test with one.

Validation: 14 new tests cover the summary wording for each verdict, noise in either
direction, small samples, expiry, exceptions, Markdown escaping, trend grouping and policy
marks, the API before and after retention, dashboard pages and downloads, an empty project,
tenant isolation and the CLI. The tenant route audit covers the new routes.
- Full Python suite: 1,463 passed, 22 skipped (live services not configured), 1 deselected.
- The 260 cloud tests also pass on PostgreSQL 17.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
- Pages were rendered with sample history and checked in Chromium on desktop, and at 390 px
  in dark mode with no horizontal overflow.

## Step 17: scheduled monitor runs in hosted history (1.6, input to 1.7)

`vectraudit workbench monitor MANIFEST cloud:main --upload` runs the scheduled check against
the pinned hosted baseline and uploads the run's projection. It records a hosted check marked
`monitor`, with the same policy, exceptions and verdict as the hosted check, so the exit code
and the dashboard agree. The local history is still written first. API clients can send
`"trigger": "monitor"` to `POST .../checks`.

Monitor checks also record drift, computed on the server by the same function the local
command uses: corpus version, configuration (each changed, unchanged or unknown) and how many
questions returned different documents. They appear labelled in the timeline, hollow in
trends, and as "Scheduled monitor" summaries whose action is to check what drifted, since
nothing was released. The latest-release card ignores them.

A derived monitor status (`GET .../monitor` and a project-page card) reads the ledger, so it
outlives retention.
- **Alerting:** while the newest runs fail, error or are not comparable. The alert is
  identified by the check ID of the streak's first run, and that ID stays stable until a
  run passes.
- **Healthy:** names the run that resolved the last alert.
- **Accepted failures:** they don't alert.

That is the deduplication and resolution input a delivery channel needs (the 1.7
acceptance). Migration 0009 adds the `trigger` column to checks and decisions; existing rows
are release checks.

Not included: sending notifications (webhook, Slack, email), schedules or staleness
detection when a monitor stops running, monitor-specific baselines beyond named pins, and
customer evidence that the status matches how teams triage.

Validation: 5 new tests cover:
- monitor status streaks, resolution, accepted failures and window completeness;
- monitor wording with drift;
- hosted monitor checks with drift, a stable alert ID, resolution, retention, an identical
  repeated check keeping its trigger, and tenant isolation;
- dashboard separation of monitor runs from release decisions;
- an end-to-end CLI run (pass, fail, fail, pass) through a real Python adapter, with exit
  codes, one alert ID and local history.

Results:
- Full Python suite: 1,468 passed, 22 skipped (live services not configured), 1 deselected.
- The 266 cloud tests also pass on PostgreSQL 17, including the migration/model consistency
  test.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
- The project and trends pages were checked in Chromium.

## Step 18: signed webhook delivery (1.7)

Project owners can add up to 10 webhooks (dashboard **Notifications** or
`POST .../webhooks`). Each receives:
- **`decision`:** each release decision.
- **`monitor`:** a scheduled-monitor alert when it opens, and once more when a passing run
  resolves it. Events are keyed by the Step 17 alert ID.

Events go into an outbox in the same transaction as the check, unique per webhook and event
key, so a repeated check or a continuing outage queues nothing new. `vectraudit-cloud
deliver-webhooks` (a new compose service, `--watch 30`) sends them outside any transaction,
claims with `SKIP LOCKED` and a lease, and retries after 1 min, 5 min, 30 min, 2 h and 6 h.
Redirects and 4xx other than 408/429 fail immediately. Transport is at-least-once;
receivers dedupe on `X-Vectraudit-Event-Key`.

- **Formats:** `signed` JSON (`vectraudit.event` v1), or `slack` (a `text` message for an
  incoming webhook, so Slack works without a receiver). Both carry Stripe-style HMAC-SHA256
  signatures over `timestamp.body` with a per-webhook secret shown once and rotatable.
- **Sharing boundary:** headline, action, verdict, trigger, candidate, IDs and a dashboard
  link; no reasons, question IDs, metrics or text.
- **SSRF:**
  - Only https URLs whose resolved addresses are all public, checked at registration and
    before each send.
  - Redirects are never followed and response bodies are discarded.
  - A development-only setting allows private and http endpoints.
  - The DNS-rebinding window is documented in the threat model.
- **Governance:** management is owner-only and audited; engineers see status and
  deliveries. The URL path is never displayed or exported (Slack URLs carry tokens). Webhooks
  go with their project, and finished deliveries are pruned after 30 days by `purge`.
  Migration 0010.

Not included: email or Slack-app delivery, digest or quiet-hours settings, a second event
when an exception later turns a check into "accepted", and staleness alerts when a monitor
stops running. The first customer's channel should decide the next delivery route.

Validation: 14 new tests cover:
- URL rules (http, credentials, private, metadata, loopback, mapped IPv6);
- owner-only management, a secret shown once, rotation, audit and export without secrets or
  paths;
- one signed decision event without evidence and nothing for an identical repeat;
- a pass, fail, fail, fail, pass, pass monitor sequence notifying exactly once when opened
  and once when resolved, in Slack format, with decision-only hooks receiving nothing;
- retry backoff, permanent failure, giving up and pruning;
- an endpoint that turns private and is not called;
- a real local HTTP receiver verifying the signature, with a 302 recorded and not followed;
- the dashboard (add, secret page, test ping, remove, engineer and viewer visibility).

The tenant route audit covers the new routes with a real victim webhook.

Results:
- Full Python suite: 1,482 passed, 22 skipped (live services not configured), 1 deselected.
- The 287 cloud tests pass on PostgreSQL 17, including the migration/model consistency test
  and `SKIP LOCKED` claiming.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
- `migrate`, `deliver-webhooks` and `purge` were run against a fresh database, and the
  dashboard section was checked in Chromium.

## Step 19: calls and spend preview (1.5)

`vectraudit workbench preview MANIFEST [--mode run|optimize|monitor]` shows, per candidate,
questions x repeats = calls, cost per call and spend, then the totals, before anything
runs. It reads the manifest and dataset only and never opens an adapter (a Python adapter
import would run customer code).

- **Cost per call:**
  - **measured:** from an earlier run whose adapter reported `cost_usd` (`--measured`).
  - **declared:** `cost_per_call_usd`, a new optional manifest field kept out of the
    candidate identity.
  - **built-in hybrid baseline:** a known $0, because it runs locally.
  - **unknown:** listed by name and never counted as $0.
- **Per-production-query cost:** one production query is one call, so it's the same as the
  cost per call.
- **Optimize:** a range, because held-out validation runs only for the chosen candidate
  (and the baseline under a policy).
- **Retries:** an allowance per run, up to each run's retry budget.
- **Scope note:** customer-side model, database and hosting costs are included only if the
  adapter reports them.
- **Guards:** `workbench run`, `optimize` and `monitor` take `--max-calls` and
  `--max-spend` (with `--measured`). They refuse before opening any adapter when the
  planned calls or maximum spend exceed the limit, or when a cost is unknown under
  `--max-spend` (exit 2). Resumes are not re-checked.

Not included: a browser screen for the preview, release-file (`release run`) and calibration
budgets, token-level cost models, and an actual-versus-planned spend report after a run.

Validation: 8 new tests cover:
- call counts and unknown costs excluded from the sum;
- a measurement winning over a declaration, and a declaration winning over a partial
  measurement;
- optimization ranges with and without a baseline, and the monitor rule;
- every limit;
- an unchanged candidate identity with a declared price;
- money formatting;
- CLI preview, JSON and refusals for `run`, `optimize` and `monitor`. An adapter module that
  leaves a marker when imported proves that preview and refusals never import it; a later
  run then prices itself from its own recorded costs.

Results:
- Full Python suite: 1,490 passed, 22 skipped (live services not configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
  No frontend or hosted code changed.
- On the bundled example, `--mode optimize` reports 18 to 24 calls at $0 (built-in baseline)
  with a 48-call retry allowance.

## Step 20: cohort false-alarm control (1.4)

With many cohorts, some fail by chance even when nothing changed. Every check now reports
`cohort_judging`: how many cohorts and cohort rules it judged, how many failed, and the
control. It appears as a "Cohorts:" line in Markdown and HTML reports and in the hosted
decision summary (ledger version 4). Checks warn from 5 rules judged without control.

- **Policy option `cohorts.control: "bonferroni"`:** a cohort `max_drop` fails only when
  its interval, widened to 1 − 0.05/m for the m cohort `max_drop` rules judged, lies
  entirely beyond the allowed drop. The interval is the family-mean change ± z × family
  standard error; tail percentiles of 2,000 bootstrap resamples would be unstable, and an
  extra resampling pass would shift existing seeded intervals.
- **Unchanged:** `min` floors, suite-level metrics and exception rule keys. Policies without
  the option keep their hashes.
- **Lead summary:** an uncontrolled cohort failure is flagged as possibly chance, and the
  failure still stands.
- **Workbench:** the policy editor has a "Correct for many cohorts (Bonferroni)" checkbox.
  It no longer drops a policy's `exceptions` block when saving, a bug found while adding the
  checkbox.

Measured A/A effect (20 cohorts of 10 families with noise, an unchanged system):
- Without control, 12 of 12 pairwise comparisons raised chance cohort failures (52 in all).
- With Bonferroni, 1 of 12 did, and 1 of 30 with six recordings.
- A real 0.4 drop in one cohort still failed under the control.

Not included:
- cohorts derived from declared context fields (tenant, language), because runs do not
  record context values and that would cross the sharing boundary;
- Holm or FDR procedures;
- t-based small-sample intervals;
- versioned non-inferiority semantics, the remaining 1.4 item.

Validation: 7 new Python tests cover:
- counting and the warning;
- chance failures controlled while a real drop fails;
- no adjustment with one rule;
- policy validation and unchanged hashes;
- the report line;
- calibration with and without control;
- the lead note.

One frontend test covers the control and exceptions round trip.

Results:
- Full Python suite: 1,496 passed, 22 skipped (live services not configured), 1 deselected.
- The 287 cloud tests pass on PostgreSQL 17.
- Ruff, mypy on Python 3.12 and 3.10 and the API type generation check (no drift) passed.
- TypeScript, Oxlint, 86 frontend tests, the production build and the release Chromium
  journey passed.

## Step 21: opt-in non-inferiority policy (rest of 1.4)

Not detecting harm is not evidence of safety. With `"non_inferiority": true`, a suite-level
`max_drop` rule passes only when the whole 95% interval of the change stays above
`-max_drop`.
- **Inconclusive:** an interval reaching past it fails with its own wording ("not shown to
  be within tolerance (inconclusive)") and `inconclusive: true` on the reason.
- **Beyond tolerance:** a clear drop is reported as such.
- **Exceptions:** both use the same rule key, so an audited exception can accept an
  inconclusive release deliberately.

The option is mutually exclusive with `require_significance`. Policies without it keep
their hashes and meaning, and cohort rules keep their own semantics.

Where it shows:
- **Check JSON:** each metric gains `tolerance.decision`.
- **Reports:** a "Non-inferiority" line.
- **Ledger summaries (version 5):** record it.
- **Lead summaries:** "shown within tolerance" replaces "not proof of safety", and
  inconclusive gets its own note.
- **Optimizer:** a screen that is only inconclusive does not exclude a candidate.
- **Calibration:** sizes the margin from the A/A interval's lower bound.
- **Workbench editor:** the significance checkbox becomes one choice of three ways to judge
  allowed drops.

Not included: a sixth `inconclusive` verdict (inconclusive fails closed, see DECISIONS),
non-inferiority for cohort rules, and per-metric choice of semantics. The execution protocol
and context-derived cohorts remain open in 1.4.

Validation: 10 new Python tests cover:
- within tolerance, inconclusive versus the measured rule, and beyond tolerance;
- policy exclusivity and unchanged hashes;
- unchanged cohort rules;
- accepting an inconclusive result by exception;
- report lines;
- lead and ledger wording;
- an optimizer screen that does not exclude;
- a calibration draft that removes A/A false alarms.

One frontend test covers the three-way choice round trip.

Results:
- Full Python suite: 1,506 passed, 22 skipped (live services not configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
- TypeScript, Oxlint, 87 frontend tests, the production build and the release Chromium
  journey passed.

## Step 22: recurring problems (1.6)

The hosted project now shows which questions keep failing across checks, not only one check
at a time. Each decision's ledger summary (version 6) keeps the IDs of the questions that
regressed on the primary metric, failed a critical rule or failed a safety rule (at most 50
each).

`GET .../recurring`, a project-page section (top five) and `/app/WORKSPACE/PROJECT/recurring`
list questions that failed in at least two of the newest checks on the current benchmark
version:
- **Failed in N of M:** the count always has its denominator.
- **How:** regressed, critical or safety.
- **Still failing:** whether it fails in the newest check, and for how many in a row.
- **Last failing decision:** linked.
- **Shared regression case:** linked, with its status when shared.

Release checks and monitor runs both count. Question text appears only from cases shared
with text, so the sharing setting still decides what hosted pages show. Older decisions are
reported as not counted, never as passes. It survives retention. Webhooks still carry no
question IDs.

Not included: document titles (still IDs), recurrence across benchmark versions, a
recurrence alert or webhook event, and grouping questions by finding type. Validation with a
real lead remains open.

Validation: 5 new tests cover:
- counting, denominators, streaks, kinds, other benchmark versions and pre-version-6
  decisions;
- failed questions extracted from regressions, critical rules and safety rules;
- the hosted API with linked cases and shared text, survival through retention, and tenant
  isolation;
- the dashboard section and page with IDs only;
- shared text shown escaped.

Results:
- Full Python suite: 1,511 passed, 22 skipped (live services not configured), 1 deselected.
- The cloud tests pass on PostgreSQL 17 (295 with the new module).
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check (no drift) passed.
- The page was checked in Chromium.

## Step 23: sharing from the workbench UI (1.0, 1.6)

A **Share with hosted project…** panel on regression cases, frozen benchmarks and saved
policies previews, before anything leaves the machine:
- the destination (hosted URL and project);
- what is sent and what stays local;
- the revision and transfer size.

Question text is off by default. When ticked, the panel shows the text that would be sent,
taken only from the verified local draft join. **Share now** needs an explicit confirmation,
uploads the same allowlisted projection as `workbench share` (both now use
`pipelines/share.py`), and links to the hosted record. Credentials stay in the environment of
`workbench serve`; without them the panel previews only. Projection errors never echo record
content, and hosted refusals (role, conflict, size) are relayed with their status.

Not included: choosing the destination project in the browser, sharing many records at once,
showing which local records were already shared, and per-workspace sharing defaults.

Validation:
- 4 new Python tests cover:
  - preview without credentials or upload, with the local file unchanged;
  - the text opt-in through the verified join only;
  - confirmation and configuration requirements, bad and unknown IDs, and a missing token;
  - a broken record refused without echoing it;
  - a confirmed upload reaching the hosted project with a link;
  - an idempotent retry, a 409 conflict and a viewer's 403.
- 2 new frontend tests cover the destination and size wording.
- The market-fit Chromium journey now opens the panel on a tracked case and checks the
  preview, the disabled **Share now** without a hosted project, and the text opt-in. Its
  server runs without `VECTRAUDIT_CLOUD_*` variables.

Results:
- Full Python suite: 1,515 passed, 22 skipped (live services not configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, the API type generation check (no drift),
  TypeScript, Oxlint, 89 frontend tests, the production build, and the market-fit, release
  and guided Chromium journeys passed.

## Step 24: readable document titles (1.6)

Findings show `kb-4821 · Refund policy (2026)` instead of a bare ID, and case cards list
their cited documents with titles.
- **Where titles come from:** a corpus document's optional `title` (or a string
  `metadata.title`) in the corpus attached to the run's own saved benchmark, after the same
  draft and revision checks as question text.
- **Trust:** a run that recorded a different corpus version gets no titles, with the reason
  shown. A run without a recorded version gets titles marked unverified. A corpus without
  titles shows IDs quietly.
- **Never persisted:** titles are not written to runs, checks or cases, and existing corpus
  versions are unchanged.

Hosted: a shared case carries its cited documents' titles only under the explicit text
opt-in. The hosted case page shows them, or says that they were not shared. The canonical
form omits empty titles, so uploads from older clients still verify. The starter example
now has titles.

Not included: titles from the registered document lookup (it returns titles per fetch, not
for a whole run), titles on the hosted recurring-problems page, and titles in exported
release reports.

Validation:
- 6 new Python tests cover:
  - verified titles with the `metadata.title` fallback, never written to disk;
  - case titles taken from the baseline's own corpus;
  - withheld titles for a different corpus, unverified titles for external runs, and a
    changed benchmark;
  - untitled corpora with unchanged versions;
  - projection only under opt-in, limited to cited documents, with forged titles refused;
  - the hosted page with and without titles.
- 2 new frontend tests cover labels and notes.

Results:
- Full Python suite: 1,521 passed, 22 skipped (live services not configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, the API type generation check (no drift),
  TypeScript, Oxlint, 91 frontend tests, the production build, and the market-fit, guided
  and passage-review Chromium journeys passed.
- A finding chip was checked in Chromium.

## Step 25: A/A calibration in the workbench (1.10)

Under the release policy picker, **Calibrate this policy against unchanged runs (A/A)** lets
an engineer choose 2–10 complete recordings of the same benchmark and see:
- false alarms by rule, with their denominator;
- failures that are not noise;
- each proposed threshold with the measured noise behind it;
- the CLI's caveats.

Measuring writes nothing. **Save draft as a new version** goes through the normal policy
versioning with a note recording the before-and-after false alarms. Only saved or startup
policies can be calibrated.

Found while checking the panel in Chromium: a baseline and a candidate (two different
configurations) could be measured as "A/A" without any warning. Calibration, in the CLI and
the workbench, now reports `config_differences` between recordings and says when they are
not one unchanged setup.

Not included: recording the A/A runs from the panel (record them as usual first), a cost
preview inside the panel, and showing exceptions in the workbench (1.10's remaining hosted
link).

Validation:
- 8 new Python tests cover:
  - measuring without writing, and saving the draft as version 2 that then has no false
    alarms;
  - refusals for exploratory defaults, one run, duplicate runs, an unknown policy and an
    unknown run;
  - mixed benchmarks;
  - configuration differences.
- 3 new frontend tests cover the summary, run choice and change wording.
- The release Chromium journey now measures from the panel and checks the "not one
  unchanged setup" warning.

Results:
- Full Python suite: 1,529 passed, 22 skipped (live services not configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, the API type generation check (no drift),
  TypeScript, Oxlint, 94 frontend tests, the production build, and the release and guided
  journeys passed.

## Step 26: pilot offer and trust package drafts (Phase 0, 1.9)

The strategy's next concrete work after distribution is a short pilot offer with exact
scope and prerequisites, and 1.9 asks for a trust package a buyer's security reviewer can
use without a founder call. Both are drafted from the product as built:
- **[Pilot offer](pilot-offer.md):** for a prospective design partner's engineering lead.
  It covers what the pilot delivers, a supported-scope table, what is out of scope,
  prerequisites, session structure, success criteria agreed up front, and data handling.
- **[Trust package](trust-package.md):** for a security reviewer. It covers:
  - deployment modes and what leaves the environment in each;
  - a data-flow diagram;
  - what uploads, shared records and notifications contain;
  - a retention and deletion table taken from the code;
  - access control, secrets and security engineering;
  - questionnaire answers;
  - a draft compatibility and deprecation policy;
  - what must still be verified in the target deployment.

Owner decisions are marked in brackets rather than invented: price and terms, support
hours and response times, hosting region and subprocessors, encryption at rest, backup
retention, notification timelines, deprecation window, advisory channel, and certification.
Neither document makes a compliance claim. While writing them, two draft statements were
corrected against the code: run uploads are recorded with their uploader but are not audit
events, and webhook payloads include the candidate's label.

Not included: legal review, a DPA template (needs counsel), screenshots or a demo video,
and pricing.

## Step 27: one reusable CI step (1.9)

1.9 asks for install, version and checksum pinning, record and check in one reusable
GitHub step that the `init` templates call. Step 8 built only the pinned install. Now:
- **`.github/actions/retrieval-check/`** is a composite action that `vectraudit init`
  and workbench release exports both write. It installs through the existing pinned
  installer, checks out the target branch, records both sides on the same runner, checks
  them and uploads the report. Outputs are `verdict` and `exit-code`.
  - `connection` mode suits an `init` project and validates the dataset first.
  - `release` mode suits an exported release configuration and keeps the approved and
    proposed semantics.
- **Both generated workflows are now one step** after checkout and Python setup. Python
  pipelines also keep their customer-owned dependency step.
- **Secrets are scoped per command.** Each side's secrets arrive as `baseline-env` and
  `candidate-env` inputs, one `NAME=value` per line, and reach only that side's record
  command. The GitHub token reaches only `check`. The installer sees neither, and the
  download token never reaches a record or check. Inputs travel as environment
  variables, never interpolated into shell. Paths must be relative to the repository,
  and runner, loader and git variables are refused as secret names. Errors name a line,
  never a value, and values are masked in logs.
- **The pinned interpreter runs every command** (`python -m vectraudit`), never another
  `vectraudit` on `PATH`.
- **Upgrade path:** `vectraudit init --ci-only --force` rewrites only `.github/`, so an
  existing project can pick up a newer action without touching its connection, policy or
  queries.

Tests run the generated script on real git checkouts with the real CLI. The `init` project
passes, then fails when the pull request moves to a lossy index while the baseline still
reads the target branch's file. An exported release configuration passes in release mode.
Unit tests cover input validation and the environment each command sees.

Not verified: a run on GitHub's hosted runners. The action uses only documented composite
features (nested local actions, `if: always()`, step outputs), but nothing here executes
GitHub Actions itself. Run it once in a design partner's or a staging repository before
calling the wrapper accepted. GitLab stays deferred until 1.7 needs it, as the strategy
says.

## Step 28: exceptions in the workbench (1.10)

1.10's acceptance asks that the CLI, CI, workbench and hosted views all show an exception.
Step 11 covered the CLI, CI (`check cloud:`) and hosted views; the workbench showed none.
Exceptions need an audited approver, so they stay hosted-only, and the workbench shows
rather than applies them:
- **Who may accept each failure:** every comparison now returns, per failing rule, the
  role that may accept it under that comparison's policy and the longest expiry, or why
  nobody can (no `exceptions` block, kind not opened, safety never waived). Local, no
  network. Core: `evaluation.exceptions.acceptance()`.
- **What the hosted project holds:** a read-only local route,
  `POST /api/guided/exceptions/hosted`, reads the connected project's exceptions and
  policy and reports each failing rule's state:
  - covered;
  - expired, used up or revoked, so a lapsed acceptance resurfaces as blocking;
  - active but not opened by the hosted policy;
  - not applicable to errors and incomparable results;
  - none.
  It predicts the hosted verdict only when the hosted policy hash equals the comparison's.
  The local verdict and check file never change.
- **UI:** a **Failing rules** section under the comparison verdict, with a button to check
  the hosted project. The browser smoke covers the unconnected path; an ad hoc browser run
  with a stubbed hosted response checked the covered state at desktop and phone widths.

Not included: granting or revoking from the workbench. A local comparison is not a hosted
check, and exceptions are granted from a failing hosted check by an approver signed in
there. Reopening cases on expiry also stays out, as in Step 11.

## Step 29: monitor staleness alerts (1.7)

A scheduled monitor that stops running (its cron job removed, its token expired, its runner
unable to reach the service) created no run, so the hosted status kept showing its last
result as if it were current. Nothing alerted. Now:
- **A declared schedule:** per project, how often a monitor run is expected plus a grace
  period (`monitor_schedules`, migration 0011). It can be set from the dashboard
  (Notifications → Monitor schedule), `vectraudit cloud monitor --every 24h --grace 1h`
  or `PUT/DELETE .../monitor/schedule`. Engineers and owners can change it, and each change
  is audited. Workspace exports include it.
- **Overdue on read:** `GET .../monitor` returns the schedule with `due_at` and
  `overdue`. The project page shows everyone, viewers included, an **Overdue** card
  warning that the latest result may be stale. Silence before a schedule was declared
  does not count.
- **One alert per silence:** each `deliver-webhooks` pass (or `vectraudit-cloud
  check-monitors` from cron) opens a staleness alert for overdue projects and sends
  `monitor.stale_opened` to webhooks subscribed to monitor events.
  - It is keyed by the last run before the silence plus the schedule's start, so repeated
    passes notify once.
  - The next monitor run resolves it (`monitor.stale_resolved`, naming that run), and so
    does a schedule change that makes the project no longer overdue, or removing the
    schedule.
  - The schedule row is locked during evaluation, so a sweep and an upload cannot both
    open or both resolve.
  - Payloads carry times and IDs only.

The 1.7 acceptance ("a scheduled monitor regression reaches the owning team's channel
once, and resolves when the next run passes") now also covers a monitor that stops
reporting. Proving it on the first customer's real channel is still open.

Not included: per-benchmark or per-monitor schedules (one expectation per project), cron
expressions, and quiet hours. Choose from how the first customer schedules monitors.

## Step 30: actual versus planned spend (1.5)

Step 19 previewed calls and spend before a run; nothing reported what a run actually used.
Runs recorded `cost_usd` averaged over repeats, but not how many calls were made, so warm-up
calls, retries and failed attempts were invisible. Now:
- **Counted where calls happen:** the runner counts every pipeline invocation per question
  (warm-ups, repeats, retries, failures) and sums the adapter-reported cost of successful
  calls. Each question gets a `usage` block, and each run a total.
  - A resumed run sums every recorded attempt.
  - Interrupted attempts, and runs recorded before usage existed, are reported, so the
    totals are marked as lower bounds rather than silently low.
- **Unknown stays unknown:** failed calls and calls that reported no cost are counted as
  unpriced, and the spend is marked incomplete. They are never counted as $0.
- **Beside the plan:**
  - `workbench run` prints, per candidate, calls and reported spend against the Step 19
    plan, with retries, failed calls and unpriced calls named. It stores the plan in the run
    file under `usage.planned`.
  - `optimize` totals every tuning and held-out run against the planned range.
  - `monitor` prints the comparison before its JSON verdict, which stays the last line.
- **Sharing boundary:** usage is not added to hosted uploads; the allowlisted projection is
  unchanged.

Not included: a browser view of usage (guided runs record it in their run files), and
billing-grade token accounting. Spend is whatever adapters report.

## Step 31: concurrent optimization (1.5)

`workbench run` honoured each candidate's `concurrency`, `max_rps` and the retry policy, but
`optimize()` ran every candidate serially with default retries. An optimization over
several candidates on a 1,000-question suite was therefore the slowest path, and it was
measured under different conditions than the runs it is compared with.
- **Per-candidate execution:** `optimize(..., execution={name: {concurrency, max_rps}},
  retry=...)`. Every run of a candidate (tuning, held-out validation, and the baseline's
  held-out run under a policy) uses that candidate's settings. Non-thread-safe adapters are
  still refused above concurrency 1.
- **CLI:** `workbench optimize` passes the manifest's settings and takes `--max-attempts`
  and `--retry-budget`, like `workbench run`. The Step 19 plan and Step 30 usage use the
  same retry policy.
- **Honest latency:** each candidate report records its concurrency. When candidates
  differ, `latency_note` says that p95 values (and the `max_p95_ms` constraint) compare
  latency under different load. Quality ranking is unaffected.

Validation: deterministic results match serial runs apart from timing. Tracked in-flight
calls show concurrency is actually used on every run of the candidate and not on others.
The retry policy applies, and unknown candidates and invalid levels are refused. The CLI
passes manifest settings through.

Not included: running several candidates at the same time. Candidates still run one after
another, each with its own concurrency, so one endpoint's rate limits are never exceeded by
two candidates at once.

## Step 32: two evaluations at once (1.5)

The PRD targets two concurrent workspace runs; the workbench ran one evaluation at a time
and refused a second with "already running". Now:
- **Two at once:** the workspace runs up to two evaluations (`MAX_EVALUATIONS`); a third
  start gets a clear 429 naming the limit. The browser tracks each evaluation separately,
  with its own progress and cancel, and explains why **Run another evaluation** is disabled.
- **One per registered pipeline:** a second simultaneous evaluation of the same registered
  pipeline is refused (409). Its declared `concurrency` and `max_rps` describe one run, and
  two would double the load on its endpoint without anyone choosing that. Built-in hybrid
  runs are independent and may overlap. A refused or failed start frees the pipeline at
  once.
- **Honest latency:** each run records `timing.overlapped_runs`. Comparisons, locally and
  in the hosted check, warn when either side ran beside another evaluation, because its
  latency includes that load. The count is a plain integer added to the projection's timing
  allowlist so both surfaces warn alike. Verdict rules are unchanged.

Validation: two registered pipelines were held inside their adapters at the same time; the
same pipeline was refused; the overlap counts were 1 and 1, and 0 for a later run alone. A
third evaluation was refused, the pipeline was freed after a refused start, and the
comparison warning appeared locally and in the hosted projection. The four browser smokes
pass after their waits moved from the run's label (now shown while running) to its
completion. An ad hoc browser run drove two slow registered adapters side by side.

Not included: queueing a third evaluation instead of refusing it, and a cross-run rate
limiter that would let one pipeline run twice within its declared limits.

## Step 33: cohorts from declared context fields (1.4)

1.4 asks for cohorts derived from declared context fields (tenant, language, plan) as well
as tags, "so a change that helps large tenants and hurts small ones is visible". This waited
on a decision about recording context values in runs under the sharing boundary. The
decision taken (owner may revisit):
- **Opt in per field, in the benchmark:** a dataset header declares `cohort_fields`.
  Nothing changes for benchmarks that don't, and declaring is a new benchmark version.
- **Derived cohorts behave exactly like tags:** each question gets `field=value` cohorts
  (`tenant=acme`, `enterprise=true`), recorded in runs as part of its tags. A team could
  already write `tenant:acme` as a tag by hand, so no new kind of data leaves the machine.
  No other context value is recorded. The trust package and dataset docs say that declared
  fields' values become cohort names in uploads.
- **One decision everywhere:** because cohorts travel as tags, local checks, CI, the hosted
  check, calibration, exceptions (`cohort:tenant=small:ndcg.max_drop`) and `dataset stats`
  judge and count the same cohorts. Policy formats and hashes are unchanged.
- **Not written back:** derived cohorts are not authored tags, so exports and round-trips
  keep the benchmark as written. Its version comes from the header and context that
  produce them. List-valued fields (such as `groups`) are refused with a hint to tag groups
  explicitly.

Validation: a candidate that drops one tenant's results passes the suite-wide rule but fails
`tenant=small` (and `enterprise=false`), with the same failing rules on the hosted
projection. Without the declaration, the same change passes because the loss hides in the
average. Bad declarations are refused, and stats count the derived cohorts.

Not included: hashing cohort values for uploads. Pseudonymous names would differ between
local and hosted rule keys and still be guessable for short names, so the opt-in is the
control. Also not included: cohorts from list-valued fields.

## Step 34: paired execution (1.4 execution protocol)

1.4 asks to "define paired execution, cache/warm-up and endpoint/version protocols". Runs
already record repeats, concurrency, runner environment, pipeline and corpus versions,
and comparisons refuse or warn on mismatches there. What was missing is pairing. A
baseline and candidate recorded one after the other put any drift during that time
(endpoint load, cache warm-up, network) entirely on the second run.
- **`execute_paired()` and `workbench run --paired`:** two candidates run interleaved per
  question, alternating which goes first (A,B / B,A), serial and without resume. Both
  records name each other (`timing.paired_with`, `timing.order: "interleaved"`). The
  runner was refactored so paired and ordinary runs share the same record setup and
  finishing, including usage.
- **Comparisons and reports:** `latency.paired` is true only when the two runs were recorded
  with each other. Markdown and HTML reports then say so, and the fields are in the hosted
  projection so the hosted check agrees. Comparisons now warn when warm-up differed.
- **Evidence:** against a simulated endpoint that slows with every call, two identical
  pipelines recorded one after the other showed the second as more than 1.5x slower at p50.
  Paired, they agree within 25%.

Not included: pairing in CI release checks. Each side runs from its own checkout in its own
process, so interleaving needs a coordinator across two processes. Endpoint-version
capture beyond the declared `pipeline_version` is also not included.

## Step 35: discovery ledger template (Phase 0, §10)

The code-only Tier 1 work is done; what remains needs customer inputs. The strategy's next
concrete work (§12, item 3) is to start the discovery ledger. [discovery-ledger.md](discovery-ledger.md)
is the founder's template for it:
- **One row per conversation or pilot session**, with 24 columns that map to the
  strategy's funnel, buying events, alternatives, the §8 pilot metrics (setup, labelling and
  investigation minutes, accepted diagnoses, verified fixes, unprompted repeat runs,
  decisions influenced, support time) and the roadmap item each row bears on.
- **Entry rules:** observed, said and interpreted are kept apart; counts carry
  denominators; failed and abandoned pilots are recorded.
- **Qualification, conversation guide and checkpoint reading:** the §4 defer rules, the
  seven §10 questions, and a table mapping each §10 experiment to the columns that answer it
  and the decision if it fails.
- **Where data comes from:** hosted usage is read from the existing monthly usage export and
  the release timeline, with no new telemetry.

A blank header is in `docs/templates/discovery-ledger.csv`. The filled-in ledger must live
outside the repository; customer names, incidents and traces never enter it.

## Step 36: a shareable sample evidence report (Phase 0, §8)

§8 suggests a "sample evidence report" to reduce trial friction without publishing source
or the package, and §12 item 2 asks for the fail/fix/release demonstration in a form a
prospect can see. The demo existed but needed a guided walkthrough of output folders.
`examples/sample_evidence_report.py` builds one self-contained HTML page from the demo
output:
- a "synthetic demonstration" label that disclaims the numbers and any latency claim;
- the four-step story with figures read from the reports (mean nDCG, the critical question,
  the failing execution and what it returned, verdicts and exit codes) and a summary table;
- both full release reports embedded in sandboxed frames, so the page stays one file.

Checked: no scripts, external requests or local paths in the page; no horizontal scroll at
desktop or phone width in a browser. Two tests build it from generated reports and cover the
missing-demo error.

## Step 37: paired execution in CI release checks (1.4)

Step 34 paired runs in the workbench, but a CI release check still recorded the approved
and proposed sides as two separate commands, minutes apart. Pairing there needs two
processes, because each side imports its pipeline from its own checkout and both may define
the same module.
- **A worker per side:** `release_worker` opens one side's release pipeline from its own
  checkout and environment, then answers one JSON request per line on stdin/stdout. Failed
  calls carry only their stable error code, as run files do, and adapter output goes to
  stderr so it cannot corrupt the protocol.
- **`run_release_paired()` and `vectraudit release run-paired`:** these drive both workers
  through `execute_paired`, with the same options and release metadata as ordinary release
  runs (refactored into shared helpers). Side variables arrive as `VECTRAUDIT_BASELINE__NAME`
  and `VECTRAUDIT_CANDIDATE__NAME`, and each worker sees only its own as `NAME`. Approved
  releases with concurrency above 1 are refused.
- **CI action:** `paired: true` (default false) records both sides in one paired command
  and passes each side's secrets under its own prefix.

Validation: the existing two-checkout test now also runs paired. Both checkouts define
`acme_search` with different behaviour; each side returned its own checkout's result, the
runs name each other, and the check marks latency as paired with the same verdict as the
separate runs. The CLI and the concurrency refusal are covered, as are the environment split
and the action's paired branch, including that no secret appears unprefixed.

Not included: making paired the default in generated workflows. It forces serial runs, so
it stays a per-team choice.

## October 9 code audit

The strategy is deliberately broader than one engineering phase. This table separates
implemented foundations from remaining acceptance work; existing features are not counted
as missing merely because their customer validation is pending.

| Item | Current code evidence | Remaining work |
|---|---|---|
| **1.0 Evidence boundary** | Step 9: structured context/filters/history, optional source identity and coverage, allowlisted hosted projection, preview and opt-in text sharing. Step 23: preview and confirmed sharing from the workbench UI. | Extend the projection when 1.2 adds context/answer records. |
| **1.1 Connection** | Python/HTTP and framework adapters, connection diagnostics, document lookup exist. | Independent engineer setup evidence, and confirming the trace format from a partner's real export. *Updated after Steps 39, 61, 64, 66, 67, 69, 75, 77, 81 (Langfuse, LangSmith, Opik and OpenObserve pulls; OpenLLMetry and OTel GenAI OTLP): OpenInference, LangSmith and Langfuse importers with a capability report exist, and documented commands are tested against the real CLI; fixtures are written from vendor docs, not real exports.* |
| **1.2 Context/answer evidence** | Stage rankings and passage judgments exist; `PipelineResult` has no actual-context, answer or citation records. | Citation relevance to labels, answer judgments in policies or uploads, and calibration on real applications. *Updated after Steps 40–41 and 65: model-context evidence, required evidence, imported answer judgments and cited-document placement exist, kept apart from retrieval quality.* |
| **1.3 Repair loop** | Findings, regression cases, verification and bounded CLI optimization exist. | Customer-selected failure classes, recorded-candidate replay, and measured investigation time with a second engineer. *Updated after Steps 42–47 and 68 (diagnosis written back to Langfuse traces): investigate and try-fix (CLI and browser), protecting the repaired case, investigation records and shared investigations linked from recurring problems.* |
| **1.4 Policy/uncertainty** | Query-family weighting/bootstrap and cross-split protection landed; critical assertions inspect repeats. Step 13: the optimizer screens and approves under the release policy against a baseline. Step 14 averages every repeat and exposes intermittent rankings/stage losses. Step 20: cohort counts on every surface and optional Bonferroni control, measured on A/A runs. Step 21: opt-in non-inferiority with inconclusive kept distinct from pass. Step 33: cohorts from context fields a benchmark declares. Step 34: paired (interleaved) execution and a warm-up mismatch warning. Step 37: paired release runs in CI. | Protocol evidence on a customer's endpoint. |
| **1.5 Execution** | Typed errors, retry budgets, per-candidate concurrency/rate limits, release protocols, checkpoint recovery and 1,000-question artifact/UI bounds landed. Step 19: calls/spend preview and budget guards before run, optimize and monitor. Step 30: actual calls and reported spend beside the plan, retries and failures included. Step 31: optimization runs each candidate at its own concurrency, rate limit and retry policy. Step 32: two evaluations at once in the workbench, one per registered pipeline, with overlap recorded. | Declared customer 1,000-question CI-window acceptance. |
| **1.6 Shared decisions** | Local question text is joined safely. Step 10: a decision ledger keeps release decisions after retention, with a hosted timeline and API. Step 15: cases, benchmark snapshots and policy revisions are shared as immutable allowlisted records. Step 16: plain-language decision summaries and per-benchmark quality/latency trends from the ledger (dashboard, API, CLI). Step 17: scheduled monitor runs upload into the same history with drift and a monitor status. Step 22: recurring problems across checks, linked to shared cases. Step 23: sharing from the workbench UI. Step 24: document titles beside IDs locally and, when shared with text, on hosted cases. | Lead and second-engineer acceptance with real users. |
| **1.7 Delivery** | GitHub publishing exists; `pipelines/monitor.py` writes local history and machine-readable alerts. Step 17: hosted monitor status with a stable alert ID per failing streak and the run that resolved it. Step 18: signed webhooks and Slack incoming webhooks deliver each decision and each alert (opened, resolved) once, with retries. Step 29: declared monitor schedules; a monitor that stops reporting raises one overdue alert, resolved by the next run. | Proof with the first customer's real channel; Slack-app delivery only if asked; additional CI only on demand. *Updated after Step 62: email notifications exist; a live mail-server run does not.* |
| **1.8 Benchmark upkeep** | Incident imports, deduplication, passage review, snapshots and freshness foundations exist. | A customer's production traces, paraphrase matching, a hosted review queue and measured reviewer effort. *Updated after Step 57: cursor-based intake, recurrence-to-case links, a de-duplicated review queue and reviewer routing exist.* |
| **1.9 Delivery foundations** | Step 8 prepares pinned wheel-only distribution independently of source access. Step 26 drafts the trust package and support/compatibility policy. Step 27: one reusable CI step (install, record, check, report) called by both generated workflows. | Provision and verify channel/credentials, a first run of the CI step on GitHub-hosted runners, owner decisions in the drafts, a DPA template with counsel. |
| **1.10 Calibration/exceptions** | Step 11: scoped, audited, expiring exceptions with policy-declared roles, recorded in the decision ledger. Step 12: A/A calibration with false alarms by rule and a reviewable draft policy. Step 25: calibration in the workbench, with configuration differences called out. Step 28: the workbench shows who may accept each failure and the connected project's exceptions against it. | Customer-runner calibration evidence. |
| **Tier 2** | Chunking experiments, permission scenarios and CLI optimization foundations exist. | Customer-selected lineage/freshness or permission lifecycle proof on real exports and change logs. *Updated after Steps 49–59: lineage, freshness, permission paths, access changes, cache isolation, migration report, browser Checks tab and a demo exist, validated on fixtures only.* |
| **Tier 3 / commercial** | Synthetic reference demos and operations/security documentation exist. | Elasticsearch/OpenSearch connector exists (Step 63, owner-chosen, fake-cluster tests); Chroma connector (Step 85, owner-chosen, tested against a real server). Defer further breadth; obtain real pilot inputs, independent adoption, paid continuation, deployment evidence and sustainable support economics. Code cannot establish these. |

## Step 38: adversarial verdict audit (§9 correctness gate)

§9 requires that "no known correctness defect can turn incomplete or incomparable evidence
into an approved release", and the launch scorecard asked for a review before claiming it.
The comparison was probed with run files a buggy producer or a hand edit could create. Two
defects and one gap were found:
- **A "complete" run could contain failed questions** and still load. The comparison then
  crashed (TypeError) or, depending on which questions failed, judged only the rest.
  `check_run` now requires a complete run to measure every dataset question without
  failures.
- **A regression could drop out by losing its scores.** If the regressed questions had no
  metrics on the candidate side only, the comparison averaged the remaining questions and
  **passed**. A question scored on one side only now makes the result not comparable.
- **The comparison validated only newer-format runs.** Duplicate question rows or NaN and
  infinite metrics in an older-format run reached the verdict (NaN passed). Every run is now
  validated before judging, and an invalid run names its side and the problem.

Several test fixtures described impossible states (a question "correctly returning nothing"
that delivered documents); they were corrected, and one test that asserted the old lenient
handling of one-sided scores now asserts the stricter result. Nine regression tests cover
each case. Real producers were unaffected: the full suite, the three browser journeys and
the pilot demo pass unchanged.

This is a review by the same author as most of the code, not an independent one. The
scorecard keeps that gate open until someone else reviews the verdict logic.

## Step 39: OpenInference trace import and capability report (1.1)

Customer traces are not available yet, so this builds the strategy's named first candidate
(OpenInference) with the reader kept separate from matching, stage building and reporting.
If design partners use Langfuse or LangSmith, a new reader feeds the same import.

- `vectraudit workbench import-traces DATASET TRACES` reads OTLP JSON or Phoenix span
  exports, matches traces to benchmark questions by query text or a question ID attribute,
  and writes an ordinary run file, scored by the existing runner and judged by `check` like
  any run. A capability report lists stable IDs, delivered ranking, stages, document flow,
  filters and tenant context, model context, replay and latency, with counts and attribute
  names only.
- **No fabricated stages or edges.** Retrievers are `custom` unless named; rerankers connect
  to earlier stages only through their recorded input documents. A subset input becomes its
  own stage, so a cut before reranking is not blamed on the reranker; an input of unknown
  origin leaves stage coverage partial and losses unattributed.
- **Gaps fail with a cause:** `trace_missing`, `trace_ambiguous`, `trace_no_retrieval`,
  `trace_final_ambiguous`, `trace_documents_unrecorded`, `document_id_missing`. The run is
  then partial and cannot pass.
- **Latency from traces is never judged.** Runs record `timing.source = "trace"` (kept in
  uploads), and the comparison reports traced latency without applying latency rules.
- Document text, prompts and model context stay in the trace file.

One defect was found while writing tests: the reader treated timezone digits as
fractional seconds in ISO timestamps; fixed before commit. The first linking rule (exact
match only) rejected the common case of an application passing only the top retrieved
documents to the reranker; it was replaced by the input-stage rule above.

Not included: context, answer and citation import (1.2); Langfuse or LangSmith readers;
a browser view of the capability report. The fixtures are written from the OpenInference
conventions, not exported from a real application, so the acceptance gate (a supported
instrumented application imports without a custom adapter; a second engineer follows the
documented path) is still open.

Validation: 12 new tests (OTLP and span-record exports, dataframe columns with epoch
milliseconds, metadata IDs and chunk deduplication, every gap code, duplicate traces,
subset and unknown-origin reranker inputs, private text kept out of the run, report and
upload, latency never judged, CLI). Full suite, ruff, and mypy on Python 3.12 and 3.10 pass.

## Step 40: model-context evidence (1.2, first part)

The strategy's §11 story is a document retrieved correctly but missing from what the model
saw. This step makes that loss observable, kept apart from retrieval quality.

- **Contract:** optional `PipelineResult.context` (`ModelContext`: ordered items with
  document ID, optional source reference and `truncated`; `coverage` and `method`). Python
  and HTTP pipelines can report it. Runs store it per question as `model_context` (IDs and
  flags only) and validate it.
- **Analysis:** each relevant delivered document is recorded as reached in full,
  **omitted** (only under observed coverage), **truncated**, or **unknown**. Document recall
  is unchanged, so "recall passes, the model never saw it" is visible.
- **Comparisons:** triage adds `context_lost` (a regression when the baseline's context had
  the document in full), `context_loss` (informational, baseline context unknown),
  `context_still_lost` and `context_recovered`. The workbench's pipeline-run view shows a
  model-context panel.
- **Trace import:** when LLM spans record input messages, delivered documents' recorded text
  is located in the prompt (whole, cut at the start or end, or absent), labelled
  `text_match`. Without recorded document text, context is `partial` and claims nothing.
  The capability report now states for how many questions context was determined.
- Context stays local: not in uploads, and no text is written anywhere.

One defect was caught before commit: run records already use `context` for the request's
tenant context, so the new per-question field is `model_context`.

Not included: answer, citation and answer-judgment import, evidence groups ("A and B both
needed"), and release-policy rules on context losses. These change verdicts and should be
calibrated on real applications. The text-match inference has only been tested on synthetic
prompts.

Validation: 11 new Python tests (contract coherence, omitted and truncated documents beside
perfect recall, partial and missing context, triage regression and unknown-baseline cases,
validation, upload exclusion; trace prompts with trimmed, reordered, start-trimmed and
unrecorded documents and no prompt) and 2 frontend tests. Full suite (1,634 passed, 22
skipped), ruff, mypy on Python 3.12 and 3.10, TypeScript, Oxlint, 101 frontend tests and the
workbench browser journey pass; the new panel was checked in Chromium on an imported trace.

## Step 41: required evidence and imported answer judgments (1.2, second part)

This completes 1.2's separation of the four outcomes without building an answer evaluator.

- **Required evidence:** benchmark questions may declare `required_evidence`, alternative
  groups of relevant documents (any one complete group suffices). It is validated (documents
  must be relevant; not on no-answer questions), round-trips through exports, changes the
  dataset version only when present, and is dropped from permission scenarios and incident
  imports, which relabel questions.
- **Sufficiency:** runs record per question whether a complete group was delivered and
  whether one reached the model uncut (`sufficient`, `insufficient`, `unknown`,
  `unrecorded`). A trimmed exception now reads as insufficient context while document
  recall passes, which meets the strategy's first 1.2 acceptance case.
- **Answer judgments:** `vectraudit workbench attach-answers RUN JUDGMENTS --out` reads one
  judgment per line from human review or tools, keyed by question or trace ID (imported runs
  now keep each question's trace ID locally). Scores need a declared threshold. Humans
  decide over models, disagreements become `disputed`, abstentions and handoffs are kept
  apart, and model evaluators' agreement with human review is reported. Explanations are
  not read.
- **Diagnosis:** each judged question gets its first observable loss with supporting facts:
  answered wrong despite evidence, lost in context, lost downstream, not observed, context
  unknown, and the correct, abstained, disputed and unjudged cases. Comparisons add
  `answer_regressed`, `answer_recovered` and `answer_not_correct` findings. The workbench
  shows an answer panel.

Not included: citations and citation support, answer judgments in release policies or
uploads, readers for specific tool export formats (the JSONL shape is generic), and
calibration on real judgments.

Validation: 14 new tests (required-evidence validation and round trip; a missing
exception detected while recall passes; a planted context omission and a planted wrong
answer diagnosed separately, alongside downstream, unobserved and unknown-context cases;
human precedence, disputes, thresholds and agreement; abstention and handoff; trace-keyed
and unmatched judgments; triage regression; invalid input; CLI). Full suite (1,648 passed,
22 skipped), ruff, mypy on Python 3.12 and 3.10, TypeScript, Oxlint, 101 frontend tests, the
production build and the workbench browser journey pass; the answer panel was checked in
Chromium.

## Step 42: investigate a failure and try a declared fix (1.3, first part)

The repair loop from a failed question to a validated fix, on the command line; the
browser screen follows in the next step.

- **Declared fixes:** manifest candidates list `fixes`: named parameter overrides with the
  failure classes they address (`not_observed`, `lost_downstream[:stage kind]`,
  `lost_in_context`, `answered_wrong_despite_evidence`). Only declared fixes are suggested,
  permission and no-answer failures cannot be targeted, and fixes leave the configuration
  identity unchanged. The hybrid baseline declares built-in fixes it can actually run.
- **`workbench investigate`:** losses in path order (including context losses and answer
  diagnoses from Steps 40–41), hypotheses labelled as unestablished, cautions (corpus
  before embeddings; filters may be intended; permissions are never a repair), explicit
  uncertainty, matching fixes, and new-versus-baseline losses.
- **`workbench try-fix`:** current and fixed configurations interleaved on the failing
  questions plus a seeded exploratory sample, with a call plan and `--max-calls`. Results
  per question: recovered, moved to a later stage, introduced losses and nDCG changes.
  Labelled exploratory; held-out questions are refused.
- **`--validate --policy`:** both configurations on held-out questions under the saved
  policy; "approved under policy" only on a pass, with the merged parameters to promote.

One design correction before commit: comparing losses by class and document counted a loss
that moved to a later stage as both recovered and regressed. Losses are now compared per
document, and moved losses are reported separately.

Not included: the investigation screen and "try this fix" buttons in the workbench, saving
the promoted candidate into the manifest automatically, recorded-candidate replay (cheap
reranking or packing experiments without calling the pipeline), and measured investigation
time.

Validation: 9 new tests (loss, hypothesis and fix matching; a moved loss is not a recovery;
recovery and a planted regression both reported; seeded plans that refuse held-out
questions; approval only on a held-out policy pass, refusal by a regression rule, and
exploratory status without a policy; no validation without a held-out split; fixes cannot
target safety failures or change identity; hybrid built-ins; CLI with call limits). Full
suite (1,657 passed, 22 skipped), ruff, and mypy on Python 3.12 and 3.10 pass. Both commands
were also run on the shipped example workbench.

## Step 43: investigation screen with "try this fix" (1.3, second part)

The browser side of Step 42, in the guided workflow where findings already appear.

- **Investigate this question** on any finding opens a panel: question text, losses, new
  losses since the baseline, unestablished causes, cautions, uncertainty and the pipeline's
  declared fixes.
- **Try this fix** shows its planned call count and runs a paired exploratory trial as a
  workspace job; results list recovered, moved and newly lost documents per question, with
  regressions highlighted. HTTP pipelines need the usual transfer confirmation.
- **Validate** requires explicit held-out confirmation and judges the fix under the
  comparison's release policy; without one the result is labelled a measurement, not an
  approval. An approved fix shows the candidate parameters to promote.
- New local API (`/api/guided/investigate`, `/fixes/try`, `/fixes/validate`,
  `/fixes/{id}`) under the session token. A fix runs only against the configuration that
  produced the run (same benchmark revision, corpus version or registered configuration);
  otherwise the panel explains why. Held-out questions cannot be tried.

Not included: writing the promoted candidate into the manifest, cheap replays of recorded
candidates, the investigation in the hosted service, and measured investigation time.

Validation: 4 new API tests (investigate, try and validate a fix end to end with policy
approval; changed benchmark and unknown fix refused; held-out questions reserved; session
token required), 3 frontend unit tests, and the guided browser journey extended to
investigate a planted loss, try the built-in depth fix (recovered, no regressions) and
measure it on held-out questions. Full suite (1,661 passed, 22 skipped), ruff, mypy on
Python 3.12 and 3.10, TypeScript, Oxlint, 104 frontend tests, the build and the guided,
market-fit and release journeys pass.

## Step 44: protect a repaired context case, and demonstrate the whole story

The strategy's §11 story ends with "the next release now checks this case automatically",
which Steps 40–43 could not yet do: release policies ignored context.

- **`critical_context` policy rule:** a named question's required evidence must reach the
  model in full on every execution. Runs record sufficiency per repeat; a demonstrated loss
  on any execution fails, unrecorded or unknown context is not comparable, and so are
  questions without required evidence and older runs. Exceptions treat it as a critical
  rule; uploads carry per-execution states so hosted checks agree; the decision ledger
  counts it with other critical failures. Existing policy hashes are unchanged.
- **`examples/context_story_demo.py`:** a synthetic support assistant whose packing budget
  trims the account-recovery exception. Through product code: human review marks the answer
  wrong while recall is 1.00; the run shows the evidence delivered but insufficient in
  context; the diagnosis is `lost_in_context`; the declared packing fix is tried (1 of 1
  recovered, 0 regressions) and approved on a held-out question under the policy; the fixed
  release passes a `critical_context` policy and a later release reverting the budget fails
  it. It writes a self-contained page, labelled synthetic, for prospects (§8 acquisition).

One test correction: the legacy policy-hash test rebuilds the old canonical form from the
dataclass, so it now drops the new empty field as it does for earlier optional fields. The
real hash was unchanged.

Not included: an aggregate context-loss metric in policies (still needs calibration on real
applications), a policy-editor control for the rule (JSON import works), and a video or
hosted copy of the page.

Validation: 8 new tests (every-execution failure, unknown and ungrouped context never
passing, hosted parity, validation and hash stability, old runs, ledger counts, and the demo
end to end). Full suite, ruff, and mypy on Python 3.12 and 3.10 pass; the page was rendered
in Chromium.

## Step 45: investigation records for pilot measurement (1.3 acceptance)

1.3's acceptance is measured: investigation time, accepted and disputed diagnoses, and
repairs reproduced by a second engineer. §11 also asks for "a way to dispute a diagnosis or
request missing evidence". Neither existed, so a pilot could not produce these numbers
without manual timekeeping.

- **Records:** opening an investigation in the workbench (after entering a name) starts a
  record in `.workbench/investigations/`. **Accept diagnosis**, **Dispute** (reason
  required), **Request evidence**, automatic linking of each finished fix trial and
  validation (read from the workspace, same run only), and **Close investigation** with an
  outcome. *Fixed* requires an approved validation. Closed records cannot change.
- **Measures:** minutes to the first accepted diagnosis and to the first approved
  validation; reproduction by a second engineer (approved validations of the same fix by two
  names, at least one not the opener). `vectraudit workbench investigations <workspace>`
  and `GET /api/guided/investigations` summarise them with denominators and sample sizes;
  the discovery ledger now points to it.
- Records hold IDs, names, reasons and timestamps, never question or document text. Names
  are self-reported; the workbench has no sign-in.

Not included: sharing records with the hosted project, verified identities, and
investigation records from the CLI-only `investigate`/`try-fix` path.

Validation: 3 new API tests (the full loop with dispute, evidence request, linking,
refusals, reproduction, closed-record immutability, summary and CLI; results from another
run refused; reproduction rules), 1 frontend test, and the guided browser journey extended
to record a name, accept the diagnosis, link the trial and measurement automatically and
close the investigation. Full suite, ruff, mypy on Python 3.12 and 3.10, TypeScript,
Oxlint, 105 frontend tests, the build and the guided, market-fit and release journeys pass.

## Step 46: protect the repaired question, and keep context rules through edits

§11's fifth decision is "protect and return": the verified configuration and the case it
fixed should be checked automatically from then on. After Step 45 that still meant typing a
rule into the policy editor by hand, and the editor could not express Step 44's rule.

- **Defect fixed:** the workbench policy editor kept only `exceptions` among the blocks it
  does not edit, so editing and saving a policy with `critical_context` dropped the rule
  silently. It now has context assertions (**Add context assertion**), validates them, and
  keeps them through every save. Stale help text claiming critical rules judge only the first
  ranking was corrected.
- **Protect this question:** on an investigation closed as fixed, the panel adds a context
  rule and/or a rank rule for the question to a saved policy, as a new version noting the
  investigation. Duplicates are no-ops; conflicting ranks are refused for the editor;
  context rules are refused when the pipeline records no model context; ranks above the
  run's k and the read-only startup policy are refused. The investigation records which
  policy version protects the question.

Not included: protecting from the CLI-only path, and automatically selecting the new policy
version for the next comparison (the panel says to select it).

Validation: 3 new API tests (protection in a new version with existing rules kept, duplicate
and conflicting rules, context refusal without context records, startup and k limits; only a
verified fix can protect; rule builder never mutates its input), 1 frontend test for the
editor round trip, and a browser journey under a saved policy: investigate, try the fix,
validate (approved), close as fixed and protect, confirmed through the API to produce
version 2 with the rule. Full suite, ruff, mypy on Python 3.12 and 3.10, TypeScript, Oxlint,
106 frontend tests, the build and the guided, market-fit and release journeys pass.

## Step 47: shared investigations linked from recurring problems (1.6)

1.6 asks the hosted decision view for "findings that recur across runs, with links to
investigations". Recurring problems existed (Step 22), but investigations (Steps 45–46) were
local, so a lead could not see that a recurring failure was being investigated, or had been
fixed and protected.

- **A fourth shared-record kind, `investigation`:** question and run IDs, loss classes,
  diagnosis status, evidence-request count, trial and validation results, outcome,
  protecting policy versions, and times to an accepted diagnosis and a verified fix.
  Reviewer names, dispute reasons, evidence requests and notes are withheld; the
  question-text setting does not apply. The allowlist is checked on both ends as for other
  records.
- **Sharing:** `vectraudit workbench share WORKSPACE investigation ID`, or **Share** in the
  investigation panel (preview first, as for cases).
- **Hosted:** recurring problems (page and API) gain an *Investigation* column linking each
  question's newest shared investigation with its outcome and whether it is protected;
  investigations have a record page and a records filter. The trust package lists the new
  upload.

One defect avoided before commit: the hosted records list summarised every unknown kind as
a policy and would have crashed on an investigation.

Not included: live status (shared records are snapshots) and sharing names under any setting.

Validation: 4 new tests (no names or free text in the projection; tampered fields refused;
the share panel never sends text for investigations; hosted recurring problems link the
investigation, and the list and record pages render without private text). Full suite (1,679
passed, 22 skipped), ruff, mypy on Python 3.12 and 3.10, TypeScript, Oxlint, 106 frontend
tests, the build and the guided journey pass.

## Step 48: up-to-date pilot offer and an owner decision brief

No code. The remaining work is owner and customer action, so this step prepares it.

- **Pilot offer corrected.** It still listed trace import and judging the model's context
  and answers as "not part of this pilot", and named none of the investigation and repair
  work from Steps 39–47. Its capability table, exclusions, sessions and success measures now
  describe what exists, with limits stated: OpenInference only, context inferred from
  prompts where traces are used, answer judgments imported rather than computed, and
  inferred results checked against the customer's own review.
- **[Owner decisions](owner-decisions.md):** every open bracket in the pilot offer and trust
  package, with options, a recommendation and the reason, ordered by what each blocks.
  Recommendations are proposals; none is applied to customer-facing documents. The central
  one: run the first pilots local or self-hosted only, which removes most trust-package
  brackets and keeps everything inside the customer's environment. No price is recommended
  without market data; the brief proposes testing two price points instead.

## Step 49: source-to-context lineage (2.1, first part)

The owner asked to keep building past the customer-independent plan, so this starts the
strategy's preferred deeper bet. Its limits stay stated: it is validated on fixtures, not on
a customer's ingestion pipeline.

- **[`vectraudit workbench lineage`](lineage.md):** reads a customer-exported lineage
  manifest (per source: revision, extracted length, chunk ranges, index state and revision,
  optional chunk hashes) and a benchmark with passage labels, optionally with a run. Each
  passage gets its first loss: source missing, empty or truncated extraction, not chunked,
  split across chunks, not indexed, stale index, not retrieved, lost downstream, or lost in
  context.
- **Advice follows the evidence:** ingestion losses produce "fix ingestion before changing
  retrieval"; boundary splits suggest overlapping or section chunking; stale chunks suggest
  re-indexing. `--fail-on-ingestion` exits 3 for CI.
- **Corpus findings:** empty extraction, stale chunks still indexed beside current ones,
  duplicate chunks, and questions where duplicates took result slots.
- The strategy's 2.1 acceptance is reproduced on a fixture: a chunk-boundary defect
  produces a finding, and the repaired manifest recovers the passage.

One defect found by the tests: following runs through their per-question context analysis
missed trimmed chunks for span-labelled questions; lineage now reads the raw context record.

Not included: freshness over time across exports, the workbench view, linking lineage into
investigations, and evidence from a real ingestion export.

Validation: 23 new tests (every stage, missing sources, unverified revisions, the boundary
defect and its repair, run integration through retrieval, delivery, omitted and trimmed
context, source-ID runs, corpus findings, manifest validation, CLI and CI gating). Full suite
(1,702 passed, 22 skipped), ruff and mypy on Python 3.12 and 3.10 pass.

## Step 50: knowledge freshness (2.1, second part)

- **[`vectraudit workbench freshness`](lineage.md#knowledge-freshness):** reads a content
  change log (update with a new revision, replace with a successor, delete, archive; each
  with a time and optional propagation window) and a lineage export taken afterwards. Each
  change is `propagated`, `pending` (inside its window, not a failure), `overdue` (new content
  not indexed after the window) or `lingering` (old revision or removed source still
  indexed).
- **With a run:** stale chunks that reached a question's results or model context, and
  conflicting revisions of one source delivered together. Questions tagged `historical` are
  exempt, as the strategy asks customers to define precedence. `--fail-on-stale` exits 3.
- The strategy's freshness acceptance is reproduced on a fixture: a stale answer served from
  an old revision is reported, and after re-indexing the same check passes.

Not included: tracking propagation across a series of exports (time to propagate),
caches outside the index, and validation on a customer's change log.

Validation: 14 new tests (every status for updates and the window, replacements, deletions
and archives, per-change windows and superseded changes, stale results and context with
conflicts, the repair, historical exemptions, invalid change logs, CLI and CI gating). Full
suite, ruff and mypy on Python 3.12 and 3.10 pass.

## Step 51: lineage in investigations (2.1 meets 1.3)

- **`workbench serve --lineage lineage.jsonl --lineage-ids chunk|source`** registers a lineage
  export at startup, validated immediately. `workbench investigate --lineage` does the same
  on the command line.
- **The investigation panel** gains a *Source lineage* section listing each labelled
  passage's first loss. Ingestion losses become `ingestion` losses with their stage, the
  lineage explanation joins the possible causes, and a caution says to fix ingestion first.
- **Retrieval fixes are withheld** for a document whose passage was lost during ingestion
  ("raise candidate depth" cannot fetch text that is not indexed intact); fixes for other
  losses on the same question remain.

Not included: freshness findings in the panel (they need a change log registered too), and a
lineage view outside investigations.

Validation: 4 new Python tests (ingestion losses withhold fixes and add causes and cautions;
an intact passage keeps them; the workbench API with a registered export; startup refusal of
a broken export or ID mode; the CLI) and 1 frontend test. Full suite (1,720 passed, 22
skipped), ruff, mypy on Python 3.12 and 3.10, the frontend checks and build, and the guided
journey pass; the panel section was checked in Chromium with a registered export.

## Step 52: freshness in investigations (2.1 meets 1.3)

- **`workbench serve --changes changes.jsonl`** (beside `--lineage`) registers a content
  change log at startup, validated immediately; a change log without lineage is refused.
  `workbench investigate --changes` does the same on the command line.
- **`question_freshness`** judges the changes to sources one question touched (delivered,
  labelled relevant, or replacing one of those) at the time the run was recorded.
- **The investigation panel** gains a *Knowledge freshness* section. Stale content served
  becomes a `stale_content` loss (results or context), two revisions delivered together a
  `conflicting_versions` loss, with a re-index-or-purge caution; neither is a fixable class.
  Labelled content whose new revision or replacement is not indexed gets no retrieval fix.
  Pending changes, historical questions and source-level IDs are stated as uncertainty.

Not included: propagation across a series of exports, caches outside the index, and a
customer's change log.

Validation: 7 new Python tests (stale context and conflicts as losses; unindexed labels
withhold fixes; pending changes; historical exemption; source-ID mode; the workbench API;
startup refusal; the CLI) and 1 frontend test. Full suite (1,727 passed, 22 skipped), ruff,
mypy on Python 3.12 and 3.10 and the frontend checks and build pass; the panel section was
checked in Chromium on the starter example with a registered change log.

## Step 53: where forbidden documents travelled (2.2, first part)

- **`permission_path`** on every permission-checked question: each forbidden document seen
  in any execution, the stages it reached as output or input, and whether it reached the
  results or the model context (unknown when context was not recorded in full).
- **Release checks:** a forbidden document in the model context fails even when the results
  are clean; a policy's new `forbidden_stages` (stage kinds such as `rerank`) fails one that
  reaches those stages, and is not comparable on a run without a stage trace. Both use the
  existing `safety.forbidden:<question>` rule, so exceptions treat them as safety failures.
- **Fixed:** forbidden documents now match by a hit's source ID too, so a forbidden source
  returned as a chunk is caught.
- Findings and investigations name context leaks; the hosted projection carries document
  IDs, stage kinds and flags; run files validate the record; the policy editor keeps the
  field.

Not included: permission changes over time (revocation and membership updates with a
propagation window), cache isolation between principals, and a customer-built scenario.

Validation: 7 new Python tests (the path, the planted reranker-before-access-control defect
caught only by the boundary, a context leak with clean results that cannot be waived,
unrecorded context and untraced runs, chunk-level source matching, the hosted projection,
malformed records and policies) and 1 frontend test. Full suite, ruff, mypy on Python 3.12 and
3.10 and the frontend checks pass.

## Step 54: access changes over time (2.2, second part)

- **An access change log** (`revoke`, `grant`, `move`, `archive`, each with a propagation
  window) is read from the customer's identity or content system. A principal matches a
  question through its context, including list-valued groups; the latest change in effect
  per principal and document wins; changes inside their window are pending.
- **`vectraudit workbench access --write-dataset`** applies the log to a benchmark: revoked
  documents become forbidden, granted ones stop being forbidden, revoked labels are removed
  and listed for review, and affected questions are tagged `access-change`. Release checks
  then enforce the new access, including model context and `forbidden_stages` (Step 53).
- **`--run`** checks a run recorded after the changes: revoked documents still reaching a
  stage, the results or the model context are violations (`--fail-on-leak` exits 3);
  pending changes and granted documents never delivered are reported, not failed.

The strategy's acceptance, on fixtures: a planted revocation defect (the application's
access index never received the revocation) fails the release check and the run check,
and passes once the index is corrected. A revoked document handed only to the model is
still caught.

Not included: a cache-isolation scenario between principals, propagation time across a
series of runs, the workbench UI, and a customer's change log.

Validation: 12 new tests (principal matching, applying a revocation with label review,
the planted defect and its correction, a context-only leak, pending changes, moves, grants,
archives and ordering, invalid logs, the CLI). Full suite, ruff and mypy on Python 3.12 and
3.10 pass.

## Step 55: cache isolation (2.2, third part)

- **`vectraudit workbench cache-probe MANIFEST`** pairs each question that has forbidden
  documents with the same question (family or text) under another context, then calls the
  primer and the target serially through the manifest's pipeline. Any forbidden document
  the target receives, in results, model context or a recorded stage, is a leak.
- **Primer first, by design:** a test shows that asking the target first fills a text-keyed
  cache with its own clean answer and hides the defect, and another shows a normal run in
  benchmark order already leaks through such a cache, so a normal run is no clean
  reference. Each leak instead records whether the target's answer was identical to the
  primer's (a shared cache entry, or an access check ignoring the context) or not (the
  access check itself).
- Failed calls make the probe incomplete, never clean; the number of calls is bounded and
  stated before running.

The strategy's acceptance, on fixtures: a cache keyed on question text alone leaks the
payroll document to the restricted user; keyed on text and user, the probe is clean.

Not included: clearing the application's caches (probe a fresh deployment), concurrency
between principals, the workbench UI, and a customer-built scenario.

Validation: 8 new tests (pairing by family and by text, the planted defect and its
correction, why the target must not go first, why a normal run is no reference, a leak
unlike the primer's answer, failed calls, the call bound, the CLI). Full suite, ruff and
mypy on Python 3.12 and 3.10 pass.

## Step 56: migration report (2.5, "migration assurance")

- **`vectraudit workbench migration CURRENT PROPOSED DATASET`** compares two
  customer-configured systems recorded on the same benchmark (others are refused) under the
  release policy, and adds what a migration needs:
  - corpus equivalence from each system's inventory (IDs, hashes, metadata; text refused),
    with labelled documents missing from the proposed system named as a corpus gap;
  - ID mapping: IDs a system returned that its inventory lacks, or mostly unknown IDs;
  - filter semantics: returned documents whose metadata fails the question's filter under
    the portable semantics;
  - environment differences side by side, and measured spend beside declared estimates.
- **"Ready to migrate"** needs the policy to pass and no concern left; missing inventories
  are concerns, so unverified equivalence never reads as ready. `--fail-unless-ready` gates
  CI. The pilot offer now lists migration assurance, ingestion and freshness, and the
  permission lifecycle checks.

Not included: running both systems from the report (record the runs first, paired if
possible), browser views, and a customer migration.

Validation: 10 new tests (an equivalent migration is ready; an ignored filter is caught
while metrics hold; unmapped IDs; a labelled document missing from the new corpus and a
changed document; unverified checks without inventories; environment and cost disclosure;
different benchmarks refused; inventories refuse text; dotted range filters; the CLI).
Full suite, ruff and mypy on Python 3.12 and 3.10 pass.

## Step 57: production intake (1.8)

- **`vectraudit workbench intake TRACES DATASET --state DIR`** runs on each new
  OpenInference export: a cursor (newest start time plus the trace IDs at that instant)
  takes only new traces from overlapping exports.
- **Matched questions:** a production ranking with none of the question's labelled relevant
  documents is a likely recurrence, linked to the question's regression cases; verified
  cases are flagged for reopening. `--fail-on-recurrence` exits 3.
- **New questions:** a de-duplicated review queue (text plus named context fields) with
  counts, first and last seen and trace IDs; priority from frequency and feedback
  (`negative`, `escalated`, `positive`), which never becomes labels; a seeded independent
  sample chosen before any signal and kept in capped exports; reviewer routing by context;
  `promoted` once the question enters the benchmark. `--export-review` writes an unlabelled
  benchmark for passage review.
- Ambiguous matches, missing question text and unreadable rankings are counted, not judged.

Not included: matching paraphrases, measuring reviewer minutes, a hosted queue, and a
customer's production traces.

Validation: 8 new tests (recurrence and de-duplicated queue, feedback never labelling,
overlapping exports and same-instant traces, promotion, case links and reopening, an
independent sample stable under signals and kept in a capped export, ambiguous text,
invalid feedback and routes, the CLI). Full suite, ruff and mypy on Python 3.12 and 3.10
pass.

## Step 58: the deeper checks in the browser (§2 "recurring work leaves the browser")

- **Checks tab:** the workbench recognises the JSON reports of lineage, knowledge freshness,
  access changes (applied and checked), the cache probe, migration and production intake,
  from the workspace folder or a file import, and shows each as a card: pass, fail or
  needs attention, a headline and the specific findings, so a lead can read them without
  the terminal.
- **Read-only by design:** the checks still run from the command line where the customer's
  exports and credentials are; the browser never names files to run against.
- **Shared fixtures:** the frontend is tested against real reports produced by the Python
  checks on synthetic data, and a Python test fails if a report's shape drifts from its
  fixture.

Validation: 7 frontend tests and 6 fixture contract tests. Full suite (1,778 passed, 22
skipped), ruff, frontend checks and build, and the workbench and guided browser journeys
pass; all six reports were rendered in Chromium from a workspace folder.

## Step 59: a demonstration for the permission and freshness offer (§4, §8)

- **`examples/reliability_story_demo.py`** runs three planted defects through product code
  on a tiny synthetic assistant:
  - a refund-policy update whose old revision stays indexed and reaches the model context
    (`lingering`, conflicting revisions), cleared by re-indexing;
  - an access revocation from a change log that the assistant's own index never received
    (the release check fails on the forbidden document in the results and context, and
    passes after the sync);
  - a cache keyed by question text that leaks one user's payroll answer to another, clean
    once the user is in the key.
- It writes `reliability-story.html`, labelled synthetic, as a leave-behind for this entry
  offer, and the JSON reports the workbench's Checks tab reads. The failure-investigation
  offer already has its demonstration (Step 44); migration assurance has the report itself
  (Step 56).

Validation: 2 new tests (the three defects and corrections; the folder's reports are the
formats the Checks tab reads); the page and the Checks tab were checked in Chromium. Full
suite, ruff and mypy pass.

## Step 60: second verdict audit (§9 correctness gate)

Steps 49–59 added new paths into the release verdict, so the Step 38 audit was repeated on
them. Three defects were found, each with a regression test that fails on the old code:

- **A clean retry hid a leak.** When a question failed partway (repeat 1 served a forbidden
  document, repeat 2 errored), its attempt history kept only rankings, and the retried
  attempt was the only one judged, so the check **passed**. Failed attempts now keep the
  forbidden documents they served and their permission path; any leak in a failed attempt
  (results, model context or a forbidden stage) fails the safety rule. Hosted checks agree,
  because uploads carry the same fields.
- **Dropping context instrumentation passed silently.** A candidate that stopped recording
  model context could not be checked for forbidden documents reaching the model, yet the
  comparison passed. Losing context coverage the baseline had is now not comparable.
- **Freshness raised a false alarm.** Old content served while its change was inside the
  propagation window counted as failing, contradicting "pending is not a failure". It is
  now reported under `served.pending`.

Validation: 5 regression tests (each fails on the previous code). Full suite (1,785 passed,
22 skipped) and frontend tests (115) pass. One run had a single failure that did not
reproduce in two reruns; it is noted here as a possible intermittent test, not resolved.
This remains a review by the code's author; the scorecard keeps the gate open for an
independent one.

## Step 61: documented commands checked against the real CLI (1.1 acceptance support)

A pilot engineer follows the documentation literally, and the 1.1 acceptance asks a second
engineer to follow the documented connection path alone. `tests/test_doc_commands.py`
extracts every `vectraudit` invocation from fenced shell blocks in the README and `docs/`,
plus the commands the workbench UI tells users to run, and walks the real command tree for
each (nothing is executed): the subcommand must exist and every option shown must be
accepted. All 111 current commands match; a self-check plants a wrong subcommand and a
misspelled option and confirms both are caught, so the test cannot pass vacuously.

## Step 62: email notifications (PRD FR13, strategy 1.7)

The PRD's FR13 asks for scheduled evaluation with "email/Slack delivery with retries,
deduplication and alert-resolution state". Slack and signed webhooks existed (Step 18);
email did not.

- **A third style, `email`,** on the existing webhook outbox: an owner adds
  `mailto:team@example.com` in the API or the dashboard. Exactly one plain address is
  accepted; lists, display names, query strings and header injection are refused.
- **Same events, same guarantees:** release decisions and monitor alerts (opened, resolved,
  overdue) are emailed once per event key, with the same small content as Slack (headline,
  action and dashboard link; no reasons, question IDs or text), and the same retries.
- **Through the operator's mail server** (`VECTRAUDIT_CLOUD_SMTP_HOST`, `_FROM`, optional
  port, login and `starttls`/`ssl`/`none`), with Python's standard library: no new
  dependency. A refused recipient or permanent (5xx) reply fails at once; a temporary (4xx)
  reply, a lost connection, a failed login or a missing configuration retries, so an
  operator can fix it before the delivery gives up.
- The operations guide and trust package name email as a destination and the mail server as
  a processor.

Not included: digests or several addresses per destination (add one destination per
address), and a live mail-server run.

Validation: 15 new tests (address validation including header injection, a decision emailed
once without evidence, an unconfigured server, the STARTTLS and login conversation,
permanent and temporary failures, settings, the dashboard form). Full suite, ruff and mypy
pass.

## Step 63: Elasticsearch and OpenSearch connector (Tier 3, chosen by the owner)

The strategy lists Elasticsearch/OpenSearch as "the most likely next backend" for
support-search customers; the owner chose to build it ahead of customer demand.

- **One dependency-free connector over the REST APIs** for both engines, detected from
  `GET /`. Connections use `"backend": "elasticsearch"` or `"opensearch"`, `url`, `index`
  (or alias), `vector_field`, optional `id_field`, and `api_key_env` (Elasticsearch API key)
  or `basic_auth_env` (`user:password`). Credentials go only over HTTPS or to a local
  cluster; field names are validated before they reach a script.
- **Search and exact reference:** Elasticsearch's `knn` option on `dense_vector` fields
  (`num_candidates`, never below k) and OpenSearch's `knn` query on `knn_vector` fields
  (`ef_search`). The exact reference is a brute-force `script_score` with the field's own
  similarity, so ANN recall is measured on the backend's vectors and filters. A field
  mapped `index: false` supports only exact search. Filters are `term` clauses.
- **Typed errors** like the other connectors: authentication, missing index, rate limits
  and outages, timeouts, rejected queries.
- `vectraudit init --backend elasticsearch|opensearch` scaffolds a connection and workflow;
  CI provisions Elasticsearch 8.15 and OpenSearch 2.17 service containers.

Not included: hybrid (BM25 plus vector) search through the connector (a hybrid application
connects through the pipeline adapters), sparse vectors, and a customer cluster.

Validation: 20 tests against a fake cluster that really scores documents (request shapes for
both engines, exact reference, source-field IDs, a full run measuring recall, unindexed
fields, typed errors, unsafe configuration, connection files under both names), and 6
real-cluster tests run locally against Elasticsearch 8.15.3 and OpenSearch 2.17.1 in Docker:
every similarity on both engines, with the exact reference matching a brute-force answer,
ANN recall above 0.8 and filters honoured. CI now provisions both clusters. Full suite,
ruff and mypy on Python 3.12 and 3.10 pass.

## Step 64: LangSmith and Langfuse trace import (strategy 1.1)

The strategy names OpenInference first, "then a Langfuse or LangSmith source if needed";
both are the competitors whose traces a likely partner already records.

- **Converted, not re-implemented.** `pipelines/trace_formats.py` turns each LangSmith run or
  Langfuse observation into an OpenInference span (kind, `input.value`, retrieved documents
  with ID, content, metadata and score, metadata, session and user). The existing reader
  then matches questions, maps stages, records gaps, infers model context from prompts and
  writes the capability report, so a LangSmith import is judged exactly like an
  OpenInference one. The run records the export shape.
- **LangSmith**: runs as a list, JSON lines or `{"runs": [...]}`; `retriever` runs carry
  documents (`{"documents": [...]}` from LangChain or a `@traceable` list); `llm` runs carry
  prompts; `trace_id` is required for child runs. Thread keys `session_id`, `thread_id` or
  `conversation_id` in `extra.metadata` become the session.
- **Langfuse**: observations from the Observations API, a list, JSON lines or a
  `{"data": [...]}` page; `RETRIEVER` and `GENERATION` types; raw-string input and output
  are parsed.
- **No inference from names.** A retriever recorded as a plain span, chain or tool counts
  only when `--stage-kind` names it; otherwise the question fails with
  `trace_no_retrieval`. Files mixing formats are refused.
- `workbench intake` reads the same exports, so benchmark upkeep works for these teams too.

Not included: fetching from the LangSmith or Langfuse APIs (exports are read as files, so no
credentials or dependency are added); Langfuse scores and LangSmith feedback (answer
judgments already import keyed by trace ID); a real partner's export.

Validation: 7 tests from record shapes in the vendors' documentation: a LangSmith export
scored with model context inferred from the prompt and no text in the run; a `@traceable`
list with document IDs; a Langfuse API page with raw-string fields; a plain span counted
only when named; a retriever without documents as a gap; orphan runs and mixed formats
refused; the CLI and intake on the same export. Full suite, ruff and mypy on Python 3.12 and
3.10 pass.

## Step 65: Langfuse scores and LangSmith feedback as answer judgments (strategy 1.2)

Strategy 1.2: "Accept answer judgments from existing tools (Ragas, DeepEval, Langfuse
scores, human review) before building any answer evaluator." Step 64 imported these teams'
traces; their judgments still needed a conversion script.

- **Read as exported.** `workbench attach-answers` accepts a Langfuse Scores API page or
  list and LangSmith feedback as a list or JSON lines. `evaluation/judgment_formats.py`
  turns each record into an ordinary judgment keyed by trace, and the existing reader
  validates it, so verdicts, disputes, human agreement and diagnoses are unchanged.
- **The caller names the answer judgments** (`--evaluator`). Without it the command lists
  the score names in the export and reads nothing: relevance, toxicity and end-user ratings
  sit beside correctness in the same export.
- **Who judged is never guessed.** Langfuse `ANNOTATION` is human and `EVAL` is a model,
  and LangSmith `model` feedback is a model, as their documentation states; anything else
  needs `NAME=human` or `NAME=model`, otherwise the score is skipped and counted.
- **Values and keys.** Verdict-named categories, or labels mapped with `--label`; numeric
  and boolean values are scores needing a threshold. Langfuse attaches by `traceId`
  (session and dataset-run scores are skipped); LangSmith by `trace_id`, else `run_id`.
- The run's `answers` summary records the export format and the skipped counts by reason
  and score name. Comments, corrections and author IDs are never read.

Not included: LangSmith feedback on child runs without `trace_id` (matched through the
trace file in a later step if partners need it), score configs as rubric versions, and a
real partner's export.

Validation: 4 tests built from the vendors' documented record fields: Langfuse human
annotations deciding over an LLM judge with agreement measured, session, text and unnamed
scores skipped; undeclared sources skipped and declared ones read with mapped labels;
LangSmith feedback attached through the CLI to a run imported from LangSmith traces, with a
documented model source kept over a declaration and a child run unmatched; refusals for
unnamed evaluators, mixed formats and bad labels. No comment or author ID reaches the run.
Full suite, ruff and mypy on Python 3.12 and 3.10 pass.
## Step 65: citations beside answer judgments (1.2)

`attach-answers` judgments may carry `citations` (document IDs the answer cited) and
`citation_support` (`supported` or `unsupported` per cited document, from review or a tool).
- **Placement:** each cited document is `in_context`, `not_in_context` (delivered but absent
  from an observed model context), `delivered_context_unknown`, or `not_delivered`.
- **Support:** humans decide over models; disagreement is `disputed`, as for answers.
- **Conflicts:** different citation lists for one question are reported as conflicting and
  not analysed.
- **Summary:** run-level counts and a terminal line. Failed questions are not analysed.
- **Boundary:** verdicts, diagnoses, release policies and uploads are unchanged; only
  document IDs already in the run are stored.

Not included: judging a citation against the benchmark's relevance labels, citation spans or
text, citations in policies or uploads, the workbench panel, and calibration on real judgments.

Validation: 8 new tests (placement against delivery and context, unchanged verdicts and
diagnoses, support precedence and disputes, conflicting lists, 8 invalid inputs, failed and
unmatched rows, CLI output with private text kept out). Ruff and mypy pass; the full suite
passes except two failures that also occur on clean `main`
(`test_cloud_email` subject check and `test_github` CLI publishing warnings).

## Step 66: pull traces and scores from Langfuse (1.1, 1.8)

`vectraudit workbench langfuse-pull --since T --until T [--traces F] [--judgments F]` reads
a time window from a Langfuse project's public API, so a scheduled intake or import no
longer needs someone to export a file first:
- **Observations** come from `GET /api/public/v2/observations` (cursor-paged, with the
  `io` and `metadata` groups) and are written exactly as an export holds them, so
  `import-traces` and `intake` read them unchanged. Repeated records are kept once.
- **Scores** come from `GET /api/public/v3/scores` (cursor-paged, with each score's
  subject) and become `attach-answers` judgments keyed by trace ID. Booleans and the three
  verdict categories become verdicts and numerics stay scores that need a declared
  threshold. Other categories, text and correction scores, and session or experiment
  scores are skipped and counted. Annotation scores count as human review.
- **Boundary:** read-only; comments are never requested; keys from the standard
  `LANGFUSE_*` variables, sent only over https (local http excepted); error bodies not
  echoed; a window over 200,000 records or a repeated cursor is refused.
- No dependency was added (standard-library HTTP, as in the Elasticsearch connector).
  Request shapes follow Langfuse's published OpenAPI specification.

Not included: LangSmith's API (still files), writing findings back to Langfuse as scores or
comments, a stored pull cursor (intake's own cursor skips repeats, so overlapping windows
are safe), retries on rate limits, and a run against a real Langfuse project.

Validation: 17 new tests:
- an end-to-end pull from a local HTTP server that pages like Langfuse, through
  `import-traces` and `attach-answers`, with human annotation deciding over a model score;
- paging, de-duplication and request parameters for both endpoints;
- every score mapping and skip reason;
- wrong keys, rate limits, server errors and malformed responses refused without echoing
  bodies, a repeated cursor, an oversized window, unsafe hosts, missing keys, times without
  a zone, and CLI errors.

Results:
- Full Python suite: 1,969 passed, 29 skipped (live services not configured), 1 deselected,
  2 failed. The two failures (`test_cloud_email` subject check and `test_github` publishing
  warnings) also fail on clean `main`.
- Ruff, and mypy on Python 3.12 and 3.10, passed. No frontend or hosted code changed.

## Step 67: pull runs and feedback from LangSmith (1.1, 1.8)

`vectraudit workbench langsmith-pull --project NAME --since T --until T [--traces F]
[--judgments F]` is the LangSmith counterpart of Step 66:
- **Runs** come from `POST /runs/query`, cursor-paged. The window is a `trace_filter` on
  each trace's root run, so a trace that started in the window comes back whole, including
  runs that end after it. They are written as an export holds them, so `import-traces` and
  `intake` read them unchanged.
- **Feedback** on the pulled runs comes from `GET /feedback`, batched by run and
  offset-paged, and becomes `attach-answers` judgments keyed by trace (feedback on a child
  run counts for its trace). An explicit `correct`/`partial`/`incorrect` value or a boolean
  score is a verdict, and a numeric score stays a score needing a threshold. Other values,
  feedback without a score, and feedback on runs outside the pull are skipped and counted.
  `app` feedback (the LangSmith interface and annotation queues) is human review by default.
- **Boundary:** read-only; the key comes from `LANGSMITH_API_KEY` and is sent only over https
  (local http excepted); comments and corrections are never written out; error bodies are
  not echoed; a repeated cursor or more than 200,000 runs is refused. No dependency.
- **Request shapes** are copied from the official LangSmith SDK source, because LangSmith's
  published API specification could not be reached from the build environment.
- **Langfuse window edges:** checked while writing this step. A Langfuse trace cut at a
  window edge imports as a gap (missing trace, no retrieval, or unknown model context),
  never as a result. A test now pins that.

Not included: writing findings back to LangSmith or Langfuse, retries on rate limits, a
stored pull cursor, and a run against a real LangSmith workspace. `app` as the human source
type needs confirming on one.

Validation: 20 new tests:
- 17 for LangSmith:
  - an end-to-end pull from a local server that pages like LangSmith, through
    `import-traces` and `attach-answers`, with app feedback deciding over a model score
    and feedback on a child run counting for its trace;
  - the root-run window filter, cursor paging and repeated runs;
  - feedback batched by run and paged by offset;
  - every feedback mapping and skip reason;
  - an unknown project, a wrong key, rate limits, server errors and malformed responses
    refused without echoing bodies;
  - a repeated cursor, an oversized window, unsafe endpoints, a missing key, times
    without a zone, and CLI errors.
- 3 for Langfuse traces cut at a window edge.

Results:
- Full Python suite: 1,988 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- The 3 window-edge tests were added after that run and pass with their module (20 passed).
- Ruff, and mypy on Python 3.12 and 3.10, passed. No frontend or hosted code changed.

## Step 68: write the diagnosis back to Langfuse (1.3, strategy "integrate")

`vectraudit workbench langfuse-push RUN [--send]` writes each judged trace's diagnosis onto
the original Langfuse trace as a categorical score (`vectraudit.diagnosis` by default), so
engineers see Vectraudit's first-observable-loss class where they already work, and can
filter traces by it.
- **Preview first:** without `--send` it shows the counts, values and one score exactly as
  it would be sent, and makes no connection.
- **Minimal payload:** trace ID, score name, diagnosis class, optional environment, and
  metadata (source, format version, run digest). No text, document IDs or facts.
- **Idempotent by intent:** the score ID is derived from the trace and score name, so a
  re-push is meant to replace the earlier score (to confirm on a real deployment).
- **Safe failure:** batches of 100. A 207 partial rejection is counted, exits 3, and is not
  retried or echoed. Runs not imported from Langfuse, unjudged runs, and invalid names or
  environments are refused before any request.
- The Langfuse client gained a POST path (`POST /api/public/scores`, per the published
  spec), sharing the GET path's key, region, rate-limit and no-echo handling.

Not included: write-back to LangSmith, scores per observation (the diagnosis is per
question, so per trace), retrieval metrics or citation flags as scores, a Langfuse score
config for the diagnosis classes, and a run against a real Langfuse project.

Validation: 8 new tests:
- one categorical score per judged trace, with the exact payload;
- no question text, passage text, document IDs or evaluator names in what is sent;
- an end-to-end preview and send against a local server that accepts score batches: the
  preview makes no connection, the sent score matches the preview, and a re-send uses the
  same score IDs;
- a 207 partial rejection counted, exiting 3, not retried and not echoed;
- batching (205 scores in 100, 100, 5) and a malformed 207 refused;
- questions without a trace or diagnosis and repeated traces left out and counted;
- non-Langfuse and unjudged runs, bad names and environments refused;
- CLI errors.

The existing Langfuse pull tests still pass on the shared client.

Results:
- Full Python suite: 2,000 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- Ruff, and mypy on Python 3.12 and 3.10, passed. No frontend or hosted code changed.

## Step 69: pull traces and feedback scores from Opik (1.1, 1.8)

The owner asked for something from Opik (Comet) that Vectraudit lacked. Opik's README
lists tracing, LLM-as-a-judge metrics, a prompt playground, an agent optimizer,
dashboards, online evaluation rules, guardrails and a PyTest integration. Tracing, the
playground and the optimizer are on the strategy's "Do not build" list, and judge metrics
fall under "import evaluator results before building another suite". The strongest fit
was Opik as a source, so Opik users' data becomes Vectraudit's input as Langfuse's and
LangSmith's already is. A PyTest integration is the other candidate, not built here.

`vectraudit workbench opik-pull --project NAME --since T --until T [--traces F]
[--judgments F]`:
- **Traces and spans** from `GET /v1/private/traces` and `/v1/private/spans`. Traces are
  taken by creation time; spans over the window plus `--grace-minutes` (default 60), kept
  only for the traces taken. Each trace is written as its root span, so the question on
  the trace is found, and the file imports through the shared reader (`opik` shape).
- **Retrievers must be named** with `--stage-kind`, because Opik has no retriever span type.
- **Feedback scores** inline on traces and spans become judgments: verdict categories
  become verdicts, uncategorised values stay scores that need a threshold, other
  categories are skipped, and `ui` is human review. Averaged `span_feedback_scores` and
  reasons are never used.
- **Boundary:** read-only; key and workspace from the SDK's `OPIK_*` variables; a key is
  sent only over https (local http excepted); a keyless self-hosted Opik may use http;
  error bodies not echoed; windows over 200,000 records refused. No dependency.
- **Found while testing:** stopping at a page shorter than the requested size could
  silently drop records if Opik caps page size (the spec sets no maximum). Paging now
  follows the reported total.
- **Also found:** `intake` had no `--stage-kind` option (only `import-traces` did), so a
  source without a retriever span type could not show retrieval in production intake. Both
  commands now share the option.

Not included: writing the diagnosis back to Opik (its feedback-score endpoints exist), Opik
threads as multi-turn history, Opik datasets and experiments, the PyTest integration, and
a run against a real Opik deployment.

Validation: 24 new tests:
- an end-to-end pull from a local server behind Opik Cloud's headers, through
  `import-traces` and `attach-answers`, with a `ui` score deciding over an online one and a
  span's score counting for its trace;
- production intake on a pull with a named retriever, finding the likely recurrence;
- a retriever never guessed without `--stage-kind`;
- traces as root spans and spans of other traces left out;
- the grace window, paging that follows the total when the server returns smaller pages,
  and a short page ending a listing without a total;
- every score mapping and skip reason, reasons dropped;
- wrong keys, wrong URL, rate limits, server errors and malformed responses refused
  without echoing bodies, an oversized window;
- unsafe settings, a keyless self-hosted Opik over http;
- times, grace, CLI errors and mixed formats.

Results:
- Full Python suite: 2,026 passed, 29 skipped (live services not configured),
  1 deselected, 4 failed.
  - Two failures (`test_cloud_email`, `test_github`) also fail on clean `main`.
  - The other two (`test_optimize_concurrency`, `test_pipeline_retries[http_unavailable]`)
    are timing-sensitive, cover code this step does not touch, and passed three runs in a
    row on their own (25 of 25 each).
- Ruff, and mypy on Python 3.12 and 3.10, passed. No frontend or hosted code changed.

## Step 70: prompt versions as release evidence (1.1, 1.4)

The owner asked whether Vectraudit has prompt management and versioning like Langfuse. It
did not. A Langfuse-style registry (store, edit, label and serve prompts at runtime) was
not built: it sits on the customer's runtime path, competes with tools teams already use,
and is most of the "broad prompt playground" the strategy rules out. What a release
decision needs is which prompt version produced each answer, so that is what this step
reads.
- **Recorded from traces** in all four supported formats: OpenInference `prompt.id` and
  `llm.prompt_template.version`; Langfuse linked prompts (the pull now requests the
  `prompt` field group; `promptName`/`promptVersion` become the OpenInference attributes);
  LangChain hub metadata in LangSmith runs (`lc_hub_*`); Opik `opik_prompts` in span or
  trace metadata. Each convention was checked against the vendor's spec or SDK source.
- **Per question and per run:** questions record `prompts` (name and version), and
  `backend.config.prompts` holds each prompt's version, or a sorted list when a run mixed
  several. Without prompt data the config is unchanged.
- **In release checks:** a prompt change appears as a changed setting through the existing
  path, so checks, the decision ledger's changed-setting names, lead summaries and
  calibration's configuration differences all show it. New warnings cover a run mixing
  versions of one prompt and only one run recording prompt versions.
- **In diagnosis:** `attach-answers` reports diagnoses by prompt version when a run holds
  several.
- **Capability report:** a `prompt_versions` line says how many traces recorded one, names
  mixed versions, and says plainly when prompt changes cannot be seen.
- **Boundary:** names and versions only, never template text. Hosted projections
  fingerprint them like text parameters, so a hosted check shows that a prompt changed but
  not its name or version.
- **Correctness:** a question refused because it matched several traces records no prompt,
  and its traces do not count towards the run's versions.

Not included: prompt versions reported by live Python/HTTP pipelines (they can be put in
`pipeline_version` for now), a release-policy rule on prompt changes, per-prompt-version
quality metrics in checks, and a prompt registry.

Validation: 11 new tests:
- each convention yielding a name and version and never the template (OpenInference,
  LangChain hub with and without an owner, Opik commit and version-number forms, malformed
  input);
- a Langfuse import recording per-question and run-level versions that still validate;
- mixed versions named in the run and capability report;
- an unchanged config and a named gap without prompt data;
- a release check listing `backend.config.prompts.support-answer: 12 → 13`;
- warnings for mixed versions and one-sided recording, none for steady runs;
- an ambiguous question claiming no prompt (and `--latest` using the later trace);
- LangSmith hub prompts end to end, and Opik prompts end to end without template text;
- diagnoses by prompt version;
- hosted projections fingerprinting names and versions, stable on re-projection, still
  showing the change, and refusing raw names.

One existing test was updated for the Langfuse pull's added `prompt` field group.

Results:
- Full Python suite: 2,039 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- Ruff, and mypy on Python 3.12 and 3.10, passed. No frontend or hosted server code changed.

## Step 71: release checks from a pytest suite (after Opik)

The owner asked for something else from Opik. Its PyTest integration (an `@llm_unit`
decorator and a plugin registered through `pytest11`) was the remaining candidate from the
Step 69 comparison. The Vectraudit equivalent makes a retrieval release check one more test
in a team's existing suite, a way in for teams that will not add a separate CI step.
- **`vectraudit.testing`:** `assert_release(baseline, candidate, policy)` raises
  `ReleaseCheckFailed` (an `AssertionError`) unless the check passes, with the verdict,
  failing rules and changed settings in the message. `check_release` returns the result.
  Both call the same comparison as `vectraudit check`. A test asserts the same verdict and
  reasons as the CLI for pass, fail, not comparable and error, and another pins the passing
  verdicts to the CLI's exit-zero codes. Runs and policies can be paths or records. No test
  framework is imported.
- **pytest plugin** (`pytest11` entry point): lists the session's checks with test and
  verdict in the terminal summary; `--vectraudit-report PATH` writes them as one Markdown
  report. Silent without checks; `-p no:vectraudit` disables it.
- **Load cost:** about 0.2 s per pytest run where Vectraudit is installed, from importing
  the package (numpy). Accepted and documented.
- No dependency added.

Not included: recording runs inside a test, a decorator like Opik's `llm_unit`, combining
reports across `pytest-xdist` workers, and hosted (`cloud:`) baselines from tests.

Validation: 12 new tests:
- a steady candidate passing and recorded under its test ID;
- a regression failing as an `AssertionError` with reasons and changed settings;
- verdict and reasons identical to `vectraudit check` for pass, fail, not comparable and
  error, with matching exit codes;
- passing verdicts pinned to the CLI's exit-zero codes;
- paths, records, parsed and built-in policies giving the same result;
- invalid policies and runs raising the CLI's errors;
- records not mutated;
- through a real pytest subprocess with the installed entry point: the summary section, a
  failing release test failing the run, the Markdown report, a silent session without
  checks, `-p no:vectraudit`, and the option disappearing with it.

Results:
- Full Python suite: 2,051 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- Ruff, and mypy on Python 3.12 and 3.10 (`--locked` sync, so the lock file is unchanged),
  passed.

## Step 72: query drift, after Evidently (1.8)

The owner asked for something from Evidently. Most of it (tabular drift and quality,
classification and regression metrics, text descriptors, LLM judges, a monitoring UI) is
outside a retrieval release check, or already covered (ranking metrics, reports, CI
tests). The fit was its **data drift** method applied to Vectraudit's open benchmark-upkeep
question (1.8): is the benchmark still representative of production?

`vectraudit workbench drift TRACES DATASET [--context-field F] [--json] [--fail-on-drift]`:
- **Production questions** are read from traces in any supported format, the same way
  intake reads them.
- **Descriptors:** word count, script, digits, question form, unseen terms (leave-one-out
  on the benchmark) and declared context fields.
- **Each descriptor is compared by PSI** exactly as Evidently computes it (Sturges binning
  for numbers with more than 20 distinct reference values, category shares otherwise, the
  same empty-bin floor, drift at 0.1). Tests check it against hand computation.
- **What moved:** new values, the most frequent unseen terms, and the share of production
  questions that are benchmark questions word for word. A small-sample warning appears
  below about 30 questions.
- Local only. It never changes a release verdict. No dependency (numpy).

Not included: embedding drift, retrieval-score drift, drift tracking over time in the
hosted service, other statistical tests, and a run on a customer's production traces.

**Corrected in Step 73:** PSI was the wrong default. It came from Evidently's documented
example, not its automatic test selection, and it raises false alarms on stable data at
realistic sample sizes. Step 73 measures this and switches to Evidently's default tests.

Validation: 18 new tests:
- categorical PSI against the textbook sum;
- Evidently's empty-bin floor in both forms;
- Sturges binning against a direct numpy computation, and few-valued numbers compared as
  categories;
- script detection across six scripts and none;
- descriptors, including the fullwidth question mark;
- no drift on identical samples;
- a new language, tenant and vocabulary flagged and named;
- longer questions drifting on word count;
- leave-one-out unseen terms;
- missing context as its own value;
- refusals;
- production questions read from traces with context;
- the CLI (report, JSON, `--fail-on-drift`, a calm run, a missing file).

Results:
- Full Python suite: 2,069 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- Ruff and mypy on Python 3.12 passed. mypy on Python 3.10 caught a bare `np.ndarray`
  annotation that its older numpy stubs reject; after the fix (annotation only) it passes.

## Step 73: retrieval score drift, and Evidently's default drift tests (1.7, 1.8)

The next Evidently item: did the **retriever** change? `vectraudit workbench score-drift
EARLIER LATER [--stage NAME] [--fail-on-drift]` compares two windows of production traces
on the retriever's first-document score, mean score, spread and number of documents
returned. It also compares the windows' questions, and reads the result:
- **retriever shift likely:** scores moved, questions did not (embedding model, index,
  similarity or score-scale change);
- **drift with question drift:** both moved; compare recorded runs to isolate the
  retriever;
- **no drift.**

Traces without recorded scores are counted and left out; `--fail-on-drift` exits 3 only for
a likely retriever shift.

**Found while testing: Step 72's PSI default was wrong.** Two windows from the same
retriever (60 traces each) gave PSI 0.63. Measured on samples drawn from one distribution,
PSI at its 0.1 threshold raised false alarms:

| Values per side | 30 | 60 | 200 | 1,000 |
|---|---|---|---|---|
| Scores | 96% | 89% | 36% | 0% |
| Question word counts | 100% | 100% | 94% | 0% |

Evidently's automatic selection (read from its source) does not use PSI. Both drift
commands now use it:
- up to 1,000 reference values: Kolmogorov-Smirnov for numbers with more than five distinct
  values, otherwise chi-square (Z-test for two values), drift at p < 0.05;
- above that: normed Wasserstein for numbers, Jensen-Shannon otherwise, drift at 0.1.

False alarms fell to 1-7% at every size. PSI remains as `--method psi`. The query-drift
report format is now version 2: each descriptor names its test with a `p_value` or
`score` and a threshold, instead of `psi`. No dependency was added. The statistics are
numpy implementations, tested against SciPy (a dev dependency already installed):
chi-square, Z, Wasserstein and Jensen-Shannon agree to rounding, and K-S is within 0.03 of
SciPy's exact small-sample result.

Not included: embedding drift, per-document score calibration, drift history in the
hosted service, and a run on a customer's traces.

Validation: 19 new tests (8 score drift, 11 for the default tests and their selection), plus 2 query
drift tests updated for the version-2 report fields.
- **Score drift:**
  - a steady retriever not flagged;
  - rescaled scores with steady questions flagged as a likely retriever shift;
  - a smaller result set flagged on `returned`;
  - drift with changed questions not blamed on the retriever;
  - unscored traces counted and refused when none have scores;
  - a named stage;
  - small windows flagged;
  - the CLI exit codes.
- **Default tests:**
  - every statistic against SciPy;
  - a new category as chi-square drift;
  - the selection rule for seven cases;
  - false alarms under 10% with the defaults against over 50% with PSI on the same stable
    data;
  - PSI as an explicit choice.

Results:
- Full Python suite: 2,090 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email` subject check and
  `test_github` publishing warnings) also fail on clean `main`.
- Ruff, and mypy on Python 3.12 and 3.10, passed.

## Step 74: MAP and Hit Rate (after Evidently)

The last Evidently item: two standard ranking metrics Evidently reports and Vectraudit did
not.
- **Definitions:** MAP@k as trec_eval's `map_cut_k` (precision at each new relevant hit,
  over all relevant documents; duplicates earn nothing) and Hit Rate@k (any relevant
  document in the top k). Evidently's MAP divides by the relevant documents retrieved,
  which does not penalise losing one, so it was not copied. Both are in the metric
  contract.
- **Optional, so nothing recorded breaks:** they are recorded beside the four semantic
  metrics on every new run (questions, repeats and summary) but are not added to the
  required set, and the contract stays at version 2. Older runs validate and compare
  exactly as before. A policy that requires `map` or `hit_rate` against an older run is
  not comparable, never a pass.
- **Validated:** when present they must be fractions, on every repeat, and equal to the
  repeat mean.
- **Everywhere a metric goes:** policy rules, `primary_metric`, cohort rules, comparisons,
  hosted uploads (older projections unchanged), lead-summary wording, and the workbench
  policy editor.
- **Found while adding them:** the policy editor rebuilds `metrics` from a fixed list, so a
  MAP rule written by hand or by another tool would have been dropped on the next save. The
  editor now lists both, and a frontend test pins the round trip.

Not included: other Evidently ranking and recommender metrics (novelty, diversity,
serendipity, popularity bias), MAP or Hit Rate in the FAISS index debugger's own evaluation
view, and re-scoring stored runs (re-record to get the new metrics).

Validation: 13 new Python tests:
- MAP against hand computation, including duplicates, the cutoff and no labels;
- MAP falling when a relevant document is lost;
- Hit Rate with graded labels;
- repeat averaging;
- the contract;
- new runs summarising both while older runs validate unchanged;
- old against new comparing as before;
- a policy needing MAP not comparable against an older run;
- a MAP regression failing, with MAP as the primary metric;
- invalid values refused;
- incomplete repeat coverage refused;
- uploads carrying both, with older projections unchanged.

One frontend test pins MAP and Hit Rate rules (including cohort and primary metric) through
an edit. One existing test that pinned the exact metric set was updated, and now also checks
MAP and Hit Rate with a duplicate in the ranking.

Results:
- Full Python suite: 2,102 passed, 29 skipped (live services not configured),
  1 deselected, 3 failed. Two failures (`test_cloud_email`, `test_github`) also fail on
  clean `main`. The third was that pinned metric-set test, since updated (45 related tests
  pass).
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.
- Frontend: TypeScript, Oxlint, 116 frontend tests and the production build passed.
## Step 75: OpenLLMetry traces (1.1)

The owner asked what OpenLLMetry (Traceloop) offers that Vectraudit could use. It is
OpenTelemetry instrumentation for LLM applications. Instrumenting applications is general
tracing, which the strategy rules out; but its exports are traces that teams already
record, and Vectraudit's OTLP import read only OpenInference's conventions, so an
OpenLLMetry export imported with no retrieval at all. Reading them is the strategy's
"integrate rather than replace".

The OTLP reader now keeps span events and translates OpenLLMetry spans into OpenInference
attributes. Every convention was checked against the OpenLLMetry source (the
semantic-conventions package, the Pinecone, Chroma and Qdrant instrumentations, and the
SDK's decorators and tracing):
- **Vector database results** come from `db.query.result`/`db.search.result` events: id,
  score, text, and metadata (JSON, or Pinecone's Python `repr`, read as literals only).
  `distance` is never used as a score.
- **Retrievers are never inferred from names:** only result events or vector-query markers
  count. A query without result events (Qdrant) is a gap, and so is a span holding several
  queries' results (Chroma with several texts, Milvus with several query IDs), never a
  merged ranking.
- **Question** from the workflow's `traceloop.entity.input` (`{"args", "kwargs"}`).
  **Prompts** come from `gen_ai.prompt.N` for model context. **Association properties**
  become metadata, session and user, so `--context-field` works in intake and drift.
- **Managed prompt versions** (`traceloop.prompt.key`/`version`) feed Step 70's prompt
  evidence.
- OpenInference spans are untouched; recorded attributes always win over translated ones.

Not included: OpenLLMetry's newer `gen_ai.input.messages` beyond using it as the input
value, rerankers (OpenLLMetry has no reranker convention), a Traceloop API pull, and a
real OpenLLMetry export (fixtures follow the instrumentation source).

Validation: 15 new tests:
- a Pinecone trace (ids from repr metadata, scores, question, prompt version);
- a Chroma trace locating model context, with distance never a score;
- multi-query Chroma and Milvus spans as gaps;
- a Qdrant query without results as a gap;
- no retriever inferred from a name;
- OpenInference spans and recorded attributes untouched;
- association properties as context and session;
- six question-extraction cases;
- literal-only metadata parsing, refusing code and oversized input;
- the prompt version-hash fallback.

Results:
- Full Python suite: 2,105 passed, 29 skipped (live services not configured),
  1 deselected, 2 failed. The two failures (`test_cloud_email`, `test_github`) also fail on
  clean `main`.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 76: the two tests that always failed

Since Step 65 every full run reported the same two failures, noted each time as "also fail
on clean `main`" but never explained. A suite that always reports two failures hides the
next real one. Both had real causes:
- **`test_cloud_email`, a product defect.** Python's email header folding (reproduced on
  3.10.20 and 3.12.3) moved the decision email's 89-character subject whole onto a
  continuation line, so recipients got a subject starting with a space. Emails are now
  serialised without folding (`max_line_length=998`; subjects are capped at 200 characters,
  within RFC 5322's limit). A new test calls the message builder directly and fails on the
  old code.
- **`test_github`, a test-isolation defect.** The code also reads the gh CLI's `GH_TOKEN`.
  The test removed only `GITHUB_TOKEN`, so wherever `GH_TOKEN` is set (this environment, and
  any developer machine logged in to `gh`) it took the publishing path against the test's
  mock server. The fixture now removes `GH_TOKEN` too, for every GitHub test.

CI runs Python 3.10 to 3.13, so the email failure probably showed there as well; the earlier
step reports that called it environment-specific were wrong to assume so.

Validation:
- Full Python suite: **2,109 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected. This is the first full run without failures since Step 65.
- The email and GitHub test modules also pass on Python 3.10 (23 passed), and the GitHub
  tests pass with and without `GH_TOKEN` set.
- Ruff, and mypy on Python 3.12 and 3.10, passed.

## Step 77: pull traces from OpenObserve (1.1, 1.8)

The owner asked what OpenObserve could add. It is a general observability backend (logs,
metrics, traces, RUM), which is the general tracing the strategy rules out. But teams send
their LLM traces there, so it fits the same "integrate" pattern as Langfuse, LangSmith and
Opik.

`vectraudit workbench openobserve-pull --since T --until T --traces F [--stream S]`:
- **Whole traces:** traces are taken by start time (`GET .../traces/latest`), then all their
  spans are read by SQL (`POST .../_search?type=traces`, by trace ID, over the window plus
  `--grace-minutes`), so traces are not cut at the window's end.
- **Rebuilt as OTLP JSON,** so the existing reader, including Step 75's OpenLLMetry
  translation, imports it unchanged. Taken from OpenObserve's source: spans are flat rows
  with attribute keys rewritten (lower case, other characters to `_`, `attr_` on collision)
  and events kept as JSON with their original keys. The rewrite is not invertible, so the
  keys the importer reads are restored by explicit patterns (a test round-trips every
  one), and other keys keep their stored names.
- **Boundary:** read-only; Basic auth over https only (local http excepted); trace IDs
  validated as 32 hex characters before entering SQL, and organisation and stream names
  restricted; error bodies not echoed.
- **Licence:** OpenObserve is AGPL-3.0. Its source was read only to learn the stored
  format; nothing was copied or added as a dependency, and the client only calls its HTTP
  API.

Not included: OpenObserve's own LLM evaluation scores as judgments, writing back, SQL
filters beyond the time window, and a real OpenObserve instance. The fake server stores
spans by the documented flattening rules.

Validation: 21 new tests:
- every key the importer reads surviving storage and restoration, collisions, and unknown
  keys left alone;
- an OpenInference trace end to end through the CLI (ids, model context, prompt version);
- an OpenLLMetry Pinecone trace end to end (event results with repr metadata, prompt
  version, tenant context);
- the grace window and exact request shapes (times in microseconds, `type=traces`, the SQL);
- paging, with a malformed trace ID never reaching SQL;
- seven refused responses without echoing bodies;
- seven unsafe or incomplete settings;
- a token sent as given;
- the CLI without a URL or organisation.

Results:
- Full Python suite: **2,130 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 78: hubs, after Evidently's popularity bias

Asked again about Evidently: its drift detection and MAP/Hit Rate were already taken
(Steps 72-74). The remaining fit was its recommender **popularity-bias** metrics, adapted
to retrieval as **hubness**: documents an embedding retriever returns for many unrelated
questions. A new embedding model or index can create a hub without moving average quality
much, and no existing check showed it.

`vectraudit workbench hubs RUN [--baseline RUN] [--dataset BENCHMARK] [--fail-on-new-hubs]`:
- **Measures:** k-occurrence per document; Gini (Evidently's formula) and coverage (when the
  run records the corpus size); k-occurrence skewness, the standard hubness measure.
- **Hubs:** documents in the top k of at least 10% (and at least three) of the questions.
  With the benchmark, each hub's labelled relevance where it was retrieved, flagging hubs
  that are relevant for fewer than half of their questions.
- **Comparison:** hubs new to the candidate; `--fail-on-new-hubs` exits 3.
- Diagnostic only. It works on any recorded run and adds no dependency.

Not included: average-recommendation popularity (retrieval has no interaction history), hub
causes (vector norms, index structure), and hubness in hosted checks or the workbench UI.

Validation: 8 new tests:
- Gini against Evidently's formula by hand;
- skewness against SciPy;
- a spread retriever with no hubs (coverage, Gini 0, zeros counted);
- a planted irrelevant hub found with its relevance and flagged;
- new hubs in a comparison;
- failed questions and repeats within one ranking not counted;
- unknown corpus size, the three-question minimum and refusals;
- the CLI (report, JSON, `--fail-on-new-hubs`, no new hubs, a missing file).

Results:
- Full Python suite: **2,140 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 79: merge MAP and Hit Rate with main, and fix the flaky clock tests

`feat/map-hit-rate` (Step 74) had fallen behind Steps 75-78 and no longer merged: both sides
had appended to `DECISIONS.md` and this log. Merged `main` into it, keeping both sides with
Step 74 in order. No code conflicted.

The re-run after the merge failed once in `test_pipeline_retries` (it passed five times
alone), the same family as the Step 69 "timing" failures that were waved through then.
- **Cause:** the retry, concurrency and job tests faked time by patching `time.sleep` and
  `time.monotonic` on the `time` module itself, which is process-wide. Any other thread
  sleeping during the test added its sleeps to the fake clock.
- **Fix:** a small helper (`tests/fake_time.py`) replaces only the module-under-test's
  `time` reference, so other threads keep real time. Applied to all three files.
- **Regression test:** runs a busy sleeping thread beside a retried run. Against the old
  patch it recorded 1,742 seconds of sleep instead of 3 and failed; it passes now.

Validation: the merged branch passes the full suite with **2,154 passed, 0 failed**,
29 skipped (live services not configured), 1 deselected. Also passing:
- the 47 retry, concurrency and job tests, and the 21 retry tests on repeated runs;
- Ruff, and mypy on Python 3.12 and 3.10;
- the API type generation check;
- TypeScript, Oxlint and 116 frontend tests (MAP touched the policy editor).

## Step 80: DeepEval results as answer judgments (1.2)

The owner asked what DeepEval offers. Its ~50 metrics are mostly LLM judges (the strategy
says to import evaluator results, not build another suite), and its pytest integration
already has a Vectraudit equivalent (Step 71). The fit was reading its results: Step 41's
`attach-answers` needed its own JSONL, so DeepEval users had to convert by hand.

`vectraudit workbench deepeval-judgments TEST_RUN BENCHMARK --metric NAME --out F` reads
`.deepeval/.latest_test_run.json` (DeepEval's `LLMApiTestCase`/`MetricData` models, checked in
its source) and writes judgment lines:
- **Explicit metric choice:** without `--metric`, the metrics in the file are listed. The
  contextual metrics are refused because they judge retrieval, not answers.
- **Verdicts:** DeepEval's own `success` decision becomes the verdict; the score and judge
  model are kept; errored or unevaluated metrics are counted and skipped.
- **Matching:** `metadata.query_id`, `metadata.trace_id`, or an unambiguous exact question
  text; ambiguous and unmatched cases are counted.
- Reasons, inputs, outputs and contexts are never written.

Not included: DeepEval's conversational test cases, its Confident AI platform API, its
synthesizer (LLM-generated benchmarks), and red-teaming.

Validation: 6 new tests:
- verdicts matched by ID, trace and text, with errored and unevaluated metrics counted and
  no private text written;
- only chosen metrics, the three contextual metrics refused, and metric listing;
- ambiguous text not guessed;
- the judgments attaching and diagnosing a planted wrong answer;
- both file layouts, plus non-runs and invalid JSON refused;
- the CLI (listing without `--metric`, writing, refusing a contextual metric).

Results:
- Full Python suite: **2,160 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 81: OpenTelemetry GenAI retrieval conventions (OpenLIT) (1.1)

The owner asked what OpenLIT offers. It is OpenTelemetry instrumentation for LLM apps and
vector databases plus an observability platform; the platform is general tracing, ruled out.
Its source showed two things:
- Its **vector-database spans** record only counts and query text, never document IDs.
- Its **LangChain retriever spans** use the official **OpenTelemetry GenAI semantic
  conventions** (`gen_ai.operation.name = retrieval`, `gen_ai.retrieval.documents`,
  `gen_ai.retrieval.query.text`), checked against the specification and its JSON schema.

So the support targets the specification, not OpenLIT, and covers any instrumentation that
follows it. The Step 75 translation now also reads:
- **retrieval spans:** documents (`id`, `score`, `content`, `metadata`) and the question;
- **chat and completion spans:** `gen_ai.input.messages`, so model context is located.

**Found in OpenLIT's source:** its LangChain retriever records only the first three
documents, with the total in `gen_ai.retrieval.document_count`. Taking those three as the
ranking would understate recall and misattribute losses, so a list shorter than its count is
a gap. A trace holding both a retrieval span and an ID-less vector-database span is reported
as ambiguous until `--final-span` names the delivered one.

Not included: other OpenLIT platform features (dashboards, guardrails, evaluations, prompt
hub), an OpenLIT API pull, and a real OpenLIT export.

Validation: 9 new tests:
- an OpenLIT LangChain trace (question, IDs, model context);
- a three-of-five list as a gap, with a complete list importing;
- the specification's `id`/`score` shape;
- four unusable document lists as gaps;
- a nested vector-database span reported as ambiguous, then resolved with `--final-span`;
- chat spans as LLM spans, and OpenInference spans untouched.

Results:
- Full Python suite: **2,171 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 82: a policy rule for prompt changes (1.4, after Step 70)

Step 70 recorded which prompt version produced each answer and reported changes, but no
policy could gate on them. `"prompt_changes": "fail"` (default `"report"`) now:
- **Fails each changed prompt** as its own rule `prompt:<name>`, including one added,
  removed or only partly rolled out (a run mixing versions).
- **Allows audited acceptance:** a new exception kind, `prompt`, lets a policy open these
  rules to audited exceptions, so a reviewed change is accepted deliberately.
- **Refuses unverifiable runs:** a check where either run records no prompt versions is
  not comparable, never a pass.
- **Calibration:** reports prompt failures between supposedly unchanged runs as not noise.
- **Hosted checks:** gate on fingerprinted names (`prompt:sha256:…`).
- **Workbench policy editor:** keeps the option through edits (a frontend test fails
  without the fix, the same class of bug as Steps 20 and 74).
- **Hashes:** policies without the option keep their hashes. Checked by hashing two
  policies with `main` and with this change; the existing hash-pinning test gained the
  field, as for each earlier optional field.

Not included: a workbench control to switch the rule on (it is kept, not edited), per-prompt
rules (for example, gate one prompt and not another), and requiring a minimum time between
prompt changes.

Validation:
- 9 new Python tests:
  - the default reporting without failing;
  - a change failing with its rule key and message, and steady versions passing;
  - a mixed rollout;
  - added and removed prompts;
  - one-sided and missing recording as not comparable;
  - an audited exception accepting the change only when the policy opens `prompt`;
  - hashes and value validation;
  - calibration;
  - hosted projections.
- 1 new frontend test.

Results:
- Full Python suite: **2,180 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, the API type generation check, TypeScript, Oxlint and
  117 frontend tests passed.

## Step 83: importers tested against real instrumentation output

Every trace importer had been tested only against fixtures written from specifications. The
real libraries were installed in a throwaway environment (not as dependencies) and run on
one offline LangChain + Chroma pipeline (deterministic fake embeddings, fake chat model),
exporting OTLP JSON as the Collector's file exporter does:
- **OpenInference** (`openinference-instrumentation-langchain` 0.1.81): imported fully.
  Questions, document IDs and model context all as expected.
- **OpenLLMetry** (`opentelemetry-instrumentation-langchain`/`-chromadb` 0.62.4): documents
  imported, but **no question matched**. A real bug from Step 75: the LangChain
  instrumentation writes `traceloop.entity.input` as `{"inputs", "tags", "metadata",
  "kwargs"}` (and `{"query"}` on retrievers), not the SDK decorator form the fixtures
  assumed. Fixed; it now imports identically to OpenInference.
- **OpenLIT** (1.45.0): two facts the Step 81 fixtures did not show:
  - it records each span of one request as a **separate trace**, so model context cannot
    be tied to retrieval;
  - it takes document IDs only from metadata `id`/`source`.

  Both are reported as gaps (unknown context, `document_id_missing`), not reconstructed.
  With `metadata.id` set, its rankings import.

All three give the same rankings for the same run. The real exports (synthetic text only,
checked for machine details) and the scripts and versions that produced them are in
`tests/fixtures/real_traces`, so the importers stay pinned to what these libraries emit.

Not included: LangSmith, Langfuse, Opik and OpenObserve. Their data comes from hosted
services, so a real check needs a workspace. DeepEval also needs a real results file, and
LlamaIndex and other frameworks need instrumenting. This is still not a customer
application.

Validation: 10 new tests:
- the two full imports with model context;
- OpenLIT rankings without context;
- OpenLIT without IDs as a gap;
- fixtures free of machine details;
- five question-extraction cases.

Without the fix, 5 of them fail.

Results:
- Full Python suite: **2,190 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 84: promptfoo results as answer judgments (1.2)

The owner asked what promptfoo offers. It is an LLM evaluation CLI (YAML tests, about 60
assertion types, prompt and model comparison, red-teaming, CI). Assertions and judges
are a suite of their own (the strategy says to import results), and CI integration exists
(Steps 27, 71). The fit is the same as DeepEval's (Step 80): its results become
`attach-answers` judgments.

`vectraudit workbench promptfoo-judgments RESULTS BENCHMARK --assertion NAME [--provider P]
[--prompt Q] --out F`:
- **One variant:** an eval comparing several provider/prompt pairs needs one chosen;
  judgments of different systems are never mixed.
- **Chosen assertions:** by metric label or type, or `overall`. `context-recall` and
  `context-relevance` are refused, and `overall` is skipped for tests that include them.
- **Verdicts:** promptfoo's `pass`, with the score. Each repeat is its own judgment.
- **Matching:** metadata or vars IDs, else an unambiguous question variable.

**Tested on a real promptfoo file**, following Step 83. promptfoo 0.125.1 ran offline (echo
provider, deterministic assertions; 2 prompts × 3 tests × 2 repeats). It caught a defect in
the first draft, written from promptfoo's types: promptfoo sets `error` to the assertion
message on every ordinary failure. The draft skipped those as errors, which would have
dropped every failing answer and imported only passes. Errors are now read from
`failureReason` 2 only. The fixture, its config and the command are in
`tests/fixtures/promptfoo`, with the local base path removed.

Not included: promptfoo's red-teaming, its LLM-graded RAG assertions as retrieval evidence,
and its cloud sharing.

Validation: 9 new tests on the real file:
- variants, assertions and repeats as recorded;
- several variants refused;
- failures imported as failures (5 of the 9 fail with the draft's `error` check);
- matching by metadata and question;
- `overall` and the retrieval refusals;
- real errors skipped;
- disagreeing repeats attaching as disputed;
- non-promptfoo files refused;
- the CLI.

Results:
- Full Python suite: **2,199 passed, 0 failed**, 29 skipped (live services not
  configured), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.

## Step 85: Chroma connector (Tier 3, chosen by the owner)

The owner asked what VectorDBZ (an MIT desktop GUI for browsing vector databases) offers.
Browsing and embedding views overlap the existing workbench, and a database GUI is not the
product. Its list of supported databases showed the gap: Chroma, the most common store in
RAG prototypes, had no connector. The strategy defers new backends until a customer is
blocked (Tier 3), and the owner chose to build it anyway, as with Elasticsearch (Step 63).

`{"backend": "chroma", "url": …, "collection": …}` (also `tenant`, `database`, `id_field`,
`token_env`, `timeout_s`, `exact_limit`):
- **Search:** over the v2 REST API, with no dependency. Equality filters become a Chroma
  `where` clause (`$and` for several).
- **Distance and settings:** distance comes from the collection's `hnsw` or `spann`
  `space`; `ef_search` and the other index settings are recorded.
- **Exact reference:** ranks the collection's stored vectors, fetched with the same filter
  and cached, using Chroma's distance definitions. It is capped at `exact_limit`.
- **No tuning parameters:** Chroma sets `ef_search` per collection.
- **No snapshot:** `log_position` stayed 0 across writes on a real server.
- **Scaffolding:** `vectraudit init --backend chroma` writes a connection template.

**Built against a real server**, not a fake: Chroma 1.5.9 ran locally, and its OpenAPI
document gave the endpoints and shapes. Against it:
- the exact reference matched an independent brute-force ranking in all three spaces;
- filters (single and `$and`) and `id_field` behaved as specified;
- `vectraudit record` then `check` ran end to end (ANN recall measured, MAP and Hit Rate
  recorded).

CI now runs `chromadb/chroma:1.5.9` beside the other backends so the live tests keep running.

Not included: Chroma Cloud specifics beyond token headers, Chroma's hybrid `search`
endpoint, sparse vectors, and a customer collection.

Validation:
- 18 unit tests against a fake shaped like the real server's responses:
  - describe;
  - query body and `where`;
  - `id_field`;
  - exact ranking per space against numpy;
  - paging and caching of the exact reference;
  - the exact limit;
  - seven HTTP failures typed;
  - token headers and https-only;
  - empty or unsupported collections;
  - connection-file validation.
- 3 live tests (one per space) that seed their own collections through the REST API. They
  passed against the real server.

Results:
- Full Python suite: **2,219 passed, 0 failed**, 32 skipped (live services not configured,
  including the three Chroma live tests, which run in CI and passed locally against the
  server), 1 deselected.
- Ruff, mypy on Python 3.12 and 3.10, and the API type generation check passed.
- An intermediate run showed 8 CLI failures because the scratch Chroma server had been
  started on port 8765, `vectraudit serve`'s default; with it stopped they pass.

## Step 86: Ragas scores as answer judgments (1.2)

The owner asked what Ragas offers. It is an Apache-2.0 RAG evaluation library: LLM-judged
and non-LLM metrics, test-set generation, and experiments. Its test sets were already
imported (`dataset import --format ragas`). Its metrics are a judge suite of their own, so
following the strategy (import results; Steps 80, 84) its evaluation results become
`attach-answers` judgments.

`vectraudit workbench ragas-judgments RESULTS BENCHMARK --metric NAME --out F` reads
`evaluate(...).to_pandas()` saved as CSV, JSON Lines or JSON (records or pandas' column
layout):
- **Scores only:** Ragas gives no pass/fail, so judgments carry the score as
  `ragas.METRIC` and the threshold is declared in `attach-answers --threshold`. Nothing is
  assumed.
- **Chosen metrics,** with every `context` metric refused (they judge retrieval).
- **Failed metrics** (empty or `null`) skipped and counted.
- **Matching:** `query_id`/`trace_id` columns, else an unambiguous `user_input`; multi-turn
  samples only by ID.

**Built against real Ragas output**, following Steps 83 and 84. Ragas 0.4.3 ran offline
with its non-LLM metrics, plus a run with an unreachable LLM judge to record what a failure
looks like. What it showed:
- the ROUGE column is `rouge_score(mode=fmeasure)`, a name with `=` in it (attach-answers
  already splits `--threshold` on the last `=`);
- BLEU can exceed 1 by rounding, so scores are not range-checked;
- failures are empty CSV cells and JSON `null`, not `NaN` text.

The fixtures and the scripts that made them are in `tests/fixtures/ragas`.

Not included: Ragas test-set generation (needs an LLM; its output is already importable),
its experiment framework, and its context metrics as retrieval evidence.

Validation: 11 new tests on the real files:
- all three saved layouts read identically;
- judgments hold only the score;
- failed metrics skipped in CSV and JSONL;
- context metrics refused and unknown metrics listed;
- ID columns, multi-turn and ambiguous matching;
- scores turning into verdicts only with a declared threshold;
- other files refused;
- the CLI.

## Step 87: Phoenix pull and push (1.2)

The owner asked what Arize Phoenix offers. It is an LLM tracing and evaluation platform
built on OpenInference (Elastic License 2.0; it is not a dependency, only its REST API is
used). Its traces were already importable from exports (the `spans` and OTLP shapes) and
its evaluations through judgment files. What was missing, compared with Langfuse (Steps
66, 68), LangSmith (67) and Opik (69), was reading a Phoenix server directly and showing
the diagnosis there.

- `vectraudit workbench phoenix-pull --since --until [--project] [--traces F]
  [--judgments F] [--annotation NAME]`:
  - root spans that started in the window, then every span of those traces by trace ID,
    written unchanged (Phoenix's `/spans` records are the `spans` shape);
  - trace and span annotations as judgments: verdict labels with their score, unlabelled
    scores for `--threshold`, `HUMAN` as human review, other labels and notes skipped.
- `vectraudit workbench phoenix-push RUN [--send] [--name]`: one `CODE` trace annotation
  per judged trace, labelled with the diagnosis class. Previews by default. The diagnosis
  loop is shared with `langfuse-push` (`judged_diagnoses`).

**Built against a running Phoenix 21.0.0 server**, following Steps 83–86: the offline
LangChain pipeline's OpenInference traces were sent to it, annotations were added through
its API, and every response the pull received was recorded (`tests/fixtures/phoenix`, once
with tiny page sizes so cursor paging is real). What the server showed:
- `/spans` flattens document metadata (`document.metadata.doc_id`) where the OTLP export
  keeps a JSON string; both import to the same rankings and model context.
- Notes are stored as `HUMAN` annotations named `note`. Read as annotations, they would
  have become unlabelled human judgments; they are skipped and their text never copied.
- Writing annotations without `sync=true` answers 200 and silently drops those for
  unknown traces. With it, the whole batch is refused (404), so the push resends that
  batch one trace at a time and reports the unknown traces.
- An annotation with the same trace, name and identifier replaces the earlier one, so a
  re-push does not pile up annotations.

Not included: Phoenix datasets and experiments (Vectraudit's benchmarks and runs cover
them), its prompt management (prompt versions are read from traces, Step 70), and its
evaluators.

Validation: 27 new tests (17 pull, replayed from the real recordings; 10 push, against a
fake server that behaves as the real one did):
- the window and per-trace span reads;
- paged and unpaged recordings giving the same pull;
- pulled spans importing like the direct OpenInference export;
- annotations to judgments, notes skipped, name filtering;
- attached judgments with a disputed verdict;
- the auth header and HTTP failures without echoing bodies;
- cursor loops and record limits;
- unsafe settings;
- push contents with nothing private, re-push replacing, unknown traces found one by
  one, batching, preview, and both CLIs.

## Next increments

The code-only work the strategy names is done (Steps 1–47). What remains needs owner
decisions or customer inputs; the [launch scorecard](launch-scorecard.md) maps every
section 9 gate to its current evidence and what is still needed. In order:

1. **Distribution acceptance (1.9).** Make the source repository private, provision the
   distribution channel and per-customer credentials using the checklist in `RELEASING.md`,
   test a revocation, and run the reusable CI step once on GitHub-hosted runners. Local unit
   tests are not customer install evidence.
2. **Owner decisions** in the pilot offer and trust package drafts; a decision brief with options
   and recommendations is in [owner decisions](owner-decisions.md) (Step 48).
3. **Qualified conversations** recorded in the discovery ledger, with the sample evidence
   report as the leave-behind.
4. **First design partnership.** Then confirm the trace format (1.1; OpenInference import
   exists, Step 39), the context and answer
   evidence (1.2), the repair loop's failure classes (1.3), the benchmark upkeep integration
   (1.8) and the Tier 2 bet from that team's real traces, incidents and channel.

Customer evidence and commercial readiness remain unverified.
