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

## October 9 code audit

The strategy is deliberately broader than one engineering phase. This table separates
implemented foundations from remaining acceptance work; existing features are not counted
as missing merely because their customer validation is pending.

| Item | Current code evidence | Remaining work |
|---|---|---|
| **1.0 Evidence boundary** | Step 9: structured context/filters/history, optional source identity and coverage, allowlisted hosted projection, preview and opt-in text sharing. | Sharing settings in the workbench UI; extend the projection when 1.2 adds context/answer records. |
| **1.1 Connection** | Python/HTTP and framework adapters, connection diagnostics, document lookup exist. | Representative trace importer, observed/missing capability report, independent engineer setup evidence. Choose the trace format from actual inputs. |
| **1.2 Context/answer evidence** | Stage rankings and passage judgments exist; `PipelineResult` has no actual-context, answer or citation records. | Context/answer/judgment provenance and evidence sufficiency, with mixed/unknown outcomes; depends on 1.0. |
| **1.3 Repair loop** | Findings, regression cases, verification and bounded CLI optimization exist. | Supported intervention controls and linked full-policy verification for customer-selected failure classes. |
| **1.4 Policy/uncertainty** | Query-family weighting/bootstrap and cross-split protection landed; critical assertions inspect repeats. Step 13: the optimizer screens and approves under the release policy against a baseline. Step 14 averages every repeat and exposes intermittent rankings/stage losses. Step 20: cohort counts on every surface and optional Bonferroni control, measured on A/A runs. Step 21: opt-in non-inferiority with inconclusive kept distinct from pass. | Measured execution protocol and context-derived cohorts. |
| **1.5 Execution** | Typed errors, retry budgets, per-candidate concurrency/rate limits, release protocols, checkpoint recovery and 1,000-question artifact/UI bounds landed. Step 19: calls/spend preview and budget guards before run, optimize and monitor. | Optimization concurrency, two simultaneous workspace runs, actual-versus-planned spend, and declared customer 1,000-question CI-window acceptance. |
| **1.6 Shared decisions** | Local question text is joined safely. Step 10: a decision ledger keeps release decisions after retention, with a hosted timeline and API. Step 15: cases, benchmark snapshots and policy revisions are shared as immutable allowlisted records. Step 16: plain-language decision summaries and per-benchmark quality/latency trends from the ledger (dashboard, API, CLI). Step 17: scheduled monitor runs upload into the same history with drift and a monitor status. | Sharing from the workbench UI, readable document titles, recurring findings and lead/second-engineer acceptance. |
| **1.7 Delivery** | GitHub publishing exists; `pipelines/monitor.py` writes local history and machine-readable alerts. Step 17: hosted monitor status with a stable alert ID per failing streak and the run that resolved it. Step 18: signed webhooks and Slack incoming webhooks deliver each decision and each alert (opened, resolved) once, with retries. | Proof with the first customer's real channel; email or Slack-app delivery only if asked; monitor staleness alerts; additional CI only on demand. |
| **1.8 Benchmark upkeep** | Incident imports, deduplication, passage review, snapshots and freshness foundations exist. | Maintained production integration/cursors, reviewer routing, full incident-to-case links and measured upkeep effort. |
| **1.9 Delivery foundations** | Step 8 prepares pinned wheel-only distribution independently of source access. | Provision and verify channel/credentials, full reusable CI wrapper, owner-approved support and trust materials. |
| **1.10 Calibration/exceptions** | Step 11: scoped, audited, expiring exceptions with policy-declared roles, recorded in the decision ledger. Step 12: A/A calibration with false alarms by rule and a reviewable draft policy. | A workbench screen for calibration; exceptions in the workbench once cases are shared; customer-runner calibration evidence. |
| **Tier 2** | Chunking experiments, permission scenarios and CLI optimization foundations exist. | Customer-selected lineage/freshness or permission lifecycle proof; interactive experiment/migration workflow where demand supports it. |
| **Tier 3 / commercial** | Synthetic reference demos and operations/security documentation exist. | Defer optional breadth; obtain real pilot inputs, independent adoption, paid continuation, deployment evidence and sustainable support economics. Code cannot establish these. |

## Next increments

1. **Finish distribution acceptance (1.9).** Provision the selected channel and verify
   independent customer credentials using the concrete checklist in `RELEASING.md`. Review
   support/trust commitments; do not substitute local unit tests for customer install evidence.
2. **Recurring findings (rest of 1.6).** Show on the hosted project which questions fail
   again across checks, linked to their shared regression cases, so a lead sees repeat
   problems rather than one check at a time. This needs no customer input.
3. **Trace import (1.1) and context/answer evidence (1.2).** Choose the trace format from a
   design partner's real traces before building.

Pilot offer, supported scope, discovery ledger, support policy and trust materials accompany
these increments. Trace-import and deeper diagnosis choices follow representative customer
inputs. Customer evidence and commercial readiness remain unverified.
