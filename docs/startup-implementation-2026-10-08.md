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
| **1.1 Connection** | Python/HTTP and framework adapters, connection diagnostics, document lookup exist. | Representative trace importer, observed/missing capability report, independent engineer setup evidence. Choose the trace format from actual inputs. |
| **1.2 Context/answer evidence** | Stage rankings and passage judgments exist; `PipelineResult` has no actual-context, answer or citation records. | Context/answer/judgment provenance and evidence sufficiency, with mixed/unknown outcomes; depends on 1.0. |
| **1.3 Repair loop** | Findings, regression cases, verification and bounded CLI optimization exist. | Supported intervention controls and linked full-policy verification for customer-selected failure classes. |
| **1.4 Policy/uncertainty** | Query-family weighting/bootstrap and cross-split protection landed; critical assertions inspect repeats. Step 13: the optimizer screens and approves under the release policy against a baseline. Step 14 averages every repeat and exposes intermittent rankings/stage losses. Step 20: cohort counts on every surface and optional Bonferroni control, measured on A/A runs. Step 21: opt-in non-inferiority with inconclusive kept distinct from pass. Step 33: cohorts from context fields a benchmark declares. Step 34: paired (interleaved) execution and a warm-up mismatch warning. Step 37: paired release runs in CI. | Protocol evidence on a customer's endpoint. |
| **1.5 Execution** | Typed errors, retry budgets, per-candidate concurrency/rate limits, release protocols, checkpoint recovery and 1,000-question artifact/UI bounds landed. Step 19: calls/spend preview and budget guards before run, optimize and monitor. Step 30: actual calls and reported spend beside the plan, retries and failures included. Step 31: optimization runs each candidate at its own concurrency, rate limit and retry policy. Step 32: two evaluations at once in the workbench, one per registered pipeline, with overlap recorded. | Declared customer 1,000-question CI-window acceptance. |
| **1.6 Shared decisions** | Local question text is joined safely. Step 10: a decision ledger keeps release decisions after retention, with a hosted timeline and API. Step 15: cases, benchmark snapshots and policy revisions are shared as immutable allowlisted records. Step 16: plain-language decision summaries and per-benchmark quality/latency trends from the ledger (dashboard, API, CLI). Step 17: scheduled monitor runs upload into the same history with drift and a monitor status. Step 22: recurring problems across checks, linked to shared cases. Step 23: sharing from the workbench UI. Step 24: document titles beside IDs locally and, when shared with text, on hosted cases. | Lead and second-engineer acceptance with real users. |
| **1.7 Delivery** | GitHub publishing exists; `pipelines/monitor.py` writes local history and machine-readable alerts. Step 17: hosted monitor status with a stable alert ID per failing streak and the run that resolved it. Step 18: signed webhooks and Slack incoming webhooks deliver each decision and each alert (opened, resolved) once, with retries. Step 29: declared monitor schedules; a monitor that stops reporting raises one overdue alert, resolved by the next run. | Proof with the first customer's real channel; email or Slack-app delivery only if asked; additional CI only on demand. |
| **1.8 Benchmark upkeep** | Incident imports, deduplication, passage review, snapshots and freshness foundations exist. | Maintained production integration/cursors, reviewer routing, full incident-to-case links and measured upkeep effort. |
| **1.9 Delivery foundations** | Step 8 prepares pinned wheel-only distribution independently of source access. Step 26 drafts the trust package and support/compatibility policy. Step 27: one reusable CI step (install, record, check, report) called by both generated workflows. | Provision and verify channel/credentials, a first run of the CI step on GitHub-hosted runners, owner decisions in the drafts, a DPA template with counsel. |
| **1.10 Calibration/exceptions** | Step 11: scoped, audited, expiring exceptions with policy-declared roles, recorded in the decision ledger. Step 12: A/A calibration with false alarms by rule and a reviewable draft policy. Step 25: calibration in the workbench, with configuration differences called out. Step 28: the workbench shows who may accept each failure and the connected project's exceptions against it. | Customer-runner calibration evidence. |
| **Tier 2** | Chunking experiments, permission scenarios and CLI optimization foundations exist. | Customer-selected lineage/freshness or permission lifecycle proof; interactive experiment/migration workflow where demand supports it. |
| **Tier 3 / commercial** | Synthetic reference demos and operations/security documentation exist. | Defer optional breadth; obtain real pilot inputs, independent adoption, paid continuation, deployment evidence and sustainable support economics. Code cannot establish these. |

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

## Next increments

The code-only work the strategy names is done (Steps 1–40). What remains needs owner
decisions or customer inputs; the [launch scorecard](launch-scorecard.md) maps every
section 9 gate to its current evidence and what is still needed. In order:

1. **Distribution acceptance (1.9).** Make the source repository private, provision the
   distribution channel and per-customer credentials using the checklist in `RELEASING.md`,
   test a revocation, and run the reusable CI step once on GitHub-hosted runners. Local unit
   tests are not customer install evidence.
2. **Owner decisions** in the pilot offer and trust package drafts.
3. **Qualified conversations** recorded in the discovery ledger, with the sample evidence
   report as the leave-behind.
4. **First design partnership.** Then confirm the trace format (1.1; OpenInference import
   exists, Step 39), the context and answer
   evidence (1.2), the repair loop's failure classes (1.3), the benchmark upkeep integration
   (1.8) and the Tier 2 bet from that team's real traces, incidents and channel.

Customer evidence and commercial readiness remain unverified.
