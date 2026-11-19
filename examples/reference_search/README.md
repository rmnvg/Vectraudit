# Reference support-search application

This example searches 18 AI-authored, fictional support passages using Vectraudit's BM25
pipeline. Its 13 distinct questions cover 12 proposed relevance judgments and one no-answer
case. All labels are **pending human review**; `judgments.json` supplies source evidence
for review. This is an engineered retrieval failure, not a user case study.

Run from the repository with development dependencies installed:

```bash
uv run python examples/real_pipeline_case_study.py --out /tmp/reference-search-study
```

The output directory must not exist. The installed-wheel version needs only the base
Vectraudit package and `httpx`, plus these example files from the checkout or source archive.
There are no model downloads, database services or new dependencies.

`app.py` uses document text, request filters and tenant context. It never reads labels or
branches on question IDs. The runner creates three standalone source revisions:

| Revision | Actual change |
|---|---|
| baseline | Rank the whole corpus, then filter tenant/current documents and return top 2 |
| candidate | Keep only the global top 2 before applying those filters |
| fixed | Restrict the searchable corpus first, then rank and keep top 2 |

For three questions, short archived/other-tenant passages outrank longer current guidance.
The shallow candidate has no permitted documents left after filtering. The fixed revision
retrieves current guidance again. Unauthorized documents never reach final results; the
example demonstrates recall loss from filter ordering, not a final-result permission leak.
Intermediate stage evidence intentionally includes competing documents in this fictional
local corpus and must not be exposed across tenants in a real application.

The fixed implementation builds its small filtered index per call. Its end-to-end latency
includes that work; stage timings start afterward. This is a correctness demonstration,
not a claim that the fix is faster. Production integrations should use native prefiltering.

With the checked-in input version, baseline and fixed nDCG@2/recall@2 are 1.00 across the
12 labeled questions; the candidate falls to 0.75 and fails account recovery on all three
executions. The no-answer question passes in every revision. These are fixture results,
not estimates of production quality. Latencies are measured per run and preserved in
`summary.json`; the small corpus and sample do not support performance promises.

## Inspect the result

- `candidate/report.html`, `.md`, `.json`: measured regression, policy verdict and critical gate.
- `fixed/report.html`, `.md`, `.json`: fix judged under exactly the same policy.
- Each revision's `run.json`: real rankings, repeated executions and measured stage timings.
- `investigation.json`: account-recovery stage evidence across all three revisions.
- `summary.json`: metrics, source hashes, policy hash, elapsed automation time and explicit
  pending human-review/reproduction fields. Machine time is not human setup time.
- `inventory.json`: immutable fixture version and input file checksums.

The runner prints a `vectraudit workbench serve ... --manifest ...` command to open the
saved benchmark and runs. In **Compare a fix**, select baseline and candidate, choose
**Reference search regression policy**, and inspect `recover-account`. The keyword stage
already lacks the current guide in the candidate; filtering then removes the remaining
two hits. Compare baseline and fixed under the same policy to see recovery.

The runner verifies guided-API/exported-CLI agreement both on the exact guided artifacts
and newly executed runs. Each exported directory contains its own adapter source and corpus;
each executes in an isolated Python subprocess. Three distinct recorded code hashes prove
the runner did not accidentally reuse one imported implementation for every revision.

To verify the visible workflow as well, build the frontend and install Playwright Chromium,
then run from the repository root:

```bash
node scripts/reference_search_smoke.mjs
```

This creates a fresh case study, opens its persisted runs in Chromium, and compares the
candidate and fix under the saved policy. It checks visible repeat failures, the first
missing stage, blocked export for the failing candidate, and a downloaded release ZIP for
the fix. Browser response metrics, critical evidence and policy identity must match the
CLI reports. It also checks the narrow-screen layout and browser errors. Desktop/mobile
screenshots, the ZIP and a Playwright trace are kept in the printed output directory.
Set `VECTRAUDIT_REFERENCE_SMOKE_DIR` to a directory without an existing `study` child to
choose that location. The test uses local port 8821.

## CI and reproduction

The repository's **Build wheel** CI job executes this case study using its isolated wheel
installation and uploads `reference-search-case-study`, including reports and reproduction
inputs. A regression is the expected result of the candidate; the harness succeeds only
when candidate exit 3 and fixed exit 0 match the guided verdicts. CI never reports the
broken candidate as an acceptable release.

The browser CI job runs the same visible journey and retains a separate
`reference-search-browser` artifact, including failure screenshots/traces when available.
Both installed-wheel and browser journeys run locally.

Each generated directory also contains an exported GitHub workflow. That workflow is a
template for an application repository; review its pinned wheel installation and pipeline
dependency prerequisites before adopting it. This fixture does not create Git commits,
push code, or dispatch workflows.

For a second engineer, supply the example folder/script and the release wheel.
Have them run the command above in a fresh environment, inspect the reports, review the
proposed labels, and record their setup effort. Do not mark human review or independent
reproduction complete automatically. `inventory.json` must be regenerated after intentional
input edits; freeze a new version and reevaluate all three revisions together.
