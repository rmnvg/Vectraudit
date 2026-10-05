# Workbench validation — 2026-10-05

Local verification for the `feat/retrieval-workbench` implementation:

| Check | Result |
|---|---|
| Full Python regression suite | 1,076 passed, 22 skipped, 1 deselected |
| Python lint and formatting | Passed |
| Strict Python type checks | Passed on Python 3.12 and 3.10 environments |
| Frontend unit tests | 61 passed across 13 files |
| Frontend type check and production build | Passed |
| Frontend lint | Passed |
| Real HTTP adapter transport | Local loopback request/context/response test passed |
| Desktop/mobile browser regression | Passed; no browser exceptions |
| Schema upgrade and recovery fixture | 0004 → 0005, inherited permissions, governance changes, SQLite backup/restore passed |
| Enterprise sign-in | PKCE/code/userinfo flow, state rejection, unverified and uninvited identities tested with a mocked provider |
| Private wheel and source archive | Built locally; nothing published |
| Isolated installed-wheel smoke | Passed outside the checkout, including standalone UI assets and no-answer evaluation |

Browser checks use real CLI-generated artifacts and the real standalone artifact server.
They open pipeline traces, trace a document ID, inspect optimization/held-out results,
open benchmark review, and verify the mobile viewport. The README screenshot uses only
synthetic example data. `scripts/workbench_smoke.mjs` reproduces this workflow.

The 22 skipped tests require externally configured connector services or optional
integrations; the deselected test is marked slow. Those results are **not** evidence that
those deployments passed. Actual provider SSO, PostgreSQL disaster recovery, representative
customer load and operational launch acceptance remain to be verified in the target
environment. No live deployment, remote schedule, customer-data upload or external alert
was configured by this work.

## Market-fit workflows — 2026-10-05

Local validation on `feat/market-fit-workflows`:

- Python regression suite: **1,115 passed, 22 skipped, 1 deselected**. The skips still
  require external pgvector, Pinecone or Weaviate services; the slow test is deselected.
  Local-server tests were run with loopback socket access enabled.
- Python lint, formatting and mypy: passed in the Python 3.12 development environment.
- Frontend: **65 tests passed**, type check/production build and lint passed.
- Existing guided workflow browser regression: passed.
- New `scripts/market_fit_smoke.mjs`: passed at desktop and mobile sizes with no browser
  exceptions or horizontal overflow. Covers passage grading and reload, frozen versions,
  persistent case notes, incident deduplication without labels, permission scenarios and
  stale-label review. Added to the CI browser job.
- API tests additionally cover real tenant forwarding, no-answer cases, immutable snapshots,
  provenance/span/filter preservation, complete-run fix verification, recurrence, stale-write
  rejection, origin/token checks and symlink boundaries.

These results validate the local workflows. They do not establish customer adoption,
production deployment readiness, or live third-party integration acceptance. Incident
imports remain local-file based, and case recurrence is checked by explicit verification.
