# Evidence for a paid pilot

This phase improves the first evaluation and the safety of labels and verdicts. It does
not establish public-launch readiness by itself. Ship a reviewed private release and run
a guided pilot before widening distribution. Keep the record below per release and pilot.

## What this phase verifies

| Area | Automated behavior | Remaining evidence |
|---|---|---|
| Verdicts | Minimum measured pairs per metric; sparse cohorts reported; bounded bootstrap | Labels and policies reflect the customer's actual intents and release risk |
| Onboarding | Offline doctor identifies invalid inputs, absent secrets/dependencies, sparse labels and dimension mismatches | Live read-only access, correct IDs, embedding model/normalization and corpus freeze |
| Labels | Human confirmation, source spans, concurrent task transactions, stale-write conflicts, local review/export | Independent annotator guidance and representative coverage beyond pooled candidates |
| Distribution | Installed wheel exercises UI, API, doctor, annotation, export, record and check | The actual reviewed release artifact installs on the customer's target platform |
| Privacy | Local annotation and diagnostics need no remote service; browser CSP and CSRF protections | Review real run/report metadata and customer-approved hosted configuration |
| Operations | Existing cloud route audit, quotas, retention, deletion and migration tests | Deployment, monitoring, restore and deletion/backup timing verified in the target environment |

Direct hashing pooling supports a fast start; real release quality still needs credible
source-document labels. Preserve a fixed dataset version for a baseline/candidate pair.
A large suite with few labelled questions does not satisfy the policy minimum for nDCG.
Do not treat a self-check as proof of retrieval quality or a passing doctor report as proof
of backend connectivity. They verify setup and interoperability.

## Repeatable engineering checks

```bash
set -o pipefail
uv run ruff check .
uv run ruff format --check .
uv run mypy
UV_PROJECT_ENVIRONMENT=/tmp/vectraudit-py310 UV_PYTHON=3.10 uv run mypy
uv run pytest
cd frontend
npx tsc -p tsconfig.app.json --noEmit
npx oxlint src
npm test
npm run build
cd ..
node scripts/labelling_smoke.mjs
uv build
```

Use `uv sync --all-extras --locked` for a fresh development environment. The Python 3.10
side environment also needs the locked extras. The browser script requires the existing
frontend dev dependencies and Playwright Chromium (`npx playwright install chromium`
inside `frontend/`). For a local Chrome installation, set `VECTRAUDIT_BROWSER_PATH`.
Set `VECTRAUDIT_SCREENSHOT_DIR` to retain review, status and mobile screenshots. The script
uses synthetic data in a temporary directory and closes its server/browser after testing.
It checks that page requests remain on the loopback origin.

CI provisions the optional pgvector, Pinecone and Weaviate integration servers. Locally,
set `VECTRAUDIT_TEST_PG_DSN`, `VECTRAUDIT_TEST_PINECONE_URL` and
`VECTRAUDIT_TEST_WEAVIATE_URL` / `VECTRAUDIT_TEST_WEAVIATE_GRPC` only for the test services.
Unconfigured services skip their tests; record skips as unverified integrations, not passes.

CI and the private release workflow install the built wheel in isolation and run
`tests/package_smoke.py` with `python -I`; the source checkout cannot satisfy that test.
The smoke path covers the shipped FAISS UI and an annotation-generated dataset flowing
through `record` and `check` without sentence-transformers, torch or cloud extras.
Review the release workflow result on the exact tag before handing customers its wheel.
Do not publish to PyPI. Maintainers publish and merge explicitly after review.

## Guided pilot acceptance

Use one agreed retrieval project and backend. Have the customer choose representative
questions, grade the source documents, review a second-pass sample and freeze the dataset.
Run doctor offline, then record a small live suite to verify IDs, filters, embedding
dimensions, permissions and terminal status. Agree the policy before recording a candidate.

For the useful pilot result, collect all of the following:

- A baseline and a real candidate on the same versioned dataset and corpus freeze strategy.
- A report the customer can explain: verdict, affected queries, metric coverage, latency
  comparability and the evidence level of each diagnosis.
- The customer's acceptance of the labels, report and CI check; a documented owner for
  resolving false alarms or updating the frozen benchmark.
- A repeated evaluation by the customer after the first guided session. Record time to the
  first useful result and where manual help was needed, rather than assuming onboarding speed.
- A reproducible incident/support example with a response owner and support channel agreed
  before charging. Collect only the diagnostic fields the customer approves for sharing.

A 30–50-question set is a starting heuristic, not statistical power evidence. Increase
coverage and inspect intervals for smaller effects or cohorts. Add diverse retrieval pools
and independently inspect documents outside them to reduce coverage bias. Local annotator
names are attribution, not authenticated identities. Hosted annotation is still deferred.

## Operating and commercial acceptance

The deployment operator should supply target-environment evidence for tenant isolation,
credential revocation, retention purge, workspace export/deletion, TLS, monitored health,
a backup restore and backup-expiry/deletion timing. Exercise the declared pilot load and
concurrent runs against the real deployment; neither a unit fixture nor a laptop run
establishes hosted capacity or API p95 latency.

Windows task locking has an implementation but is not exercised by the current macOS/Linux
checks; network filesystems are also unverified. Use local customer-controlled storage and
verify the platform before committing support. Model licences and actual local-LLM inference
must be verified for any model offered to a customer. Optional dependencies retain their
existing licensing requirements; this phase adds no runtime dependencies.

Before taking payment, agree pilot scope, quotas, customer-side compute costs, support
coverage, cancellation, and how success will be judged. Record the customer's acceptance
and billing owner. Pricing, name/domain clearance, contractual review and a repeatable
sales motion require owner decisions and customer evidence; this branch does not invent
those outcomes. Keep the first offer narrow and review repeat usage before widening it.

## Release record

Attach the reviewed commit/tag, wheel checksum, CI result, any skipped checks, supported
platform/backend, pilot dataset hash, frozen corpus identifier, baseline/candidate run IDs,
customer acceptance, deployment/restore evidence and support owner to the release review.
Keep customer text and credentials in customer-controlled storage; share aggregate reports
only by explicit agreement. Re-run the affected evidence after changing code, dependencies,
platform or deployment configuration.
