# Threat model

Scope: the hosted control plane (`vectraudit.cloud`), the runner CLI (`vectraudit record`,
`check`, `cloud`), and the paths between them. Reviewed at Phase 17; update it whenever a
new kind of data, actor or integration is added.

## What we protect

| Asset | Where | Why it matters |
|---|---|---|
| Customer vector-database credentials | Customer CI secrets only | Full read access to their corpus. **Never sent to us.** |
| Run files | Customer CI, our database | Document ids, ranks, metrics, timings, backend configuration, commit SHAs. No document text or vectors by design. |
| Evaluation datasets | Customer repository | Query text and labels. Never uploaded; runs carry only the dataset hash. |
| Checks and reports | Our database | Reveal retrieval quality, regressions and configuration of a customer's product. |
| API tokens and sessions | Hashes in our database | Act as a user or runner. |
| Audit log | Our database | Evidence of who changed baselines, members and tokens. |
| Secret key | Service environment | Signs CSRF tokens and share links. |

## Actors

- **Members** of a workspace (owner, engineer, viewer) and **runner tokens** (one project).
- **Other tenants**: authenticated but must see nothing of other workspaces.
- **Anonymous internet**: no data access; sign-in pages and share links only.
- **Fork pull requests**: run untrusted code in customer CI; must get no secrets.
- **Operators** (us): create workspaces and run maintenance; no implicit data access.
- **A compromised dependency** in the service or runner.

## Trust boundaries and controls

| Threat | Control | Verified by |
|---|---|---|
| Tenant reads another tenant's data | Every tenant query goes through `repo.authorize`; foreign objects return the same 404 as missing ones | Route audit test: every workspace route is called with another tenant's token and session against real victim ids; unscoped routes need a reviewed entry |
| Runner token used beyond its project | Tokens bound to one project; scopes per action; runners can't pin baselines, manage members or read audit logs | `test_cloud_accounts`, route audit |
| Stolen token database | Only SHA-256 hashes stored; plaintext shown once | `security.py`, tests |
| Token or password guessing | 256-bit random secrets; rate limits on failed authentication and sign-in (Phase 17) | `test_cloud_ratelimit` |
| Session theft | Random session ids, only hashes stored; HttpOnly, SameSite=Lax, Secure over HTTPS; revocable; 14-day expiry | `test_cloud_web` |
| Cross-site request forgery | HMAC CSRF tokens on dashboard forms; session cookies authorize API reads only | `test_cloud_web` |
| Cross-site scripting | All values HTML-escaped; no scripts; strict CSP on pages and reports; reports sandboxed in an iframe | report and dashboard tests with hostile labels |
| Clickjacking | `frame-ancestors 'none'` (pages), `'self'` (embedded report), `X-Frame-Options: DENY` | `test_cloud_ops` |
| Open redirect after sign-in | Only `/app…` paths accepted as `next` | `test_cloud_web` |
| Share-link forgery or extension | HMAC over check id and expiry; max 30 days; off by default; secret rotation revokes all | `test_cloud_history` |
| Secrets in logs | Access log records path only (no query, no headers); uvicorn access log off | `test_cloud_ops` |
| Tampered or corrupted upload | Body digest header; schema validation (`check_run`); idempotency conflict on changed content | `test_cloud_runs` |
| Baseline moved silently | Only engineers/owners pin, only complete runs, every pin audited with previous run | `test_cloud_checks` |
| Resource exhaustion | Body size limit, per-run query cap, plan quotas | `test_cloud_quotas` |
| Fork PR exfiltrates secrets | Workflow uses `pull_request`, never `pull_request_target`; publishing degrades to job summary | template and docs |
| Database credentials of customers leak via connection files | Literal secrets rejected; only `*_env` references | `test_connectors` |
| API keys for hosted vector databases leak in transit | Pinecone and Weaviate connectors refuse to send keys over plain HTTP except to a local emulator; URLs with embedded credentials are rejected | `test_connector_pinecone`, `test_connector_weaviate` |
| SQL injection via pgvector connector | Identifier validation and quoting; bound parameters; read-only transactions | `test_connector_pgvector` |
| Vulnerable dependency | `pip-audit` and `npm audit` in CI (Phase 17) | CI |
| Webhooks used to reach internal services (SSRF) or exfiltrate data | Owner-only management, audited; https only; every resolved address must be public, checked at registration and before each send; redirects never followed; response bodies discarded; payload limited to headline, verdict, IDs and a link | `test_cloud_webhooks` |
| Forged or replayed webhook events at the receiver | HMAC-SHA256 over timestamp and body with a per-webhook secret shown once and rotatable; receivers reject stale timestamps and dedupe on the event key | `test_cloud_webhooks` |
| Poisoned documents end up in a customer's production index | Vectraudit never writes to a backend (connectors are read-only). A poison suite is generated locally from fixed templates (no network, no model) with harmless sentinel claims. It runs either in memory beside the corpus or against a staging index the customer loaded; a candidate with an `overlay` must declare `"environment": "staging"`, and a poison run refuses one that does not | `test_poison`, `test_poison_run`, `test_poisoning` |
| Document or answer text leaks through a poisoning report | Suites, reports, safety records and red-team imports hold IDs, canary tokens, offsets, counts and flags only; the corpus scan, canary check and red-team import never write a document, prompt, answer or reason; excerpts print to the terminal only | `test_corpus_scan`, `test_canary`, `test_promptfoo_redteam`, `test_poisoning_demo` |
| Hostile document text attacks the operator's terminal (escape sequences, bidirectional controls) | `--show-excerpts` prints at most 120 characters with every non-printable character escaped; excerpts never enter a file | `test_corpus_scan` |
| A held-out question is tuned against by a poisoning check | Poison suites, controls, review queues and claim exports use exploratory questions only; held-out questions are refused by name and never sampled, and a held-out question repeated under a new ID refuses the check | `test_poison`, `test_poisoning` |
| A capacity run overloads, or is pointed at, the wrong backend | A run is read-only searches only, bounded before it starts by a duration (default 30 s) and a global requests-per-second ceiling (default 50, at most 1000), and prints the most it can send first. It refuses to start unless `--confirm-target` equals the connection's collection, index or table, and refuses a target or connection named like `prod` without `--allow-production`. An authentication or configuration error stops it on the first rejected request rather than repeating it. It sends nothing for the refusal checks beyond reading the backend's description | `test_capacity` |
| A crafted benchmark directory harms the machine or poisons a reference | `dataset import-vdbbench` only reads local Parquet files (pyarrow, no pickle, `.npy` read with `allow_pickle=False`), never downloads, never writes into the source directory, and records every source file's SHA-256. A reference is verified against the hash of its neighbour array on load, its distance, `k`, filter and queries are checked against the run, and neighbours outside the corpus or breaking the filter rule are refused | `test_vdbbench_import`, `test_external_reference` |
| Data retained after a customer leaves | Workspace export and deletion; retention purge; backup expiry documented (Phase 17) | `test_cloud_deletion` |

## Residual risks and assumptions

- **TLS termination** is the deployment's job (reverse proxy). Running over plain HTTP
  outside localhost exposes tokens; the client refuses to send tokens over HTTP to remote
  hosts.
- **Rate limits are per process.** With several workers or instances an attacker gets a
  multiple of the limit; still far below what 256-bit tokens would need. Put a proxy-level
  limit in front for production.
- **Backups** contain deleted data until they expire; customer terms must state backup
  retention.
- **Operators with database access** can read customer data. Mitigate with access control
  on the database host and audit of operator actions; consider encryption at rest from the
  database provider.
- **Run files can contain identifiers** a customer considers sensitive (document ids,
  backend names, commit SHAs). Documented; customers control what ids they use.
- **A poisoning check shows only what its attacks did.** Templates are fixed and public, so
  a pass does not mean a pipeline resists a tailored attack, and an absent canary shows only
  that these tokens were not echoed. Findings are labelled hypothesis, supported or verified,
  and a poisoning check never approves a release.
- **The corpus scan is a set of known patterns.** It can miss a hijack or a paraphrased
  instruction, and ordinary text can trip its weak detectors; findings mark documents for a
  person to read.
- **GitHub OAuth** trusts GitHub's verified-email claim.
- **Webhook DNS rebinding.** A host can resolve to a public address when checked and a
  private one when connected. The window is one request, and the response is discarded,
  but a deployment with sensitive internal HTTP services should also restrict the
  delivery worker's egress at the network level. Webhook secrets are stored in the
  database to sign requests.
