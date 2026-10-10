# Running Vectraudit Cloud

The hosted control plane stores uploaded runs, pinned baselines, checks and reports for
workspaces of customers. It is one Docker image plus PostgreSQL, with no cloud-specific
services, so it runs on any provider. Customer credentials never reach it: runners execute
in the customer's CI and upload an allowlisted projection of each run file (see below).

## What an upload contains

Uploads send a versioned projection (`release-metadata-v1`), never the raw run file, which
stays unchanged on disk. The projection keeps what a release check needs: query, document
and family IDs, tags, run labels, metrics, timings, safety assertions, retry and attempt
rankings, corpus and runner identity, numeric and boolean search parameters, text search
parameters as SHA-256 fingerprints, and pipeline identity (adapter kind, name, configuration
hash and versions). It excludes request context and filters, vectors, endpoint hosts, module
paths and other adapter configuration, raw stage evidence and source observations, error
messages, dataset descriptions and provenance, and unknown fields. Query text and
conversation history are excluded unless `--share-query-text` is given explicitly. IDs, tags
and labels can themselves be sensitive; review them before uploading.

Preview a transfer without credentials or a network call, and optionally write the exact
payload to a new file for review:

```bash
vectraudit cloud preview runs/candidate.run.json
vectraudit cloud preview runs/candidate.run.json --out review/candidate.projection.json
vectraudit cloud push runs/candidate.run.json            # sends that projection
```

The service projects every upload again, so an older client that sends a raw run file still
stores only the projection, and it rejects a projection with fields outside its profile or a
digest mismatch. Uploading the same run ID with different sharing settings conflicts rather
than rewriting history. `check cloud:<baseline>` compares projections on both sides, so its
local report and the hosted check show the same verdict and the same changed settings. Runs
stored before this change keep their original content.

## Deploy with Docker Compose

```bash
cd deploy/cloud
cp .env.example .env          # set POSTGRES_PASSWORD, VECTRAUDIT_CLOUD_SECRET_KEY, PUBLIC_URL
docker compose up -d --build  # PostgreSQL, then migrations, then the API on 127.0.0.1:8800
curl -s localhost:8800/readyz # {"status":"ready","schema":"0003"}
```

Put a TLS-terminating reverse proxy in front of port 8800 (Caddy, nginx, or your cloud's
load balancer) and set `VECTRAUDIT_CLOUD_PUBLIC_URL` to the `https://` address. Over HTTPS
the service sends HSTS and marks session cookies `Secure`.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `VECTRAUDIT_CLOUD_DATABASE_URL` | yes | PostgreSQL URL (`postgresql://user:pass@host:5432/db`). URL-encode special characters in the password. |
| `VECTRAUDIT_CLOUD_SECRET_KEY` | yes | 32+ random characters. Signs CSRF tokens and share links; rotating it signs everyone out of forms and revokes all share links. |
| `VECTRAUDIT_CLOUD_PUBLIC_URL` | yes in production | The address users and runners use; appears in links. |
| `VECTRAUDIT_CLOUD_GITHUB_CLIENT_ID` / `_SECRET` | no | "Sign in with GitHub". Create a GitHub OAuth app with callback `<PUBLIC_URL>/auth/github/callback`. Without it, members sign in by pasting a personal token. |
| `VECTRAUDIT_CLOUD_SHARE_LINKS` | no | `true` enables signed, expiring public report links (off by default). |
| `VECTRAUDIT_CLOUD_MAX_UPLOAD_BYTES` | no | Request body limit (default 20 MB). |
| `VECTRAUDIT_CLOUD_MAX_QUERIES_PER_RUN` | no | Hard cap across all plans (default 10,000). |
| `VECTRAUDIT_CLOUD_WEBHOOK_ALLOW_PRIVATE` | no | Development only: lets webhooks use http and private, loopback or link-local addresses. Leave unset in production. |

## First workspace

```bash
docker compose run --rm api create-admin you@yourcompany.com      # operator token
docker compose run --rm api create-workspace acme --name "Acme" --owner-email lead@acme.com
```

`create-workspace` prints the owner's personal token once. The owner signs in to the
dashboard, invites members (`POST /api/v1/workspaces/acme/members` or ask an operator),
creates a project and a runner token, and stores the runner token as the CI secret
`VECTRAUDIT_CLOUD_TOKEN`. Only invited emails can sign in. Operators (admins) can create
workspaces but have no access to customer data.

## Plans and billing

```bash
docker compose run --rm api set-plan acme team --limit runs_per_month=2000
docker compose run --rm api usage --month 2026-10 > usage-2026-10.csv   # for invoicing
```

Plans set runs per month, queries per run, stored runs and retention days (pilot: 500,
2,000, 1,000 and 30 days). Over-limit uploads get a clear 413/429 error; retries of an
already stored run are never blocked.

## Decision ledger and release timeline

Retention deletes runs and the checks that reference them, but not the decisions. Every
release check also writes a small ledger row: verdict, policy digest, baseline name, run
IDs and digests, candidate label and commit, per-metric changes with intervals, latency,
critical-question counts, the first ten reasons and the names of changed settings. It holds
no query text, document text or raw evidence, only what the upload projection already
contains. Rows are kept until their project or workspace is deleted, appear in workspace
exports, and are not counted against stored-run limits.

The project page's **Release timeline** shows decisions and baseline pins, newest first,
with a **Summary** link and a link to the full report while the check still exists
("Report expired under retention" afterwards). The same timeline is available as JSON, paginated like the other histories:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$VECTRAUDIT_CLOUD_URL/api/v1/workspaces/acme/projects/search/decisions?limit=50"
```

The purge records any missing decision before it deletes a check. After upgrading an
existing deployment, record decisions for checks made before the ledger existed once:

```bash
docker compose run --rm api backfill-decisions
```

## Release summaries and trends for the lead

These views read only the decision ledger, so they keep working after runs expire.

**Plain-language summary.** For each decision: a headline and the action it implies, and
then:
- **what changed:** quality metrics in words, p95 latency and the names of changed settings;
- **why it matters:** failing rules, accepted exceptions, critical questions and questions
  that scored lower;
- **what is uncertain:** changes whose 95% interval includes zero, fewer than 30 independent
  question families, warnings, and a report that has expired.

A failure that is within the noise still reads as a failure, because the policy judges the
measured change. An accepted failure reads "fail, accepted", never pass. The project page
shows the latest decision this way. The summary page (`/app/WORKSPACE/PROJECT/decisions/CHECK_ID`)
offers a Markdown download, and the same summary is available from
`GET .../decisions/CHECK_ID/summary` and the command line:

```bash
vectraudit cloud summary CHECK_ID            # Markdown; `--json` for the structured form
vectraudit cloud summary "$CHECK_ID" >> "$GITHUB_STEP_SUMMARY"
```

`check cloud:main --upload` prints the matching `cloud summary` command. Labels and reasons
are escaped in the Markdown, so an uploaded label cannot inject links or HTML into an issue
or chat.

**Trends.** `/app/WORKSPACE/PROJECT/trends` (and `GET .../trends?limit=200`) plots each
check's candidate value against its baseline, coloured by verdict, plus p95 latency, oldest
on the left.
- Each benchmark version (dataset SHA-256) is a separate series. Scores on different
  questions are not comparable, so no line crosses a benchmark change.
- A dotted rule marks a policy change: verdicts on either side were judged by different
  rules.
- Charts are inline SVG, so the dashboard stays script-free.
- Points are spaced by check, not by time, and a rejected candidate is plotted as checked;
  the line shows what was proposed, not only what shipped.

## Scheduled monitor history

`vectraudit workbench monitor MANIFEST cloud:main --upload` records each scheduled run as a
hosted check marked `monitor` (see the [workbench guide](retrieval-workbench.md#record-monitor-runs-in-the-hosted-project)).
API clients can create the same with `"trigger": "monitor"` on `POST .../checks`.

- **Drift:** monitor checks also record what drifted: corpus version, configuration (each
  `changed`, `unchanged` or `unknown`) and how many questions returned different documents.
- **Where they appear:** in the release timeline (labelled **Monitor**), in trends (hollow
  points) and in plain-language summaries ("Scheduled monitor: … now breaks 2 release
  rules"; nothing was released, so the action is to check what drifted).
- **Release card:** the project page's latest-release card ignores monitor runs.

**Monitor status.** The project page and `GET .../monitor` derive the monitor state from
the ledger, so it outlives run retention.
- **`alerting`:** while the newest runs fail, error or are not comparable. The alert opens at
  the first run of that unbroken streak, and its `alert_id` (that run's check ID) stays the
  same until a run passes.
- **`healthy`:** after a pass, with the most recently resolved alert and the run that
  resolved it.
- **Accepted failures:** a failure accepted by an audited exception neither opens nor
  extends an alert.

A notifier can therefore deliver each alert once and announce its resolution. Vectraudit
does not send notifications itself yet.

## Notifications (webhooks)

A project owner can add up to 10 webhooks under **Notifications** on the project page, or
with `POST .../projects/PROJECT/webhooks` (`url`, `style`, `events`).

- **Events:**
  - `decision`: each release decision.
  - `monitor`: a scheduled monitor alert when it opens, and again when a passing run
    resolves it (see [monitor history](#scheduled-monitor-history)).
- **Delivered once each:** an event is queued once per webhook, keyed by
  `decision:<check>` or `monitor.alert_opened|alert_resolved:<alert>`. A repeated check
  queues nothing, and a monitor that keeps failing does not notify again.
- **Formats:**
  - `signed`: Vectraudit JSON (`format: vectraudit.event`) with event, verdict, trigger,
    candidate, headline, action, check ID, alert details and a dashboard link.
  - `slack`: the same headline, action and link as a `text` message for a Slack incoming
    webhook.
- **What is sent:** no reasons, question IDs, metric details or text. Receivers with
  dashboard access follow the link.
- **Security:**
  - Only https URLs whose host resolves to public addresses are accepted, checked when the
    webhook is added and again before every delivery.
  - Redirects are not followed.
  - Responses are not stored beyond their status code.
  - The URL's path is never shown back, because Slack-style URLs carry a token.
  - Webhooks appear in exports without their secret or full URL.
- **Retries:** after 1 min, 5 min, 30 min, 2 h and 6 h, then the delivery is marked failed.
  Redirects and 4xx responses other than 408 and 429 fail at once. **Last delivery** on the
  project page and `GET .../webhooks/ID/deliveries` show each attempt's outcome.
- **Roles:** owners add, test (`POST .../webhooks/ID/test` sends a `ping`), rotate the secret
  (`POST .../webhooks/ID/rotate-secret`) and remove webhooks; each change is audited.
  Engineers can see webhooks and deliveries; viewers and runner tokens cannot.

Delivery is at-least-once, like any webhook system: dedupe on `X-Vectraudit-Event-Key`. The
secret is shown once. Every request carries `X-Vectraudit-Timestamp` and
`X-Vectraudit-Signature: v1=<hex HMAC-SHA256 of "<timestamp>.<body>">`. Verify it:

```python
import hashlib, hmac, time


def verify(secret: str, headers: dict, body: bytes) -> bool:
    stamp = headers["X-Vectraudit-Timestamp"]
    expected = (
        "v1=" + hmac.new(secret.encode(), stamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    )
    fresh = abs(time.time() - int(stamp)) <= 300
    return fresh and hmac.compare_digest(expected, headers["X-Vectraudit-Signature"])
```

Deliveries are sent by `vectraudit-cloud deliver-webhooks`, which the compose file runs as the
`webhooks` service (see [scheduled jobs](#scheduled-jobs)). An exception granted after a check
does not send a second event for that check.

## Audited exceptions

An approver can accept one failing rule of a hosted check when the project policy opens that
kind of rule (see [release checks](release-checks.md#audited-exceptions)). On the check page
the **Failing rules** table offers an accept form to members with the required role; from
the command line:

```bash
vectraudit cloud accept CHECK_ID metric:ndcg.max_drop \
  --reason "Accepted for the reranker launch; fix tracked in SEARCH-142" \
  --owner lead@example.com --expires 2026-11-15 [--max-checks 5]
vectraudit cloud exceptions [--all]
vectraudit cloud revoke-exception EXCEPTION_ID
```

- Only a rule that actually failed in that check can be accepted, with a reason, an owner
  and an expiry within the policy's `max_days`. One active exception per rule per project.
- The check it came from is re-evaluated at once, and later checks of the project are
  covered while the exception is active (not revoked, not expired, under `--max-checks`).
  A check whose failures are all covered reads "fail, accepted"; any other failure still
  fails it.
- Expired, used-up or revoked exceptions stop applying, so the failure blocks again.
  Checks they covered keep their record.
- Runner tokens and viewers cannot accept. Changing the policy's `exceptions` block needs
  the owner role, while other policy edits stay engineer-level.
- Every grant and revocation is an audit event and appears in the release timeline; each
  decision records which exceptions covered it and its verdict before exceptions. The
  project page lists exceptions under **Accepted regressions** with a revoke button.

## Shared cases, benchmarks and policies

A second engineer or lead can read the workbench records that matter for a release without
access to the first engineer's machine. Share one saved regression case, benchmark snapshot
or policy revision with a project:

```bash
vectraudit workbench share WORKSPACE_DIR case CASE_ID --preview     # no credentials, no upload
vectraudit workbench share WORKSPACE_DIR case CASE_ID --project acme/search
vectraudit workbench share WORKSPACE_DIR benchmark SNAPSHOT_ID [--share-query-text]
vectraudit workbench share WORKSPACE_DIR policy POLICY_ID
```

- The upload is an allowlisted projection (profile `workbench-metadata-v1`), validated on
  both ends. Cases keep IDs, status, finding kinds and document IDs, the policy identity
  and the verification history; snapshots keep dataset identity and judgments (relevant,
  irrelevant and forbidden IDs, spans, no-answer, split, family); policies keep their rules,
  hash and lineage. Owners, investigation notes, passages, context and any unknown field
  never leave the machine, and the server rejects a record carrying them without echoing it.
- Question text is sent only with `--share-query-text`; the record and the page say so.
- Each local revision is immutable once uploaded: retrying the same content returns the same
  record, while different content under the same revision is refused (409). A new local
  revision uploads as a new record, so the history of a case stays readable.
- Runner tokens and engineers can upload; viewers can read. Records are listed under
  **Shared cases** on the project page (`/app/WORKSPACE/PROJECT/records`, filterable by kind)
  and through `GET .../records` and `GET .../records/ID/file`. Each upload is an audit event.
- Records are kept after run retention, included in workspace export, and deleted with their
  project or workspace. A record is at most 4 MiB.

Shared records are read-only evidence from the workbench. They do not synchronize, and a
case marked verified locally is not a hosted release approval; only hosted checks and the
decision ledger are.

## Scheduled jobs

| When | Command | Why |
|---|---|---|
| Daily | `docker compose run --rm api purge` | Deletes runs past retention; pinned baselines and ledger decisions are kept. Also prunes finished webhook deliveries older than 30 days. |
| Continuously | the compose `webhooks` service (`deliver-webhooks --watch 30`), or `deliver-webhooks` from cron every minute | Sends queued webhook events and retries failures. |
| Nightly | `./backup.sh` and copy `backups/` off the host | Restore point (beta targets: RPO 24 h, RTO 8 h). |

## Backups and restore

`backup.sh` writes a compressed `pg_dump` to `backups/` and never leaves an empty file
behind on failure. To restore:

```bash
docker compose stop api
./restore.sh backups/vectraudit-YYYYmmdd-HHMMSS.dump
docker compose start api
```

Test a restore regularly: a backup you have not restored is a hope, not a backup. Deleted
data persists in backups until they expire; state your backup retention in customer terms.

## Upgrades

Pull or build the new image and `docker compose up -d`: the `migrate` service applies
migrations before the API starts. `/readyz` returns 503 `migrations_pending` while the
schema is behind the code. With several API instances, run `vectraudit-cloud migrate` once
as a release step instead of `serve --migrate`.

## Monitoring

- `GET /healthz`: liveness. `GET /readyz`: database reachable and schema current.
- Logs are one JSON line per request (`request_id`, method, path, status, ms) on stdout.
  Query strings and headers are never logged, so tokens and share-link signatures stay
  out of logs. Send `X-Request-ID` to correlate with your proxy.

## Customer data requests and incidents

| Need | How |
|---|---|
| Export a workspace | Owner: dashboard "Download everything", or `GET /api/v1/workspaces/SLUG/export` |
| Delete a workspace or project | Owner: dashboard (type the slug) or `DELETE` with `{"confirm": "SLUG"}` |
| Delete a person | `docker compose run --rm api delete-user EMAIL --yes` (refused for the last owner of a workspace) |
| Contain a compromise | `docker compose run --rm api revoke-access --workspace SLUG --yes`, or `--yes` alone for everything |
| Investigate | `docker compose run --rm api audit SLUG` (JSON lines), request logs by `request_id` |

See [incident-response.md](incident-response.md) for the full process and
[threat-model.md](threat-model.md) for what the service defends against.

## Security notes

- Tokens are stored as SHA-256 hashes; the plaintext is shown once. Sessions store only a
  hash of the cookie value.
- Every tenant-owned query is scoped by workspace; other workspaces' objects return the
  same 404 as missing ones.
- Dashboard pages and reports send a strict Content-Security-Policy with no scripts.
- Session cookies can read the JSON API but never write to it; writes need a token.
- Failed sign-ins and token guesses are limited per address (20 per 5 minutes, then 429);
  add a proxy-level limit for production.
