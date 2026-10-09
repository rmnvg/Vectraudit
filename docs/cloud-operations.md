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

## Scheduled jobs

| When | Command | Why |
|---|---|---|
| Daily | `docker compose run --rm api purge` | Deletes runs past retention; pinned baselines are kept. |
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
