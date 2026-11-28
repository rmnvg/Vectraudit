# Deployment and administration

## Profiles

- **Local engineering:** `vectraudit workbench serve DIRECTORY`, bound to loopback,
  plus trusted CLI/Python adapters. No authentication surface or uploads are added.
- **Private scheduled runner:** the same installed package and a controlled checkout;
  use `deploy/monitor/workbench-check.yml` or cron. Keep adapter credentials in runner secrets.
- **Self-hosted control plane:** existing `deploy/cloud/docker-compose.yml` (PostgreSQL,
  migration job, API) behind a TLS proxy. Evaluation stays on the user runner. Existing
  project-scoped runner tokens, workspace membership and audit logging remain authoritative.

## Single sign-on (OIDC)

Set `VECTRAUDIT_CLOUD_PUBLIC_URL` to the external HTTPS origin and
`VECTRAUDIT_CLOUD_SSO_JSON` to a JSON object with these five fields:

```json
{"client_id":"registered-client","client_secret":"SECRET_FROM_YOUR_SECRET_STORE","authorize_url":"https://identity.example/authorize","token_url":"https://identity.example/token","userinfo_url":"https://identity.example/userinfo"}
```

Register `https://YOUR_PUBLIC_ORIGIN/auth/sso/callback` as the exact redirect URI. The
provider must support authorization codes, S256 PKCE, form-encoded client-secret token
exchange, bearer access tokens and userinfo with `sub`, `email`, `email_verified:true`.
The login page shows single sign-on when configured. Only existing active users can
sign in; invitation, membership, role and tenant checks remain unchanged.

This is a fixed-provider OAuth code/userinfo integration. It does not parse ID tokens or
claim general OIDC/SAML compatibility. State is signed and expires after ten minutes;
PKCE binds the code exchange. Provider URLs must be HTTPS without embedded credentials,
query strings or fragments. Redirects, environment proxies and unbounded response bodies
are rejected. Codes, tokens and profile responses stay out of artifacts and access logs.
Choose a provider whose verified email ownership is an acceptable account identity policy.
Before launch, run positive/negative login and account-disable acceptance tests with the
actual provider. Mocked-provider tests do not certify a vendor configuration.

## Project permissions and governance

Existing workspace owners manage these endpoints with personal API tokens. Cookie sessions
still cannot authorize JSON API writes.

`PUT /api/v1/workspaces/{workspace}/projects/{project}/access`

```json
{"roles":{"EXISTING_MEMBER_USER_ID":"viewer"}}
```

A non-null map restricts project access to those existing members (viewer or engineer).
Their effective role is the lower of workspace and project role. An empty map denies all
non-owner human members; `{"roles":null}` restores workspace inheritance. Owners retain
administration access. Project-scoped runner tokens retain their separately authorized
scope; revoke them explicitly when removing automation access. Denied projects disappear
from project listings and return 404 to nonmembers. All changes are audited.

`PUT /api/v1/workspaces/{workspace}/governance`

```json
{"exports_enabled":false,"retention_days":90}
```

Retention must be 1–3650 days and uses the existing purge mechanism and baseline preservation
rules. Schedule the existing purge command. `exports_enabled` controls the bulk workspace
export endpoint, including owners; it does not revoke normal read access or prevent a reader
from saving information they can view. Re-enable exports through the governance endpoint.
Shared-report links retain their existing separate deployment setting (off by default).

## Upgrade and recovery procedure

Schema revision `0005` adds nullable project access maps and an enabled-by-default workspace
export switch. Existing workspaces preserve their access behavior after upgrade.

1. Record the image digest and current schema revision. Test the upgrade against a copy of
   the database using the new image and `vectraudit-cloud migrate`.
2. Take a database backup with the existing `deploy/cloud/backup.sh`; retain the old image
   and deployment configuration in your approved secret/backup stores.
3. Quiesce writes, run migrations, start the new API and require `/readyz` to pass.
4. Verify both tenants: owner read/write, viewer denial, project restrictions, revoked
   runner token, SSO login and one frozen release comparison.
5. Rehearse restore into an isolated database using `restore.sh`, then verify readiness,
   workspace/project/run/baseline counts, access rules and a known comparison verdict.
6. If recovery is needed, stop API writers, restore the matching backup and image together,
   verify readiness and repeat tenant checks before reopening traffic. Never restore over
   an active production database merely to test a backup.

`test_upgrade_and_backup_restore_preserve_governance` automatically upgrades revision 0004,
checks inherited access defaults, changes governance, backs up and restores a SQLite fixture,
and verifies schema and controls. Existing tests cover tenant isolation and hosted check
compatibility.

**PostgreSQL rehearsal (2026-11-28, Docker Compose, PostgreSQL 17).** `rehearse.py` brought
up a fresh deployment, created a workspace, uploaded a recorded run, backed up, deleted the
project and restored it: the run came back, and the restore took 5 to 13 seconds on a laptop
with almost no data. An upgrade from schema `0010` to `0011` with `start.sh` kept existing
workspaces and projects. Rolling that upgrade back by restoring the pre-upgrade backup failed
with the original `restore.sh` (`pg_restore --clean` cannot drop a table a newer migration
references), which is why `restore.sh` now drops the schema and replays the dump in one
transaction; the rollback, the old version starting on the restored data and a second upgrade
then succeeded, and a truncated dump was refused with the data unchanged. CI repeats the
rehearsal before each release. Not rehearsed: restore duration at user scale, TLS routing,
a real identity provider's sign-in and disaster-recovery objectives.
