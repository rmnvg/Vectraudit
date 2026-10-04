# Incident response

For Vectraudit Cloud. Commands assume the Compose deployment in `deploy/cloud`; prefix them
with `docker compose run --rm api` (shown as `vac`).

## Severity

| Level | Examples | Response |
|---|---|---|
| **SEV1** | Customer data exposed to another tenant or the internet; secret key, database or backup leaked; active exploitation | Start now, all hands; customer notice within 72 hours of awareness |
| **SEV2** | A token or account compromised in one workspace; data deleted by mistake; service down | Start within 1 hour; notify the affected workspace's owners |
| **SEV3** | Vulnerable dependency without known exploitation; degraded performance | Fix in the normal release cycle, record the decision |

## Roles

One person **leads** (decides, keeps the timeline), one **operates** (runs commands), one
**communicates** (customers, internal). In a small team one person may hold several roles;
the lead still writes the timeline as they go.

## Steps

1. **Detect and open a record.** Note the time, what was seen and by whom. Keep a timeline
   in the incident record from here on (UTC timestamps, actions, decisions).
2. **Triage.** Which workspaces, what data, since when? Read the audit log
   (`vac audit SLUG`) and the request logs (JSON lines with `request_id`, path, status:
   no tokens or query strings are logged).
3. **Contain** (pick what fits):
   - Leaked runner or personal token: the owner revokes it in the dashboard, or
     `DELETE /api/v1/.../runner-tokens/{id}`.
   - Compromised account or workspace: `vac revoke-access --workspace SLUG --yes` (runner
     tokens, plus members' personal tokens and sessions).
   - Service-wide compromise: `vac revoke-access --yes` (every token and session), then
     rotate `VECTRAUDIT_CLOUD_SECRET_KEY` (invalidates CSRF tokens and all share links)
     and the database password, and restart.
   - Share links abused: set `VECTRAUDIT_CLOUD_SHARE_LINKS=false` (all links stop working
     at once) or rotate the secret key.
   - Vulnerable code path: deploy a fix or disable the feature; if needed take the API
     offline (`docker compose stop api`) while keeping the database.
4. **Eradicate.** Fix the cause (patch, configuration, access on the host), and add a test
   that would have caught it (the route audit is the place for authorization bugs).
5. **Recover.** Restore data if needed (`./restore.sh backups/<file>.dump`, see
   cloud-operations), then issue new tokens to the affected owners
   (`vac create-token EMAIL`); they re-create runner tokens and update CI secrets.
6. **Notify.** Tell affected workspace owners what happened, what data was involved, what
   we did and what they should do (rotate their CI secrets; their database credentials were
   never held by us). For personal data, check notification duties under the applicable law
   (e.g. GDPR: supervisory authority within 72 hours when required).
7. **Review** within a week, blameless: timeline, root cause, what worked, what didn't,
   follow-up actions with owners and dates. Update the threat model.

## Scenarios

| Scenario | First actions |
|---|---|
| Runner token in a public log | Revoke it; check `audit` for uploads from its prefix after the leak; the token can't read other projects or pin baselines |
| Personal token leaked | Revoke it (`/account` or `revoke-access --workspace`); review audit for member, token and baseline changes by that user |
| Baseline moved unexpectedly | `audit` shows who pinned which run (with previous run); re-pin the right run |
| Workspace deleted by mistake | The deletion record shows who and when; restore the latest backup into a staging database and export the workspace from there, or restore production if the whole service can roll back |
| Cross-tenant access suspected | SEV1. Reproduce with the route audit, contain with `revoke-access`, review request logs by path for the affected ids |
| Secret key leaked | SEV1. Rotate it (share links and CSRF tokens become invalid), revoke sessions, check logs for share-link use |
| Database or backup leaked | SEV1. Token and session values are stored as hashes; run files contain ids and metrics, not documents. Revoke everything, rotate all secrets, notify all customers |
