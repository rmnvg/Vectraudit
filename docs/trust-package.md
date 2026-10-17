# Vectraudit trust package

**Draft for owner review.** Written for a buyer's security reviewer. Items in *[brackets]*
need an owner decision or a fact that depends on the chosen deployment. Vectraudit makes
**no compliance claim** (for example SOC 2 or ISO 27001): no independent audit has been
performed. Technical detail: [threat model](threat-model.md),
[cloud operations](cloud-operations.md), [deployment](enterprise-deployment.md),
[incident response](incident-response.md).

## Deployment modes and what leaves your environment

| Mode | Where evaluation runs | What leaves your environment by default |
|---|---|---|
| **Local** (workbench and CLI) | Your workstation or CI runner | Nothing. The workbench binds to loopback and has a per-session token. It calls only the pipelines and stores you register, plus the hosted project when you explicitly share a record |
| **Private scheduled runner** | Your CI runner or a host you control | Nothing, unless you enable upload |
| **Hosted project, self-hosted** (you run the control plane) | Your runner | Projections go to *your* service; nothing reaches us |
| **Hosted project, operated by us** *[if offered]* | Your runner | Projections and shared records you explicitly upload; notifications you configure |

In every mode **your documents, passages, request context (tenant, user, groups), filters,
vectors, credentials, endpoint hosts and error messages stay on your side**. Evaluation always
runs where your application runs; the hosted service never calls your application.

## Data flows

```
Your application ──(questions, context)──▶ Vectraudit runner (your machine/CI)
        ▲                                            │ raw run files, local only
        └──────(ranked IDs, optional stages)─────────┤
                                                     │ optional, explicit upload
                                                     ▼
                                   allowlisted projection (preview first)
                                                     │ HTTPS, project-scoped token
                                                     ▼
                                     Hosted control plane ──▶ your webhook / Slack
                                     (PostgreSQL)              (headline, verdict, link)
```

**Uploads** (`record --upload`, `check --upload`, `cloud push`, `workbench share`) send a
versioned, allowlisted projection, never the raw file:
- **Runs:** query, document and family IDs, tags (including cohorts derived from context
  fields a benchmark declares in `cohort_fields`, such as `tenant=acme`), run labels,
  metrics, timings, safety
  assertions, numeric search parameters, text parameters as SHA-256 fingerprints, and pipeline
  identity.
- **Shared cases, benchmarks and policies:** IDs, status, judgments and policy rules.
- **Shared investigations:** question and run IDs, loss classes, diagnosis status, counts of
  evidence requests, fix trial and validation results, outcome, protecting policy versions,
  and times. Reviewer names, dispute reasons, evidence requests and notes are never sent.
- **Question text and document titles:** sent only with an explicit opt-in. The workbench
  shows the text before you send it.

`vectraudit cloud preview` and the workbench share panel show the exact payload without any
network call. The service re-validates every upload against the same allowlist and rejects
anything outside it. IDs, tags and labels can themselves be sensitive; review them before
uploading.

**Notifications** (webhooks you configure) carry a headline, the verdict, the candidate's
label, the check ID and a dashboard link. They carry no reasons, question IDs or text, and go
only to https endpoints on public addresses.

## Retention and deletion (hosted)

| Data | Kept | Deleted by |
|---|---|---|
| Raw run projections and checks | Plan retention: 30 days (pilot), 90 (team), 365 (enterprise); pinned baselines are kept | Daily purge job |
| Release decisions (compact ledger) and shared records | Until the project or workspace is deleted | Project or workspace deletion |
| Webhook deliveries | 30 days after delivery or failure | Daily purge job |
| Dashboard sessions | 14 days, revocable | Expiry, sign-out, or `revoke-access` |
| Audit events | With the workspace | Workspace deletion |
| Backups | *[Operator's backup retention]*: deleted data persists in backups until they expire | Backup expiry |

Owners can export everything in a workspace, and delete a project, a workspace or a
person, at any time. Deletion is immediate in the database and reaches backups when they
expire.

## Access control and secrets

- Workspaces are isolated tenants. Every route is checked against the caller's
  membership, and a test calls every workspace route with another tenant's credentials.
- **Roles:** viewer, engineer, owner. Runner tokens are bound to one project with scoped
  permissions and cannot pin baselines, manage members or webhooks, or read audit logs.
- **Token storage:** tokens and sessions are stored only as hashes, and shown once.
  Webhook signing secrets are stored to sign requests, shown once and rotatable.
- **Sign-in:** GitHub OAuth, or a fixed-provider OAuth code flow with PKCE for enterprise
  sign-in. This is not general SAML or OIDC support.
- **Audit:** baseline pins, audited exceptions, policy changes, webhook changes,
  shared-record uploads, membership changes and deletions are audit events, exportable per
  workspace. Run uploads and checks are recorded with who made them, not as separate audit
  events.
- **Secrets:** the client refuses to send tokens over plain HTTP to non-local hosts, and
  connection files accept environment-variable references only, never literal secrets.
  In the generated GitHub step, each side's backend credentials reach only that side's
  record command and the GitHub token only the check command. The installer sees neither,
  and the download token reaches only the installer, which keeps it out of pip's environment.

## Security engineering

- **Dashboard:** strict Content-Security-Policy with no scripts, CSRF tokens on every form,
  HTML-escaped output, and clickjacking protection.
- **Webhooks (SSRF):** https only, public addresses checked at registration and before
  every send, redirects never followed. The residual DNS-rebinding risk is documented.
- **Dependencies:** dependency audits (`pip-audit`, `npm audit`) in CI, and new
  dependencies limited to permissive licences.
- **Abuse and resilience:** rate limits on authentication, request-size limits, plan
  quotas, structured errors that do not echo submitted content, and an access log without
  query strings.

## Common questionnaire answers

| Question | Answer |
|---|---|
| Do you access our documents or production data? | No. Evaluation runs on your infrastructure; documents, passages and context are never uploaded |
| Do you train models on our data? | No. Vectraudit does not train models, and the hosted service stores only the projections listed above |
| Where is data stored? | Self-hosted: where you run it. Operated by us: *[region and hosting provider]* |
| Subprocessors | Self-hosted: none from Vectraudit. Operated by us: *[hosting provider; email provider if used]* |
| Encryption in transit | HTTPS through your TLS proxy (self-hosted) or *[ours]*; the client refuses plain-HTTP tokens to remote hosts |
| Encryption at rest | Provided by the database host or volume encryption *[confirm per deployment]* |
| Can access be revoked quickly? | Yes. Runner tokens and sessions are revocable individually; `revoke-access` revokes a whole workspace's credentials |
| Incident response | [incident-response.md](incident-response.md): severity levels, roles, containment and customer notification *[owner to confirm notification timelines]* |
| Penetration test or certification | None yet *[owner decision]*. No compliance claim is made |
| Licensing and distribution | Proprietary. Pinned, checksummed wheels from a private channel with credentials per customer and no source access *[channel being provisioned]* |

## Supported versions and compatibility (draft policy)

- **Supported:** Python 3.10–3.13 on Linux and macOS, tested in CI. Windows is not
  verified.
- **Versioned file formats:**
  - run files (`vectraudit.run`, `format_version` 1, with metric contract versions);
  - check results (`format_version` 1);
  - policies (`version` 1, with a content hash shown wherever a policy judges);
  - release files (`format_version` 1);
  - hosted projections (`release-metadata-v1`, `workbench-metadata-v1`).

  New fields are added only when they leave existing hashes and digests unchanged, and old
  files keep loading.
- **Deprecation:** a format or command is announced in the changelog at least *[N
  months]* before removal, with upgrade notes. Security fixes are released for the
  *[latest minor version]*. Advisories go to *[security contact / channel]*.
- **Upgrades:** pin an exact version and checksum, upgrade in a branch, and rerun the
  release check before merging. Rollback reinstalls the previous pinned wheel. See
  [RELEASING.md](RELEASING.md).

## What a reviewer should verify in their own environment

These need evidence from the actual deployment; tests on our side do not establish them:
- tenant isolation against the deployed service;
- credential revocation;
- the retention purge;
- export and deletion;
- TLS configuration;
- a backup restore and the backup expiry.

The deployment checklist is in [launch readiness](launch-readiness.md).
