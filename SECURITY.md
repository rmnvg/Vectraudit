# Security

Vectraudit has three parts with different security models:

- **The release-check runner** (`vectraudit record`, `check`, `cloud`) runs in your CI.
  It holds your vector-database credentials, which it reads from environment variables
  (connection files reject literal secrets) and never sends to us. Run files contain ids,
  metrics and timings, not document text or vectors.
- **Vectraudit Cloud** (`vectraudit-cloud`), the hosted service: workspaces, tokens, runs,
  baselines, checks and reports. Its threats and controls are in
  [docs/threat-model.md](docs/threat-model.md); operating it is described in
  [docs/cloud-operations.md](docs/cloud-operations.md) and incidents in
  [docs/incident-response.md](docs/incident-response.md).
- **The local FAISS debugger** (`vectraudit serve`): single-user and unauthenticated. It
  binds to `127.0.0.1`; don't expose it publicly except with `--demo-mode`, which is
  read-only and caps expensive operations. FAISS index files are loaded with FAISS's own
  reader: only open index files you trust.

## Reporting a vulnerability

Email the maintainer with the details and how to reproduce; please don't open an issue.
We acknowledge within two working days, agree on a fix timeline based on severity, and
credit you if you wish. Fixes ship in the next release and, for the hosted service, are
deployed as soon as they are verified.

## Dependencies

CI scans the locked Python (`pip-audit`) and frontend (`npm audit`) dependencies on every
pull request, on every push to `main` and weekly. When an advisory affects a dependency:

1. Check whether Vectraudit uses the affected code path (note the finding in the pull
   request either way).
2. Upgrade the dependency (`uv lock --upgrade-package NAME` / `npm update NAME`), run the
   full test suite and release.
3. If no fix exists and the path is used, mitigate (disable the feature, add a guard) and
   record the decision in `docs/DECISIONS.md` until a fix ships.
4. For the hosted service, follow the incident process if the vulnerability may have been
   exploitable in production.
