# Security

Vectraudit is a local, single-user tool. `vectraudit serve` binds to `127.0.0.1` and has no
authentication, so anyone who can reach the port can use it. Don't bind it to a public
interface; for a public deployment use `--demo-mode`, which is read-only and caps the
expensive operations.

FAISS index files are loaded with FAISS's own reader. Only open index files you trust.

## Reporting a vulnerability

Report vulnerabilities to the maintainer directly. This repository is private, so GitHub's
private vulnerability reporting (a public-repository feature) is not available. Fixes go into
the latest release.
