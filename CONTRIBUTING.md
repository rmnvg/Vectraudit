# Contributing to Vectraudit

Thanks for your interest. Bug reports, documentation fixes and pull requests are welcome.

## Before you start

- For anything larger than a small fix, open an issue first so the approach can be agreed
  before you spend time on it.
- Report security problems privately as described in [SECURITY.md](SECURITY.md), not in a
  public issue.

## Development setup

```bash
git clone https://github.com/rmnvg/Vectraudit && cd Vectraudit
uv sync --all-extras                       # Python 3.12 is pinned for development
cd frontend && npm install && npm run build && cd ..
```

## Checks to run before opening a pull request

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
cd frontend && npx tsc -p tsconfig.app.json --noEmit && npx oxlint src && npm test
```

Tests that need a live vector backend skip themselves unless their `VECTRAUDIT_TEST_*`
variable is set. To run them, start the same servers CI uses (Docker required; add a
`--port-offset` if ports such as 5432 are taken):

```bash
python scripts/live_backends.py up --track newest --port-offset 10000 > live.env
set -a && . ./live.env && set +a
uv run pytest tests/test_connector_*.py tests/test_acceptance.py tests/test_tune.py
python scripts/live_backends.py down
```

On a pull request, these run in the **Live backends** workflow when you change a connector,
against the oldest and newest server versions listed in the script.

CI also type-checks on Python 3.10 with an older numpy; see [CLAUDE.md](CLAUDE.md) for the
side-environment commands. After changing frontend runtime dependencies, run
`node scripts/third-party-notices.mjs` in `frontend/` (CI fails on drift).

## Conventions

- `vectraudit.core` is a plain library and must not import FastAPI or anything web-related.
- Type hints everywhere and docstrings on public functions; write tests with the code.
- Record each non-obvious technical decision as one line in
  [docs/DECISIONS.md](docs/DECISIONS.md), and add user-visible changes to
  [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.
- New dependencies need a permissive licence (MIT, BSD, Apache-2.0, ISC or similar).
- Use conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

## Licence

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same licence as the project.
