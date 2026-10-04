# Releasing Vectraudit

Vectraudit is distributed privately. Nothing is published to PyPI; release builds are
attached to a GitHub release in this private repository, so only people with access to the
repository can download them.

## Each release

1. Update `version` in `pyproject.toml` and add a `## [x.y.z] - date` section to
   `CHANGELOG.md`.
2. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "vectraudit X.Y.Z"
   git push origin main vX.Y.Z
   ```

3. `release.yml` runs the reusable CI workflow against the tagged commit (Python checks,
   frontend checks, browser tests and an installed-wheel smoke test). It then builds the UI,
   sdist and release wheel, checks the tag matches the version, and installs the release
   wheel in a fresh environment to verify its UI, API and CLI. The private GitHub release
   is created only if these checks succeed.

Install a release wheel from the private release:

```bash
gh release download vX.Y.Z --repo rmnvg/Vectraudit --pattern '*.whl'
pip install ./vectraudit-X.Y.Z-py3-none-any.whl
```

## Hugging Face Space

Copy `deploy/hf-space/Dockerfile` and `deploy/hf-space/README.md` into a Docker Space (private
unless a public demo is a deliberate decision) and add a `GITHUB_TOKEN` Space secret with
read-only access to this repository. Set `VECTRAUDIT_REF` to the release tag for a
reproducible build. See [deploy/hf-space/README.md](../deploy/hf-space/README.md).
