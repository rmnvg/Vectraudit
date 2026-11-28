# Releasing Vectraudit

Releases are wheels attached to GitHub releases of this repository, together with a
`SHA256SUMS` file, and the same wheel on [PyPI](https://pypi.org/project/vectraudit/). The
wheel includes the built web UI. The CI workflows that `vectraudit init` and workbench
release exports generate install the wheel from the GitHub releases, pinned to an exact
version and SHA-256.

## One-time PyPI setup

PyPI uses trusted publishing: no API token is stored in the repository. Before the first
release, add a trusted publisher on pypi.org (a "pending publisher" while the project does
not exist yet) with owner `rmnvg`, repository `Vectraudit`, workflow `release.yml` and
environment `pypi`, and create a `pypi` environment in the repository settings
(optionally with required reviewers, so each upload waits for approval). Without this,
the PyPI job fails but the GitHub release is still published.

## Branches and review

Create development branches without inheriting the main branch's upstream:

```bash
git fetch origin main
git switch --no-track -c feat/your-change origin/main
git for-each-ref --format='%(refname:short) upstream=%(upstream)' refs/heads/feat/your-change
```

The upstream should be empty until the branch is published under its own matching name.
Review the change before merging it.

## Each release

1. Update `version` in `pyproject.toml` (then `uv lock`), turn `[Unreleased]` in
   `CHANGELOG.md` into a `## [x.y.z] - date` section above a new empty `[Unreleased]`, and
   update the install commands that name a version (`README.md`, the example workflow
   under `examples/github/`). `tests/test_distribution.py` fails until they all match.
2. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "vectraudit X.Y.Z"
   git push origin main vX.Y.Z
   ```

3. `release.yml` runs the reusable CI workflow against the tagged commit (Python checks,
   frontend checks, browser tests and an installed-wheel smoke test). It then builds the UI
   and release wheel, checks the tag matches the version, and installs the release wheel
   in a fresh environment to verify its UI, API and CLI. `scripts/stage_release.py` stages
   exactly one wheel plus `SHA256SUMS` in a new directory, and the workflow attaches both to
   a GitHub release whose notes come from `CHANGELOG.md`. Finally it uploads the same
   wheel (not `SHA256SUMS`) to PyPI. Existing release tags are not overwritten, and PyPI
   never accepts the same version twice; corrections require a new version.

## Install and upgrade in CI

`vectraudit init` and workbench release exports both include the same two local composite
actions. `.github/actions/retrieval-check/` is the single step both workflows call: it
installs the pinned release, records the target branch and the pull request on the same
runner, checks them and uploads the report. It uses `.github/actions/install-vectraudit/`
for the install. Commit both with the workflow, and set the workflow's exact
`VECTRAUDIT_VERSION` (without `v`) and `VECTRAUDIT_SHA256` (from the release's
`SHA256SUMS`). The generated checksum placeholder deliberately fails until replaced.

The install action downloads exactly the named wheel from the GitHub release (this
repository by default; set the action's `repository` input for a fork or mirror),
verifies its bytes and package/version metadata, and only then invokes pip. Runner tokens
are removed from pip's environment. Backend dependencies still use your configured pip
index; the wheel pin does not lock all transitive dependencies, so mirror or pin those
separately for offline CI.

The check action passes each side's secrets (`baseline-env`, `candidate-env`, one
`NAME=value` per line) only to that side's record command and the GitHub token only to
`check`; the installer sees neither. To upgrade an `init` project, install the new release
locally, run `vectraudit init --backend BACKEND --ci-only --force` (only `.github/` is
rewritten), then update both pins. Release exports regenerate the actions on each export.

For release checks, `paired: true` on the action records both sides in one
`vectraudit release run-paired` command: each side runs in its own worker process from its
own checkout, and questions alternate which side goes first, so drift during the run (a
deployment warming up, a busy endpoint) affects both alike. Each side's secrets reach only
its own worker. Paired runs are serial, so the approved release must use concurrency 1.

The installer is standard-library-only and requires Python 3.10+. It can also be run by
hand:

```bash
python install.py --version X.Y.Z --sha256 WHEEL_SHA256
# Or use the same verification with a pre-downloaded wheel:
python install.py --wheel ./vectraudit-X.Y.Z-py3-none-any.whl --version X.Y.Z --sha256 WHEEL_SHA256
```

For an upgrade, review the changelog and update version and checksum together in a pull
request. Keep the previous pins for rollback. Test existing run, benchmark and policy files
with the proposed version before switching CI; new optional fields can be rejected by older
readers. Roll back using the previous wheel in a clean environment and retained input
files; do not assume an older tool can read newer files.

## Hugging Face Space

Copy `deploy/hf-space/Dockerfile` and `deploy/hf-space/README.md` into a Docker Space.
Set `VECTRAUDIT_REF` to the release tag for a reproducible build. See
[deploy/hf-space/README.md](../deploy/hf-space/README.md).
