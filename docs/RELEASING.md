# Releasing Vectraudit

Vectraudit is distributed privately. Nothing is published to PyPI. Customer delivery uses
a separate private, non-fork GitHub repository containing release wheels and checksums.
The source repository's releases are for maintainers only; never give a customer access to
it for installation. A Python wheel contains executable Python modules; this separates
repository access and development materials, and does not obfuscate the shipped code.

The implementation supports this channel but does not provision it. Before an outside
install, complete the setup and live acceptance checks below.

## Branches and review

Create development branches without inheriting the main branch's upstream:

```bash
git fetch origin main
git switch --no-track -c feat/your-change origin/main
git for-each-ref --format='%(refname:short) upstream=%(upstream)' refs/heads/feat/your-change
```

The upstream should be empty until the maintainer publishes the feature branch under its
own matching name. Review the change before merging it. See [launch-readiness.md](launch-readiness.md)
for pilot acceptance and operating evidence; successful CI alone does not establish them.

## Each release

1. Update `version` in `pyproject.toml` and add a `## [x.y.z] - date` section to
   `CHANGELOG.md`.
2. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "vectraudit X.Y.Z"
   git push origin main vX.Y.Z
   ```

3. `release.yml` runs the reusable CI workflow against the tagged commit (Python checks,
   frontend checks, browser tests and an installed-wheel smoke test). It then builds the UI
   and release wheel, checks the tag matches the version, and installs the release wheel
   in a fresh environment to verify its UI, API and CLI. `prepare_distribution.py` stages
   exactly one wheel plus `SHA256SUMS` in a new directory; source archives and other build
   files cannot join the release. Maintainer and customer releases use these same assets.
   Existing release tags are not overwritten; corrections require a new version.

## Provision the private channel once

1. Create an empty **private, non-fork** distribution repository, separate from this one.
   Keep application source, history, tests, customer content and private release notes out
   of it. GitHub automatically offers archives of the distribution repository itself, so
   its history must also remain free of source. The workflow verifies private/non-fork and
   different repository identity; an owner must verify its contents and access rules.
2. In the source repository, set repository variable `VECTRAUDIT_DISTRIBUTION_REPOSITORY`
   to `OWNER/DISTRIBUTION`. Leave it unset until the channel is ready; the distribution
   job skips when unset. There is no fallback customer download from the source repository.
3. Configure environment `private-distribution` and secret
   `VECTRAUDIT_DISTRIBUTION_PUBLISH_TOKEN`, with write access only to the distribution
   repository. Apply the owner's release-review rules to the environment. Customer
   download credentials must never use this publishing token.
4. Issue separate read-only distribution credentials per customer under the chosen
   GitHub account/app access model. Record owner, expiry, customer, repository scope and
   revocation procedure outside this repository. Do not reuse a maintainer's source token
   or share one download credential between customers.
5. Publish a tested release through the tag workflow. Deliver the version and SHA-256
   through the reviewed release handoff; committing a hash obtained blindly at install
   time would not pin the reviewed artifact.

## Install and upgrade

`vectraudit init` and workbench release exports both include the same local composite
action at `.github/actions/install-vectraudit/`. Commit it with the workflow. Set customer
repository variable `VECTRAUDIT_DISTRIBUTION_REPOSITORY`, secret
`VECTRAUDIT_DOWNLOAD_TOKEN`, and the workflow's exact `VECTRAUDIT_VERSION` (without `v`)
and `VECTRAUDIT_SHA256`. The generated checksum placeholder deliberately fails until
replaced. The action downloads exactly the named wheel, verifies bytes and package/version
metadata, and only then invokes pip. Its download credential is removed from pip's
environment. Backend dependencies still use the customer's configured pip index; the wheel
pin does not lock all transitive dependencies. Mirror/pin those separately for offline CI.

For a first local install, obtain the wheel through the private channel and the reviewed
installer script through onboarding. The installer is standard-library-only and requires
Python 3.10+; online download additionally requires `gh` and `GH_TOKEN`:

```bash
python install.py --repository OWNER/DISTRIBUTION --version X.Y.Z --sha256 REVIEWED_WHEEL_SHA256
# Or use the same verification with a pre-downloaded wheel:
python install.py --wheel ./vectraudit-X.Y.Z-py3-none-any.whl --version X.Y.Z --sha256 REVIEWED_WHEEL_SHA256
```

The installer script itself is trusted onboarding material: review it before execution.
It ships in the wheel and is also generated by `init`/release export. Do not execute an
unreviewed downloaded script to obtain the first trusted copy.

For an upgrade, review the changelog and update version and checksum together in a pull
request. Preserve the previous pins and artifacts for rollback. Test existing run,
benchmark and policy files with the proposed version before switching CI; new optional
fields can be rejected by older readers. Roll back using the previous wheel in a clean
environment and retained input files; do not assume an older tool can read newer files.
There are no runtime phone-home licence checks. Revocation stops future downloads, not
execution of wheels already delivered or copied into a customer cache.

## Live acceptance (required before customer use)

- Customer A can install the exact reviewed wheel in a fresh CI environment, with no
  permission to view the source repository.
- Customer B can independently install it using a different credential.
- Revoking A's credential makes a fresh download fail while B's download still works.
  Test with an empty cache; installed software is unaffected by download revocation.
- Wrong hashes, changed bytes and missing credentials fail before pip runs.
- The release has only the wheel and checksum assets; no source archives or source history
  were copied to the distribution repository. Upgrade and rollback succeed with retained pins.

Record the repository, artifact digest, date, credential identifiers (never tokens) and
test results in the pilot evidence log. Local tests cover installer behavior and staging;
they do not prove deployed authorization or revocation. The broader support/compatibility
commitment and buyer trust package from strategy item 1.9 still require owner review.

## Hugging Face Space

This is a maintainer-operated demo deployment, not a customer installation path.
Copy `deploy/hf-space/Dockerfile` and `deploy/hf-space/README.md` into a Docker Space (private
unless a public demo is a deliberate decision) and add a `GITHUB_TOKEN` Space secret with
read-only access to this repository. Set `VECTRAUDIT_REF` to the release tag for a
reproducible build. See [deploy/hf-space/README.md](../deploy/hf-space/README.md).
