# Evidence a reviewer can verify

A security, compliance or audit reviewer may ask how a retrieval release was tested, on what
data, and under which rule. A CI log expires, and a screenshot proves nothing. An evidence
bundle is a zip that a reviewer outside the team can check with one command, without access
to your data.

```bash
vectraudit check baseline.json candidate.json --policy policy.json \
  --bundle evidence/release-42.zip --bundle-runs
```

```text
Evidence bundle: evidence/release-42.zip (manifest SHA-256 2cbda1cd…)
```

The reviewer runs:

```bash
vectraudit verify release-42.zip
```

```text
ok          hash: all 10 files match the manifest
ok          policy: policy SHA-256 6f0e3a91c2b4… recomputed
ok          verdict: re-running the comparison reproduces fail and every rule's outcome (3 reasons)
Manifest SHA-256 2cbda1cd…; recorded verdict fail
Verified: hashes hold and the comparison reproduces the verdict.
```

## What is in the bundle

The bundle holds IDs and numbers only:

| File | Holds |
|---|---|
| `check.json` | The verdict, every reason with its rule key, metrics, deltas and intervals |
| `policy.json` | The policy's rules and their SHA-256 |
| `runs/baseline.header.json`, `runs/candidate.header.json` | Run IDs, labels, times, status, backend, `k` and parameters, the commit, and the summary metrics |
| `runs/baseline.json`, `runs/candidate.json` | Only with `--bundle-runs`: per question, the IDs returned and the metrics |
| `dataset.json` | The dataset version (SHA-256), its counts and metadata |
| `exceptions.json` | Every audited exception applied: ID, rule, who accepted it, why, until when |
| `versions.json` | Vectraudit and Python versions, the check format and both runs' metric contracts |
| `inputs.json` | The SHA-256 of the run and policy files given to `check` |
| `manifest.json` | The SHA-256 and size of every other file |

Runs enter the bundle through the same sharing projection the hosted service uses. Text
parameters, such as a prompt, become SHA-256 fingerprints. Question text, document text,
model context, endpoints and credentials are never included. The check in the bundle is
recomputed on those projections, and `check` refuses to write the bundle if that does not
give the verdict of record. A failing, accepted or not comparable check writes its bundle as
well, and the exit code is unchanged.

## What `verify` checks

1. **Every hash.** A file that changed, is missing or was added fails verification, and the
   file is named.
2. **The policy.** Its rules hash to the recorded policy SHA-256.
3. **The verdict.** When the bundle holds the runs (`--bundle-runs`), the comparison is run
   again with the same policy and audited exceptions, and must reproduce the verdict and
   every rule's outcome. Without runs in the bundle, the reviewer can pass the original files
   (`--baseline`, `--candidate`). Their SHA-256 must match `inputs.json`.
4. **The dataset**, when given with `--dataset`: its version must be the one the runs were
   measured on.

The exit code is 0 when the bundle is **verified**: the hashes hold and the comparison
reproduces the verdict. It is 2 for **hashes consistent, verdict not reproduced**, a bundle
without runs that was not re-run. It is 1 when verification **failed**, or the file is not a
bundle. A bundle without runs is never reported as verified.

```bash
vectraudit verify release-42.zip --baseline baseline.json --candidate candidate.json \
  --dataset dataset.jsonl --json verification.json
```

When the comparison does not reproduce and the bundle was made by another Vectraudit
version, `verify` says so, because the metrics or rules may have changed between versions.

## Signing

Vectraudit does not sign bundles (decision D18: it adds no cryptography dependency). Each
file's hash is in `manifest.json`, so signing that one file, or recording its SHA-256 where
the reviewer can see it, covers the bundle. Without that, someone who edits a file and
rewrites the manifest to match passes the hash check, so the manifest hash matters. Even
then, a bundle with runs fails `verify` unless the runs, the check and the verdict all agree.
Use your CI's signing, for example Sigstore `cosign sign-blob manifest.json`, `minisign`, or a
GPG detached signature, or attach the zip to a signed release. `check` prints the manifest
hash, and `verify` prints it again for comparison.
