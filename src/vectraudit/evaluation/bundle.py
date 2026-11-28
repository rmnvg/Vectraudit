"""Evidence bundles: a release check a reviewer outside the team can verify.

:func:`build_bundle` writes a zip with the check's verdict and reasons, the policy, both runs'
headers, the dataset's version, the audited exceptions applied and by whom, the tool and
metric-contract versions, the SHA-256 of the input files, and ``manifest.json`` listing the
SHA-256 of every other file. IDs and numbers only: runs enter through the hosted sharing
projection (:mod:`vectraudit.integrations.sharing`), so text parameters are fingerprints and
no question text, document text or endpoint is included. The check stored is recomputed on
those projections, and the bundle is refused if that does not give the verdict of record.

:func:`verify_bundle` recomputes every hash, names any file that changed, is missing or was
added, re-runs the comparison when the runs are in the bundle (``--bundle-runs``) or are
supplied, and checks it reproduces the recorded verdict and every rule's outcome. A bundle
whose hashes hold but whose verdict was not re-run is "hashes consistent, verdict not
reproduced", never "verified". Nothing is signed here (decision D18): the manifest's SHA-256
is printed so your CI can sign ``manifest.json`` or record the hash where a reviewer can find
it.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from vectraudit import __version__
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.exceptions import apply_exceptions
from vectraudit.evaluation.policy import (
    DEFAULT_POLICY,
    Policy,
    PolicyError,
    parse_policy,
    policy_sha256,
)

FORMAT = "vectraudit.evidence_bundle"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
MAX_ENTRIES = 64
MAX_BYTES = 512 * 1024 * 1024
_DATE = (1980, 1, 1, 0, 0, 0)


class BundleError(ValueError):
    """A bundle cannot be written faithfully, or a file is not a readable bundle."""


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, as recorded under ``inputs``."""
    return _sha(Path(path).read_bytes())


def _outcomes(result: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    return [(r.get("rule"), r.get("verdict")) for r in result.get("reasons", [])]


def _exceptions(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The audited exceptions a result applied, with who accepted them."""
    out = []
    for reason in result.get("reasons", []):
        accepted = reason.get("accepted")
        if accepted:
            out.append(
                {
                    "id": accepted["exception_id"],
                    "rule": reason["rule"],
                    "accepted_by": accepted["accepted_by"],
                    "reason": accepted["reason"],
                    "expires_at": accepted["expires_at"],
                }
            )
    return out


def recheck(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Policy,
    exceptions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The check on two runs' sharing projections, with the recorded exceptions applied."""
    from vectraudit.integrations.sharing import project_run

    result = compare_runs(project_run(baseline), project_run(candidate), policy)
    if exceptions:
        result = apply_exceptions(result, exceptions, policy)
    return result


def _policy_rules(policy: Policy) -> dict[str, Any]:
    """The policy as policy-file JSON: its canonical form without fields files cannot set."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            out = {k: strip(v) for k, v in value.items()}
            if isinstance(out.get("metrics"), dict):
                out["metrics"] = {
                    name: {k: v for k, v in rule.items() if k != "required"}
                    for name, rule in out["metrics"].items()
                }
            return out
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    rules: dict[str, Any] = strip({k: v for k, v in policy.as_dict().items() if k != "is_default"})
    return rules


def _policy_entry(policy: Policy) -> dict[str, Any]:
    digest = policy_sha256(policy)
    if policy.is_default:
        return {"default": True, "sha256": digest}
    rules = _policy_rules(policy)
    try:
        same = policy_sha256(parse_policy(rules)) == digest
    except PolicyError:
        same = False
    if not same:
        raise BundleError(
            "the policy cannot be written so that it hashes to the same rules; report this"
        )
    return {"default": False, "rules": rules, "sha256": digest}


def _policy_from(entry: Mapping[str, Any]) -> Policy:
    if entry.get("default") is True:
        return DEFAULT_POLICY
    return parse_policy(entry["rules"])


def build_bundle(
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Policy,
    out: str | Path,
    *,
    include_runs: bool = False,
    inputs: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Write the evidence bundle for a check; returns its manifest and the manifest's SHA-256.

    ``result`` is the check as reported (exceptions applied). ``inputs`` maps names such as
    ``baseline_file`` to the SHA-256 of the files given to the check.
    """
    data, written = bundle_bytes(
        result, baseline, candidate, policy, include_runs=include_runs, inputs=inputs
    )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return written


def bundle_bytes(
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Policy,
    *,
    include_runs: bool = False,
    inputs: Mapping[str, str | None] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """The evidence bundle as zip bytes, with its manifest and the manifest's SHA-256.

    Runs that are already hosted sharing projections are used as they are.
    """
    from vectraudit.integrations.sharing import project_run

    exceptions = _exceptions(result)
    base_p, cand_p = project_run(baseline), project_run(candidate)
    stored = recheck(base_p, cand_p, policy, exceptions)
    if stored["verdict"] != result["verdict"] or _outcomes(stored) != _outcomes(result):
        raise BundleError(
            "the check recomputed on the runs' sharing projections does not give the verdict of "
            f"record ({stored['verdict']} against {result['verdict']}), so a bundle of IDs and "
            "numbers cannot reproduce it"
        )
    files: dict[str, bytes] = {
        "check.json": _dump(stored),
        "policy.json": _dump(_policy_entry(policy)),
        "runs/baseline.header.json": _dump({k: v for k, v in base_p.items() if k != "queries"}),
        "runs/candidate.header.json": _dump({k: v for k, v in cand_p.items() if k != "queries"}),
        "dataset.json": _dump(
            {
                "baseline": base_p.get("dataset"),
                "candidate_sha256": (cand_p.get("dataset") or {}).get("sha256"),
            }
        ),
        "exceptions.json": _dump(exceptions),
        "versions.json": _dump(
            {
                "vectraudit": __version__,
                "python": platform.python_version(),
                "check_format": [stored.get("format"), stored.get("format_version")],
                "metric_contract": {
                    "baseline": base_p.get("metric_contract"),
                    "candidate": cand_p.get("metric_contract"),
                },
                "sharing_profile": (base_p.get("sharing") or {}).get("profile"),
                "runs_by": {
                    "baseline": ((base_p.get("provenance") or {}).get("runner") or {}).get(
                        "vectraudit"
                    ),
                    "candidate": ((cand_p.get("provenance") or {}).get("runner") or {}).get(
                        "vectraudit"
                    ),
                },
            }
        ),
        "inputs.json": _dump(dict(inputs or {})),
    }
    if include_runs:
        files["runs/baseline.json"] = _dump(base_p)
        files["runs/candidate.json"] = _dump(cand_p)
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "verdict": stored["verdict"],
        "baseline_run_id": base_p.get("run_id"),
        "candidate_run_id": cand_p.get("run_id"),
        "policy_sha256": policy_sha256(policy),
        "includes_runs": include_runs,
        "files": [
            {"path": name, "sha256": _sha(data), "bytes": len(data)}
            for name, data in sorted(files.items())
        ],
    }
    manifest_bytes = _dump(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in [*sorted(files.items()), (MANIFEST, manifest_bytes)]:
            info = zipfile.ZipInfo(name, date_time=_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue(), {"manifest": manifest, "manifest_sha256": _sha(manifest_bytes)}


def _read(path: str | Path) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise BundleError(f"{path} is not a readable zip file: {e}") from e
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ENTRIES:
            raise BundleError(f"the bundle has {len(infos)} entries; at most {MAX_ENTRIES}")
        if sum(i.file_size for i in infos) > MAX_BYTES:
            raise BundleError("the bundle expands beyond 512 MiB")
        out: dict[str, bytes] = {}
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            if name.startswith("/") or "\\" in name or ".." in parts or info.is_dir():
                raise BundleError(f"unsafe or unexpected entry {name!r}")
            if name in out:
                raise BundleError(f"the bundle repeats {name!r}")
            out[name] = archive.read(info)
    return out


def _json(data: bytes, name: str) -> Any:
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BundleError(f"{name} is not JSON: {e}") from e


def verify_bundle(
    path: str | Path,
    *,
    baseline: str | Path | None = None,
    candidate: str | Path | None = None,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    """Check a bundle; returns its status and each check as ok, failed or not checked.

    Status is ``verified`` (hashes hold and the verdict reproduces), ``hashes_consistent``
    (hashes hold, no runs to re-run the comparison) or ``failed``.
    """
    checks: list[dict[str, str]] = []

    def note(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    entries = _read(path)
    if MANIFEST not in entries:
        raise BundleError("the bundle has no manifest.json")
    manifest_bytes = entries.pop(MANIFEST)
    manifest = _json(manifest_bytes, MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BundleError("manifest.json is not an evidence bundle manifest")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"unsupported bundle format_version {manifest.get('format_version')!r}")
    listed = {f["path"]: f for f in manifest.get("files", [])}
    changed = [p for p, f in listed.items() if p in entries and _sha(entries[p]) != f["sha256"]]
    missing = sorted(set(listed) - set(entries))
    extra = sorted(set(entries) - set(listed))
    for p in sorted(changed):
        note("hash", "failed", f"{p} changed: its SHA-256 differs from the manifest")
    for p in missing:
        note("hash", "failed", f"{p} is listed in the manifest but missing")
    for p in extra:
        note("hash", "failed", f"{p} is in the bundle but not in the manifest")
    result: dict[str, Any] = {
        "format": "vectraudit.bundle_verification",
        "bundle": str(path),
        "manifest_sha256": _sha(manifest_bytes),
        "verdict": manifest.get("verdict"),
        "checks": checks,
    }
    if changed or missing or extra:
        result["status"] = "failed"
        return result
    note("hash", "ok", f"all {len(listed)} files match the manifest")
    check = _json(entries["check.json"], "check.json")
    policy_entry = _json(entries["policy.json"], "policy.json")
    inputs = _json(entries["inputs.json"], "inputs.json")
    exceptions = _json(entries["exceptions.json"], "exceptions.json")
    failed = False
    try:
        policy = _policy_from(policy_entry)
    except (PolicyError, KeyError, TypeError) as e:
        note("policy", "failed", f"policy.json is not a valid policy: {e}")
        result["status"] = "failed"
        return result
    if policy_sha256(policy) != policy_entry.get("sha256") or policy_entry.get(
        "sha256"
    ) != manifest.get("policy_sha256"):
        note("policy", "failed", "the policy's rules do not hash to the recorded policy SHA-256")
        failed = True
    else:
        note("policy", "ok", f"policy SHA-256 {policy_entry['sha256'][:12]}… recomputed")
    if check.get("verdict") != manifest.get("verdict"):
        note("verdict", "failed", "check.json and the manifest record different verdicts")
        failed = True
    runs: dict[str, Any] = {}
    for side, raw in (("baseline", baseline), ("candidate", candidate)):
        if raw is not None:
            from vectraudit.evaluation.runfile import load_run

            want = inputs.get(f"{side}_file")
            got = file_sha256(raw)
            if want is None:
                note(f"{side} file", "not checked", "the bundle recorded no file hash for it")
            elif got != want:
                note(f"{side} file", "failed", f"{raw} is not the file the check was run on")
                failed = True
                continue
            else:
                note(f"{side} file", "ok", "SHA-256 matches the file the check was run on")
            runs[side] = load_run(raw)
        elif f"runs/{side}.json" in entries:
            runs[side] = _json(entries[f"runs/{side}.json"], f"runs/{side}.json")
    if dataset is not None:
        from vectraudit.evaluation.dataset import load_dataset

        recorded = (_json(entries["dataset.json"], "dataset.json").get("baseline") or {}).get(
            "sha256"
        )
        if load_dataset(dataset).sha256 == recorded:
            note("dataset", "ok", "the dataset file has the version the runs were measured on")
        else:
            note("dataset", "failed", f"{dataset} is not the dataset version the runs used")
            failed = True
    reproduced = False
    if len(runs) == 2 and not failed:
        try:
            again = recheck(runs["baseline"], runs["candidate"], policy, exceptions)
        except ValueError as e:
            note("verdict", "failed", f"the runs could not be compared again: {e}")
            failed = True
        else:
            same_ids = (again["baseline"]["run_id"], again["candidate"]["run_id"]) == (
                manifest.get("baseline_run_id"),
                manifest.get("candidate_run_id"),
            )
            if not same_ids:
                note("verdict", "failed", "the runs are not the ones the check compared")
                failed = True
            elif again["verdict"] == check["verdict"] and _outcomes(again) == _outcomes(check):
                note(
                    "verdict",
                    "ok",
                    f"re-running the comparison reproduces {again['verdict']} and every "
                    f"rule's outcome ({len(_outcomes(again))} reasons)",
                )
                reproduced = True
            else:
                versions = _json(entries["versions.json"], "versions.json")
                detail = (
                    f"re-running the comparison gives {again['verdict']}, not "
                    f"{check['verdict']}, or different rule outcomes"
                )
                if versions.get("vectraudit") != __version__:
                    detail += (
                        f" (bundle made by vectraudit {versions.get('vectraudit')}, verified "
                        f"with {__version__})"
                    )
                note("verdict", "failed", detail)
                failed = True
    elif not failed:
        note(
            "verdict",
            "not checked",
            "no runs in the bundle or supplied: the verdict was not re-run",
        )
    result["status"] = "failed" if failed else "verified" if reproduced else "hashes_consistent"
    return result
