"""Install a checksum-pinned private wheel; runs before Vectraudit is installed.

Standard library only. The download credential is used by gh, never passed to pip.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

SOURCE_REPOSITORY = "rmnvg/vectraudit"
SUPPORTED_EXTRAS = {
    "faiss-cpu",
    "qdrant",
    "pgvector",
    "weaviate",
    "cloud",
    "text",
    "umap",
    "parquet",
    "label-llm",
    "all",
}


def validate_pins(version: str, sha256: str, extras: str) -> str:
    """Reject mutable versions, missing hashes and unexpected pip argument syntax."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?", version):
        raise ValueError("Use an exact version without the v prefix (for example 0.1.0).")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("Set the reviewed release wheel's 64-character SHA-256 pin.")
    if extras and any(extra not in SUPPORTED_EXTRAS for extra in extras.split(",")):
        raise ValueError("Unsupported installation extras.")
    return f"vectraudit-{version}-py3-none-any.whl"


def validate_repository(repository: str) -> None:
    """Require an explicit distribution repository, never the source repository."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Set the distribution repository as owner/name.")
    if repository.casefold().removesuffix(".git") == SOURCE_REPOSITORY:
        raise ValueError("Customer downloads must not use the source repository.")


def verify_wheel(wheel: Path, version: str, sha256: str) -> None:
    """Verify bytes and package identity before allowing any installation code to run."""
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if not hmac.compare_digest(digest.hexdigest(), sha256.lower()):
        raise ValueError("Wheel checksum mismatch; nothing was installed.")
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1 or archive.getinfo(metadata[0]).file_size > 1024 * 1024:
                raise ValueError("Wheel must contain one bounded package metadata record.")
            package = BytesParser().parsebytes(archive.read(metadata[0]))
            if package.get("Name") != "vectraudit" or package.get("Version") != version:
                raise ValueError("Wheel package/version does not match the reviewed pin.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid wheel archive.") from exc


def install(
    *, repository: str, version: str, sha256: str, extras: str = "", wheel: Path | None = None
) -> None:
    """Download (or copy an offline wheel), verify, then install the exact artifact."""
    filename = validate_pins(version, sha256, extras)
    # Even a supplied offline wheel is copied into an isolated directory before verification.
    with tempfile.TemporaryDirectory(prefix="vectraudit-install-") as directory:
        target = Path(directory) / filename
        if wheel is None:
            validate_repository(repository)
            if not os.environ.get("GH_TOKEN"):
                raise ValueError("Set GH_TOKEN to a customer-specific distribution credential.")
            # Do not permit an inherited enterprise host to redirect the download credential.
            download_env = {**os.environ, "GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1"}
            try:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        f"v{version}",
                        "--repo",
                        repository,
                        "--pattern",
                        filename,
                        "--dir",
                        directory,
                    ],
                    env=download_env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise ValueError(
                    "Wheel download failed; check the release, distribution access "
                    "and gh installation."
                ) from exc
        else:
            import shutil

            shutil.copyfile(wheel, target)
        verify_wheel(target, version, sha256)
        requirement = str(target) + (f"[{extras}]" if extras else "")
        pip_env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "VECTRAUDIT_DOWNLOAD_TOKEN"}
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                requirement,
            ],
            env=pip_env,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="")
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--extras", default="")
    parser.add_argument(
        "--wheel", type=Path, help="Use a pre-downloaded wheel (no gh/token needed)."
    )
    args = parser.parse_args()
    try:
        install(**vars(args))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        # The download exception is deliberately static; never echo gh's credential-bearing output.
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
