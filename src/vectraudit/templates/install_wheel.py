"""Install a checksum-pinned Vectraudit release wheel; runs before Vectraudit is installed.

Standard library only. The wheel is downloaded from a public GitHub release (Vectraudit's
own by default; pass another repository for a fork or mirror) and verified against the
pinned version and SHA-256 before pip sees it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

DEFAULT_REPOSITORY = "rmnvg/Vectraudit"
MAX_WHEEL_BYTES = 256 * 1024 * 1024
_CREDENTIALS = {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"}
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


def validate_repository(repository: str) -> str:
    """Return the GitHub repository (owner/name) whose releases hold the wheel."""
    repository = repository or DEFAULT_REPOSITORY
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Set the release repository as owner/name.")
    return repository.removesuffix(".git")


def release_url(repository: str, version: str, filename: str) -> str:
    """The public download URL of one release asset."""
    return f"https://github.com/{repository}/releases/download/v{version}/{filename}"


def download(url: str, target: Path) -> None:
    """Stream a release asset to ``target``, refusing anything larger than a wheel can be."""
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
            total = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if total > MAX_WHEEL_BYTES:
                    raise ValueError("Downloaded file is too large to be a Vectraudit wheel.")
                out.write(block)
    except OSError as exc:
        raise ValueError(
            "Wheel download failed; check the repository, version and network access."
        ) from exc


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
            download(release_url(validate_repository(repository), version, filename), target)
        else:
            shutil.copyfile(wheel, target)
        verify_wheel(target, version, sha256)
        requirement = str(target) + (f"[{extras}]" if extras else "")
        pip_env = {key: value for key, value in os.environ.items() if key not in _CREDENTIALS}
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
    parser.add_argument(
        "--repository", default="", help=f"Release repository (default {DEFAULT_REPOSITORY})."
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--extras", default="")
    parser.add_argument(
        "--wheel", type=Path, help="Use a pre-downloaded wheel instead of downloading it."
    )
    args = parser.parse_args()
    try:
        install(**vars(args))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
