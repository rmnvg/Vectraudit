"""Stage only the tested wheel and its checksum for a customer-visible release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def prepare(source: Path, destination: Path) -> Path:
    """Use a fresh directory so stale source archives can never join the release assets."""
    wheels = list(source.glob("vectraudit-*-py3-none-any.whl"))
    if len(wheels) != 1 or len(list(source.glob("*.whl"))) != 1:
        raise ValueError("Expected exactly one Vectraudit release wheel.")
    if destination.exists():
        raise ValueError("Distribution destination must be a new directory.")
    destination.mkdir(parents=True)
    wheel = destination / wheels[0].name
    shutil.copyfile(wheels[0], wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (destination / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dist"))
    parser.add_argument("--out", type=Path, default=Path("customer-dist"))
    args = parser.parse_args()
    prepare(args.source, args.out)
