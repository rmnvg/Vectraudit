"""Packages the notices script treats as not bundled really are not in the built web UI."""

from pathlib import Path

import pytest

ASSETS = Path(__file__).parents[1] / "src" / "vectraudit" / "static" / "assets"


def test_unbundled_packages_are_in_no_emitted_chunk() -> None:
    chunks = list(ASSETS.glob("*.js")) if ASSETS.is_dir() else []
    if not chunks:
        pytest.skip("the frontend is not built (npm run build)")
    for chunk in chunks:
        assert "duckdb" not in chunk.read_text(errors="ignore").lower(), chunk.name
