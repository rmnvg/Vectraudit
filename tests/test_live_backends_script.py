"""scripts/live_backends.py: one table of server versions, used the same way in CI and locally."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "live_backends_script", ROOT / "scripts" / "live_backends.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


tb = load()


def test_every_track_names_an_image_for_every_server():
    names = {server.name for server in tb.SERVERS}
    for track, images in tb.TRACKS.items():
        assert set(images) == names, track
        for image in images.values():
            assert ":" in image.rsplit("/", 1)[-1], f"{track}: {image} has no tag"
    for track in ("oldest", "newest"):
        floating = [i for i in tb.TRACKS[track].values() if i.endswith((":latest", ":pg18"))]
        assert not floating, f"{track} must pin exact versions: {floating}"


def test_port_offset_moves_every_host_port_and_the_variables_with_it():
    env = tb.test_env(10000)
    assert env["VECTRAUDIT_TEST_PG_DSN"].endswith("@localhost:15432/va")
    assert env["VECTRAUDIT_TEST_WEAVIATE_GRPC"] == "http://localhost:60051"
    assert env["VECTRAUDIT_TEST_OPENSEARCH_URL"] == "http://localhost:19201"
    servers = {s.name: s for s in tb.SERVERS}
    pinecone = tb.docker_run_args(servers["pinecone"], "img:1", 10000)
    # Pinecone Local serves each index on its own port, so the range maps one to one.
    assert "127.0.0.1:15080-15090:15080-15090" in pinecone
    assert "PORT=15080" in pinecone
    assert pinecone[-1] == "img:1"
    opensearch = tb.docker_run_args(servers["opensearch"], "img:2", 10000)
    assert "127.0.0.1:19201:9200" in opensearch
    assert "cluster.routing.allocation.disk.threshold_enabled=false" in opensearch


def test_headroom_fails_early_with_a_reason_and_passes_with_room():
    with pytest.raises(SystemExit, match=r"Only 3\.5 GiB free .* need 10 GiB.*read-only|stop"):
        tb.check_headroom(10, free=lambda: 3.5)
    tb.check_headroom(10, free=lambda: 50.0)


def test_ci_postgres_and_workflow_tracks_follow_the_script():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert f"image: {tb.TRACKS['oldest']['postgres']}" in ci
    backends = (ROOT / ".github/workflows/backends.yml").read_text()
    matrix_line = backends.split("track:")[1].split("\n")[0]
    named = set(re.findall(r'"(\w+)"', matrix_line))
    assert named == set(tb.TRACKS)


def test_readme_names_the_tested_versions():
    readme = (ROOT / "README.md").read_text()
    for track in ("oldest", "newest"):
        for image in tb.TRACKS[track].values():
            version = image.rsplit(":", 1)[1]
            version = version.split("-pg")[0]  # pgvector tags are 0.8.7-pg17
            assert version in readme, f"README does not name {image}"
