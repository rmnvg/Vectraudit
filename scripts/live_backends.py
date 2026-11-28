"""Start, check and stop the vector backends the live connector tests run against.

One table of server versions serves CI (``.github/workflows/backends.yml``) and local runs.
Each track starts PostgreSQL with pgvector, Pinecone Local, Weaviate, Elasticsearch,
OpenSearch and Chroma in Docker, waits until each answers, and prints the
``VECTRAUDIT_TEST_*`` variables that turn the live tests on::

    python scripts/live_backends.py up --track newest --port-offset 10000 > live.env
    set -a && . ./live.env && set +a
    uv run pytest tests/test_connector_*.py tests/test_acceptance.py tests/test_tune.py
    python scripts/live_backends.py down

``oldest`` and ``newest`` bound the server versions the connectors are tested against;
``latest`` follows floating tags (nightly only) so an upstream release that breaks a
connector shows up without blocking pull requests. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "va-test-"

TRACKS: dict[str, dict[str, str]] = {
    "oldest": {
        "postgres": "pgvector/pgvector:0.8.7-pg17",
        "pinecone": "ghcr.io/pinecone-io/pinecone-local:v1.0.0.rc0",
        "weaviate": "cr.weaviate.io/semitechnologies/weaviate:1.32.4",
        "elasticsearch": "docker.elastic.co/elasticsearch/elasticsearch:8.15.3",
        "opensearch": "opensearchproject/opensearch:2.17.1",
        "chroma": "chromadb/chroma:1.5.9",
    },
    "newest": {
        "postgres": "pgvector/pgvector:0.8.7-pg18",
        "pinecone": "ghcr.io/pinecone-io/pinecone-local:v1.0.0.rc0",
        "weaviate": "cr.weaviate.io/semitechnologies/weaviate:1.41.0",
        "elasticsearch": "docker.elastic.co/elasticsearch/elasticsearch:9.5.7",
        "opensearch": "opensearchproject/opensearch:3.10.1",
        "chroma": "chromadb/chroma:1.5.9",
    },
    # Floating tags where the project publishes one; Elastic publishes no `latest`, and
    # pgvector's newest PostgreSQL major is picked by hand.
    "latest": {
        "postgres": "pgvector/pgvector:pg18",
        "pinecone": "ghcr.io/pinecone-io/pinecone-local:latest",
        "weaviate": "cr.weaviate.io/semitechnologies/weaviate:latest",
        "elasticsearch": "docker.elastic.co/elasticsearch/elasticsearch:9.5.7",
        "opensearch": "opensearchproject/opensearch:latest",
        "chroma": "chromadb/chroma:latest",
    },
}

# Search engines and Weaviate stop accepting writes on a nearly full disk; fail before that.
MIN_FREE_GIB = 10


@dataclass(frozen=True)
class Server:
    """How to run one backend and how to tell it is ready."""

    name: str
    ports: tuple[tuple[int, int], ...]  # (host port before the offset, container port)
    env: dict[str, str] = field(default_factory=dict)
    ready_path: str = ""  # HTTP path on the first port; empty means use `command`
    command: tuple[str, ...] = ()  # docker exec probe when there is no HTTP check
    headers: dict[str, str] = field(default_factory=dict)
    any_status: bool = False  # ready once it answers at all (Pinecone Local's 401)


HEALTH = "_cluster/health?wait_for_status=yellow&timeout=1s"
SEARCH_ENGINE = {
    "discovery.type": "single-node",
    # A nearly full runner made the disk watermarks block index creation; the headroom
    # check below fails first instead.
    "cluster.routing.allocation.disk.threshold_enabled": "false",
}
SERVERS = (
    Server(
        "postgres",
        ((5432, 5432),),
        {"POSTGRES_PASSWORD": "va-test", "POSTGRES_DB": "va"},
        command=("pg_isready", "-U", "postgres", "-h", "127.0.0.1"),
    ),
    Server("pinecone", (), {}, "/indexes", headers={"Api-Key": "pclocal"}, any_status=True),
    Server(
        "weaviate",
        ((8080, 8080), (50051, 50051)),
        {
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
            "DEFAULT_VECTORIZER_MODULE": "none",
            "CLUSTER_HOSTNAME": "node1",
            "DISK_USE_WARNING_PERCENTAGE": "100",
            "DISK_USE_READONLY_PERCENTAGE": "100",
        },
        "/v1/.well-known/ready",
    ),
    Server(
        "elasticsearch",
        ((9200, 9200),),
        {**SEARCH_ENGINE, "xpack.security.enabled": "false", "ES_JAVA_OPTS": "-Xms512m -Xmx512m"},
        f"/{HEALTH}",
    ),
    Server(
        "opensearch",
        ((9201, 9200),),
        {
            **SEARCH_ENGINE,
            "DISABLE_SECURITY_PLUGIN": "true",
            "DISABLE_INSTALL_DEMO_CONFIG": "true",
            "OPENSEARCH_JAVA_OPTS": "-Xms512m -Xmx512m",
        },
        f"/{HEALTH}",
    ),
    Server("chroma", ((8000, 8000),), {}, "/api/v2/heartbeat"),
)
PINECONE_PORT = 5080  # Pinecone Local serves each index on the next ports up to +10


def test_env(offset: int) -> dict[str, str]:
    """The ``VECTRAUDIT_TEST_*`` variables for servers started with ``offset``."""
    return {
        "VECTRAUDIT_TEST_PG_DSN": f"postgresql://postgres:va-test@localhost:{5432 + offset}/va",
        "VECTRAUDIT_TEST_PINECONE_URL": f"http://localhost:{PINECONE_PORT + offset}",
        "VECTRAUDIT_TEST_WEAVIATE_URL": f"http://localhost:{8080 + offset}",
        "VECTRAUDIT_TEST_WEAVIATE_GRPC": f"http://localhost:{50051 + offset}",
        "VECTRAUDIT_TEST_ELASTICSEARCH_URL": f"http://localhost:{9200 + offset}",
        "VECTRAUDIT_TEST_OPENSEARCH_URL": f"http://localhost:{9201 + offset}",
        "VECTRAUDIT_TEST_CHROMA_URL": f"http://localhost:{8000 + offset}",
    }


def free_gib(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 2**30


def check_headroom(min_gib: float, free: Callable[[], float] = free_gib) -> None:
    """Exit with a clear message when the disk is too full for the search engines."""
    available = free()
    if available < min_gib:
        raise SystemExit(
            f"Only {available:.1f} GiB free on / (the live backends need {min_gib:g} GiB): "
            "Elasticsearch, OpenSearch and Weaviate stop accepting writes on a nearly full "
            "disk and the tests then fail with timeouts or HTTP 403. Free some space first "
            "(on a GitHub runner, remove unused toolchains such as /usr/share/dotnet)."
        )


def docker_run_args(server: Server, image: str, offset: int) -> list[str]:
    args = ["docker", "run", "-d", "--rm", "--name", PREFIX + server.name]
    env = dict(server.env)
    if server.name == "pinecone":
        port = PINECONE_PORT + offset
        env |= {"PORT": str(port), "PINECONE_HOST": "localhost"}
        args += ["-p", f"127.0.0.1:{port}-{port + 10}:{port}-{port + 10}"]
    for host, container in server.ports:
        args += ["-p", f"127.0.0.1:{host + offset}:{container}"]
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    return [*args, image]


def _host_port(server: Server, offset: int) -> int:
    return PINECONE_PORT + offset if server.name == "pinecone" else server.ports[0][0] + offset


def is_ready(server: Server, offset: int) -> bool:
    if server.command:
        probe = ["docker", "exec", PREFIX + server.name, *server.command]
        return subprocess.run(probe, capture_output=True, check=False).returncode == 0
    url = f"http://localhost:{_host_port(server, offset)}{server.ready_path}"
    request = urllib.request.Request(url, headers=server.headers)
    try:
        with urllib.request.urlopen(request, timeout=3):
            return True
    except urllib.error.HTTPError:
        return server.any_status
    except (urllib.error.URLError, OSError):
        return False


def up(track: str, offset: int, timeout_s: float, min_gib: float) -> dict[str, str]:
    check_headroom(min_gib)
    images = TRACKS[track]
    for server in SERVERS:
        print(f"Starting {server.name} ({images[server.name]})", file=sys.stderr)
        subprocess.run(
            docker_run_args(server, images[server.name], offset),
            check=True,
            stdout=subprocess.DEVNULL,
        )
    waiting = list(SERVERS)
    deadline = time.monotonic() + timeout_s
    while waiting and time.monotonic() < deadline:
        waiting = [s for s in waiting if not is_ready(s, offset)]
        if waiting:
            time.sleep(2)
    if waiting:
        names = ", ".join(s.name for s in waiting)
        for server in waiting:
            subprocess.run(["docker", "logs", "--tail", "40", PREFIX + server.name], check=False)
        raise SystemExit(f"Not ready after {timeout_s:g} s: {names}.")
    print(f"All backends ready ({track}).", file=sys.stderr)
    return test_env(offset)


def down() -> None:
    names = [PREFIX + s.name for s in SERVERS]
    subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("up", help="start one track and print its VECTRAUDIT_TEST_* lines")
    start.add_argument("--track", choices=sorted(TRACKS), default="newest")
    start.add_argument(
        "--port-offset",
        type=int,
        default=0,
        help="add this to every host port (to avoid ports already in use)",
    )
    start.add_argument("--timeout", type=float, default=240.0)
    start.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    sub.add_parser("down", help="remove the containers `up` started")
    headroom = sub.add_parser("headroom", help="only check the free disk space")
    headroom.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    args = parser.parse_args(argv)

    if args.command == "down":
        down()
    elif args.command == "headroom":
        check_headroom(args.min_free_gib)
        print(f"{free_gib():.1f} GiB free.", file=sys.stderr)
    else:
        env = up(args.track, args.port_offset, args.timeout, args.min_free_gib)
        lines = "".join(f"{key}={value}\n" for key, value in env.items())
        if github_env := os.environ.get("GITHUB_ENV"):
            with Path(github_env).open("a", encoding="utf-8") as f:
                f.write(lines)
        sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
