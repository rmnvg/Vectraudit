"""Rehearse a Vectraudit Cloud deployment end to end, then a backup and a restore.

For a scratch deployment only (a fresh `./start.sh`, a staging host, or CI): the restore
replaces the whole database with the backup taken a moment earlier. From a checkout, with
the stack running::

    uv run python deploy/cloud/rehearse.py --scratch

It creates a workspace, a project and a runner token, records a small FAISS release run
and uploads it with `vectraudit cloud push`, takes a backup with ``backup.sh``, deletes the
project (so the run is gone), restores the backup with ``restore.sh`` while the API is
stopped, and checks that the run is back. It prints how long the restore took.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8800"


def step(message: str) -> None:
    print(f"==> {message}", flush=True)


def compose(*args: str, capture: bool = False) -> str:
    done = subprocess.run(
        ["docker", "compose", *args], cwd=HERE, check=True, text=True, capture_output=capture
    )
    return done.stdout if capture else ""


def api(method: str, path: str, token: str, body: dict[str, object] | None = None) -> object:
    request = urllib.request.Request(
        URL + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def run_exists(workspace: str, project: str, run_id: str, token: str) -> bool:
    try:
        api("GET", f"/api/v1/workspaces/{workspace}/projects/{project}/runs/{run_id}", token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    return True


def wait_ready(timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(URL + "/readyz", timeout=3) as response:
                return response.read().decode()
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise SystemExit("The API did not become ready; see: docker compose logs api")


def make_run(folder: Path) -> Path:
    """Record a small, real release run against an exact FAISS index."""
    import faiss
    import numpy as np

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((500, 16)).astype(np.float32)
    faiss.write_index(_flat(faiss, vectors), str(folder / "index.faiss"))
    (folder / "connection.json").write_text(
        json.dumps({"backend": "faiss", "name": "rehearsal", "index": "index.faiss"})
    )
    with (folder / "dataset.jsonl").open("w") as f:
        for i in range(40):
            q = vectors[i] + rng.normal(0, 0.05, 16).astype(np.float32)
            f.write(
                json.dumps({"query_id": f"q{i}", "vector": q.tolist(), "relevant": {str(i): 1}})
            )
            f.write("\n")
    out = folder / "rehearsal.run.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vectraudit",
            "record",
            "--connection",
            str(folder / "connection.json"),
            "--dataset",
            str(folder / "dataset.jsonl"),
            "--out",
            str(out),
            "--label",
            "rehearsal",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return out


def _flat(faiss: object, vectors: object) -> object:
    index = faiss.IndexFlatL2(16)  # type: ignore[attr-defined]
    index.add(vectors)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Confirm this is a scratch deployment: the restore replaces the whole database.",
    )
    args = parser.parse_args()
    if not args.scratch:
        raise SystemExit(
            "Refusing to run without --scratch: the restore step replaces the whole database "
            "with a backup taken during the rehearsal. Use a fresh or staging deployment."
        )

    step(f"API ready: {wait_ready()}")
    workspace, project = f"rehearsal-{int(time.time())}", "release-checks"
    output = compose(
        "run",
        "--rm",
        "-T",
        "api",
        "create-workspace",
        workspace,
        "--name",
        "Rehearsal",
        "--owner-email",
        f"{workspace}@example.com",
        capture=True,
    )
    owner = output.strip().splitlines()[-1].strip()
    step(f"Created workspace {workspace}")
    api(
        "POST",
        f"/api/v1/workspaces/{workspace}/projects",
        owner,
        {"slug": project, "name": "Release checks"},
    )
    runner = api(
        "POST",
        f"/api/v1/workspaces/{workspace}/projects/{project}/runner-tokens",
        owner,
        {"name": "rehearsal runner"},
    )
    assert isinstance(runner, dict)
    step("Created a project and a runner token")

    with tempfile.TemporaryDirectory() as tmp:
        run_file = make_run(Path(tmp))
        run_id = json.loads(run_file.read_text())["run_id"]
        env = {
            **os.environ,
            "VECTRAUDIT_CLOUD_URL": URL,
            "VECTRAUDIT_CLOUD_TOKEN": str(runner["token"]),
            "VECTRAUDIT_CLOUD_PROJECT": f"{workspace}/{project}",
        }
        subprocess.run(
            [sys.executable, "-m", "vectraudit", "cloud", "push", str(run_file)],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
        )
    if not run_exists(workspace, project, run_id, owner):
        raise SystemExit("The uploaded run is not visible through the API.")
    step(f"Recorded and uploaded run {run_id}")

    backup = subprocess.run(
        ["./backup.sh"], cwd=HERE, check=True, text=True, capture_output=True
    ).stdout.strip()
    step(f"Backed up to {backup}")

    api("DELETE", f"/api/v1/workspaces/{workspace}/projects/{project}", owner, {"confirm": project})
    if run_exists(workspace, project, run_id, owner):
        raise SystemExit("Deleting the project did not remove the run.")
    step("Deleted the project: the run is gone")

    started = time.monotonic()
    compose("stop", "api", "webhooks")
    subprocess.run(["./restore.sh", backup], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    compose("start", "api", "webhooks")
    wait_ready()
    took = time.monotonic() - started
    if not run_exists(workspace, project, run_id, owner):
        raise SystemExit("The run is missing after the restore.")
    step(f"Restored {backup} in {took:.0f} s: the project and its run are back")
    print("Rehearsal passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
