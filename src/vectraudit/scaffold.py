"""``vectraudit init``: scaffold the files a team needs for its first release check."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from vectraudit import __version__

BACKENDS = (
    "faiss", "qdrant", "pgvector", "pinecone", "weaviate", "elasticsearch", "opensearch", "chroma",
)  # fmt: skip
INSTALL_ACTION = ".github/actions/install-vectraudit"
CHECK_ACTION = ".github/actions/retrieval-check"


@dataclass(frozen=True)
class _Backend:
    extras: str
    secret: str | None


_SETTINGS = {
    "faiss": _Backend("faiss-cpu", None),
    "qdrant": _Backend("qdrant", "QDRANT_API_KEY"),
    "pgvector": _Backend("pgvector", "RETRIEVAL_PG_DSN"),
    "pinecone": _Backend("", "PINECONE_API_KEY"),  # REST API: no extra package needed
    "weaviate": _Backend("weaviate", "WEAVIATE_API_KEY"),
    # REST APIs: no extra package needed.
    "elasticsearch": _Backend("", "ELASTICSEARCH_API_KEY"),
    "opensearch": _Backend("", "OPENSEARCH_BASIC_AUTH"),
    "chroma": _Backend("", "CHROMA_TOKEN"),
}


def _template(name: str) -> str:
    return files("vectraudit.templates").joinpath(name).read_text(encoding="utf-8")


def side_env(baseline: dict[str, str], candidate: dict[str, str]) -> str:
    """The check action's per-side secret inputs (``NAME: secret``), indented for a step."""
    lines = []
    if baseline or candidate:
        lines.append("          # Each side's secrets reach only its record command.")
    for side, values in (("baseline", baseline), ("candidate", candidate)):
        if values:
            lines.append(f"          {side}-env: |")
            lines.extend(
                f"            {name}=${{{{ secrets.{secret} }}}}" for name, secret in values.items()
            )
    return "".join(f"{line}\n" for line in lines)


def render_workflow(backend: str) -> str:
    """The GitHub Actions workflow for one backend."""
    settings = _SETTINGS[backend]
    secrets = {settings.secret: settings.secret} if settings.secret else {}
    return (
        _template("retrieval-check.yml.tmpl")
        .replace("__EXTRAS__", settings.extras)
        .replace("__VERSION__", __version__.split("+")[0])
        .replace("__SIDE_ENV__\n", side_env(secrets, secrets))
    )


def ci_files() -> dict[str, str]:
    """The reusable check action and the pinned install action it calls.

    Shared by ``init`` and workbench release exports; both workflows call the check action.
    """
    return {
        f"{INSTALL_ACTION}/action.yml": _template("install-action.yml"),
        f"{INSTALL_ACTION}/install.py": _template("install_wheel.py"),
        f"{CHECK_ACTION}/action.yml": _template("check-action.yml"),
        f"{CHECK_ACTION}/check.py": _template("retrieval_check.py"),
    }


def scaffold_files(backend: str) -> dict[str, str]:
    """Relative path -> content for every file ``init`` writes."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    secret = _SETTINGS[backend].secret
    secret_note = (
        f" Credentials come from the environment variable `{secret}`, never the file."
        if secret
        else " Paths are relative to this folder."
    )
    ci_secret = f" Add repository secret `{secret}` (a read-only credential)." if secret else ""
    return {
        **ci_files(),
        "retrieval/connection.json": _template(f"connection-{backend}.json"),
        "retrieval/policy.json": _template("policy.json"),
        "retrieval/queries.csv": _template("queries.csv"),
        "retrieval/README.md": _template("README.md").format(
            backend=backend, secret_note=secret_note, ci_secret=ci_secret
        ),
        ".github/workflows/retrieval-check.yml": render_workflow(backend),
    }


def init_project(
    root: Path, backend: str, *, force: bool = False, ci_only: bool = False
) -> tuple[list[Path], list[Path]]:
    """Write the scaffold under ``root``. Returns (written, skipped because they exist).

    ``ci_only`` writes just the workflow and actions under ``.github/``, for upgrading an
    existing project without touching its connection, policy or queries.
    """
    written, skipped = [], []
    for rel, content in scaffold_files(backend).items():
        if ci_only and not rel.startswith(".github/"):
            continue
        path = root / rel
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written, skipped
