"""``vectraudit init``: scaffold the files a team needs for its first release check."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from vectraudit import __version__

BACKENDS = ("faiss", "qdrant", "pgvector", "pinecone", "weaviate")
INSTALL_ACTION = ".github/actions/install-vectraudit"


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
}


def _template(name: str) -> str:
    return files("vectraudit.templates").joinpath(name).read_text(encoding="utf-8")


def render_workflow(backend: str) -> str:
    """The GitHub Actions workflow for one backend."""
    settings = _SETTINGS[backend]
    env_block = ""
    if settings.secret:
        env_block = (
            f"        env:\n          {settings.secret}: ${{{{ secrets.{settings.secret} }}}}\n"
        )
    return (
        _template("retrieval-check.yml.tmpl")
        .replace("__EXTRAS__", settings.extras)
        .replace("__VERSION__", __version__.split("+")[0])
        .replace("__ENV_BLOCK__", env_block)
    )


def installer_files() -> dict[str, str]:
    """Shared, self-contained install action for init and workbench release exports."""
    return {
        f"{INSTALL_ACTION}/action.yml": _template("install-action.yml"),
        f"{INSTALL_ACTION}/install.py": _template("install_wheel.py"),
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
    ci_secret = f" and `{secret}` (a read-only credential)" if secret else ""
    return {
        **installer_files(),
        "retrieval/connection.json": _template(f"connection-{backend}.json"),
        "retrieval/policy.json": _template("policy.json"),
        "retrieval/queries.csv": _template("queries.csv"),
        "retrieval/README.md": _template("README.md").format(
            backend=backend, secret_note=secret_note, ci_secret=ci_secret
        ),
        ".github/workflows/retrieval-check.yml": render_workflow(backend),
    }


def init_project(root: Path, backend: str, *, force: bool = False) -> tuple[list[Path], list[Path]]:
    """Write the scaffold under ``root``. Returns (written, skipped because they exist)."""
    written, skipped = [], []
    for rel, content in scaffold_files(backend).items():
        path = root / rel
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written, skipped
