"""Portable release configurations: one benchmark, pipeline and policy for every surface.

A release configuration binds the exact frozen query split, the adopted pipeline, k and
repeats, and a reviewed policy, each identified by content hash. The workbench creates one
from a passing comparison and exports it; ``vectraudit release`` validates, runs and checks
it locally and in CI. Every surface decides with :func:`compare_runs` under the same parsed
policy, so the same run artifacts get the same verdict everywhere.

In CI the *approved* configuration is the target branch's: its benchmark, k and policy
judge both sides, so a pull request cannot loosen its own gate. Each side executes the
pipeline defined by its own checkout, importing its own code.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from vectraudit import __version__
from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.dataset import Dataset, DatasetError, load_dataset
from vectraudit.evaluation.policy import Policy, PolicyError, load_policy, policy_sha256
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import HTTPPipeline, Pipeline

RELEASE_FORMAT = "vectraudit.release"
RELEASE_DIR = "retrieval/release"
WORKFLOW_PATH = ".github/workflows/retrieval-release-check.yml"
PIPELINE_URL_ENV = "RETRIEVAL_PIPELINE_URL"
_PRECEDENCE = [Verdict.ERROR, Verdict.NOT_COMPARABLE, Verdict.FAIL]


class ReleaseError(ValueError):
    """A release configuration or one of its files is missing, invalid or changed."""


class ReleasePipeline(Contract):
    """The adopted pipeline. Credentials and per-deployment URLs come from the environment."""

    name: str = Field(min_length=1, max_length=128)
    kind: Literal["python", "http", "hybrid"]
    target: str | None = Field(default=None, max_length=2048)
    """Python ``module:function``, or an HTTP URL when ``target_env`` is not used."""
    target_env: str | None = Field(default=None, max_length=128)
    """HTTP only: read the URL from this variable, so each side names its own deployment."""
    token_env: str | None = Field(default=None, max_length=128)
    timeout: float = Field(default=30, gt=0, le=300)
    corpus: str | None = Field(default=None, max_length=1024)
    """Hybrid only: JSONL corpus path, relative to the release file."""
    python_path: list[str] = Field(default_factory=list, max_length=16)
    """Python only: import roots relative to the release file, searched before site-packages."""
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configured(self) -> ReleasePipeline:
        if self.kind == "python":
            module, sep, name = str(self.target).partition(":")
            if not (self.target and sep and module and name.isidentifier()):
                raise ValueError("a python pipeline needs target module:function")
        if self.kind == "http" and not (self.target or self.target_env):
            raise ValueError("an http pipeline needs target or target_env")
        if self.kind == "hybrid" and not self.corpus:
            raise ValueError("a hybrid pipeline needs corpus")
        return self


class BenchmarkRef(Contract):
    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    """Dataset content version (``vectraudit dataset validate``), not a file-bytes hash."""
    split: Literal["explore", "heldout"]
    n_queries: int = Field(ge=1)


class PolicyRef(Contract):
    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    """Hash of the effective rules (:func:`policy_sha256`)."""
    name: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)


class ReleaseFile(Contract):
    format: Literal["vectraudit.release"] = "vectraudit.release"
    format_version: Literal[1] = 1
    configuration: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    benchmark: BenchmarkRef
    policy: PolicyRef
    pipeline: ReleasePipeline
    k: int = Field(ge=1, le=10000)
    repeats: int = Field(default=3, ge=1, le=100)
    concurrency: int = Field(default=1, ge=1, le=32)
    """Queries in flight on each side. The approved file's value runs both sides, so latency
    stays comparable and a pull request cannot change its own measurement protocol."""
    max_rps: float | None = Field(default=None, gt=0, le=1000)
    """Optional cap on pipeline calls per second for each side, retries included."""
    corpus_version: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    vectraudit_version: str = __version__


_EXECUTION_DEFAULTS = {"concurrency": 1, "max_rps": None}


def release_json(release: ReleaseFile) -> str:
    """Serialize a release file. Execution settings appear only when changed from their
    defaults, so default exports stay byte-identical and readable by earlier versions."""
    data = json.loads(release.model_dump_json())
    for key, default in _EXECUTION_DEFAULTS.items():
        if data.get(key) == default:
            del data[key]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def load_release(path: str | Path) -> ReleaseFile:
    """Parse a release file; the error names the field, never secret values."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ReleaseError(f"Release configuration not found: {p}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ReleaseError(f"Cannot read {p}: {e}") from e
    try:
        return ReleaseFile.model_validate_json(text)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(x) for x in err['loc']) or 'file'}: {err['msg']}" for err in e.errors()
        )
        raise ReleaseError(f"{p} is not a valid release configuration: {problems}") from e


def _within(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def release_benchmark(release: ReleaseFile, root: Path) -> Dataset:
    """The frozen query split; fails when the file is missing or no longer the approved one."""
    path = _within(root, release.benchmark.path)
    try:
        ds = load_dataset(path)
    except DatasetError as e:
        raise ReleaseError(f"Benchmark {release.benchmark.path}: {e}") from e
    if ds.sha256 != release.benchmark.sha256:
        raise ReleaseError(
            f"Benchmark {release.benchmark.path} changed (version {ds.sha256[:12]}, approved "
            f"{release.benchmark.sha256[:12]}). Review it and run `vectraudit release update`."
        )
    other = sorted({q.split for q in ds.queries} - {release.benchmark.split})
    if other:
        raise ReleaseError(
            f"Benchmark mixes the {release.benchmark.split} split with {', '.join(other)} "
            "queries; exploratory and held-out queries must stay separate."
        )
    return ds


def release_policy(release: ReleaseFile, root: Path) -> Policy:
    """The reviewed policy; fails when its rules no longer match the recorded hash."""
    try:
        policy = load_policy(_within(root, release.policy.path))
    except PolicyError as e:
        raise ReleaseError(f"Policy {release.policy.path}: {e}") from e
    actual = policy_sha256(policy)
    if actual != release.policy.sha256:
        raise ReleaseError(
            f"Policy {release.policy.path} changed (rules {actual[:12]}, approved "
            f"{release.policy.sha256[:12]}). Review it and run `vectraudit release update`."
        )
    return policy


def read_corpus(path: Path) -> list[dict[str, Any]]:
    """Validated JSONL documents, in file order."""
    from vectraudit.pipelines.guided import Document

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise ReleaseError(f"Corpus not found: {path}") from e
    try:
        docs = [
            Document.model_validate_json(line).model_dump(exclude_none=True)
            for line in lines
            if line.strip()
        ]
    except ValidationError as e:
        raise ReleaseError(f"{path} is not a valid JSONL corpus") from e
    if not docs or len({d["doc_id"] for d in docs}) != len(docs):
        raise ReleaseError(f"{path} needs at least one document and unique doc_id values")
    return docs


def corpus_version(docs: list[dict[str, Any]]) -> str:
    """The same corpus version the guided workbench records for its built-in pipeline."""
    return hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()


def pipeline_sha256(pipeline: ReleasePipeline) -> str:
    return hashlib.sha256(json.dumps(pipeline.model_dump(), sort_keys=True).encode()).hexdigest()


def _import_roots(pipeline: ReleasePipeline, root: Path) -> list[Path]:
    return [root.resolve(), *(_within(root, p) for p in pipeline.python_path)]


def _import_from(module: str, roots: list[Path]) -> Any:
    """Import ``module`` from ``roots`` first, discarding a copy cached from another checkout."""
    top = module.split(".")[0]
    for name in [n for n in sys.modules if n == top or n.startswith(top + ".")]:
        location = getattr(sys.modules[name], "__file__", None)
        if location and not any(Path(location).resolve().is_relative_to(r) for r in roots):
            del sys.modules[name]
    for folder in reversed(roots):
        # Kept on sys.path: adapters may import lazily while they run. One side per process.
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.path.insert(0, str(folder))
    importlib.invalidate_caches()
    return importlib.import_module(module)


def open_release_pipeline(
    pipeline: ReleasePipeline, root: Path, env: Mapping[str, str] | None = None
) -> tuple[Pipeline, dict[str, Any]]:
    """Open the pipeline this checkout defines; returns it with its run configuration."""
    env = os.environ if env is None else env
    config: dict[str, Any] = {
        "name": pipeline.name,
        "adapter": pipeline.kind,
        "sha256": pipeline_sha256(pipeline),
    }
    if pipeline.kind == "hybrid":
        docs = read_corpus(_within(root, str(pipeline.corpus)))
        config["corpus_sha256"] = corpus_version(docs)
        return hybrid_pipeline(docs), config
    if pipeline.kind == "http":
        url = pipeline.target
        if pipeline.target_env:
            url = env.get(pipeline.target_env) or None
            if not url:
                raise ReleaseError(
                    f"Set {pipeline.target_env} to the URL of the deployment this side measures."
                )
        token = None
        if pipeline.token_env:
            token = env.get(pipeline.token_env)
            if not token:
                raise ReleaseError(f"Required environment variable {pipeline.token_env} is unset")
        parts = urlsplit(str(url))
        config["destination"] = f"{parts.scheme}://{parts.hostname or ''}"
        return HTTPPipeline(str(url), token=token, timeout=pipeline.timeout), config
    module, _, name = str(pipeline.target).partition(":")
    roots = _import_roots(pipeline, root)
    try:
        imported = _import_from(module, roots)
    except ImportError as e:
        raise ReleaseError(
            f"Cannot import {module} from this checkout: {e}. Add its folder to python_path "
            "or install its dependencies."
        ) from e
    fn = getattr(imported, name, None)
    if not callable(fn):
        raise ReleaseError(f"{pipeline.target} is not a callable pipeline")
    # Fingerprint the code this side actually executed, so a check shows when the two
    # sides ran the same adapter source.
    location = getattr(imported, "__file__", None)
    if location:
        config["code"] = {
            "module": module,
            "sha256": hashlib.sha256(Path(location).read_bytes()).hexdigest(),
        }
    pipe: Pipeline = fn
    return pipe, config


def run_release(
    release_path: Path,
    *,
    approved_path: Path | None = None,
    label: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run this checkout's pipeline on the approved benchmark (default: its own)."""
    release = load_release(release_path)
    approved = load_release(approved_path) if approved_path else release
    approved_root = (approved_path or release_path).parent
    ds = release_benchmark(approved, approved_root)
    pipeline, config = open_release_pipeline(release.pipeline, release_path.parent, env)
    config["release"] = {"configuration": release.configuration, "version": release.version}
    options = PipelineOptions(
        k=approved.k,
        repeats=approved.repeats,
        warmup=0,
        label=label or f"{release.name} v{release.version}",
        params=release.pipeline.params,
        concurrency=approved.concurrency,
        max_rps=approved.max_rps,
    )
    run = execute_pipeline(
        ds, pipeline, options, config=config, corpus_version=config.get("corpus_sha256")
    )
    run["release"] = {
        "configuration": release.configuration,
        "version": release.version,
        "approved_configuration": approved.configuration,
        "approved_version": approved.version,
        "benchmark_sha256": approved.benchmark.sha256,
        "split": approved.benchmark.split,
        "policy_sha256": approved.policy.sha256,
    }
    return run


def _flag(result: dict[str, Any], verdict: Verdict, message: str) -> None:
    result["reasons"].append({"verdict": verdict.value, "message": message})
    verdicts = {Verdict(r["verdict"]) for r in result["reasons"]}
    result["verdict"] = next((v for v in _PRECEDENCE if v in verdicts), Verdict.PASS).value


def check_release(
    approved_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    proposed_path: Path | None = None,
) -> dict[str, Any]:
    """Decide under the approved policy; report what a proposed configuration would change."""
    approved = load_release(approved_path)
    policy = release_policy(approved, approved_path.parent)
    result = compare_runs(baseline, candidate, policy)
    for side, run in (("baseline", baseline), ("candidate", candidate)):
        if run["dataset"]["sha256"] != approved.benchmark.sha256:
            _flag(
                result,
                Verdict.NOT_COMPARABLE,
                f"The {side} run used dataset {run['dataset']['sha256'][:12]}, not the approved "
                f"benchmark {approved.benchmark.sha256[:12]}.",
            )
    changes: list[str] = []
    if proposed_path is not None:
        proposed = load_release(proposed_path)
        if proposed.policy.sha256 != approved.policy.sha256:
            changes.append(
                f"This change proposes policy {proposed.policy.name} v{proposed.policy.version}; "
                f"the approved {approved.policy.name} v{approved.policy.version} decided this "
                "check. The proposed policy applies after review and merge."
            )
        if proposed.benchmark.sha256 != approved.benchmark.sha256:
            changes.append(
                "This change proposes a new benchmark version; the approved benchmark measured "
                "both sides. The new version applies after review and merge."
            )
        if (proposed.k, proposed.repeats) != (approved.k, approved.repeats):
            changes.append("This change proposes a different k or repeat count.")
        if (proposed.concurrency, proposed.max_rps) != (approved.concurrency, approved.max_rps):
            changes.append(
                "This change proposes different concurrency or rate limits; the approved "
                "settings measured both sides and apply until it is merged."
            )
    result["warnings"].extend(changes)
    result["release"] = {
        "configuration": approved.configuration,
        "name": approved.name,
        "version": approved.version,
        "policy": {
            "name": approved.policy.name,
            "version": approved.policy.version,
            "sha256": approved.policy.sha256,
        },
        "benchmark_sha256": approved.benchmark.sha256,
        "split": approved.benchmark.split,
        "proposed_changes": changes,
    }
    return result


def validate_release(path: Path, env: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    """Offline readiness: errors block a run; warnings name prerequisites to confirm."""
    env = os.environ if env is None else env
    issues: list[dict[str, str]] = []

    def add(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    try:
        release = load_release(path)
    except ReleaseError as e:
        return [{"level": "error", "message": str(e)}]
    root = path.parent
    for step in (release_benchmark, release_policy):
        try:
            step(release, root)
        except ReleaseError as e:
            add("error", str(e))
    p = release.pipeline
    paths = [release.benchmark.path, release.policy.path, *p.python_path]
    paths += [p.corpus] if p.corpus else []
    for relative in paths:
        if Path(relative).is_absolute():
            add("warning", f"{relative} is an absolute path; it will not exist on a CI runner.")
    if p.kind == "hybrid" and p.corpus:
        corpus = _within(root, p.corpus)
        if not corpus.is_file():
            add("error", f"Corpus {p.corpus} is missing. Place your corpus JSONL there.")
        elif release.corpus_version:
            try:
                if corpus_version(read_corpus(corpus)) != release.corpus_version:
                    add("warning", "The corpus differs from the one the comparison measured.")
            except ReleaseError as e:
                add("error", str(e))
    if p.kind == "http":
        if p.target_env and not env.get(p.target_env):
            add("warning", f"Set {p.target_env} to the deployment URL before running.")
        if p.target and not p.target_env and _local_host(p.target):
            add("warning", f"{p.target} is a local address; a CI runner cannot reach it.")
    if p.token_env and not env.get(p.token_env):
        add("warning", f"Secret {p.token_env} is not set in this environment.")
    if p.kind == "python":
        module = str(p.target).partition(":")[0]
        module_path = Path(*module.split("."))
        roots = _import_roots(p, root)
        found = any(
            (r / module_path).with_suffix(".py").is_file()
            or (r / module_path / "__init__.py").is_file()
            for r in roots
        )
        for folder in p.python_path:
            if not _within(root, folder).is_dir():
                add("error", f"python_path entry {folder} does not exist in this checkout.")
        if not found:
            add(
                "warning",
                f"Module {module} is not under the release folder or python_path; it must be "
                "installed, and then both sides run the same installed code.",
            )
    if release.concurrency > 1 and p.kind == "python":
        add(
            "warning",
            f"concurrency {release.concurrency} calls {p.target} from several threads; "
            "confirm the adapter is thread-safe.",
        )
    if release.vectraudit_version != __version__:
        add(
            "warning",
            f"Created with vectraudit {release.vectraudit_version}; running {__version__}.",
        )
    return issues


def update_release(path: Path) -> tuple[ReleaseFile, list[str]]:
    """Re-approve reviewed edits to the benchmark or policy as a new configuration version."""
    release = load_release(path)
    root = path.parent
    try:
        ds = load_dataset(_within(root, release.benchmark.path))
        policy = load_policy(_within(root, release.policy.path))
    except (DatasetError, PolicyError) as e:
        raise ReleaseError(str(e)) from e
    splits = {q.split for q in ds.queries}
    if len(splits) != 1:
        raise ReleaseError("A release benchmark must contain exactly one split.")
    changes = []
    benchmark = release.benchmark
    if ds.sha256 != benchmark.sha256 or splits != {benchmark.split}:
        changes.append(f"benchmark {benchmark.sha256[:12]} -> {ds.sha256[:12]}")
        benchmark = benchmark.model_copy(
            update={"sha256": ds.sha256, "split": splits.pop(), "n_queries": len(ds)}
        )
    policy_ref = release.policy
    digest = policy_sha256(policy)
    if digest != policy_ref.sha256:
        changes.append(f"policy {policy_ref.sha256[:12]} -> {digest[:12]}")
        policy_ref = policy_ref.model_copy(
            update={"sha256": digest, "version": policy_ref.version + 1}
        )
    if not changes:
        return release, []
    updated = release.model_copy(
        update={
            "benchmark": benchmark,
            "policy": policy_ref,
            "version": release.version + 1,
            # Earlier comparison evidence does not cover the new benchmark or policy.
            "evidence": {"superseded": release.evidence, "note": "Re-approved locally."},
            "vectraudit_version": __version__,
        }
    )
    path.write_text(release_json(updated), encoding="utf-8")
    return updated, changes


def _local_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in {"localhost", "::1", "0.0.0.0"} or host.startswith("127.")


# --- export -----------------------------------------------------------------------------


def render_release_files(
    release: ReleaseFile,
    *,
    benchmark: str,
    policy: Mapping[str, Any],
    corpus: str | None,
    prerequisites: list[str],
    evaluated_destination: str | None = None,
) -> dict[str, str]:
    """Relative path -> content for the exported release check. Nothing is written here."""
    files = {
        f"{RELEASE_DIR}/release.json": release_json(release),
        f"{RELEASE_DIR}/{release.benchmark.path}": benchmark,
        f"{RELEASE_DIR}/{release.policy.path}": json.dumps(policy, indent=2) + "\n",
        f"{RELEASE_DIR}/README.md": _readme(release, prerequisites, evaluated_destination),
        WORKFLOW_PATH: render_release_workflow(release.pipeline),
    }
    if corpus is not None and release.pipeline.corpus:
        files[f"{RELEASE_DIR}/{release.pipeline.corpus}"] = corpus
    return files


def _secret_env(pipeline: ReleasePipeline, side: str) -> str:
    lines = []
    if pipeline.target_env:
        source = "RETRIEVAL_BASELINE_URL" if side == "baseline" else "RETRIEVAL_CANDIDATE_URL"
        lines.append(f"          {pipeline.target_env}: ${{{{ secrets.{source} }}}}")
    if pipeline.token_env:
        lines.append(f"          {pipeline.token_env}: ${{{{ secrets.{pipeline.token_env} }}}}")
    return "        env:\n" + "\n".join(lines) + "\n" if lines else ""


def render_release_workflow(pipeline: ReleasePipeline) -> str:
    """The GitHub Actions workflow that runs this release check on every pull request."""
    from importlib.resources import files as resources

    template = resources("vectraudit.templates").joinpath("release-check.yml.tmpl")
    install = ""
    if pipeline.kind == "python":
        install = (
            "      - name: Install pipeline dependencies (customer-owned)\n"
            "        run: |\n"
            "          # Install what your adapter imports, e.g. pip install -r requirements.txt.\n"
            "          # Do not install the pipeline package itself: python_path makes each\n"
            "          # side import its own checkout.\n"
            "          true\n\n"
        )
    return (
        template.read_text(encoding="utf-8")
        .replace("__VERSION__", __version__.split("+")[0])
        .replace("__RELEASE__", f"{RELEASE_DIR}/release.json")
        .replace("__INSTALL_PIPELINE__", install)
        .replace("__BASELINE_ENV__", _secret_env(pipeline, "baseline"))
        .replace("__CANDIDATE_ENV__", _secret_env(pipeline, "candidate"))
    )


def _readme(
    release: ReleaseFile, prerequisites: list[str], evaluated_destination: str | None
) -> str:
    p = release.pipeline
    split = (
        "exploratory"
        if release.benchmark.split == "explore"
        else "held-out (running it on every change gradually tunes against it; prefer an "
        "exploratory release check and keep held-out runs for final validation)"
    )
    secrets = []
    if p.target_env:
        secrets.append(
            "`RETRIEVAL_BASELINE_URL`: the deployment the target branch represents "
            "(usually production)"
        )
        secrets.append(
            "`RETRIEVAL_CANDIDATE_URL`: the deployment built from the pull request "
            "(for example a preview environment)"
        )
    if p.token_env:
        secrets.append(f"`{p.token_env}`: a read-only credential for the pipeline")
    secrets.append("`VECTRAUDIT_DOWNLOAD_TOKEN`: read access to the Vectraudit release")
    lines = [
        f"# Retrieval release check: {release.name} (configuration v{release.version})",
        "",
        "Generated by the Vectraudit workbench from a passing comparison. Review these files,",
        "then commit them; nothing here was pushed or published.",
        "",
        "## What it checks",
        "",
        f"- Benchmark `{release.benchmark.path}`: {release.benchmark.n_queries} {split} "
        f"queries, version `{release.benchmark.sha256[:12]}`.",
        f"- Policy `{release.policy.path}`: {release.policy.name} v{release.policy.version}, "
        f"rules `{release.policy.sha256[:12]}`.",
        f"- Pipeline: {p.name} ({p.kind}), k={release.k}, {release.repeats} repeats"
        + (
            f", {release.concurrency} queries in flight"
            + (f" at most {release.max_rps:g} calls/s" if release.max_rps else "")
            if release.concurrency > 1 or release.max_rps
            else ""
        )
        + ".",
        "",
        "On each pull request the workflow runs the target branch's pipeline and the pull",
        "request's pipeline on the target branch's benchmark and judges them with the target",
        "branch's policy. A pull request that edits the policy or benchmark is checked under",
        "the approved versions and its proposal is reported; it takes effect after merge.",
        "",
        "## Prerequisites",
        "",
        *(f"- {item}" for item in prerequisites or ["None beyond the files in this folder."]),
        "",
        "## Repository secrets",
        "",
        *(f"- {item}" for item in secrets),
        "",
    ]
    if evaluated_destination:
        lines += [
            f"The workbench comparison measured the endpoint at `{evaluated_destination}`.",
            "",
        ]
    lines += [
        "## Run it locally",
        "",
        "```sh",
        f"vectraudit release validate {RELEASE_DIR}/release.json",
        f"vectraudit release run {RELEASE_DIR}/release.json --out runs/candidate.run.json",
        "# With a baseline run recorded from the approved checkout:",
        f"vectraudit release check {RELEASE_DIR}/release.json "
        "runs/baseline.run.json runs/candidate.run.json",
        "```",
        "",
        "Exit codes: 0 pass, 1 error, 3 fail, 4 not comparable. After a reviewed edit to",
        f"`{release.policy.path}` or `{release.benchmark.path}`, run",
        f"`vectraudit release update {RELEASE_DIR}/release.json` to record a new configuration",
        "version.",
        "",
    ]
    return "\n".join(lines)
