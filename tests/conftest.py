"""Shared fixtures: small synthetic indexes of every type, built once per test session."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

SMALL = {"n": 2_000, "d": 32, "n_centers": 20, "n_queries": 50, "nlist": 16, "hnsw_m": 16}


def _load_make_synthetic() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_synthetic"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def make_synthetic() -> ModuleType:
    """The ``examples/make_synthetic.py`` module."""
    return _load_make_synthetic()


@pytest.fixture(scope="session")
def synthetic(
    tmp_path_factory: pytest.TempPathFactory, make_synthetic: ModuleType
) -> dict[str, Path]:
    """Paths to small synthetic indexes and side files (see ``make_synthetic.build_dataset``)."""
    out = tmp_path_factory.mktemp("synthetic")
    return make_synthetic.build_dataset(out, **SMALL)


@pytest.fixture(scope="session")
def binary_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A binary index file (unsupported by vectraudit)."""
    path = tmp_path_factory.mktemp("binary") / "binary_flat.index"
    faiss.write_index_binary(faiss.IndexBinaryFlat(64), str(path))
    return path


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Never touch the real ~/.cache during tests.

    Session-scoped so it is active before any module-scoped fixture creates a Session.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VECTRAUDIT_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        yield


@pytest.fixture(scope="session")
def uniform_ivf(tmp_path_factory) -> tuple[Path, Path, Any]:
    """IVF-Flat over unclustered data, where a low nprobe really misses neighbours."""
    from vectraudit.evaluation.dataset import parse_dataset

    out = tmp_path_factory.mktemp("uniform")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2_000, 16)).astype(np.float32)
    queries = rng.normal(size=(20, 16)).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(16), 16, 16)
    index.train(x)
    index.add(x)
    faiss.write_index(index, str(out / "ivf.index"))
    np.save(out / "x.npy", x)
    rows = [{"query_id": f"q{i}", "vector": q.tolist()} for i, q in enumerate(queries)]
    return out / "ivf.index", out / "x.npy", parse_dataset("\n".join(map(json.dumps, rows)))
