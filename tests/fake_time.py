"""Fake clocks scoped to one module, so other threads keep real time.

Patching ``time.sleep`` or ``time.monotonic`` on the ``time`` module itself (as
``monkeypatch.setattr("pkg.module.time.sleep", ...)`` does) changes them for the whole
process: any other thread that sleeps meanwhile, such as a server left from another test,
records its sleeps on the fake clock and makes assertions about it fail at random under
load. Replacing the module's own ``time`` reference keeps the fake inside that module.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class _ModuleTime:
    def __init__(self, **overrides: Callable[..., Any]) -> None:
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(time, name)


def patch_time(monkeypatch: Any, module: str, **overrides: Callable[..., Any]) -> None:
    """Give ``module`` a ``time`` whose named functions are replaced, and nothing else."""
    monkeypatch.setattr(f"{module}.time", _ModuleTime(**overrides))
