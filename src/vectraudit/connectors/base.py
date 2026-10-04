"""The contract every vector backend connector implements.

A connector answers one question portably: the ranked document ids for a query vector.
Everything else a backend can offer (exact search, filters, configuration, tuning
parameters, query plans) is declared in :class:`Capabilities`; a missing capability is
reported as unavailable, never as an empty successful result. Failures are
:class:`ConnectorError` with a code, so a broken connection is never mistaken for a
quality regression.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
FilterValue = str | int | float | bool
ParamValue = int | float


class ErrorCode(str, Enum):
    """Why a connector call failed. Only ``QUERY`` and ``UNSUPPORTED`` are per-query."""

    CONFIG = "config"
    DEPENDENCY = "dependency"
    CONNECTION = "connection"
    AUTH = "auth"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    QUERY = "query"


class ConnectorError(Exception):
    """A typed connector failure with an optional hint for the user."""

    def __init__(self, code: ErrorCode, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, str | None]:
        return {"code": self.code.value, "message": self.message, "hint": self.hint}


class Distance(str, Enum):
    """Distance the backend ranks by. Scores are never compared across distances."""

    L2 = "l2"
    IP = "ip"
    COSINE = "cosine"


@dataclass(frozen=True)
class Capabilities:
    """What a connector can do, each declared independently."""

    search: bool = True
    filters: bool = False
    exact_reference: bool = False
    """Can run an exact (brute-force) search over the same corpus for ANN recall."""
    exact_reference_note: str | None = None
    """Caveat about the exact reference, e.g. vectors reconstructed from compressed codes."""
    config_inspection: bool = False
    query_plans: bool = False
    vector_export: bool = False
    tuning_params: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tuning_params"] = list(self.tuning_params)
        return d


@dataclass(frozen=True)
class BackendInfo:
    """A non-secret description of the backend a run measured."""

    backend: str
    distance: Distance
    dim: int
    count: int | None
    target: dict[str, Any] = field(default_factory=dict)
    """Which collection / table / file, without credentials."""
    backend_version: str | None = None
    client_version: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    """Index and search configuration as the backend reports it."""
    snapshot: str | None = None
    """An immutable content identifier, or ``None`` when the backend has none."""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["distance"] = self.distance.value
        return d


@dataclass(frozen=True)
class SearchRequest:
    """One top-k query. ``params`` are backend tuning parameters (e.g. ``nprobe``)."""

    vector: FloatArray
    k: int
    filter: Mapping[str, FilterValue] = field(default_factory=dict)
    params: Mapping[str, ParamValue] = field(default_factory=dict)
    exact: bool = False


class Connector(abc.ABC):
    """Base class for backend connectors. Use as a context manager to close connections."""

    name: str = "connector"

    @abc.abstractmethod
    def describe(self) -> BackendInfo:
        """Describe the backend, corpus and configuration (no secrets)."""

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """What this connector supports for the configured target."""

    @abc.abstractmethod
    def _search(self, request: SearchRequest) -> list[str]:
        """Backend-specific search; called after :meth:`check_request` passed."""

    def search(self, request: SearchRequest) -> list[str]:
        """Ranked document ids (best first, at most ``k``, no duplicates)."""
        self.check_request(request)
        ids = self._search(request)
        seen: set[str] = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out[: request.k]

    def check_request(self, request: SearchRequest) -> None:
        """Reject requests this connector can't honour, before touching the backend."""
        caps = self.capabilities()
        if request.k < 1:
            raise ConnectorError(ErrorCode.QUERY, "k must be >= 1.")
        if request.filter and not caps.filters:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"{self.name} does not support filters.",
                "Remove the filter from these queries or evaluate them on another backend.",
            )
        if request.exact and not caps.exact_reference:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"{self.name} has no exact reference search."
            )
        unknown = set(request.params) - set(caps.tuning_params)
        if unknown:
            allowed = ", ".join(caps.tuning_params) or "none"
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"Unknown search parameter(s) for {self.name}: {', '.join(sorted(unknown))}.",
                f"Supported: {allowed}.",
            )

    def evidence(
        self, request: SearchRequest, found: list[str], exact: list[str] | None
    ) -> dict[str, Any] | None:
        """Backend-specific facts about one search (e.g. a query plan), or ``None``.

        ``found`` is what :meth:`search` returned and ``exact`` the exact reference, when
        measured. Implementations must not change results and should stay small: the
        output is stored per query. Every dict carries ``kind`` and ``source`` (how the
        facts were obtained, e.g. ``"replayed"`` or ``"planner estimate"``).
        """
        return None

    def close(self) -> None:  # noqa: B027 - optional hook; most connectors hold no resources
        """Release connections. Safe to call twice."""

    def __enter__(self) -> Connector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
