"""FAISS connector: an index file, plus optional raw vectors for the exact reference."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    Distance,
    ErrorCode,
    FilterValue,
    SearchRequest,
)
from vectraudit.core._faiss import FaissNotInstalledError, import_faiss
from vectraudit.core.filtering import AllowedSet, allowed_from_metadata, filter_fingerprint
from vectraudit.core.ivf import Assignments, assignments
from vectraudit.core.loader import IndexLoadError, load_index
from vectraudit.core.metadata import Metadata, MetadataError, load_metadata
from vectraudit.core.search import GroundTruth, MissReason, probed_miss_reason, search
from vectraudit.core.types import IndexKind, LoadedIndex, Metric
from vectraudit.core.vectors import VectorMismatchError, VectorSource, from_arrays, reconstruct_all


class FaissConnector(Connector):
    """Search a FAISS index file.

    The exact reference is brute force over the raw ``vectors`` (``.npy``, row ``i`` is id
    ``i`` unless ``ids`` is given). A flat index is its own exact reference. With
    ``reconstructed_reference`` the vectors are decoded from the index instead, which
    includes any compression error, so the reference is marked as weaker.

    With ``metadata`` (``.jsonl``/``.csv``/``.parquet`` keyed by id) the connector supports
    equality filters: they are applied inside the FAISS search, the exact reference applies
    the same filter, and ``count`` is the number of ids whose metadata satisfies it.
    """

    name = "FAISS"

    def __init__(
        self,
        index: str | Path,
        *,
        vectors: str | Path | None = None,
        ids: str | Path | None = None,
        reconstructed_reference: bool = False,
        metadata: str | Path | Metadata | None = None,
    ) -> None:
        try:
            self.li: LoadedIndex = load_index(index)
        except FaissNotInstalledError as e:
            raise ConnectorError(ErrorCode.DEPENDENCY, str(e), e.hint) from e
        except (FileNotFoundError, IndexLoadError) as e:
            raise ConnectorError(ErrorCode.CONFIG, str(e)) from e
        if not self.li.is_supported:
            raise ConnectorError(ErrorCode.UNSUPPORTED, f"{self.li.unsupported_reason}")
        if ids is not None and vectors is None:
            raise ConnectorError(ErrorCode.CONFIG, "ids needs vectors.")
        self.path = Path(index)
        self._sha256: str | None = None
        self._assign: Assignments | None = None
        self._truth: GroundTruth | None = None
        self._reconstruct = False
        self._note: str | None = None
        self._metadata: Metadata | None = None
        self._allowed: dict[str, AllowedSet] = {}
        if metadata is not None:
            try:
                self._metadata = load_metadata(metadata)
            except MetadataError as e:
                raise ConnectorError(ErrorCode.CONFIG, str(e), e.hint) from e
        if vectors is not None:
            self._truth = GroundTruth(self._load_vectors(Path(vectors), ids), self.li.metric)
        elif reconstructed_reference and self.li.kind is not IndexKind.FLAT:
            # Decoded lazily on the first exact search: it can use as much memory as raw vectors.
            self._reconstruct = True
            self._note = "Vectors reconstructed from the index: includes compression error."

    def _load_vectors(self, path: Path, ids_path: str | Path | None) -> VectorSource:
        try:
            vectors = np.load(path, allow_pickle=False)
            ids = None if ids_path is None else np.load(Path(ids_path), allow_pickle=False)
            return from_arrays(self.li, vectors, ids)
        except FileNotFoundError as e:
            raise ConnectorError(ErrorCode.CONFIG, f"Vectors file not found: {e.filename}") from e
        except VectorMismatchError as e:
            raise ConnectorError(ErrorCode.CONFIG, str(e), e.hint) from e
        except ValueError as e:
            raise ConnectorError(ErrorCode.CONFIG, f"Cannot read {path}: {e}") from e

    def _exact(self) -> GroundTruth | None:
        if self._truth is None and self._reconstruct:
            try:
                self._truth = GroundTruth(reconstruct_all(self.li), self.li.metric)
            except (RuntimeError, ValueError) as e:
                raise ConnectorError(
                    ErrorCode.UNSUPPORTED, f"Cannot reconstruct vectors from this index: {e}"
                ) from e
        return self._truth

    def capabilities(self) -> Capabilities:
        kind = self.li.kind
        tuning: tuple[str, ...] = ()
        if kind.is_ivf:
            tuning = ("nprobe",)
        elif kind.is_hnsw:
            tuning = ("ef_search",)
        return Capabilities(
            filters=self._metadata is not None,
            filter_count=self._metadata is not None,
            exact_reference=self._truth is not None or self._reconstruct or kind is IndexKind.FLAT,
            exact_reference_note=self._note,
            config_inspection=True,
            vector_export=self._truth is not None,
            tuning_params=tuning,
        )

    def describe(self) -> BackendInfo:
        faiss = import_faiss()
        li = self.li
        return BackendInfo(
            backend="faiss",
            distance=Distance.L2 if li.metric is Metric.L2 else Distance.IP,
            dim=li.d,
            count=li.ntotal,
            target={"index": self.path.name},
            backend_version=str(faiss.__version__),
            client_version=str(faiss.__version__),
            config={
                "kind": li.kind.value,
                "class_chain": li.class_chain,
                "params": li.params.as_dict(),
                "id_map": li.has_id_map,
                "refine_k_factor": li.refine_k_factor,
                "transforms": [t.name for t in li.transforms],
            },
            snapshot=f"sha256:{self._file_sha256()}",
        )

    def _file_sha256(self) -> str:
        if self._sha256 is None:
            h = hashlib.sha256()
            with self.path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._sha256 = h.hexdigest()
        return self._sha256

    def _corpus_ids(self) -> Any:
        if self.li.ids is not None:
            return self.li.ids
        if self._truth is not None:
            return self._truth.source.ids
        return np.arange(self.li.ntotal, dtype=np.int64)

    def _allowed_for(self, filter: Mapping[str, FilterValue]) -> AllowedSet | None:
        """The ids that satisfy ``filter`` (cached per filter); ``None`` for no filter."""
        if not filter:
            return None
        assert self._metadata is not None  # check_request refused filters without metadata
        key = filter_fingerprint(filter)
        if key not in self._allowed:
            unknown = sorted(set(filter) - set(self._metadata.columns))
            if unknown:
                raise ConnectorError(
                    ErrorCode.QUERY,
                    f"Filter field(s) {', '.join(unknown)} are not columns of the metadata.",
                    f"The metadata has: {', '.join(self._metadata.columns) or 'no columns'}.",
                )
            self._allowed[key] = allowed_from_metadata(self._metadata, filter, self._corpus_ids())
        return self._allowed[key]

    def _count(self, filter: Mapping[str, FilterValue]) -> int:
        allowed = self._allowed_for(filter)
        return self.li.ntotal if allowed is None else len(allowed)

    def _search(self, request: SearchRequest) -> list[str]:
        allowed = self._allowed_for(request.filter)
        try:
            # A flat index has no exact helper: its own search is already exact.
            truth = self._exact() if request.exact else None
            if truth is not None:
                found = truth.search(request.vector, request.k, allowed=allowed)
                return [str(i) for i in found.valid_ids]
            result = search(
                self.li,
                request.vector,
                request.k,
                nprobe=_int_param(request.params, "nprobe"),
                ef_search=_int_param(request.params, "ef_search"),
                allowed=allowed,
            )
        except ValueError as e:
            raise ConnectorError(ErrorCode.QUERY, str(e)) from e
        return [str(i) for i in result.valid_ids]

    def evidence(
        self, request: SearchRequest, found: list[str], exact: list[str] | None
    ) -> dict[str, Any] | None:
        """IVF: replay the probe order and say why each missed exact neighbour was missed."""
        li = self.li
        if li.ivf is None or exact is None:
            return None
        nprobe = _int_param(request.params, "nprobe") or int(li.ivf.nprobe)
        nlist = int(li.ivf.nlist)
        q = li.to_core_space(np.asarray(request.vector, dtype=np.float32).reshape(1, -1))
        _, order = li.ivf.quantizer.search(q, nlist)
        probe_rank = np.full(nlist, nlist, dtype=np.int64)
        valid = order[0] >= 0
        probe_rank[order[0][valid]] = np.arange(nlist)[valid]
        exact_ids = np.array([int(i) for i in exact], dtype=np.int64)
        lists = self._assignments().lookup(exact_ids)
        found_set = set(found)
        probed_miss = probed_miss_reason(li).value
        missed: dict[str, str] = {}
        ranks: dict[str, int] = {}
        worst = 0
        for nid, list_no in zip(exact, lists, strict=True):
            rank = int(probe_rank[list_no]) if list_no >= 0 else nlist
            worst = max(worst, rank)
            if nid not in found_set:
                ranks[nid] = rank
                missed[nid] = MissReason.CELL_NOT_PROBED.value if rank >= nprobe else probed_miss
        counts: dict[str, int] = {}
        for reason in missed.values():
            counts[reason] = counts.get(reason, 0) + 1
        return {
            "kind": "faiss_ivf",
            "source": "replayed: coarse-quantizer probe order and list assignments",
            "nprobe": nprobe,
            "nlist": nlist,
            "min_nprobe_for_all": min(nlist, worst + 1),
            "missed": missed,
            "missed_by_reason": counts,
            # Position of each missed neighbour's list in the probe order (0 = probed first).
            "probe_rank": ranks,
        }

    def _assignments(self) -> Assignments:
        if self._assign is None:
            self._assign = assignments(self.li)
        return self._assign


def _int_param(params: Any, key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConnectorError(ErrorCode.CONFIG, f"{key} must be a positive integer.")
    return value
