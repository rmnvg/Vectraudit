"""Read FAISS vector stores saved by LangChain and LlamaIndex.

Both frameworks save a FAISS index next to a document store, and their document stores
are what makes an index readable: each vector's text, source and metadata. This module
finds the index file in a saved folder and turns the document store into ``Metadata``
keyed by the index ids, so the debugger shows the chunks behind every neighbour.

- **LangChain** ``FAISS.save_local(folder, index_name)`` writes ``<name>.faiss`` and
  ``<name>.pkl``: a pickle of ``(InMemoryDocstore, {row: docstore id})``. Unpickling can
  run arbitrary code, so the pickle is read by an unpickler that only rebuilds LangChain's
  docstore and ``Document`` as plain data and refuses every other class. Nothing from
  LangChain is imported.
- **LlamaIndex** ``StorageContext.persist(folder)`` with a ``FaissVectorStore`` writes the
  FAISS index as ``default__vector_store.json`` (a binary file despite its name), and the
  JSON ``docstore.json`` and ``index_store.json``, which maps FAISS rows to node ids.
"""

from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.core.metadata import Metadata, MetadataError

LANGCHAIN = "LangChain"
LLAMAINDEX = "LlamaIndex"
LLAMAINDEX_INDEX = "default__vector_store.json"
MAX_STORE_BYTES = 2 * 1024**3  # a document store this large is not a saved folder we read


class SavedStoreError(MetadataError):
    """A folder looks like a saved vector store but cannot be read."""


@dataclass(frozen=True)
class SavedStore:
    """A framework's saved FAISS folder: where its index is and what each vector holds."""

    framework: str
    folder: Path
    index_path: Path
    metadata: Metadata
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """One line for the terminal, such as ``LangChain FAISS folder (6 documents)``."""
        return f"{self.framework} FAISS folder ({len(self.metadata):,} documents)"


def is_saved_store(path: Path) -> bool:
    """Whether ``path`` is a folder holding a LangChain or LlamaIndex FAISS save."""
    return path.is_dir() and (bool(_langchain_pairs(path)) or (path / LLAMAINDEX_INDEX).is_file())


def open_saved_store(folder: str | Path, index_name: str | None = None) -> SavedStore:
    """Find the FAISS index in ``folder`` and read its document store.

    ``index_name`` picks one LangChain save when the folder holds several (the
    ``index_name`` given to ``save_local``; default ``index``).
    """
    path = Path(folder)
    if not path.is_dir():
        raise SavedStoreError(f"Not a folder: {path}", "Pass the folder the framework saved.")
    if (path / LLAMAINDEX_INDEX).is_file():
        return _llamaindex(path)
    pairs = _langchain_pairs(path)
    if pairs:
        return _langchain(path, pairs, index_name)
    raise SavedStoreError(
        f"{path} is not a saved LangChain or LlamaIndex FAISS folder.",
        "Expected index.faiss with index.pkl (LangChain save_local) or "
        f"{LLAMAINDEX_INDEX} with docstore.json (LlamaIndex persist), or pass the index file.",
    )


# LangChain ----------------------------------------------------------------------------


def _langchain_pairs(folder: Path) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.faiss") if p.with_suffix(".pkl").is_file())


class _Plain:
    """Stand-in for an allowed LangChain class: keeps the pickled state, runs no code."""

    state: dict[str, Any]

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2:  # (__dict__, slots) form
            state = {**(state[0] or {}), **(state[1] or {})}
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("Unexpected state in the LangChain docstore.")
        # Pydantic models pickle their fields under __dict__; plain classes are the dict.
        inner = state.get("__dict__", state)
        self.state = dict(inner) if isinstance(inner, dict) else {}


class _Docstore(_Plain):
    pass


class _Document(_Plain):
    pass


_SAFE_BUILTINS = {"set": set, "frozenset": frozenset}


class _DocstoreUnpickler(pickle.Unpickler):
    """Rebuilds only LangChain's docstore and documents; any other class is refused."""

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in ("langchain", "langchain_core", "langchain_community"):
            if name == "InMemoryDocstore":
                return _Docstore
            if name == "Document":
                return _Document
        if module == "builtins" and name in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[name]
        raise pickle.UnpicklingError(
            f"The docstore refers to {module}.{name}, which Vectraudit does not load "
            "(only LangChain's InMemoryDocstore and Document are read, as plain data)."
        )


def _langchain(folder: Path, names: list[str], index_name: str | None) -> SavedStore:
    if index_name is not None:
        if index_name not in names:
            raise SavedStoreError(
                f"No {index_name}.faiss with {index_name}.pkl in {folder}.",
                f"Saved indexes here: {', '.join(names)}.",
            )
        name = index_name
    elif len(names) == 1:
        name = names[0]
    elif "index" in names:
        name = "index"
    else:
        raise SavedStoreError(
            f"{folder} holds several LangChain saves: {', '.join(names)}.",
            "Pass the .faiss file of the one to open, next to its .pkl.",
        )
    return read_langchain(folder / f"{name}.faiss")


def read_langchain(index_path: Path) -> SavedStore:
    """Read the docstore saved beside a LangChain ``.faiss`` file."""
    pkl = index_path.with_suffix(".pkl")
    raw = _read_bytes(pkl)
    try:
        loaded = _DocstoreUnpickler(io.BytesIO(raw)).load()
    except pickle.UnpicklingError as e:
        raise SavedStoreError(str(e), "Open the index file alone, without its .pkl.") from e
    except Exception as e:  # truncated or foreign pickles fail in many ways
        raise SavedStoreError(
            f"{pkl.name} is not a LangChain FAISS docstore ({type(e).__name__}: {e}).",
            "Open the index file alone, without its .pkl.",
        ) from e
    if not (isinstance(loaded, tuple) and len(loaded) == 2):
        raise SavedStoreError(
            f"{pkl.name} does not hold (docstore, index_to_docstore_id).",
            "Is this a folder written by LangChain's FAISS.save_local?",
        )
    docstore, row_to_id = loaded
    documents = docstore.state.get("_dict") if isinstance(docstore, _Docstore) else None
    if not isinstance(documents, dict) or not isinstance(row_to_id, dict):
        raise SavedStoreError(
            f"{pkl.name} has no in-memory docstore.",
            "Only LangChain's InMemoryDocstore (the default) is supported.",
        )
    rows, missing = [], 0
    for row, doc_id in sorted(row_to_id.items()):
        doc = documents.get(doc_id)
        if not isinstance(doc, _Document):
            missing += 1
            continue
        fields = doc.state
        rows.append(_row(int(row), doc_id, fields.get("page_content"), fields.get("metadata")))
    notes = [f"{missing} vectors have no document in the docstore."] if missing else []
    return SavedStore(LANGCHAIN, index_path.parent, index_path, Metadata(rows), notes)


# LlamaIndex ---------------------------------------------------------------------------


def _llamaindex(folder: Path) -> SavedStore:
    stores = _read_json(folder / "index_store.json").get("index_store/data")
    if not isinstance(stores, dict):
        raise SavedStoreError(
            "index_store.json has no index_store/data.",
            "Is this a folder written by LlamaIndex's StorageContext.persist?",
        )
    vector_indexes = []
    for index_id, entry in stores.items():
        if isinstance(entry, dict) and entry.get("__type__") == "vector_store":
            data = entry.get("__data__")
            vector_indexes.append((index_id, json.loads(data) if isinstance(data, str) else data))
    if len(vector_indexes) != 1:
        raise SavedStoreError(
            f"index_store.json holds {len(vector_indexes)} vector indexes; expected one.",
            "Persist one VectorStoreIndex per folder.",
        )
    nodes_dict = vector_indexes[0][1].get("nodes_dict") if vector_indexes[0][1] else None
    if not isinstance(nodes_dict, dict):
        raise SavedStoreError(
            "The vector index in index_store.json has no nodes_dict.",
            "It may store text in the vector store; only FaissVectorStore folders are read.",
        )
    nodes = _read_json(folder / "docstore.json").get("docstore/data", {})
    rows, missing = [], 0
    for row, node_id in sorted(nodes_dict.items(), key=lambda kv: int(kv[0])):
        node = nodes.get(node_id, {}).get("__data__") if isinstance(nodes, dict) else None
        if not isinstance(node, dict):
            missing += 1
            continue
        source = (node.get("relationships") or {}).get("1")  # NodeRelationship.SOURCE
        extra = {"ref_doc_id": source.get("node_id")} if isinstance(source, dict) else {}
        rows.append(_row(int(row), node_id, node.get("text"), node.get("metadata"), extra))
    notes = [f"{missing} vectors have no node in docstore.json."] if missing else []
    return SavedStore(LLAMAINDEX, folder, folder / LLAMAINDEX_INDEX, Metadata(rows), notes)


# Shared -------------------------------------------------------------------------------

_RESERVED = ("id", "text", "docstore_id", "ref_doc_id")


def _row(
    row: int,
    doc_id: Any,
    text: Any,
    metadata: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"id": row, "text": text, "docstore_id": str(doc_id), **(extra or {})}
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            out[f"metadata_{key}" if key in _RESERVED else str(key)] = value
    return out


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise SavedStoreError(f"Missing {path.name} in {path.parent}.", "Copy the whole folder.")
    if path.stat().st_size > MAX_STORE_BYTES:
        raise SavedStoreError(
            f"{path.name} is larger than {MAX_STORE_BYTES // 1024**3} GiB.",
            "Open the index file alone, or export the documents to a metadata file.",
        )
    return path.read_bytes()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(_read_bytes(path))
    except json.JSONDecodeError as e:
        raise SavedStoreError(
            f"{path.name} is not valid JSON: {e}", "Copy the whole folder."
        ) from e
    if not isinstance(data, dict):
        raise SavedStoreError(f"{path.name} is not a JSON object.", "Copy the whole folder.")
    return data
