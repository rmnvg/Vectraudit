"""LangChain and LlamaIndex FAISS folders open directly, with their documents as metadata.

The fixtures were written by the libraries themselves (tests/fixtures/frameworks/README.md).
"""

from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.core.frameworks import (
    LANGCHAIN,
    LLAMAINDEX,
    SavedStoreError,
    is_saved_store,
    open_saved_store,
)

pytest.importorskip("faiss")

FIXTURES = Path(__file__).parent / "fixtures" / "frameworks"
LC, LI = FIXTURES / "langchain", FIXTURES / "llamaindex"


def test_langchain_folder_maps_every_row_to_its_document():
    store = open_saved_store(LC)
    assert store.framework == LANGCHAIN
    assert store.index_path == LC / "index.faiss"
    assert len(store.metadata) == 6
    assert store.notes == []
    assert store.metadata.get(0) == {
        "text": "Refunds are issued within 14 days of a returned item reaching us.",
        "docstore_id": "lc-0",
        "source": "handbook.md",
        "chunk": 0,
    }
    assert store.metadata.snippet(2) == {
        "title": "faq.md",
        "text": "Orders over 50 euros ship free inside the European Union.",
    }
    assert store.describe() == "LangChain FAISS folder (6 documents)"


def test_llamaindex_folder_maps_rows_to_nodes_and_their_source_documents():
    store = open_saved_store(LI)
    assert store.framework == LLAMAINDEX
    assert store.index_path.name == "default__vector_store.json"
    assert len(store.metadata) == 6
    first = store.metadata.get(0)
    assert first is not None
    assert first["ref_doc_id"] == "doc-handbook.md"
    assert first["text"].startswith("Refunds are issued")
    assert store.metadata.snippet(4) is not None
    assert store.metadata.snippet(4)["title"] == "policy.md"  # file_name is a title column


def test_detection_and_a_clear_refusal_for_other_folders(tmp_path):
    assert is_saved_store(LC)
    assert is_saved_store(LI)
    assert not is_saved_store(tmp_path)
    assert not is_saved_store(LC / "index.faiss")
    with pytest.raises(SavedStoreError, match="not a saved LangChain or LlamaIndex") as e:
        open_saved_store(tmp_path)
    assert "index.pkl" in e.value.hint


class _RunsCode:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return (os.system, (f"touch {self.sentinel}",))


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        ("code", "system"),  # posix.system or nt.system
        ("datetime", "datetime.datetime"),
    ],
)
def test_the_docstore_pickle_never_loads_other_classes(tmp_path, payload, named):
    folder = tmp_path / "store"
    shutil.copytree(LC, folder)
    sentinel = tmp_path / "ran"
    if payload == "code":
        data = pickle.dumps((_RunsCode(sentinel), {0: "lc-0"}))
    else:  # a harmless class that is still not on the allow-list
        data = pickle.dumps(({"when": datetime(2026, 1, 1)}, {0: "lc-0"}))
    (folder / "index.pkl").write_bytes(data)
    with pytest.raises(SavedStoreError, match=named) as e:
        open_saved_store(folder)
    assert "only LangChain's InMemoryDocstore and Document" in str(e.value)
    assert not sentinel.exists()


def test_several_langchain_saves_need_a_name(tmp_path):
    folder = tmp_path / "store"
    folder.mkdir()
    for name in ("docs", "faq"):
        shutil.copy(LC / "index.faiss", folder / f"{name}.faiss")
        shutil.copy(LC / "index.pkl", folder / f"{name}.pkl")
    with pytest.raises(SavedStoreError, match="several LangChain saves: docs, faq"):
        open_saved_store(folder)
    assert open_saved_store(folder, index_name="faq").index_path.name == "faq.faiss"
    with pytest.raises(SavedStoreError, match=r"No other\.faiss"):
        open_saved_store(folder, index_name="other")


def test_llamaindex_folder_without_one_vector_index_is_refused(tmp_path):
    folder = tmp_path / "store"
    shutil.copytree(LI, folder)
    (folder / "index_store.json").write_text('{"index_store/data": {}}')
    with pytest.raises(SavedStoreError, match="0 vector indexes"):
        open_saved_store(folder)


def test_session_uses_the_docstore_unless_metadata_is_given():
    from vectraudit.session import InputError, Session

    session = Session(LC, disk_cache=False)
    assert session.saved_store is not None
    assert session.saved_store.framework == LANGCHAIN
    assert session.metadata is session.saved_store.metadata
    assert session.metadata_coverage() == 1.0

    own = Session(LI, metadata=[{"id": i, "text": f"mine {i}"} for i in range(6)], disk_cache=False)
    assert own.metadata is not None
    assert own.metadata.get(0) == {"text": "mine 0"}

    with pytest.raises(InputError, match="not a saved"):
        Session(FIXTURES, disk_cache=False)


def test_search_results_carry_the_chunk_text():
    from fastapi.testclient import TestClient

    from vectraudit.server.app import create_app
    from vectraudit.session import Session

    client = TestClient(create_app(Session(LC, disk_cache=False)))
    body = client.post("/api/search", json={"query": {"id": 0}, "k": 3}).json()
    texts = [r["snippet"]["text"] for r in body["results"]]
    assert len(texts) == 3
    assert all(t for t in texts)


def test_info_names_the_framework_and_its_columns():
    result = CliRunner().invoke(app, ["info", str(LI)])
    assert result.exit_code == 0, result.output
    assert "LlamaIndex FAISS folder (6 documents)" in result.output
    assert "ref_doc_id" in result.output
