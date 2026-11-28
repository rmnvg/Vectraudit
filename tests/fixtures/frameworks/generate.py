"""Write the LangChain and LlamaIndex FAISS folders under tests/fixtures/frameworks.

Run in a separate environment (none of these are Vectraudit dependencies); see README.md.
Embeddings are deterministic fakes and all text is synthetic, so nothing is downloaded.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import faiss  # before torch-free LlamaIndex imports, as elsewhere in the repo

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
DIM = 8
PASSAGES = [
    ("handbook.md", "Refunds are issued within 14 days of a returned item reaching us."),
    ("handbook.md", "Store credit never expires and can be combined with discount codes."),
    ("faq.md", "Orders over 50 euros ship free inside the European Union."),
    ("faq.md", "Express shipping arrives in one to two working days."),
    ("policy.md", "Personal data is deleted 30 days after an account is closed."),
    ("policy.md", "Support tickets are answered within one working day."),
]


def langchain(folder: Path) -> None:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    docs = [
        Document(page_content=text, metadata={"source": source, "chunk": i}, id=f"lc-{i}")
        for i, (source, text) in enumerate(PASSAGES)
    ]
    FAISS.from_documents(docs, DeterministicFakeEmbedding(size=DIM)).save_local(str(folder))


def llamaindex(folder: Path) -> None:
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    Settings.embed_model = MockEmbedding(embed_dim=DIM)
    Settings.llm = None
    sources: dict[str, list[str]] = {}
    for source, text in PASSAGES:
        sources.setdefault(source, []).append(text)
    documents = [
        Document(text="\n\n".join(texts), id_=f"doc-{name}", metadata={"file_name": name})
        for name, texts in sources.items()
    ]
    store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(DIM))
    context = StorageContext.from_defaults(vector_store=store)
    # Small chunks so each source document becomes several nodes, as in real use.
    Settings.chunk_size, Settings.chunk_overlap = 24, 0
    VectorStoreIndex.from_documents(documents, storage_context=context)
    context.persist(persist_dir=str(folder))


if __name__ == "__main__":
    for name, write in (("langchain", langchain), ("llamaindex", llamaindex)):
        target = OUT / name
        shutil.rmtree(target, ignore_errors=True)
        write(target)
        print(f"Wrote {target}")
