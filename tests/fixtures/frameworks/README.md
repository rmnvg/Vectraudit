# FAISS folders saved by LangChain and LlamaIndex

Written by the libraries themselves, so `vectraudit.core.frameworks` is tested against what
they actually save. Six synthetic passages from three source files, embedded with each
library's deterministic fake embedding (8 dimensions, `IndexFlatL2`).

| Folder | Written by | Versions |
|---|---|---|
| `langchain/` | `FAISS.from_documents(...).save_local(...)` | `langchain-community` 0.4.2, `langchain-core` 1.8.5 |
| `llamaindex/` | `VectorStoreIndex.from_documents(...)` with a `FaissVectorStore`, then `persist(...)` | `llama-index-core` 0.14.26, `llama-index-vector-stores-faiss` 0.8.0 |

Common: `faiss-cpu` 1.16.0, Python 3.12. To regenerate (in a separate environment; none of
these are Vectraudit dependencies):

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install langchain-core langchain-community \
  faiss-cpu llama-index-core llama-index-vector-stores-faiss
.venv/bin/python -I generate.py .
```

LlamaIndex node ids are random UUIDs, so they change on every run; the tests do not depend
on them.
