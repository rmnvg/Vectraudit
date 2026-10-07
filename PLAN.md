# Vectraudit: Build Plan

> **Status (2026-10-04):** this is the original build plan for the FAISS debugger (released
> internally as 0.1.0 under the name faissight). Its phases are complete. The project is now a
> private commercial product: steps below that publish to PyPI or set up public distribution
> no longer apply. The next phases come from the product requirements, not this file.
>
> - Import name: `vectraudit`
> - CLI command: `vectraudit`
> - Repo: `github.com/rmnvg/Vectraudit` (private)
> - Distribution: private GitHub releases (no PyPI)
> - Current milestone (October 7): [pilot release validation](docs/pilot-release-validation.md).
>   The October 6 market-fit engineering priorities are implemented; validate the complete
>   release journey and use observed pilot friction to choose subsequent work.

---

## 0. How to use this plan (instructions for Claude Code)

- Work **phase by phase, in order**. Do not start a phase until the previous phase's acceptance criteria pass.
- At the start, create a `CLAUDE.md` in the repo root summarising: project goal, repo layout, commands (`uv run pytest`, `uv run ruff check`, `npm run build`), and coding conventions from section 5.
- Make **one git commit per completed step** with a conventional commit message (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Write tests alongside code, not after. Every public function in `vectraudit.core` needs at least one test.
- Before adding any dependency not listed in section 4, stop and ask.
- When a FAISS API behaves differently than this plan assumes, trust the installed FAISS version, adapt, and leave a short comment explaining why.
- Keep a `docs/DECISIONS.md` log: one line per non-obvious technical decision.

---

## 1. Problem & positioning

### The problem
FAISS is the default vector index in countless RAG systems, but it has **no UI**. When retrieval is bad, engineers can't easily answer:
- Which cluster/cell did my query probe, and were the true nearest neighbours even in those cells?
- Is my IVF index badly imbalanced (a few giant lists, many empty ones)?
- How much accuracy did PQ compression cost me?
- What `nprobe` / `efSearch` gives me 95% recall at acceptable latency?
- Why did this irrelevant chunk rank #1 and the relevant one not appear?

### Prior art (mention honestly in README)
- **Feder** (zilliztech/feder): JS visualizer for Faiss IVF_Flat and HNSWlib indexes. Educational, shows clusters and search process. Limitations: Faiss support only for IVF_Flat, parses index files in JavaScript (memory-heavy), no recall/tuning or RAG debugging.
- **Embedding projectors** (TensorFlow Projector, Nomic Atlas, Renumics Spotlight, Arize Phoenix): visualise embeddings, not index internals.

### vectraudit's angle
A **Python-first retrieval debugger for FAISS**:
1. Works on real indexes (IVF-Flat, IVF-PQ, IVF-SQ, HNSW, Flat, with IDMap/PreTransform wrappers).
2. Explains **misses** against exact ground truth, not just shows pictures.
3. Tunes the **recall vs latency** trade-off interactively.
4. Attaches **chunk text** so it doubles as a RAG debugging tool.

### Goals (v0.1.0)
- `pip install "vectraudit[faiss-cpu]"` then `vectraudit serve my.index` opens a local web UI.
- `vectraudit.launch(index, ...)` works from Python and renders inline in Jupyter.
- IVF overview + query explorer + recall tuner + RAG text panel.
- HNSW graph view with animated search trace.
- PQ reconstruction-error view.
- Live hosted demo on Hugging Face Spaces.

### Non-goals (v0.1.0)
- GPU indexes (users can `faiss.index_gpu_to_cpu` first; document this).
- Binary indexes, LSH, IndexIVFFastScan internals (show "unsupported: basic stats only").
- Editing/rebuilding indexes from the UI.
- Multi-user hosting or auth.

---

## 2. Architecture

```
┌────────────────────────────── vectraudit (Python package) ──────────────────────────────┐
│                                                                                         │
│  CLI (Typer)            Python API                    Jupyter helper                    │
│  vectraudit serve/info   vectraudit.launch(...)         IFrame to local server            │
│        │                       │                             │                          │
│        └───────────────┬───────┴─────────────────────────────┘                          │
│                        ▼                                                                │
│                 Session (in-memory state: index, vectors, metadata, caches)             │
│                        │                                                                │
│        ┌───────────────┼───────────────────────────────┐                                │
│        ▼               ▼                               ▼                                │
│  core.loader     core.ivf / core.hnsw / core.pq   core.search (trace, ground truth,     │
│  (read, unwrap,  (extract structure, stats)       sweep)                                │
│   detect type)                                                                          │
│        │                                                                                │
│        ▼                                                                                │
│  core.projection (PCA default, UMAP optional, sampling, disk cache)                     │
│                        │                                                                │
│                        ▼                                                                │
│        FastAPI server (JSON API + serves built frontend from package static/)           │
└────────────────────────┬────────────────────────────────────────────────────────────────┘
                         │ HTTP (localhost)
                         ▼
          React + TypeScript frontend (Vite build, bundled into the wheel)
          deck.gl for large scatter/graph views, Recharts for charts
```

Key principle: **all FAISS work happens in Python.** The frontend only receives compact JSON (projected 2D/3D coordinates, ids, small stats). Never ship raw high-dimensional vectors to the browser.

---

## 3. Repository layout

```
vectraudit/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                # lint, typecheck, test (py 3.10–3.12), frontend build
│   └── release.yml           # build frontend → build wheel → publish to PyPI (trusted publishing)
├── src/vectraudit/
│   ├── __init__.py           # exports launch, Session, __version__
│   ├── cli.py                # Typer app: serve, info, sweep
│   ├── session.py            # Session class holding loaded state
│   ├── jupyter.py            # notebook display helper
│   ├── core/
│   │   ├── loader.py         # read_index, unwrap wrappers, detect IndexKind
│   │   ├── types.py          # dataclasses / pydantic models shared across core
│   │   ├── ivf.py            # centroids, list sizes, imbalance, members
│   │   ├── hnsw.py           # graph extraction, levels, neighbours
│   │   ├── pq.py             # PQ/SQ reconstruction error
│   │   ├── search.py         # search w/ params, ground truth, IVF trace, miss reasons
│   │   ├── hnsw_trace.py     # Python re-implementation of HNSW greedy search for tracing
│   │   ├── sweep.py          # recall@k and latency vs nprobe / efSearch
│   │   ├── projection.py     # PCA/UMAP, sampling, caching
│   │   ├── metadata.py       # load jsonl/csv/parquet chunk metadata
│   │   └── embed.py          # optional text → vector (sentence-transformers or callable)
│   ├── server/
│   │   ├── app.py            # FastAPI app factory
│   │   ├── routes.py         # API endpoints (section 7)
│   │   └── schemas.py        # pydantic response models
│   └── static/               # built frontend (gitignored, produced by build)
├── frontend/
│   ├── package.json
│   ├── vite.config.ts        # build output → ../src/vectraudit/static
│   └── src/
│       ├── api/              # typed fetch client
│       ├── views/            # Overview, ClusterMap, QueryExplorer, Tuner, HnswGraph, Quantization
│       ├── components/
│       └── App.tsx
├── examples/
│   ├── make_synthetic.py     # gaussian-blob indexes of each type (for tests + quick demos)
│   ├── make_rag_demo.py      # real text dataset → embeddings → several indexes + chunks.jsonl
│   └── notebook_demo.ipynb
├── tests/
│   ├── conftest.py           # fixtures build small indexes in tmp dirs
│   ├── test_loader.py
│   ├── test_ivf.py
│   ├── test_hnsw.py
│   ├── test_search.py
│   ├── test_hnsw_trace.py
│   ├── test_sweep.py
│   ├── test_projection.py
│   └── test_api.py
├── deploy/hf-space/
│   ├── Dockerfile
│   └── README.md             # HF Space card metadata
└── docs/
    ├── DECISIONS.md
    └── screenshots/
```

---

## 4. Tech stack & dependencies

**Python (3.10+)**
- Tooling: `uv` (env + build), `hatchling` (build backend), `ruff` (lint + format), `mypy` (strict on `vectraudit.core`), `pytest`, `pytest-cov`, `pre-commit`.
- Core deps: `numpy`, `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `typer`, `rich`.
- Extras:
  - `vectraudit[faiss-cpu]` → `faiss-cpu>=1.8`. FAISS is **not** a hard dependency, because conda and GPU users install FAISS differently and a pip `faiss-cpu` would clash. On import failure, raise a clear error telling the user how to install it.
  - `vectraudit[umap]` → `umap-learn` (heavy; PCA is the default projection).
  - `vectraudit[text]` → `sentence-transformers` (text queries).
  - `vectraudit[parquet]` → `pyarrow`.
  - `vectraudit[qdrant]` → `qdrant-client` (Phase 14 connector).
  - `vectraudit[pgvector]` → `psycopg[binary]` (Phase 14 connector; LGPL, used unmodified).
  - `vectraudit[weaviate]` → `weaviate-client` (BSD-3) (Phase 19). Pinecone needs no extra: the connector uses its REST API with a pinned API version.
  - `vectraudit[cloud]` → `sqlalchemy>=2`, `alembic`, `psycopg[binary]` (Phase 16 hosted service).
  - `vectraudit[all]` → everything above.
  - `vectraudit[dev]` → test/lint tools.

**Frontend**
- React 18 + TypeScript + Vite.
- Tailwind CSS.
- `deck.gl` (ScatterplotLayer, LineLayer, OrbitView) for cluster maps and HNSW graphs; handles 100k+ points via WebGL.
- `recharts` for bar/line charts (list sizes, recall curves).
- `@tanstack/react-query` for API state.
- No component library needed; keep UI clean and minimal.

---

## 5. Conventions

- `src/` layout; type hints everywhere; docstrings on public functions.
- `vectraudit.core` must **not** import FastAPI or anything web-related. It's a plain library usable without the server.
- All IDs exposed to the UI are the **user-facing ids** (after IDMap), never internal offsets. Keep a mapping in the Session.
- Distances: always report the metric (`L2` or `IP`) alongside values. For IP, "closer" = larger.
- Large arrays: never return more than `max_points` (default 50,000) points to the frontend; sample deterministically (seeded) and stratified by IVF list when applicable.
- Every expensive computation (projection, ground truth, sweep) is cached on the Session, and projections are also cached on disk at `~/.cache/vectraudit/<index_sha1>/`.
- Errors surface to the UI as structured JSON `{error_code, message, hint}`.

---

## 6. Inputs the tool accepts

| Input | Required | Formats | Purpose |
|---|---|---|---|
| FAISS index | yes | file path or in-memory `faiss.Index` | the thing being inspected |
| Raw vectors | optional but strongly recommended | `.npy` (float32, shape `n×d`, row i ↔ id i or aligned with `--ids`) | exact ground truth, PQ error |
| Ids for raw vectors | optional | `.npy` int64 | when row order ≠ ids |
| Metadata | optional | `.jsonl` / `.csv` / `.parquet` with an `id` column plus e.g. `text`, `source`, `title` | RAG debugging text |
| Embedder | optional | sentence-transformers model name (CLI) or Python callable `str -> np.ndarray` (API) | type text queries |
| Query vectors | optional | `.npy` | batch evaluation / sweep queries |

If raw vectors are **not** given: ground truth falls back to exact search over **reconstructed** vectors (`reconstruct_n` after `make_direct_map`). The UI must show a visible banner: "Ground truth computed on reconstructed vectors; PQ/SQ error is not measured."

If no query set is given for the sweep: sample 200 stored vectors as queries (seeded), and exclude each query's own id from its results.

---

## 7. API endpoints (FastAPI)

All under `/api`. Response models in `server/schemas.py`.

| Method | Path | Returns |
|---|---|---|
| GET | `/info` | index kind, class chain (wrappers), d, ntotal, metric, is_trained, params (nlist, nprobe, M, efSearch, PQ m/nbits…), which optional inputs are loaded |
| GET | `/ivf/lists` | per-list sizes, empty count, imbalance factor, min/median/max, top-20 largest lists |
| GET | `/ivf/list/{list_no}` | member ids (paginated), with metadata snippets |
| GET | `/projection?kind=points\|centroids&method=pca\|umap&dims=2\|3` | coordinates + ids + list assignment (sampled) |
| POST | `/search` | body: `{query: {id \| vector \| text}, k, nprobe?, efSearch?}` → results with distance, list_no, rank, metadata; plus ground truth, recall@k, and miss explanations |
| POST | `/trace/ivf` | probe order of cells with centroid distances, which were probed, where each true neighbour lives and the probe rank its cell would have needed |
| POST | `/trace/hnsw` | per-layer visited nodes in order, edges traversed, entry point, final candidates |
| GET | `/hnsw/graph?level=L&limit=N` | nodes + edges at a level (sampled for layer 0) with projected coords |
| GET | `/hnsw/stats` | nodes per level, avg/min/max degree per level, entry point, max level |
| POST | `/sweep` | body: `{param: nprobe\|efSearch, values: [...], k, n_queries}` → recall@k, mean/p95 latency per value |
| GET | `/pq/error` | reconstruction error distribution (histogram), per-list mean error, worst-50 ids |
| GET | `/metadata/{id}` | metadata row |
| GET | `/health` | ok |

Everything else (`/`, `/assets/*`) serves the built frontend.

---

## 8. Phases & steps

### Phase 0: Scaffold & reserve the name
1. `uv init --lib vectraudit` with `src/` layout; configure `pyproject.toml` (hatchling, metadata, extras from section 4, `[project.scripts] vectraudit = "vectraudit.cli:app"`).
2. Add ruff, mypy, pytest config; `.pre-commit-config.yaml`.
3. Scaffold `frontend/` with Vite React-TS + Tailwind; set Vite `build.outDir` to `../src/vectraudit/static` and `emptyOutDir: true`; add `static/` to `.gitignore` but include it in wheel via hatch `artifacts`.
4. Minimal `cli.py` with `vectraudit --version`.
5. CI workflow: Python 3.10/3.11/3.12 matrix, `uv sync --all-extras`, ruff, mypy, pytest; separate job builds the frontend.
6. **Manual step for the human** (note it in README TODO, do not attempt): publish a `0.0.1` placeholder to PyPI to reserve the name, and set up PyPI trusted publishing for `release.yml`.

**Acceptance:** `uv run vectraudit --version` works; CI green; `uv build` produces a wheel that contains `vectraudit/static/index.html`.

### Phase 1: Test fixtures & loader
1. `examples/make_synthetic.py`: generate `n=20_000, d=64` gaussian blobs (seeded, 50 centres) and save indexes: `IndexFlatL2`, `IndexIVFFlat(nlist=128)`, `IndexIVFPQ(nlist=128, m=8, nbits=8)`, `IndexIVFScalarQuantizer(QT_8bit)`, `IndexHNSWFlat(M=16)`, `IndexIDMap(IndexFlatL2)`, an `IndexPreTransform(PCA→IVFFlat)`, plus an IP-metric IVFFlat on normalised vectors. Also save the raw vectors `.npy` and a fake `chunks.jsonl`.
2. `tests/conftest.py`: session-scoped fixtures building **small** versions (n≈2,000) of each in `tmp_path_factory` so tests are fast.
3. `core/loader.py`:
   - `load_index(path_or_index) -> LoadedIndex` using `faiss.read_index`.
   - Unwrap chain: walk `IndexPreTransform`, `IndexIDMap`/`IndexIDMap2`, `IndexRefine` using `faiss.downcast_index`; record the class chain for display.
   - Use `faiss.extract_index_ivf(index)` (in a try/except) to find an IVF core.
   - Detect `IndexKind`: `FLAT`, `IVF_FLAT`, `IVF_PQ`, `IVF_SQ`, `HNSW_FLAT`, `HNSW_OTHER`, `UNSUPPORTED`.
   - Read params: `d`, `ntotal`, `metric_type`, `nlist`, `nprobe`, `pq.M`, `pq.nbits`, `hnsw.efSearch`, `hnsw.efConstruction`, M.
   - Id mapping: if IDMap, `faiss.vector_to_array(index.id_map)`.
4. `vectraudit info my.index` prints a rich table of the above.

**Acceptance:** loader tests pass for every synthetic index type; `info` output is correct and readable; unsupported types (e.g. a binary index) produce a clear message rather than a crash.

### Phase 2: IVF inspection
1. `core/ivf.py` using `faiss.contrib.inspect_tools` (`get_invlist_sizes`, `get_invlist`):
   - `list_sizes() -> np.ndarray`
   - `imbalance_factor()` (FAISS formula: `nlist * Σ size² / (Σ size)²`); cross-check against `invlists.imbalance_factor()` if available.
   - `list_members(list_no) -> ids`
   - `centroids()` via `ivf.quantizer.reconstruct_n(0, nlist)`
   - `assignments() -> ids→list_no` (build once, cache).
2. Map internal ids to user-facing ids when wrapped in IDMap.

**Acceptance:** Σ list sizes == ntotal; every id appears in exactly one list; centroids shape `(nlist, d)`; tests cover IVFFlat, IVFPQ, IVFSQ, and PreTransform-wrapped IVF (centroids live in the transformed space, so document that in DECISIONS.md).

### Phase 3: Search, ground truth, IVF trace
1. `core/search.py`:
   - `search(query, k, nprobe=None, efSearch=None)` using `faiss.SearchParametersIVF` / `faiss.SearchParametersHNSW` (don't mutate the shared index params; fall back to `faiss.ParameterSpace().set_index_parameter` only if params objects are unsupported by the index type).
   - `ground_truth(query, k)`: exact `IndexFlat` with the **same metric** built over raw vectors (or reconstructed vectors, with the flag set). Build lazily, cache.
   - `recall_at_k(results, truth)`.
   - Query resolution: by stored id (reconstruct, then exclude self from results), by raw vector, or by text via embedder.
2. IVF trace:
   - Coarse search: `quantizer.search(q, nlist)` → full probe order with centroid distances (apply the PreTransform to the query first if present).
   - Mark first `nprobe` as probed.
   - For each ground-truth neighbour: its list, the **probe rank** of that list, and a miss reason:
     - `CELL_NOT_PROBED` (needed nprobe ≥ rank+1)
     - `QUANTIZATION` (its cell was probed but approximate distance pushed it out of top-k; applies to PQ/SQ)
     - `FOUND`
   - Return "minimum nprobe to find all true top-k" as a headline number.
3. `vectraudit.core` public API summary in `__init__` of core.

**Acceptance:** on IVFFlat with nprobe=nlist, recall@10 == 1.0 against ground truth; miss reasons are correct on a hand-constructed test where a neighbour is placed in a far cell; self-exclusion works.

### Phase 4: Projection
1. `core/projection.py`:
   - PCA via numpy SVD (no sklearn dependency) to 2D/3D.
   - UMAP if `umap-learn` installed (`n_neighbors=15, min_dist=0.1, random_state=42`).
   - Fit on a sample (≤ 50k), project centroids and query vectors with the same fitted transform (for UMAP use `.transform`).
   - Stratified sampling by IVF list so small lists still appear.
   - Disk cache keyed by sha1 of index bytes + method + dims + sample size.
2. Background computation: projection runs in a thread when the server starts; API returns `202 {status: "computing"}` until ready; UI shows progress.

**Acceptance:** PCA of 20k×64 in < 2 s; cache hit on second run; query vector lands near its neighbours in projection space (sanity test: nearest projected point overlaps top-10 more than random).

### Phase 5: Server & CLI serve
1. `server/app.py` app factory taking a `Session`; mount `static/` with SPA fallback to `index.html`.
2. Implement endpoints from section 7 that are backed by phases 1–4 (`info`, `ivf/*`, `projection`, `search`, `trace/ivf`, `metadata`, `health`).
3. `vectraudit serve INDEX [--vectors v.npy] [--ids ids.npy] [--meta chunks.jsonl] [--embedder all-MiniLM-L6-v2] [--queries q.npy] [--host 127.0.0.1] [--port 8765] [--no-browser] [--max-points 50000]`. Opens the browser unless `--no-browser`.
4. Validate inputs up front with actionable errors (dimension mismatch between vectors and index, metadata missing `id` column, etc.).
5. `tests/test_api.py` using FastAPI `TestClient`.

**Acceptance:** `vectraudit serve examples/data/ivfflat.index --vectors ... --meta ...` starts, `/api/info` and `/api/search` return correct data; API tests pass.

### Phase 6: Frontend MVP
Layout: left sidebar (index info card + nav), main panel (active view). Light and dark theme.

1. **Overview view**: index card (kind, class chain, d, ntotal, metric, params), list-size histogram, imbalance factor with a plain-English verdict ("healthy" / "skewed: top 5% of lists hold X% of vectors"), empty-list count, top-20 largest lists table.
2. **Cluster Map view**: deck.gl scatter of sampled points coloured by list; centroids as larger outlined markers; hover shows id + metadata snippet; click a centroid → side panel with that list's members. Toggle 2D/3D, PCA/UMAP.
3. **Query Explorer view** (the core feature):
   - Query input: text (if embedder), stored id, or paste vector.
   - Controls: k, nprobe (or efSearch), "compare with exact" toggle.
   - Results table: rank, id, distance, list, metadata text, badge if also in ground truth.
   - Ground-truth table with miss reasons and colour-coded badges.
   - Headline: recall@k, min nprobe needed.
   - On the map: query star, probed cells highlighted, returned results vs missed true neighbours in different colours.
   - Probe-order chart: centroid distance vs probe rank, with the nprobe cut-off line and dots for cells holding true neighbours.
4. Loading/computing states, empty states, error toasts from structured errors.

**Acceptance:** full flow works on the synthetic IVFFlat and IVFPQ indexes; UI stays responsive with 50k points; Lighthouse performance reasonable on the built bundle.

### Phase 7: Recall / latency tuner
1. `core/sweep.py`: for each value in the sweep (default nprobe `[1,2,4,8,16,32,64,128,…≤nlist]`, efSearch `[16,32,64,128,256,512]`) run the query set, measure recall@k against cached ground truth and latency (warm-up run first; `time.perf_counter`; report mean and p95 per query; pin `faiss.omp_set_num_threads(1)` during timing and restore afterwards so numbers are stable and comparable).
2. Endpoint `/sweep` runs in a background thread with progress polling.
3. **Tuner view**: recall@k vs param line chart, latency vs param chart, and a recall-vs-latency Pareto chart; "target recall" input highlights the cheapest value meeting it; a copyable code snippet: `index.nprobe = 24` / `faiss.SearchParametersIVF(nprobe=24)`.
4. CLI: `vectraudit sweep INDEX --vectors ... --queries ... --param nprobe` prints a rich table (useful without the UI and in CI).

**Acceptance:** recall is monotonic non-decreasing in nprobe on IVFFlat (test); recommended value meets target recall on the test set.

### Phase 8: Python API & Jupyter
1. `vectraudit.launch(index, vectors=None, ids=None, metadata=None, embedder=None, port=None, open_browser=None) -> Viewer`:
   - Accepts in-memory `faiss.Index`, `np.ndarray`, pandas DataFrame or list of dicts for metadata, a callable embedder.
   - Starts uvicorn in a daemon thread on a free port.
   - In Jupyter: returns an object whose `_repr_html_` renders an `IFrame` of the UI (height 800). Outside Jupyter: opens the browser.
   - `Viewer.stop()` shuts down the server; `Viewer.url` property.
2. `examples/notebook_demo.ipynb`: build an index, launch, run a query.
3. Note in docs: works in JupyterLab and VS Code notebooks locally; Colab needs `google.colab.output.serve_kernel_port_as_iframe`; add a small helper for that and detect Colab automatically.

**Acceptance:** launching twice in one kernel uses different ports and doesn't crash; `stop()` frees the port; notebook demo runs top to bottom.

### Phase 9: HNSW view & search trace
1. `core/hnsw.py` (via `faiss.contrib.inspect_tools.get_hnsw_links` or direct `hnsw.levels/offsets/neighbors` arrays with `faiss.vector_to_array`):
   - node level for each id (FAISS stores `levels` as level+1).
   - neighbours per node per level; entry point; max level.
   - stats: nodes per level, degree distribution per level.
2. `core/hnsw_trace.py`: re-implement FAISS HNSW search in numpy over the extracted graph using stored vectors (reconstructed from HNSW storage):
   - Upper levels: greedy descent (ef=1) from entry point.
   - Level 0: beam search with `efSearch`, candidate min-heap + result max-heap.
   - Record visit order, distance at each visit, edges traversed, per level.
   - Note FAISS internals (e.g. `search_bounded_queue`, visited-table details) may differ slightly; this is a **reconstructed trace**. Label it so in the UI.
3. Validation test: trace results must overlap FAISS's own `search` results by ≥ 95% on average across 100 queries at efSearch=64. If lower, investigate and document differences in DECISIONS.md.
4. **HNSW Graph view**:
   - Stacked layer view (deck.gl OrbitView): each level a plane, top levels full, layer 0 sampled around the query neighbourhood.
   - Stats panel: nodes/level, degree histograms, entry point.
   - "Play search" animation: step through the trace (play/pause/step/speed), highlighting current node, frontier, visited set, and final top-k; drop from layer to layer.
   - Missed ground-truth neighbours highlighted with "not reached from the visited frontier" explanation.
5. efSearch support in the Query Explorer and Tuner for HNSW indexes.

**Acceptance:** trace overlap test passes; animation runs smoothly for efSearch up to 256; view works on the synthetic HNSWFlat index.

### Phase 10: Quantization (PQ / SQ) view
1. `core/pq.py`:
   - Requires raw vectors; otherwise the view shows "Provide --vectors to measure quantization error."
   - `make_direct_map()` on the IVF, reconstruct stored vectors in batches, compute per-vector error `‖x − x̂‖²` (and relative error).
   - Aggregate: histogram, per-list mean error, worst-50 ids.
   - Distance distortion: for sampled pairs, true vs approximate distance scatter.
2. **Quantization view**: error histogram, per-list error bar chart (sortable), true-vs-approx distance scatter with y=x line, worst vectors table with metadata text.
3. In Query Explorer, mark `QUANTIZATION` misses with a link to this view.

**Acceptance:** error is ~0 for IVFFlat, clearly > 0 for IVFPQ, lower for IVF-SQ8 than IVFPQ(m=8) in synthetic tests.

### Phase 11: RAG demo dataset & polish
1. `examples/make_rag_demo.py`:
   - Take ~20k passages from a public, permissively licensed text dataset (e.g. from Hugging Face `datasets`; choose one and record the licence in the README).
   - Chunk (≈200 tokens), embed with `sentence-transformers/all-MiniLM-L6-v2` (normalised, IP metric).
   - Build and save `IVFFlat(nlist=256)`, `IVFPQ(nlist=256, m=48, nbits=8)`, `HNSWFlat(M=32)`, plus `vectors.npy`, `chunks.jsonl`, and ~200 sample queries.
2. UI polish: keyboard shortcut to focus query box, shareable URL state (query + params in URL hash), export results as JSON/CSV, screenshot-friendly layout.
3. Performance pass: profile `/projection` and `/search` on 1M-vector synthetic index; ensure server memory stays reasonable (document limits in README).

**Acceptance:** a newcomer can go from `pip install` to seeing the RAG demo in under 2 minutes following the README.

### Phase 12: Docs, demo deployment, release 0.1.0
1. README:
   - One-line pitch, animated GIF of Query Explorer + HNSW animation, install, 3-command quick start, Python/Jupyter usage, supported index types table, "How it compares" section (Feder, embedding projectors, honest and short), limitations, roadmap, contributing.
2. `deploy/hf-space/Dockerfile`: Python slim, install `vectraudit[all]` from the repo, bake in the RAG demo artifacts, `CMD vectraudit serve ... --host 0.0.0.0 --port 7860 --no-browser`. Include a read-only "demo mode" flag that disables uploads and heavy sweeps beyond a query cap.
3. `release.yml`: on tag `v*`, build frontend, `uv build`, publish via PyPI trusted publishing, create GitHub release with changelog.
4. Tag `v0.1.0`.

**Acceptance:** `pip install "vectraudit[all]"` in a clean venv + README quick start works; HF Space loads and runs a query; PyPI page renders README correctly.

### Phase 13 (optional, after 0.1.0): VS Code extension wrapper
1. Separate folder `vscode-extension/` (TypeScript, `yo code` scaffold).
2. Command + explorer context menu on `*.index` / `*.faiss` files: "Open in vectraudit".
3. Uses the Python interpreter from the VS Code Python extension, checks `vectraudit` is installed (offer `pip install` if not), spawns `vectraudit serve FILE --no-browser --port <free>`, and shows the UI in a webview panel.
4. Kill the process when the panel closes.
5. Publish to VS Code Marketplace and Open VSX.

### Phase 14: Release-check core (FAISS, Qdrant, pgvector)
Goal: the customer-side half of the product. A team runs `vectraudit run` against a baseline
and a candidate retrieval setup in its own CI, then `vectraudit check` decides pass / fail /
error / not comparable under a policy and writes a report. No hosted service yet.
New code lives in `vectraudit.connectors` (backend access) and `vectraudit.evaluation`
(datasets, runs, metrics, comparison, policy, reports); both are strict-typed and web-free.

1. **Dataset format.** JSONL, one query per line: `query_id` (string, unique), `vector`,
   optional `text`, `relevant` (`{doc_id: grade}`, grades >= 0), `filter` (portable:
   field -> scalar equality, AND-ed), `tags` (cohorts). Validation with line numbers; the
   dataset version is the SHA-256 of its canonical form, so any edit is a new version.
2. **Connector interface + FAISS connector.** `Connector` protocol: `describe()` (backend,
   versions, metric, dimension, count, configuration, snapshot id), `capabilities()`
   (search, filters, exact reference, config inspection, query plans, tuning params; each
   declared separately), `search(vector, k, filter, params, exact)` returning ranked
   user-facing ids as strings. Errors are typed (`config`, `connection`, `auth`, `timeout`,
   `unsupported`, `query`) and never look like empty results. Connection files are JSON;
   secrets only through `*_env` references. FAISS: index file + optional raw vectors for the
   exact reference; filters unsupported.
3. **Runner + run format.** Execute a dataset against a connection: warm-up, repeats,
   per-query latency, exact reference when available, per-query errors kept apart from
   results, status `complete | partial | failed | cancelled` (Ctrl-C writes a cancelled run).
   Run JSON stores dataset hash, backend description, capabilities, k, params, timing
   protocol, metric contract version, provenance (commit SHA, CI run, runner environment,
   user labels such as embedding revision and chunking version) and per-query ranks and
   metrics. Semantic metrics (recall, precision, MRR, nDCG@k) only with labels; ANN recall
   only with an exact reference, marked separately.
4. **Comparison + policy.** Refuse with reasons when dataset hash, k, metric contract or
   query set differ; incomplete runs give `error`. Paired per-query deltas, seeded bootstrap
   95% intervals, cohorts by tag, latency p50/p95 deltas with an environment-mismatch
   warning. JSON policy: per-metric max drop / minimum, latency limits, minimum queries,
   max regressed queries, max error rate, cohort rules. Verdict `pass | fail | error |
   not_comparable` with every reason listed.
5. **Reports.** Self-contained escaped HTML (no scripts), JSON, and a Markdown summary for
   `$GITHUB_STEP_SUMMARY`: verdict, reasons, aggregate table with intervals, worst regressed
   queries with rank changes, cohorts, configuration and provenance diff.
6. **CLI + CI.** `vectraudit dataset validate`, `vectraudit run`, `vectraudit check`
   (exit 0 pass, 1 error, 3 fail, 4 not comparable, matching `sweep --baseline`). Example
   GitHub Actions workflow and a README section.
7. **Qdrant connector** (`vectraudit[qdrant]`): named vectors, payload filters, `hnsw_ef`,
   exact search reference, collection config. Tests use qdrant-client's local mode.
8. **pgvector connector** (`vectraudit[pgvector]`): table, id and vector columns; cosine,
   L2, inner product; read-only transactions with `statement_timeout`; validated
   identifiers; `hnsw.ef_search` / `ivfflat.probes`; exact reference by disabling index
   scans; index definitions and extension version. Integration tests against
   `pgvector/pgvector` (Docker locally, a service container in CI), skipped without a DSN.
9. **End-to-end acceptance.** One labelled fixture on every backend: baseline passes against
   itself; a controlled quality regression fails; a latency regression fails; a broken
   connection gives error; a different dataset gives not comparable.

**Acceptance:** all of step 9 passes in CI; `vectraudit check` exit codes and reports match
the fixture; no secret appears in run files or logs.

### Phase 15: Investigation evidence, diagnoses, GitHub checks, dataset tooling
Goal: make a failing check explain itself and land where engineers look (the pull request),
and make building a labelled dataset easy enough for a two-week pilot. Still customer-side
only; the hosted control plane comes after this.

1. **Backend evidence per query.** Optional `Connector.evidence(request, found, exact)`:
   FAISS IVF explains each exact neighbour the search missed (list not probed vs ranked out
   by quantization), FAISS HNSW reports efSearch; pgvector records the query plan (scan
   types, index used, cost) via EXPLAIN without ANALYZE; Qdrant reports the indexing state.
   `record --evidence` (default on for regressed-query investigation: stored per query, kept
   small). Evidence is labelled as measured or replayed/simulated.
2. **Diagnoses.** `check` derives findings from the comparison, each with an evidence level
   (`verified`: directly measured; `supported`: consistent with a measured experiment;
   `hypothesis`) and links to the facts behind it. Rules include: approximation loss
   (ANN recall fell with config change), relevance change without approximation loss
   (embedding / corpus / chunking change), filter-related losses, plan changes (index ->
   sequential scan), latency attributable to search parameters, nondeterminism, corpus
   drift. Shown first in reports; never stated as certain without measurement.
3. **GitHub check runs and PR comments.** `check --github-check` creates a check run on
   the pull request head commit with the verdict, summary and regressed queries using the
   workflow's `GITHUB_TOKEN` (`checks: write`); `--github-comment` upserts one PR comment.
   Error, fail and not comparable all conclude as failure; fork pull requests without
   write tokens degrade to the job summary with a warning, never an error.
4. **Dataset tooling.** `dataset import` from CSV or JSONL with column mapping (query id,
   text, relevant ids with optional grades, tags, filter columns); `dataset embed` adds
   vectors with a sentence-transformers model and records the model in the dataset;
   `dataset diff` lists added, removed and changed queries between versions;
   `dataset stats` reports label coverage, grade distribution and per-tag counts with
   warnings (too few labels, cohorts below the policy minimum).
5. **Pilot kit.** `vectraudit init` scaffolds `retrieval/` (connection, policy, dataset
   template, workflow) for a chosen backend; a "first check in 15 minutes" guide; a
   reproducible before/after investigation demo (`examples/release_check_demo.py`) that
   produces a real report for sales and docs.

**Acceptance:** a regressed FAISS IVF candidate's report attributes lost neighbours to
unprobed lists with `verified` evidence; a pgvector candidate whose plan stops using the
index is diagnosed; a check run payload is produced and posted against a mocked GitHub API;
CSV import + embed + record + check works end to end from a fresh `vectraudit init`.

### Phase 16: Hosted control plane (provider-neutral)
Goal: the hosted half of the paid MVP (PRD FR01, FR05, FR07-FR10): workspaces and roles,
runner tokens, run upload, hosted baselines and checks, history, private reports, quotas.
Provider-neutral: one Docker image, any PostgreSQL. New code lives in `vectraudit.cloud`
(FastAPI app, SQLAlchemy models, Alembic migrations), separate from the local UI server and
installed with the `vectraudit[cloud]` extra. Customer credentials never reach it; runners
upload ids, metrics and timings only (query text and vectors are not in run files).

1. **Service skeleton.** Settings from environment, SQLAlchemy 2 models, Alembic migrations,
   app factory, health and readiness endpoints, structured `{error_code, message, hint}`
   errors, request size limits. Tests run on SQLite and on PostgreSQL when a DSN is set.
2. **Workspaces, members, tokens, audit.** Users, workspaces, memberships (owner, engineer,
   viewer), projects. Personal API tokens and project-scoped runner tokens, stored as
   SHA-256 hashes with a visible prefix, revocable, last-used tracking. Every query is
   scoped by workspace in one repository layer; audit events for membership, token and
   baseline changes. An admin CLI (`vectraudit-cloud`) bootstraps the first owner.
   Cross-tenant negative tests for every object type and endpoint.
3. **Run upload.** `POST /api/v1/projects/{id}/runs` accepts a run file validated with
   `check_run`, with a SHA-256 content digest header; idempotent on run id (same content:
   200, different content: 409). Size and per-run query limits. Client: `vectraudit record
   --upload` and `vectraudit cloud push`.
4. **Hosted baselines, policies and checks.** Pin a baseline per project and branch name
   (owner or engineer; audited, never automatic); store the project policy; server-side
   `POST /checks` compares a candidate run with the pinned baseline and stores the result.
   Client: `vectraudit check --baseline cloud:main`; failed/partial runs can never become
   baselines.
5. **History and private reports.** Search runs and checks by commit, dataset version,
   backend, status and label, paginated. HTML reports rendered from stored results behind
   workspace auth, plus signed expiring share links (off by default).
6. **Dashboard.** Server-rendered pages (no tokens in browser storage): projects, run
   history, checks with verdicts and findings, baselines with their audit trail, tokens.
   Browser sign-in through a pluggable identity provider (OIDC) with secure session cookies.
7. **Quotas, retention and usage.** Per-workspace plan limits (runs per month, queries per
   run, stored runs, concurrent uploads), 30-day default retention with a purge command,
   usage export (CSV) for manual pilot invoicing.
8. **Operations.** Dockerfile and docker-compose (API + PostgreSQL), configuration and
   backup/restore docs, structured logs, readiness that checks the database and migrations.

**Acceptance:** a runner token uploads two runs, a pinned baseline and a server-side check
produce the same verdict as the CLI; every endpoint and object type rejects other
workspaces' credentials; a viewer can read but not pin baselines; revoked tokens fail;
retries never duplicate runs or checks.

### Phase 17: Security hardening before the first paid customer
Goal: the PRD's launch security requirements for the hosted service, plus the account
management a pilot needs, before real customer data arrives.

1. **Threat model and route audit.** `docs/threat-model.md` (assets, actors, trust
   boundaries, threats and mitigations, residual risks). An automated test enumerates every
   route with a `{workspace}` parameter and proves another tenant's token and session get
   404 (or 401/403), so new routes can't skip authorization unnoticed.
2. **Brute-force protection.** Rate limits on sign-in, token authentication failures and
   the GitHub callback: per client IP and per token prefix, sliding window, 429 with
   `Retry-After`; failed attempts are logged without secrets.
3. **Deletion and export.** Owners export a workspace (JSON: projects, runs, baselines,
   checks, audit log) and delete a project or the whole workspace (typed confirmation;
   audited outside the deleted workspace). Operators delete a user (memberships, tokens,
   sessions). Expired sessions and old revoked tokens are purged.
4. **Members and tokens in the dashboard.** Owners invite members, change roles and
   remove them; every user manages their personal tokens on a settings page. CSRF and
   last-owner rules as in the API.
5. **Supply chain.** `pip-audit` and `npm audit` in CI against the locked dependencies;
   documented process for advisories.
6. **Incident response.** `docs/incident-response.md`: severity levels, containment
   (token and session revocation, secret rotation), customer notification, restore from
   backup, post-incident review; operator commands to revoke all tokens/sessions of a
   workspace.

**Acceptance:** the route audit covers every workspace route; brute-force attempts hit 429;
a deleted workspace leaves no rows behind; exports contain everything the owner can see and
nothing else; CI fails on known-vulnerable dependencies.

### Phase 18: Embedding and chunking experiments (PRD FR12)
Goal: compare several candidate retrieval setups (embedding model, chunking, index) on one
labelled query set, fairly, in one report, and hand the winner to release checks.
New code lives in `vectraudit.experiments` (plain library, strict types).

1. **Corpus and chunking.** A corpus is JSONL source documents (`doc_id`, `text`, optional
   metadata). Chunkers (`words` with size/overlap, `paragraphs` merged up to a size) emit
   chunks with their source `doc_id` and character offsets, so labels survive re-chunking.
2. **Embedders.** A dependency-free `hashing` embedder (lexical baseline, also used in
   tests) and `sentence-transformers` models with query/document prefixes, batching and a
   disk cache keyed by model, chunking and corpus hash.
3. **Labels across chunkings.** Labels stay on source documents (`relevant`) or text spans
   (new optional `relevant_spans`). Document-level evaluation maps retrieved chunks to
   their documents (first occurrence wins) and searches deeper until k distinct documents;
   span-level evaluation counts a chunk relevant when it overlaps a labelled span. Old
   chunk ids are never reused across chunkings.
4. **Experiment runner.** A JSON manifest lists the corpus, queries, k and candidates. For
   each candidate: chunk, embed documents and queries, build the index (exact by default,
   or a FAISS factory string with search parameters and a per-candidate exact reference),
   search, and write a standard run record (ids are document ids; backend `experiment`
   with the candidate's configuration), so any candidate can be compared, uploaded and
   pinned like any other run.
5. **Comparison report.** All candidates ranked on nDCG/recall/MRR with paired bootstrap
   intervals against the leader, cohort winners, and costs (chunks, dimensions, index
   size, embedding time, query p50/p95); pairwise diagnoses reuse `compare_runs`. HTML,
   JSON and Markdown. CLI: `vectraudit experiment run MANIFEST --out DIR`.
6. **Real demo and case study.** Embedding models and chunk sizes on the RAG corpus from
   the Phase 15 case study, with the hashing baseline.

**Acceptance:** on a fixture where the right answer is known (a lexical-only candidate vs a
semantic one; small vs large chunks), the report ranks candidates correctly with intervals;
chunking changes are evaluated against document labels, not stale chunk ids; the winner's
run file passes `vectraudit check` against itself and uploads to the cloud.

### Phase 19: Pinecone and Weaviate connectors (PRD FR11)
Goal: certify the two "next release" backends against the same connector contract,
contract tests and reference dataset as FAISS, Qdrant and pgvector.

1. **Pinecone** (no extra; REST API with a pinned version): index by name or host, namespace, metadata
   filters (portable equality filters as `$eq`), API key via `api_key_env`. Black box per
   the PRD: no exact reference, no tuning parameters, no snapshot; describe reports metric,
   dimension, vector counts (index and namespace) and serverless/pod spec. Typed errors for
   auth, missing index, rate limits, timeouts and connection failures.
2. **Weaviate** (`vectraudit[weaviate]`): collection, optional named vector and tenant,
   document id from a property or the object UUID, portable filters as property equality,
   API key via `api_key_env`. Exact reference only for collections with a flat index;
   describe reports the vector index type, distance and HNSW settings.
3. **Contract tests and CI.** Unit tests with recorded client behaviour; integration tests
   against `pinecone-local` and `semitechnologies/weaviate` containers (skipped without
   their URLs), added as CI services; the Phase 14 acceptance suite runs on both.
4. **Docs and scaffolding.** Connection fields in the release-check guide, `vectraudit init
   --backend pinecone|weaviate`, threat-model and changelog updates.

**Acceptance:** the acceptance suite (baseline passes, noisy embeddings fail, broken
connection is an error, different dataset is not comparable, latency regression fails)
passes on Pinecone and Weaviate; unsupported capabilities are reported, never faked.

### Phase 20: Local labelling assistant

1. Candidate pooling from CSV/JSONL questions, corpus, run files and experiment candidates;
   randomized source-document pools in a content-hashed local task.
2. Loopback-only labelling UI with escaped HTML, strict CSP, CSRF protection, atomic
   autosave, grades, skip, keyboard shortcuts and character spans.
3. Named annotators, sampled second passes, binary/graded Cohen kappa and explicit review.
4. Versioned dataset export with provenance, preserving existing dataset hashes.
5. Optional callable and LLM suggestions, disabled by default and human-confirmed.
6. **Deferred:** optional hosted labelling; this phase delivers only the local workflow,
   without creating any hosted text-upload surface.
7. Workflow/privacy guide, first-check integration and three-candidate RAG demo.

Full lint, format, both Python type checks and test gates precede each step commit;
UI changes also require a headless Chrome visual check. Never push.

### Phase 21: Paid-pilot launch readiness

1. **Trustworthy verdicts.** Enforce policy minimums on measured query pairs per
   quality metric, disclose sparse coverage and keep bootstrap memory bounded. Cohort
   metrics below their own minimum are reported without judging.
2. **Offline onboarding check.** `vectraudit doctor` inspects installation, local UI,
   connection files, required secret presence, dataset labels/vectors and policy readiness
   without opening a connector, loading a model or sending data. JSON output and strict
   warning mode support CI; align the product introduction with implemented capabilities.
3. **Safe, usable annotation.** Serialize task writes across processes, reject stale
   saves, preserve saved spans, resume unfinished work, show label readiness and offer
   reviewer resolution plus dataset download in the browser. Cover concurrent tabs,
   conflicts, malformed forms and local-only operation.
4. **Distribution and launch evidence.** Installed-wheel smoke coverage for onboarding
   and labelling, repeatable browser regression checks and a paid-pilot readiness guide
   distinguishing automated checks from customer, operating and commercial evidence.

No new runtime dependencies, hosted corpus upload, deployment, push or merge. Work on
`feat/launch-readiness` created with `--no-track`; verify it still has no upstream at the
end. Full Python gates precede each step commit, with visual/browser checks for UI changes.

---

## Retrieval workbench phases 22–26 (2026-10-05)

Implemented on `feat/retrieval-workbench`, extending the existing release-check and workspace foundations:

- **22:** Text/vector Python and HTTP adapters, explicit stage graph traces, recorded loss analysis,
  client latency, compatible run artifacts and CLI execution.
- **23:** Explicit no-answer labels, permission contexts, forbidden-result gates, cohort coverage
  guidance and backward-compatible dataset serialization. No-answer/permission authoring is JSONL;
  the existing annotation UI remains focused on positive document judgments.
- **24:** BM25/vector RRF and phrase-reranker baselines, custom rerankers through adapters,
  bounded candidate search with latency/cost/memory constraints and one held-out validation.
- **25:** Shared workbench UI, standalone local serving, benchmark duplicates/coverage and
  exploratory human-judgment prioritization. CLI remains the execution/annotation entrypoint.
- **26:** Scheduler-compatible frozen checks, local history, separate drift dimensions,
  project access maps, retention/bulk-export controls, enterprise code/PKCE sign-in,
  upgrade/restore fixture and deployment profiles.

Acceptance artifacts: `tests/test_pipelines.py`, `tests/test_workbench.py`,
`tests/test_cloud_governance.py`, `scripts/workbench_smoke.mjs`, and the runnable
`examples/workbench/` suite. See `docs/retrieval-workbench.md` and
`docs/enterprise-deployment.md` for exact boundaries. Actual identity-provider certification,
production load tests and PostgreSQL restore rehearsals remain deployment acceptance work;
these are not inferred from local tests. No push, deployment or production configuration change.

---

## 9. Testing strategy

- **Unit** (`tests/test_*.py`): every core module on tiny synthetic indexes; tests must run in < 60 s total.
- **Correctness anchors**: recall=1.0 at nprobe=nlist; Σ list sizes = ntotal; HNSW trace overlap ≥ 95%; PQ error ordering (Flat < SQ8 < PQ).
- **API**: `TestClient` covering each endpoint, including error cases (bad id, dimension mismatch, unsupported index).
- **Frontend**: Vitest for API client and pure helpers; one Playwright smoke test (load app → run query → results visible) in CI against the synthetic index.
- **Coverage target**: ≥ 85% on `vectraudit.core`.

---

## 10. Performance budgets

| Operation | Target (20k × 384, laptop CPU) |
|---|---|
| `vectraudit serve` to UI visible | < 3 s (projection may still be computing) |
| PCA projection | < 2 s |
| UMAP projection | < 60 s, cached afterwards |
| Single query with IVF trace + ground truth | < 150 ms |
| Sweep, 200 queries × 8 nprobe values | < 20 s |
| Frontend scatter, 50k points | ≥ 30 fps pan/zoom |

For 1M+ vectors: sampling kicks in, ground-truth index is built lazily, and the README documents RAM needs (raw vectors + flat ground truth ≈ 2 × n × d × 4 bytes).

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| FAISS SWIG API differences across versions | Pin minimum `faiss-cpu>=1.8`; use `faiss.contrib.inspect_tools` where possible; CI tests against latest and the minimum version |
| HNSW trace doesn't match FAISS exactly | Label as reconstructed; enforce overlap test; document differences |
| Huge indexes blow up memory | Sampling, lazy ground truth, `--max-points`, clear RAM guidance |
| UMAP is slow / heavy | PCA default, UMAP optional extra, disk cache |
| PreTransform indexes (OPQ/PCA) confuse the geometry | Always apply the transform to queries; show "transformed space" note in UI |
| Name collision later | Reserve on PyPI in Phase 0 |

---

## 12. Roadmap after 0.1.0 (put in README)

- IVF-HNSW (HNSW coarse quantizer) and IndexIVFFastScan support.
- Compare two indexes side by side (e.g. IVFFlat vs IVFPQ on the same queries).
- Evaluation with labelled query→relevant-chunk pairs (true retrieval relevance, not just ANN recall).
- Integrations: load a FAISS store directly from LangChain / LlamaIndex save folders.
- VS Code extension (Phase 13) if not already done.
