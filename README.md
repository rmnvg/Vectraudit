# Vectraudit

**Retrieval evaluation and release checks for RAG and search. Test your pipeline. Explain
failures. Measure fixes. Guard every release.**

[![CI](https://github.com/rmnvg/Vectraudit/actions/workflows/ci.yml/badge.svg)](https://github.com/rmnvg/Vectraudit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Vectraudit tells you whether a change to your retrieval system made it better or worse,
and why. It measures a baseline and a candidate on the same versioned, labelled questions,
across the whole pipeline: query handling, keyword and vector retrieval, filters, fusion,
reranking, context packing and the answer. It traces where each relevant document was lost,
checks permission and no-answer behaviour, and backs every decision with query-level
evidence. It works with eight vector backends, LangChain and LlamaIndex retrievers, any
Python or HTTP pipeline, and traces from the observability tools you already run. Evaluation
runs on your infrastructure; only run files are uploaded, and only if you choose to.

## What it does

| You want to | Vectraudit gives you | Start with |
|---|---|---|
| **Build and maintain a benchmark** | Guided labelling with grades, no-answer and permission cases, held-out splits and frozen versions; imports from BEIR, TREC, MTCB, Ragas, DeepEval, LangSmith and Langfuse; a label audit that finds labels likely to be wrong | [`workbench serve`](#retrieval-workbench), [`label`](docs/labelling.md), [`dataset import`](docs/release-checks.md#bringing-labels-you-already-have) |
| **Evaluate a whole pipeline** | Stage-by-stage traces, the stage where each document was lost, a per-component scoreboard, side-by-side comparison of a fix | [`workbench run`](#retrieval-workbench) |
| **Measure what is in production** | Runs built from OpenInference, OpenTelemetry, Phoenix, Langfuse, LangSmith, Opik and OpenObserve traces; answer judgments from your evaluation tools; a review queue of new questions; question, embedding and score drift | [`import-traces`, `intake`, `drift`](#from-production-traces-and-the-tools-you-already-use) |
| **Block regressions in CI** | A pass, fail, error or not-comparable verdict under a release policy, with paired significance tests, cohort rules, critical questions, and HTML and GitHub reports | [`init`, `record`, `check`](#release-checks) |
| **Explain a failure** | Investigation of one question, declared fixes tried and validated on held-out data, lineage and freshness, permission lifecycle, fusion replay, claim-level evidence | [Investigate deeper](#investigate-deeper) |
| **Choose a setup** | Embedding models, chunkers and indexes compared on one labelled set; recall under filters; capacity under concurrent load; migration reports | [`experiment`](docs/experiments.md), [`capacity`](docs/capacity.md), [`migration`](docs/migration-report.md) |
| **Check safety** | Permission and tenant assertions, cache-leak probes, corpus scans for injected instructions, poisoning checks | [Permissions](docs/permission-lifecycle.md), [poisoning](docs/poisoning.md) |
| **Tune a search setting on any backend** | The cheapest `hnsw_ef`, `ef_search`, `num_candidates` or `probes` that reaches your recall target against an exact reference, the neighbours each query missed, filtered and batch curves, two setups compared | [`tune`, `tune-compare`](docs/tuning.md) |
| **See inside a FAISS index** | Missed neighbours and why, quantization loss, HNSW search replay, recall against latency | [`serve INDEX`](#faiss-index-debugger) |
| **Share results with a team** | Pinned baselines, history, reports and pull-request checks on a control plane you host yourself | [Vectraudit Cloud](docs/cloud-operations.md) |

## Install

Download the wheel from the [latest GitHub release](https://github.com/rmnvg/Vectraudit/releases)
(it includes the built web UI; check it against the release's `SHA256SUMS`), or work from
a checkout with `uv sync --all-extras` and prefix the commands below with `uv run`:

```bash
pip install "./vectraudit-0.1.0-py3-none-any.whl[all]"
```

Extras install only what you use: the backends `[faiss-cpu]`, `[qdrant]`, `[pgvector]` and
`[weaviate]` (Pinecone, Elasticsearch, OpenSearch and Chroma need none), `[text]` for
sentence-transformers text queries, `[umap]` for the cluster map, `[parquet]` for Parquet
inputs, `[hdf5]` for ann-benchmarks datasets and `[cloud]` for the self-hosted control plane.
`[all]` includes all of these except the optional LLM labelling helper (`[label-llm]`).
Python 3.10 to 3.13 is tested in CI.

Nothing runs from the open-source projects Vectraudit learns from (ranx, trec_eval, MTEB,
RAGChecker and others): the methods are reimplemented and credited in
[docs/credits.md](docs/credits.md), and no dataset is ever downloaded for you.

## Retrieval workbench

Test the whole pipeline, not just the index: query rewriting, hybrid retrieval, filters,
rerankers and permissions. The workbench runs locally and opens on a guided workflow:

```bash
vectraudit workbench serve workbench-runs   # opens http://127.0.0.1:8766
```

1. **Connect a pipeline:** the built-in BM25/hybrid baseline over a JSONL corpus, your
   existing [LangChain or LlamaIndex retriever](docs/retrieval-workbench.md#langchain-and-llamaindex-retrievers),
   or Python and HTTP adapters registered at startup with `--manifest pipelines.json`.
2. **Build a benchmark:** questions, relevance grades, explicit no-answer judgments, tenant
   context, filters, forbidden documents, cohort tags and an exploratory/held-out split.
   Read local passages and click grades, preview CSV/JSON imports, or build permission
   scenarios. Freeze versions and review labels when documents change.
3. **Evaluate and inspect:** a cancellable local run records each stage's results.
   Follow any document through every stage to the one where it was lost.
4. **Compare a fix:** run a candidate configuration, get a release verdict, and step
   through both runs' traces side by side. Track a finding, record investigation notes,
   and verify a later fix against the same comparison suite.

Use **Use starter example** for a first run in under a minute. Held-out queries stay out of
exploratory runs and need explicit confirmation, so tuning doesn't contaminate them.

![Guided workflow comparing a baseline and a reranked candidate with side-by-side stage traces](docs/screenshots/guided-workflow.png)

The same engine runs from the command line for automation: bounded optimization with one
held-out validation, benchmark audits (duplicates, missing categories, judgment queues)
and scheduled monitoring against a frozen baseline.

```bash
vectraudit workbench run examples/workbench/manifest.json --out workbench-runs
vectraudit workbench optimize examples/workbench/manifest.json --out workbench-runs/optimization.json
vectraudit workbench benchmark examples/workbench/queries.jsonl --runs workbench-runs
vectraudit workbench monitor manifest.json baseline.run.json
```

Evaluations save query progress locally. Resume from **Saved progress** in the browser
or use `workbench run --resume CHECKPOINT_DIRECTORY`; failed-query retries are explicit.
See [recovery details](docs/retrieval-workbench.md#resume-an-evaluation).

![Workbench run view: summary, the component scoreboard and stage evidence for one question](docs/screenshots/workbench.png)

<table>
<tr>
<td><img alt="Checks tab: a label audit listing two labels to review, an embedding drift report naming a new topic, and a failed poisoning check" src="docs/screenshots/checks.png"></td>
<td><img alt="Corpus atlas: a map of 20,000 documents with a coverage hypothesis that 12% of documents form a group no benchmark question is close to" src="docs/screenshots/atlas.png"></td>
</tr>
</table>

![Component scoreboard: 71.4% of question families first lost their evidence at the reranker, two questions not observed](docs/screenshots/component-scoreboard.png)

See the [workbench guide](docs/retrieval-workbench.md) for adapter manifests, stage
evidence, no-answer and permission assertions, and data-handling boundaries.

### From production traces and the tools you already use

An application that is already traced can be measured without writing an adapter, and
answer grades from your existing evaluation tools can be attached to the same questions.

```bash
vectraudit workbench phoenix-pull --since 2026-10-19T00:00:00Z --until 2026-10-20T00:00:00Z \
  --traces spans.jsonl --judgments judgments.jsonl          # or langfuse-, langsmith-, opik-pull
vectraudit workbench import-traces benchmark.jsonl spans.jsonl --id-field doc_id --out current.run.json
vectraudit workbench attach-answers current.run.json judgments.jsonl --out current.judged.run.json
vectraudit workbench phoenix-push current.judged.run.json --send   # or langfuse-push
```

- **Trace formats** ([import guide](docs/retrieval-workbench.md#import-traces-openinference-langsmith-langfuse)):
  OpenInference, OpenLLMetry and OpenTelemetry GenAI spans as OTLP JSON, Phoenix span
  records, LangSmith runs, Langfuse observations and Opik pulls. Each trace import reports
  what the traces can and cannot show (document IDs, stages, model context, prompt
  versions), and a missing fact is a gap, never a guess. The importers are tested on
  exports written by the real instrumentation libraries and servers.
- **Direct pulls** ([production intake](docs/production-intake.md)): `langfuse-pull`,
  `langsmith-pull`, `opik-pull`, `phoenix-pull` and `openobserve-pull` read a time window
  over each tool's API, with its scores, feedback or annotations as answer judgments.
  Read-only; reasons and explanations are never written out.
- **Answer judgments** from human review or tools: `deepeval-judgments`,
  `promptfoo-judgments` and `ragas-judgments` convert those tools' results. You choose
  which metrics judge answers, retrieval metrics are refused (Vectraudit measures retrieval
  itself), and scores without a verdict need a declared `--threshold`.
- **Diagnosis** ([answer judgments](docs/retrieval-workbench.md#required-evidence-and-answer-judgments)):
  each judged question gets the first observable loss between retrieval and answer
  (`lost_downstream`, `lost_in_context`, `answered_wrong_despite_evidence`, ...). Cited
  documents are placed against what reached the model, and disagreeing judges show as
  disputed. `langfuse-push` and `phoenix-push` write the diagnosis back onto each trace,
  previewing by default.
- **Prompt versions** recorded in traces (OpenInference, Langfuse, LangChain hub, Opik) are
  compared between runs, and a release policy can fail on a changed prompt.
- **Keeping the benchmark current:** `intake` turns each new trace export into a
  de-duplicated review queue and flags likely recurrences. `drift` checks whether
  production questions still look like the benchmark's, in their descriptors and, with
  `--embeddings`, in embedding space; `embedding-drift` compares two windows or two corpus
  exports; and `score-drift` whether the retriever's scores moved between two windows
  ([drift reports](docs/production-intake.md#is-the-benchmark-still-representative)).

### Investigate deeper

| Command | Answers | Guide |
|---|---|---|
| `investigate`, `try-fix` | What did one question lose and where, and does a declared fix recover it? | [Investigate a failure](docs/retrieval-workbench.md#investigate-a-failure-and-try-a-fix) |
| `lineage`, `freshness` | Where was each required passage first lost between source and model context? Did a content change reach the index, results and context? | [Lineage](docs/lineage.md) |
| `access`, `cache-probe` | Do permission changes hold in a later run? Does one context's question warm another's cached answer? | [Permission lifecycle](docs/permission-lifecycle.md) |
| `migration` | Is a new system as good as the old one on the same benchmark, under the release policy? | [Migration report](docs/migration-report.md) |
| `fusion-replay`, `optimize` fusion grid | Would a different way of combining keyword and vector rankings have helped, measured on your questions with paired significance tests? | [Rank fusion](docs/fusion.md) |
| `components` | Where along the pipeline (retrieval, filter, fusion, rerank, cut-off, context, answer) do questions first lose their evidence? | [Component scoreboard](docs/retrieval-workbench.md#component-scoreboard-where-questions-lose-their-evidence) |
| `decision-queue` | Which unjudged documents could still change this release check, and when can judging stop? | [Decision queue](docs/decision-queue.md) |
| `attribute` | A release changed several settings at once: which one caused the regression, which interact, and which single revert recovers the most? | [Attribution](docs/attribution.md) |
| `label-audit`, `review-labels` | Which of the benchmark's relevance labels are likely wrong or missing, so reviewers check those first? | [Label audit](docs/label-audit.md) |
| `review-sample`, `attach-answers --sample` | What is the true rate of correct answers when a model judged every question and people reviewed a random few, with an honest interval? | [Judge-corrected estimates](docs/judge-corrected-estimates.md) |
| `ragchecker-input`, `ragchecker-judgments`, `attach-answers` | Which claims in an answer are unsupported by the context, and which retrieved documents does that point to? | [Claim-level evidence](docs/claim-evidence.md) |
| `workbench atlas`, `workbench benchmark --corpus`, `dataset import-vdbbench`, `dataset import-annb`, `sweep --filter`, `sweep --pass-rates`, `capacity` | How does recall fall as a metadata filter keeps fewer documents? How do throughput, latency and errors change as concurrent load grows on a staging target? | [Corpus atlas](docs/corpus-atlas.md), [VectorDBBench and ann-benchmarks datasets](docs/vectordbbench.md), [Filtered search](docs/filtered-search.md), [Capacity](docs/capacity.md) |
| `corpus-scan`, `poison generate`, `poison run`, `canary-judgments`, `promptfoo-redteam` | Does the corpus hold text that tries to instruct a model? When poisoned documents are present, are they retrieved, do they reach the model, do they push out the right answer, and did the answer follow them? | [Poisoning checks](docs/poisoning.md) |

## Release checks

Measure a baseline and a candidate retrieval setup on the same labelled queries, and let CI
decide: pass, fail, error or not comparable.

```bash
vectraudit init --backend qdrant          # or faiss, pgvector, pinecone, weaviate, elasticsearch, opensearch, chroma
vectraudit doctor --connection retrieval/connection.json --dataset retrieval/dataset.jsonl --policy retrieval/policy.json
vectraudit record --connection main.json --dataset dataset.jsonl --out baseline.json --label main
vectraudit record --connection pr.json   --dataset dataset.jsonl --out candidate.json --label pr-42
vectraudit check baseline.json candidate.json --policy policy.json --html report.html
```

`init` scaffolds a connection, policy, query sheet and GitHub workflow. `doctor` inspects
the setup offline without loading a model or opening a database connection. `check` exits 0
(pass), 1 (error), 3 (fail) or 4 (not comparable), writes the GitHub job summary, and the
HTML report lists the regressed queries with the ids they lost.

Semantic metrics (recall, precision, MRR, nDCG, MAP and hit rate) need relevance labels;
ANN recall needs an exact reference; neither is ever substituted for the other. Failing
checks explain themselves: each finding is marked verified, supported or hypothesis, with
the facts behind it. A policy can also fail on a changed prompt version, and
`vectraudit workbench hubs` reports documents returned for many unrelated questions
([metrics](docs/release-checks.md#metrics), [hubs](docs/release-checks.md#hubs)).
Credentials are referenced by environment variable and never written into config files.

Teams that gate merges on pytest can make the check one more test
([details](docs/release-checks.md#from-a-pytest-suite)):

```python
from vectraudit.testing import assert_release


def test_retrieval_release():
    assert_release("baseline.json", "candidate.json", policy="retrieval/policy.json")
```

| Backend | Connector | Extra |
|---|---|---|
| FAISS | index file | `[faiss-cpu]` |
| Qdrant | collection | `[qdrant]` |
| pgvector | table and columns in PostgreSQL | `[pgvector]` |
| Pinecone | index over HTTPS | none |
| Weaviate | collection | `[weaviate]` |
| Elasticsearch 8 / OpenSearch 2 | index with a `dense_vector` / `knn_vector` field | none |
| Chroma 1.x | collection over the v2 REST API | none |

### Bring what you have, take results where you need them

- **Labels you already have:** `vectraudit dataset import` reads BEIR folders, TREC qrels,
  MTCB chunking benchmarks, Ragas test sets, DeepEval goldens and LangSmith/Langfuse exports
  ([formats](docs/release-checks.md#bringing-labels-you-already-have)).
- **No labels yet:** the [local labelling assistant](docs/labelling.md) (`vectraudit label`)
  pools candidate documents from your questions and corpus, then supports blind second
  passes, agreement and reviewer resolution before export. The
  [label audit](docs/label-audit.md) later points reviewers at labels that look wrong.
- **Other IR tooling:** runs and labels move in and out in the TREC formats that `trec_eval`,
  pytrec_eval, ir-measures and ranx read (`runs export-trec`, `runs import-trec`,
  `dataset export-qrels`). An imported run says it is external and is never judged on latency
  it did not measure. Every quality metric is tested to equal trec_eval's, and reports show
  each metric's ir-measures name (`nDCG@10`, `RR@10`)
  ([details](docs/release-checks.md#moving-runs-and-labels-to-and-from-other-ir-tools-trec)).
- **Your own notebooks:** `vectraudit runs table` writes one row per question and repeat
  (CSV or Parquet, no text), and `runs summary` validates a run of any size in a few MiB of
  memory ([large runs](docs/release-checks.md#large-runs-summaries-and-tables)).
- **Public ANN benchmarks:** `dataset import-vdbbench` and `dataset import-annb` turn a
  VectorDBBench or ann-benchmarks dataset's published neighbours into the reference for ANN
  recall ([details](docs/vectordbbench.md)).
- **Choosing a setup before you guard it:** [experiments](docs/experiments.md)
  (`vectraudit experiment`) compare embedding models, chunkings and indexes on one labelled
  set, from your own corpus, a BEIR folder or an MTCB benchmark. Reports show MTCB's passage
  recall beside span recall, and `experiment export-mteb` writes each candidate in MTEB's
  result layout, marked as not like for like with leaderboard scores. `experiment project`
  projects each candidate's recall from a 10% sample to the whole corpus, with an interval
  ([corpus projection](docs/corpus-projection.md)).

- [Your first release check in 15 minutes](docs/first-check.md)
- [Release-check guide](docs/release-checks.md): dataset, connection and policy formats
- [Setup diagnostics](docs/doctor.md): `vectraudit doctor` for text-only queries and CI
- [Pull-request workflow](examples/github/retrieval-check.yml) with check runs and comments
- [Case study: a latency fix that quietly broke retrieval](docs/case-study-nprobe.md)
- [Case study: choosing an embedding model and chunk size](docs/case-study-experiment.md)

## FAISS index debugger

One of the eight backends gets a deeper tool. For FAISS users, a local UI explains what
happens inside one index: which true neighbours the search missed and why, how much
quantization costs, how balanced the lists are, and the cheapest setting that reaches your
recall target.

```bash
vectraudit demo                       # builds a real RAG demo once (~1 min), opens the UI
vectraudit serve my.index --vectors vectors.npy --meta chunks.jsonl
```

`vectraudit demo` downloads 12,000 public passages, embeds them with `all-MiniLM-L6-v2` and
builds IVF-Flat, IVF-PQ and HNSW indexes. Type a question and see which true neighbours
the index missed, and why.

![Query explorer: recall 0.40 at nprobe 4, five neighbours missed because their cells weren't probed, one lost to quantization; nprobe 25 would find them all](docs/screenshots/query-explorer.png)

| View | What it shows |
|---|---|
| **Query explorer** | Search by text, stored id or vector. Compares with exact ground truth and gives a reason for every missed neighbour: *cell not probed*, *quantization*, *transform* (PCA/OPQ), or *ranked out*, plus the smallest nprobe that finds them all. |
| **Tuner** | Recall@k and latency across nprobe or efSearch, the cheapest value for your target recall, the worst queries, and suggested next steps backed by the measurements. On the command line, `sweep --batch` adds batch throughput, kept apart from single-query latency. |
| **Compare** | Two indexes over the same vectors on identical queries: recall with intervals, mean/p95 latency, serialized size, and which queries' neighbours changed. |
| **HNSW graph** | The layers of an HNSW graph and an animated, step-by-step replay of the search. |
| **Quantization** | How far PQ/SQ codes are from your raw vectors and how much they distort near-neighbour distances. |
| **Overview** | Inverted-list sizes and imbalance, with a **Diagnose my index** guide. |
| **Cluster map** | Stored vectors projected with PCA or UMAP (2D/3D), with IVF centroids. |
| **Relevance** | Recall, MRR and nDCG@k from labelled judgements, before and after exact reranking. |
| **Run history** | Saved sweeps, comparisons and evaluations kept across restarts. |

<details>
<summary>More screenshots and supported index types</summary>

<table>
<tr>
<td><img alt="Tuner: no nprobe reaches 0.95 recall on IVF-PQ; the suggested next step explains that compression caps recall near 0.62 and to re-rank with exact distances" src="docs/screenshots/tuner.png"></td>
<td><img alt="Animated HNSW search: greedy descent through the upper layers, then candidate expansion on level 0" src="docs/screenshots/hnsw-search.png"></td>
</tr>
<tr>
<td><img alt="Quantization: PQ error, compression and distance fidelity" src="docs/screenshots/quantization.png"></td>
<td><img alt="Cluster map with one inverted list selected" src="docs/screenshots/cluster-map.png"></td>
</tr>
</table>

| Index | Support |
|---|---|
| `IndexFlat` (L2 / IP) | search, map, ground truth |
| `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer` | everything: list balance, probe trace, miss reasons, tuner, quantization |
| `IndexHNSWFlat`, `IndexHNSWSQ`/`PQ` | graph view, search trace, tuner (efSearch), quantization for SQ/PQ storage |
| Wrappers: `IndexIDMap`/`IDMap2`, `IndexPreTransform` (PCA, OPQ), `IndexRefine` | unwrapped transparently; ids stay yours |
| Binary, GPU, IVF FastScan, others | basic stats only (convert GPU indexes with `faiss.index_gpu_to_cpu`) |

</details>

From Python or Jupyter (JupyterLab, VS Code notebooks and Colab; see [Jupyter notes](docs/jupyter.md)):

```python
import vectraudit

viewer = vectraudit.launch(index, vectors=xb, metadata=chunks)  # any faiss.Index
report = viewer.session.query(id=42, k=10, nprobe=4)
report.recall, report.ivf_trace.min_nprobe_for_all
```

The [FAISS debugger reference](docs/faiss-debugger.md) covers inputs, labelled relevance,
every command-line option (`serve`, `sweep`, `compare`, `evaluate`, `runs diff`, `cache`),
saved baselines, measurement methodology, caching and limitations.

## Self-hosted control plane and deployment

- [Running Vectraudit Cloud](docs/cloud-operations.md): a control plane you run yourself for
  shared baselines, history, reports and the dashboard (`vectraudit record --upload`,
  `check cloud:main --upload`), with plain-language release summaries and quality trends
  (`cloud summary`)
- [Deployment](docs/deployment.md): local, private-runner and self-hosted profiles, SSO,
  roles and recovery
- [Threat model](docs/threat-model.md) and [incident response](docs/incident-response.md)
- [Release rehearsal](examples/release_demo.py): reproduce an improving-average regression,
  inspect a critical repeat failure, and verify the fix through exported checks
- [Reference support search](examples/reference_search/README.md): measure a real BM25/filter
  regression on fictional documents, compare independent application revisions, and inspect
  the exported reports (proposed labels await human review)

## Development

```bash
git clone https://github.com/rmnvg/Vectraudit && cd Vectraudit
uv sync --all-extras
cd frontend && npm install && npm run build && cd ..
uv run pytest && uv run ruff check . && uv run mypy
uv run python examples/make_synthetic.py && uv run vectraudit serve examples/data/ivf_pq.index --vectors examples/data/vectors.npy
```

Frontend development: run `npm run dev` in `frontend/` next to a running `vectraudit serve`.
The frontend's API types are generated from the server's OpenAPI schema; after changing
`server/schemas.py`, run `uv run python scripts/generate_api_types.py` (CI fails on drift).
After adding or upgrading a frontend runtime dependency, run
`node scripts/third-party-notices.mjs` in `frontend/` to refresh
`src/vectraudit/THIRD_PARTY_NOTICES.txt`, which ships in the wheel (CI fails on drift).
Browser tests run on Chromium, Firefox and WebKit: `npx playwright install chromium firefox
webkit`, build the UI, then `npm run e2e` in `frontend/`. The workbench journeys run with
`node scripts/workbench_smoke.mjs` and `node scripts/guided_smoke.mjs`; the newer flows have
`scripts/atlas_smoke.mjs` (the atlas, needs a GPU-capable Chrome), `scripts/checks_smoke.mjs` (the
check reports in the Checks tab), `scripts/filtered_sweep_smoke.mjs` (the Tuner's pass-rate
card), `scripts/estimates_smoke.mjs` and `scripts/components_smoke.mjs` (the run view's
judge-corrected estimates and component scoreboard) and `scripts/large_run_smoke.mjs` (a run too
large to open). Slow tests, such as the 200 MiB run, are skipped by default: `uv run pytest -m slow`.
Design decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md). Contributions are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [Apache License 2.0](LICENSE); see also [NOTICE](NOTICE). The demo corpus
is "RAG Dataset 12000" by Neural Bridge AI (Apache-2.0), with passages from Falcon RefinedWeb
(ODC-By 1.0).

The methods Vectraudit reimplements, and where they come from, are credited in [docs/credits.md](docs/credits.md).
