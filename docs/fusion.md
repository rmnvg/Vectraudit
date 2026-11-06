# Rank fusion

`vectraudit.core.fusion` combines several rankings of one query (keyword, vector, sparse,
or any named stage) into one. It is a plain library with no server or FAISS dependency, and
the built-in hybrid baseline uses it for its fusion stage.

```python
from vectraudit.core.fusion import fuse

fused = fuse(
    {"keyword": [("doc-3", 12.1), ("doc-9", 8.4)], "vector": [("doc-9", 0.82), ("doc-1", 0.77)]},
    "rrf",
    weights={"keyword": 0.5, "vector": 0.5},
    rrf_k=60,
)
# [("doc-9", 0.0163...), ("doc-3", 0.0081...), ("doc-1", 0.0080...)]
```

## Methods

| Method | Uses | Score of a document |
|---|---|---|
| `rrf` | ranks | Σ w / (rrf_k + rank) |
| `isr` | ranks | (Σ w / rank²) × number of stages returning it |
| `bordafuse` | ranks | Σ w × Borda points: n − rank + 1 over the n candidates; (n − len + 1) / 2 if a stage did not return it |
| `wsum` | scores | Σ w × normalised score (0 where a stage did not return it) |
| `combsum` | scores | Σ normalised score (same as `wsum` with equal weights) |
| `combmnz` | scores | `combsum` × number of stages returning it |
| `combmax` | scores | max over the stages that returned it |

Score methods normalise each stage's scores first, with `norm`:

| `norm` | Value |
|---|---|
| `min_max` (default) | (s − min) / (max − min) |
| `max` | s / max |
| `sum` | (s − min) / (Σs − n·min) |
| `zmuv` | (s − mean) / standard deviation |
| `rank` | 1 − position / n |
| `borda` | 1 − position / candidates, and 0.5 − (n − 1) / (2 · candidates) for unreturned candidates |
| `none` | s |

Denominators are floored at 1e-9, so a stage whose scores are all equal normalises to 0.

## Rules

- Results are ordered by score, then by document ID, so fusion is deterministic.
- Rank methods ignore scores and `norm`. A score method given a hit without a finite score
  raises an error that suggests `rrf` or `bordafuse`.
- `weights` must name every stage, be finite and non-negative, and include at least one
  positive weight. A stage weighted 0 contributes nothing, not even candidates.
- `depth` keeps each input's top `depth` hits before fusing. A document listed twice in
  one ranking keeps its best position.

## In pipelines

The built-in hybrid baseline (`"adapter": {"kind": "hybrid", ...}`) fuses its keyword and
vector stages with `fuse()`, weighting them `1 − vector_weight` and `vector_weight`. Its
candidate `params` choose the method:

| Param | Default | Values |
|---|---|---|
| `fusion` | `"rrf"` | any method above |
| `fusion_k` | `60` | integer 0–10000, used by `rrf` |
| `fusion_norm` | `"min_max"` | any normalisation above, used by score methods |

The defaults reproduce the baseline's earlier output exactly (`tests/test_hybrid_golden.py`).
Once a candidate fuses vectors (`vector_weight` above 0), investigations offer the built-in
fix **switch to weighted score fusion** (`{"fusion": "wsum"}`) for documents that were never
observed or were lost at the fusion stage. Rank fusion ignores how far apart two stages'
scores are, while score fusion lets a document one stage scores far above the rest climb.

A Python pipeline can declare the same measured stage without writing fusion code:

```python
from vectraudit.pipelines.runtime import PythonPipeline, Stage, fusion_stage

pipeline = PythonPipeline(
    [
        Stage("keyword", "keyword", bm25),
        Stage("vector", "vector", dense),
        fusion_stage(
            "fused", ["keyword", "vector"], "wsum", weights={"keyword": 0.4, "vector": 0.6}
        ),
        Stage("final", "truncate", lambda q, p: p["fused"][: q.k], ("fused",)),
    ]
)
```

`fusion_stage` checks its options when the pipeline is built, and the stage is recorded with
kind `fusion` and its parents like any other measured stage.

## Replay: try other fusion settings without re-running

`vectraudit workbench fusion-replay` re-fuses the stage hits a run already recorded
(`queries[].evidence.stages`), so no pipeline call is made:

```bash
vectraudit workbench fusion-replay current.run.json --dataset benchmark.jsonl \
  --methods rrf,wsum,combmnz --grid "vector_weight=0.3,0.5,0.7;fusion_k=20,60" \
  --out replay.json
```

Each configuration is scored with the run's metric contract and compared with the run's
recorded final ranking, question by question, then averaged within families. Intervals are
seeded 95% bootstrap intervals over family deltas, as in a release comparison. Grid keys are
`<input>_weight` (the other of two inputs gets `1 - weight`, as `vector_weight` does in the
baseline), `fusion_k`, `fusion_norm` and `depth`; a setting a method ignores (`fusion_k` for
`wsum`, `fusion_norm` for rank methods) is not repeated. At most 200 configurations.

A replay is a **hypothesis**, never a measurement, and it says where it cannot be trusted.
The summary counts each limit and every question shows its own:

| Limit | What it means |
|---|---|
| Excluded | No stage trace, no labels (or a no-answer question), a failed execution, or a held-out split. Held-out questions are found with the optimiser's identity guard, so a held-out question reused as an exploratory one under a new id is refused. |
| `truncated_inputs` | An input stage recorded fewer than k hits, so deeper candidates may exist that the replay cannot see. |
| Downstream `none` | Only the final truncation follows fusion, so the replay is exact. |
| Downstream `filter` | Only filters follow. The result is a bound: *optimistic* keeps every replayed document the recorded filters did not drop, *pessimistic* keeps only documents they are known to have kept. A filter is assumed to judge each document on its own. |
| `downstream_unknown` | A reranker, a custom stage or a branch follows. The replayed ranking is the one before those stages and is not compared with the recorded one. |

A forbidden document in a replayed top-k is reported per configuration and question. Its
absence is never read as safe, because later stages were not replayed.

The report (`vectraudit.fusion_replay`, version 1, evidence level `hypothesis`) holds mean
metrics and paired deltas per configuration, the documents each configuration recovers or
loses against the recorded top-k, and a next step. The next step names `try-fix` only when
the candidate (`--manifest`, `--candidate`) already declares a fix whose params are exactly
the best configuration; otherwise it shows the fix to declare. Parameter names are the
built-in baseline's; map them to your own pipeline's. The best configuration is chosen by
mean delta on exact replays, or on the pessimistic bound for filtered ones. Measure it on the
real pipeline with `try-fix` before believing it.

## Fusion settings in the optimiser

`workbench optimize` can search fusion settings of the built-in hybrid baseline:

```bash
vectraudit workbench optimize manifest.json \
  --fusion-grid "fusion=rrf,wsum;vector_weight=0.3,0.5,0.7" \
  --policy retrieval/policy.json --baseline current
```

Each grid point becomes a candidate named `<base>+fusion=wsum,vector_weight=0.3`, with the
base candidate's adapter and params plus the grid's overrides (its declared fixes are not
copied). The base is `--grid-base`, else `--baseline`, else the manifest's only candidate,
and must be a hybrid candidate: any other pipeline would ignore these params and the points
would silently be identical. Keys are `fusion`, `fusion_k`, `fusion_norm` and
`vector_weight`. A setting the effective method ignores is not repeated, and a point equal
to the base is dropped. The grid plus the manifest's candidates must fit `max_candidates`,
or the command stops and says by how many.

Generated candidates are ordinary candidates: they run on exploratory questions only, and
only the chosen candidate is validated, once, on held-out questions (plus the baseline when a
policy is given), exactly as without a grid. The report gains a `fusion_grid` entry.

`--shortlist-from replay.json --top 3` keeps only the grid points among a fusion replay's
three best configurations, so the real runs go where the recorded stage hits already point.
The replay must be of the manifest's dataset and cover the same settings, or the command
stops. The report records which replay chose the points. A replay is a hypothesis: it saves
runs, and the held-out check still decides.

## Conformance

Every method and normalisation is checked against ranx 0.3.21's `fuse()` on 40 seeded
queries with two or three stages, including negative scores and single-hit stages
(`tests/test_fusion.py`, fixture in `tests/fixtures/ranx/`). There is one deliberate
difference. ranx's CombMAX gives a document a stage did not return a score of −1e9, and
that placeholder can win when every real normalised score is lower (all-negative scores
under `max` normalisation). Vectraudit takes the maximum only over the stages that returned
the document.

## Credits

The methods and normalisations follow [ranx](https://github.com/AmenRa/ranx) (Elias
Bassani, MIT), reimplemented without its code. RRF: Cormack, Clarke and Büttcher,
"Reciprocal rank fusion outperforms Condorcet and individual rank learning methods", SIGIR
2009. CombSUM, CombMNZ, CombMAX: Fox and Shaw, "Combination of multiple searches", TREC-2,
1994. Borda fusion: Aslam and Montague, "Models for metasearch", SIGIR 2001. ISR: Mourão,
Martins and Magalhães, "Multimodal medical information retrieval with unsupervised rank
fusion", Computerized Medical Imaging and Graphics, 2015.
