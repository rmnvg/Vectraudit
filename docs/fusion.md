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
