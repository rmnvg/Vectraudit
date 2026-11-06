# Reference outputs from ranx

Metric values computed by [ranx](https://github.com/AmenRa/ranx) 0.3.21 (MIT, commit
`7363db0c35e9`) on seeded synthetic queries, so `tests/test_metric_conformance.py` checks
Vectraudit's metrics against an independent implementation. ranx is not a Vectraudit
dependency: it brings numba, pandas, scipy and more, so it only runs here, in a throwaway
environment. No ranx code is copied; only its outputs are committed.

| File | Contents |
|---|---|
| `metrics.json` | 200 queries: graded labels 0–3, the ranking, and ranx's `recall`, `precision`, `mrr`, `ndcg`, `map` and `hit_rate` at k = 1, 5, 10, 100 |
| `fusion.json` | 40 queries with two or three stages, and ranx's `fuse()` output for `rrf` (k = 20, 60), `isr`, `bordafuse`, and `wsum`, `sum`, `mnz`, `max` under every normalisation |

The queries cover single-relevant queries, rankings shorter than k, unjudged documents and
tied scores. ranx breaks score ties with an unstable sort, so ties are resolved by
`(-score, id)` before ranx sees the run, and ranx gets strictly decreasing scores. Both
sides then score the same ranking (`tied_scores` marks the queries that had ties).

To regenerate (the output is deterministic, so an unchanged script reproduces the files
byte for byte):

```bash
uv venv /tmp/ranx-venv -p 3.12 && VIRTUAL_ENV=/tmp/ranx-venv uv pip install ranx==0.3.21
/tmp/ranx-venv/bin/python -W ignore tests/fixtures/ranx/generate/make_fixtures.py
```
