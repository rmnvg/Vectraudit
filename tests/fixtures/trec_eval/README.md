# Reference outputs from trec_eval

Metric values from [pytrec_eval](https://github.com/cvangysel/pytrec_eval) 0.5 (MIT; it wraps
the NIST `trec_eval`, which is licensed separately and is **not** copied here) for the same 200
seeded queries as `tests/fixtures/ranx/metrics.json`, so `tests/test_metric_conformance.py`
checks Vectraudit's metrics against a second, independent implementation. pytrec_eval is not a
Vectraudit dependency; only its outputs are committed.

`metrics.json` holds, per query id, `recall`, `precision`, `ndcg`, `map`, `hit_rate` and `mrr`
at k = 1, 5, 10, 100, with `source_alignment_sha256` pinning the labels and rankings it scored
(the ranx fixture's), so the two files cannot drift apart.

Mapping (relevance level 1: a label of 1 or more is relevant; graded labels are the gains for
nDCG): `recall` = `recall_k`, `precision` = `P_k`, `ndcg` = `ndcg_cut_k`, `map` = `map_cut_k`,
`hit_rate` = `success_k`, and `mrr` = `recip_rank` on the ranking cut to k. Scores handed to
trec_eval are strictly decreasing, so its tie-breaking (by document id, descending) never applies.

To regenerate (deterministic):

```bash
uv venv /tmp/trec-venv -p 3.12 && VIRTUAL_ENV=/tmp/trec-venv uv pip install pytrec_eval numpy
/tmp/trec-venv/bin/python tests/fixtures/trec_eval/generate/make_fixtures.py
```
