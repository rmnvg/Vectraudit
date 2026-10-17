# Migration report

Compare your current and proposed retrieval systems on the same labelled benchmark before
a migration: pgvector to Qdrant, a new embedding model, a provider change that forces
re-indexing. The release policy decides, and the report adds the checks a migration needs
because the two systems were built separately.

```bash
vectraudit workbench migration current.run.json proposed.run.json questions.jsonl \
  --policy retrieval/policy.json \
  --current-inventory current-docs.jsonl --proposed-inventory proposed-docs.jsonl \
  --costs costs.json --out migration-report.md --fail-unless-ready
```

Record both runs on the same benchmark first, with both systems as candidates in one
manifest: `vectraudit workbench run pipelines.json --paired` interleaves them so both see
the same conditions. Runs on different benchmarks are refused.

## What it checks

| Check | Needs | Flags |
|---|---|---|
| **Release decision** | `--policy` (otherwise exploratory) | The same verdict as any release check: metrics, cohorts, critical questions, permissions |
| **Corpus equivalence** | Both inventories | Documents missing, added or changed (by `sha256`). Labelled documents missing from the proposed system are a corpus gap, not worse retrieval |
| **ID mapping** | The runs; inventories for the direct check | A system returning IDs that are not in its own inventory. Without inventories: most returned IDs unknown to the labels and to the other system |
| **Filter semantics** | Inventories with `metadata` | Returned documents whose metadata does not satisfy the question's filter (equality, list membership, numeric ranges; dotted fields allowed) |
| **Environment** | The runs | Runner, backend, configuration, corpus version, k, embedder, concurrency, repeats, latency source and order, side by side. Differences are disclosed, not corrected for |
| **Cost** | `--costs` for estimates | Measured spend from each run (unknown when calls are unpriced), beside costs you declare, such as re-indexing or hosting, labelled as estimates |

**Ready to migrate** means the policy passes *and* no concern remains. A missing
inventory is itself a concern, because it leaves corpus equivalence or filter semantics
unverified. `--fail-unless-ready` exits 3 otherwise; `--json` writes everything.

## Inputs

An inventory lists one document per line, exported from each system:

```json
{"id": "kb-42", "sha256": "…", "metadata": {"tenant": "acme", "lang": "en", "year": 2025}}
```

Only `id`, `sha256` and `metadata` are accepted. A line with document text is refused, so
inventories never carry content. Declared costs:

```json
{"current": {"monthly_hosting_usd": 900}, "proposed": {"reindex_usd": 1200, "monthly_hosting_usd": 650}}
```

## Limits

- **No data handling:** the report never copies, re-indexes or reads production data.
- **Latency:** compared only as the policy allows across environments; the report shows
  the differences but does not normalise them.
- **Embeddings:** choose them by relevance on your labels, not by nearest-neighbour recall
  across different embedding spaces.
