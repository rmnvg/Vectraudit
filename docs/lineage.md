# Source-to-context lineage

A required passage can be lost long before retrieval: extraction drops a table, a chunk
boundary splits the clause that mattered, or the index still holds an older revision. A
different embedding model fixes none of these. `vectraudit workbench lineage` follows each
labelled passage from source to model context and names the **first stage that lost it**.

## What you provide

1. **A benchmark with passage labels** (`relevant_spans`: source ID and character range in
   the extracted text), as for chunking experiments.
2. **A lineage manifest** exported from your ingestion pipeline: one JSON object per
   source document.

```json
{"source_id": "refund-policy", "revision": "r8", "extracted_chars": 1840,
 "chunks": [{"chunk_id": "refund-policy#0", "start": 0, "end": 900, "indexed_revision": "r8"},
            {"chunk_id": "refund-policy#1", "start": 900, "end": 1840, "indexed_revision": "r8",
             "text_sha256": "…"}]}
```

| Field | Meaning |
|---|---|
| `revision` | The source's current revision. Omit it if you do not track revisions; staleness is then reported as unverified, never as current. |
| `extracted_chars` | Length of the extracted text. Span and chunk offsets refer to this text. |
| `chunks[].start`, `end` | The chunk's character range in the extracted text. |
| `chunks[].indexed` | `false` for a chunk that exists but is not in the index (default `true`). |
| `chunks[].indexed_revision` | The source revision the chunk was indexed from. |
| `chunks[].text_sha256` | Optional; identical hashes mark duplicate chunks. |

The manifest holds offsets, IDs, revisions and hashes, never document text. Vectraudit does
not re-run your ingestion; findings are only as complete as the export.

3. **Optionally, a run** of your pipeline, to continue past the index. `--ids chunk` (the
   default) means the run's result IDs are the manifest's chunk IDs; `--ids source` means
   they are source document IDs.

```bash
vectraudit workbench lineage lineage.jsonl questions.jsonl --run current.run.json --json lineage.json
```

## First-loss stages, in order

| Stage | Meaning |
|---|---|
| `source_missing` | The source is not in the export |
| `empty_extraction` | Extraction produced no text |
| `beyond_extraction` | The passage lies past the end of the extracted text |
| `not_chunked` | Part of the passage is in no chunk |
| `split_across_chunks` | No single chunk holds the whole passage |
| `not_indexed` | The chunk holding it is not indexed |
| `stale_index` | That chunk was indexed from an older revision |
| `not_retrieved` | Indexed intact, but in no recorded retrieval stage |
| `lost_downstream` | Retrieved, then dropped before delivery |
| `lost_in_context` | Delivered, but left out of or cut short in the model context |

The first seven are ingestion losses. When any are found, the report says to fix ingestion
before changing retrieval, and `--fail-on-ingestion` exits 3 so CI can block on them. The
report also lists corpus problems visible in the export alone: sources with empty extraction,
chunks from an older revision still indexed beside current ones (a stale answer waiting to
happen), and identical chunks, with the questions where duplicates took result slots.

## Limits

- Offsets must refer to the same extracted text in the benchmark and the export.
- A run without stage traces cannot distinguish "not retrieved" from "dropped before
  delivery"; the report says so.
- Freshness over time (an update reaching the index within a propagation window) needs
  repeated exports; this check reads one export.
