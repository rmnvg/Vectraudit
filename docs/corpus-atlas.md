# Corpus atlas

A 2D map of the corpus and the benchmark questions, for deciding where to look. Each point is
categorised from your run records, so the picture shows what happened (not a guess about what
might have): which relevant documents were lost and where, which questions failed or
regressed, which documents are hubs or were flagged.

```bash
vectraudit workbench atlas corpus.jsonl benchmark.jsonl --out runs/ --name release \
  --run candidate.json --baseline main.json \
  --hubness hubness.json --scan scan.json --poison-suite suite/poison-suite.json
vectraudit workbench serve runs/   # the atlas is served with the other artifacts
```

The corpus is the JSONL the hybrid baseline already reads (`doc_id`, `text`, `metadata`,
`vector`). Documents or questions without vectors need `--embedder <model>` (the benchmark's
own embedding model, `vectraudit[text]`); nothing is embedded otherwise.

## Categories

| Documents | Meaning |
|---|---|
| Document | no finding |
| Relevant | labelled relevant to a question |
| Lost: never retrieved | relevant, and in no recorded stage of the question that needed it |
| Lost: retrieved, then dropped | relevant, in a recorded stage, missing from the delivered results |
| Hub | retrieved for many questions (a hubness report) |
| Flagged | named by the corpus scan or a poison suite |

| Questions | Meaning |
|---|---|
| Passed | every relevant document delivered |
| Failed | a relevant document missing, or the question errored (never counted as passed) |
| Regressed | worse than in `--baseline` |
| No answer | expects no answer |
| Not judged | no run entry, no label, or held out |

A document with several findings takes the first of: flagged, lost (never retrieved before
dropped), hub, relevant. Categories use `--run`; without it nothing is called lost or failed.

## Held-out questions

Held-out questions are not drawn unless you pass `--show-heldout`, and then they are one
uncoloured category. They never colour themselves, or any document, by result: the map cannot
be used to tune against the held-out set.

## Sampling and size

The cap (`--max-points`, default 50,000) bounds plain, unlabelled documents only. Every
question and every document in another category is always kept, so a failure is never sampled
out, and the cap can be exceeded when there are more notable points than that (the sidecar says
so). Sampling is seeded (`--seed`). Going above 50,000 needs `--large-atlas` (up to
1,000,000); above 200,000 points the sidecar omits titles to stay servable.

## Coverage regions

`vectraudit workbench benchmark benchmark.jsonl --corpus corpus.jsonl` adds
`embedding_coverage` to the audit, and the atlas draws the same regions. Documents are grouped
by seeded k-means on the **full-dimensional** vectors (about the square root of the corpus size,
at most 48 groups), never on the 2D map. Each region reports its documents, the questions whose
nearest region it is, the documents labelled relevant to a question, and its top terms (TF-IDF
over local titles and first lines).

A group of regions becomes a finding, at evidence level `hypothesis`, when all of these hold:

- no question's nearest region is any of them;
- their documents are far from every question: the median distance to the nearest question is
  more than twice that of the regions that do have a question;
- together (neighbouring regions are merged, since k-means cuts one topic into several pieces)
  they hold at least 5% of the corpus.

For example: "About 16% of documents form a group no benchmark question is close to (about:
refund, policy)." Only exploratory questions count: a held-out question near the group does not
cover it. A finding says the benchmark may not ask about that part of the corpus; it does not
say the benchmark is wrong, and a question can sit far from its answers in an embedding space.
With no exploratory question vectors, coverage is reported as not measured.

## Files

`atlas-<name>.json` (format `vectraudit.atlas` v1) holds identifiers, titles, the legend, counts
and the SHA-256 of `atlas-<name>.bin`, which holds little-endian `float32 x[n]`,
`float32 y[n]`, `uint8 category[n]` (padded to 4 bytes) and `uint32 row[n]` (the row in the
corpus or dataset). The workbench serves the binary at `/api/workbench/atlas/<name>`.

The layout is PCA, fitted on the whole corpus; questions are placed by the same projection. **A
2D layout distorts distance**: points that look close may not be. Treat the map as a pointer.
The sidecar contains document titles from your corpus, so it is marked `local_only` and is never
part of an upload.
