# Label audit

A benchmark's relevance labels are measurements too, and some are wrong: a document that answers
the question but was never labelled, or a label on one that does not. Every metric inherits
those errors, and a release check on wrong labels can pass a regression or fail a fix. The label
audit lists the labels most likely to be wrong or missing so a person can check them first.

It **never changes a label**. Each suspect is a hypothesis; a reviewer's decision makes a new
dataset version, and the original file is untouched.

## The round trip

```bash
# 1. Audit: candidate runs on this benchmark version, plus corpus vectors and a labelling task
#    if you have them (each adds a signal).
vectraudit workbench label-audit benchmark.jsonl --runs workbench-runs \
  --corpus corpus.jsonl --labels task.json --out label-audit.json

# 2. Optional: put the suspects in the benchmark's judgment queue, beside unjudged pooled
#    documents and claim-evidence items, ordered by priority.
vectraudit workbench benchmark benchmark.jsonl --runs workbench-runs \
  --label-audit label-audit.json --out benchmark-audit.json

# 3. A reviewer decides each suspect they looked at, and a new version is written.
vectraudit workbench review-labels benchmark.jsonl label-audit.json \
  --decisions decisions.jsonl --reviewer "Dana" --out benchmark.v2.jsonl
```

`decisions.jsonl` holds one line per decided suspect: a new grade from 0 to 3, or `keep`.

```json
{"query_id": "q-114", "doc_id": "kb-2031", "decision": 2}
{"query_id": "q-114", "doc_id": "kb-0457", "decision": "keep"}
```

Every decision, `keep` included, is recorded in the new version's header under
`provenance.label_review` with the reviewer's name, the signals that raised it and the version
audited. A later audit does not raise a decided pair again (`--recheck-reviewed` raises them).
Only suspects from this audit of this exact version can be decided, and a decision that would
break the question is refused: a document in the question's required evidence, or its last
relevant document, needs the question edited instead. A document set to 0 loses its relevant
spans.

## The three signals

| Signal | Needs | Finds | Measured on planted errors |
|---|---|---|---|
| **Rank** (confident learning) | Two or more candidate runs on this benchmark version | A relevant label the runs rank no higher than the question's not-relevant results; an unjudged or 0-graded document they rank as high as the relevant ones | Precision 0.74 to 0.81, recall 0.71 to 0.86 |
| **Near-duplicate** | Corpus vectors (`--corpus`) | Documents at cosine 0.995 or more labelled differently for one question | Precision 0.58 to 0.90, recall 0.24 to 0.31 |
| **Annotators** | A labelling task with several judgments per item (`--labels`) | Items annotators split on that no reviewer has resolved | Flags 97 to 100% of items with a wrong majority, but only 9 to 13% of the items it flags have one (three annotators) |

**Rank.** Each run's ranking gives every returned document a score, `1 - (rank - 1) / k`,
averaged over the runs (0 when a run did not return it). As in confident learning, each class
gets a per-question threshold: the mean score of the documents labelled with it. A question needs
at least two relevant labels (one gives no stable threshold) and at least two runs, and is
skipped, with the reason counted, when no run returns any of its relevant documents: there a
labelling error and a retrieval failure look identical. Even where it applies, the signal cannot
tell a wrong label from a document every run is wrong about; both are worth a look. Unjudged
documents are counted as not relevant, as every metric counts them, so a suspect on one says
"the label may be missing".

**Near-duplicate.** When both duplicates are judged and disagree, one label is wrong and both
are listed. When one is relevant and its duplicate unjudged, the duplicate is listed as possibly
relevant. It sees only errors on documents that have a duplicate, so its recall is low.

**Annotators.** A split vote is a reason to look, not evidence of an error: with three
annotators most splits have a right majority. It orders the review, nothing more.

Suspects are ordered by priority (rank and another signal agreeing first, then rank alone, then
the others), then by the strength of the leading signal. The score is a ranking, not a
probability.

## What the numbers mean

The rates above come from synthetic corpora with planted errors (`scripts/label_audit_spike.py`,
seeds 1 to 5, five candidate runs), the bar for adopting the audit at all being at least 70% of
planted errors found with at most 30% of flags false. The rank signal clears it alone; with the
near-duplicate signal beside it, 84 to 96% of planted errors were found with 24 to 30% of flags
false. Real corpora are harder: planted errors are
independent of retrieval, while real labelling mistakes tend to sit where retrieval is also
hard. Read the rates as an upper bound, and the audit as a way to spend review time well.

## Boundaries

- Held-out questions, and exploratory questions that repeat one under another id, are never
  audited or queued unless you pass `--include-heldout`.
- Runs recorded on another version of the benchmark are ignored and counted; the same run given
  twice counts once. Imported runs (`runs import-trec`) work: the signal needs rankings only.
- Documents a question forbids are never suspects: a forbidden document is a permission
  assertion, not a relevance label.
- The report holds question and document IDs, labels, counts and scores, never text.
- An audit that cannot measure any signal is an error, not an empty report.
