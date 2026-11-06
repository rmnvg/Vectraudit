# Claim-level evidence

Vectraudit does not grade answers. If your team already checks answers claim by claim with
[RAGChecker](https://github.com/amazon-science/RAGChecker) (Ru et al., NeurIPS 2024
Datasets and Benchmarks; Apache-2.0), Vectraudit can keep what its output says about
**which delivered document supported the answer and which one misled it**, next to the run's
own evidence. It never runs RAGChecker or any model, and writes no claim, answer or
document text.

## Import

Run RAGChecker yourself (on your machine, with your models), then:

```bash
vectraudit workbench ragchecker-judgments ragchecker_output.json \
  --dataset benchmark.jsonl --run current.run.json \
  --extractor "llama3-70b" --checker "llama3-70b" \
  --metrics f1,faithfulness,hallucination --threshold ragchecker.f1=0.6 \
  --out judgments.jsonl
vectraudit workbench attach-answers current.run.json judgments.jsonl --out answered.run.json
```

- **Scores.** One judgment per question and chosen metric (`ragchecker.f1`), a `model`
  judgment with the score. A verdict exists only where you declare a pass mark with
  `--threshold ragchecker.METRIC=VALUE`: a score at or above it is correct, except for
  `hallucination` and the two noise sensitivities, where lower is better. `self_knowledge`
  has no pass mark (a correct claim from the model's own knowledge is neither good nor bad).
  `claim_recall` and `context_precision` are refused: they measure retrieval, which
  Vectraudit measures against your benchmark's labels itself.
- **Claim facts.** A `ragchecker.claims` line per question, with no verdict:

  | Field | Meaning |
  |---|---|
  | `reference`, `response` | Claims extracted from the reference answer and from the response |
  | `response_incorrect` | Response claims the reference answer does not entail |
  | `reference_in_context` | Reference claims entailed by some delivered chunk (RAGChecker's claim recall, as a count) |
  | `reference_in_response` | Reference claims the response entails (its recall) |
  | `reference_used` | Reference claims that were in the context and in the response |
  | `unsupported_incorrect` | Incorrect response claims no chunk entails: hallucination |
  | `unsupported_correct` | Correct response claims no chunk entails: the model's own knowledge |
  | `from_relevant_incorrect` | Incorrect claims entailed by a chunk that supports a reference claim |
  | `from_irrelevant_incorrect` | Incorrect claims entailed only by chunks that support none |
  | `context_documents`, `useful_documents` | Chunks, and chunks that entail a reference claim |

  and `claim_support`, per delivered document: `reference` and `response` (claims it
  entails), `response_incorrect` (incorrect response claims it entails) and
  `contradicts_reference` (reference claims it contradicts). RAGChecker counts a
  contradiction like a neutral label; a delivered document that contradicts the reference
  answer often means a stale document, so it is kept on its own.

The counts are re-derived from RAGChecker's labels with its own formulas, so they can be
checked against its metrics: `unsupported_incorrect / response` is its hallucination,
`reference_in_context / reference` its claim recall, `from_relevant_incorrect / response` its
noise sensitivity in relevant chunks. The test suite does this on RAGChecker's example
output.

## Matching questions and documents

A row matches a benchmark question by `query_id`, else by its `query` text when exactly one
question has that text (case and spacing ignored). Unmatched, ambiguous and repeated rows
are counted and skipped. Rows with malformed labels or shapes are refused and counted.

Chunk evidence needs `--run` and a `doc_id` on every `retrieved_context` chunk. Each must be a
document the run delivered or put in the model context, and none may repeat. Otherwise that
question is still scored, but without claim facts, and the command says why (no run,
question not in the run, no chunks, a chunk without `doc_id`, a repeated or unknown
`doc_id`). Use the same IDs as the benchmark and the run.

## What is attached

`attach-answers` puts each question's counts and per-document support under
`answer.claim_evidence`, bounded in size (at most 500 documents, counts up to 100,000, and
counts that must be consistent with each other). A question with two claim judgments is an
error: import one extractor and checker per file. Claim evidence never changes a verdict or
the diagnosis, and a file with only claim lines gives no verdicts.

## Findings

When a question has claim evidence, `attach-answers` adds `claim_findings` beside the
diagnosis (and prints their counts). The findings are read from the counts; they never change
a verdict or the diagnosis class.

| Finding | When | Points to |
|---|---|---|
| `reference_not_in_context` | Reference claims no delivered document entails | A retrieval gap. If a complete `required_evidence` group was delivered, the finding says the label or the required evidence may be incomplete or stale |
| `evidence_unused` | Claims in the delivered documents that the response left out (needs `reference_used`) | Generation or context packing |
| `misled_by_irrelevant_context` | Incorrect claims entailed only by documents that support no reference claim; the documents are listed | Noise, hubs, poisoned or stale documents |
| `misled_by_relevant_context` | Incorrect claims entailed by documents that do support reference claims; listed | Conflicting content inside relevant documents |
| `unsupported_incorrect` | Incorrect claims no delivered document supports | The model invented content |
| `unsupported_correct` | Correct claims no delivered document supports | Correct, but not from the documents |
| `contradicting_document` | Delivered documents that contradict the reference answer; listed | A stale document: check it with `workbench freshness` |

Each finding carries its count, the documents it names (at most 20, with how many more),
its evidence level, and `disputed`. The level is always `supported` (model-judged), never
`verified`. `disputed` is true when human review disagrees with a model verdict on the
question, or when the human verdicts disagree with each other, so a reader sees that the
claim-level evidence is contested rather than being quietly overruled. Only documents the
run delivered or put in the model context are named; any others in the claim support are
listed as `unmatched_documents` on the evidence (the file was probably made against another
run). Documents are listed in the investigation view with a link to passage review. Where
the claim support has no per-document `response_incorrect` count, a finding keeps its count
but names no documents.

## Evidence level

Everything here is model-judged. It is evidence to review, not a measurement: it depends on
the extractor and checker models you chose, which are recorded with each judgment
(`evaluator_version`).
