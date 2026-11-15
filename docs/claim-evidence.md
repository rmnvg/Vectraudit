# Claim-level evidence

Vectraudit does not grade answers. If your team already checks answers claim by claim with
[RAGChecker](https://github.com/amazon-science/RAGChecker) (Ru et al., NeurIPS 2024
Datasets and Benchmarks; Apache-2.0), Vectraudit can keep what its output says about
**which delivered document supported the answer and which one misled it**, next to the run's
own evidence. It never runs RAGChecker or any model, and writes no claim, answer or
document text.

## The round trip

```bash
# 1. Build RAGChecker's input from the run (private text; stays on this machine)
vectraudit workbench ragchecker-input current.run.json --dataset benchmark.jsonl \
  --corpus corpus.jsonl --references references.jsonl --answers answers.jsonl \
  --out ragchecker_input.json
# 2. Run RAGChecker's own CLI on that file, with your own extractor and checker models
# 3. Import its output, then attach the judgments (see below)
vectraudit workbench ragchecker-judgments ragchecker_output.json ... --out judgments.jsonl
vectraudit workbench attach-answers current.run.json judgments.jsonl --out answered.run.json
```

`ragchecker-input` writes, per question, RAGChecker's `query_id`, `query`, `gt_answer`,
`response` and `retrieved_context` (`doc_id` and `text`). The context is what the model was
given: the run's observed model context when it recorded one, otherwise the documents
retrieval delivered, in the recorded order. Document text comes from `--corpus` (JSONL with
`doc_id` and `text`) or `--manifest` (its `document_lookup` resolver, the one passage review
uses, scoped to each question's tenant, filters and the run's corpus version). Reference
answers are JSONL lines `{"query_id": ..., "reference": ...}` and responses
`{"query_id": ..., "answer": ...}`.

A question is exported only when everything RAGChecker judges is present: question text, a
reference, a response, a context and the text of every context document. Otherwise it is
skipped and counted, never exported with a partial context, since RAGChecker would then judge
something other than what the model saw. Held-out questions are not exported, so claim
evidence is never gathered for them.

**The file contains private text** (questions, answers and documents). It is written
readable by you only and the command says so; keep it local. Vectraudit never uploads it and
never calls a model.

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
  entails), `response_incorrect` (incorrect response claims it entails),
  `contradicts_reference` (reference claims it contradicts) and `reference_claims` (the
  positions, never the text, of the reference claims it entails; up to 1,000 claims per
  answer). RAGChecker counts a
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

## Review queue

Claim evidence also says something about the benchmark's labels. Run the benchmark audit with
runs that went through `attach-answers`:

```bash
vectraudit workbench benchmark benchmark.jsonl --runs workbench-runs --out audit.json
```

- **Documents to judge.** A delivered document that entails at least one reference claim but
  has no judgment for the question joins the judgment queue as *possibly relevant,
  unjudged*, most claims first (priority 3). Documents already judged, forbidden for the
  question or marked irrelevant in a draft are not queued, and a document the pool had
  already queued becomes one item with the higher priority.
- **Labels to recheck.** A document labelled relevant that entails none of the reference
  claims, while another delivered document entails some, is queued as *label may be stale or
  wrong* (priority 4, above new suggestions).
- **Proposed required evidence.** `required_evidence_proposals` lists, per question, the
  documents that together entail the most reference claims (a greedy cover over claim
  positions, never including a forbidden or irrelevant-marked document), the fraction of
  claims covered, which documents are already labelled relevant and which still need
  judgment, and the `required_evidence` group ready to paste. Status `proposed`.

Items are suggestions for a human: model output never becomes a relevance label. Reviewing a
queued document in the workbench's passage review, and saving, creates a new benchmark version
as for any judgment. To accept a proposed group by name:

```bash
vectraudit workbench accept-evidence benchmark.jsonl audit.json --question q1 --out next.jsonl
```

This writes a new dataset version, adding the group beside any the question already has, and
records it in the header's `provenance`. It accepts nothing you do not name, refuses a group
until every document in it is labelled relevant (a required document must be a relevant
one), refuses another version's proposals, and leaves the original file alone.

**Held-out questions never get items or proposals,** and `accept-evidence` refuses one even if
a proposal file is edited by hand. Judging documents for held-out questions would let their
labels be tuned to what the pipeline returns, which is what held-out questions are for.

## Can you trust the findings?

Claim findings depend on the extractor and checker models, so check them against people
before relying on them. When some questions also have human verdicts (`attach-answers` with
`"kind": "human"` judgments), the attach summary and `answers.claim_evidence.meta_evaluation`
report, for each finding, how many of the questions that had it a human judged incorrect,
beside the same figure for questions with no finding and for all of them:

```text
Human review of the answers, by claim finding (incorrect / judged):
  all questions: 12 of 40 judged incorrect (60 questions)
  no finding: 1 of 15 judged incorrect (22 questions)
  unsupported incorrect: 9 of 11 judged incorrect (14 questions)
```

Read it as you would a diagnostic test. A finding is worth acting on when answers that have
it are judged incorrect much more often than answers without any finding, and when enough
questions were judged (`human_judged`; a rate over a handful is noise). If the rates do not
separate, that extractor and checker pair does not predict what your reviewers call wrong, so
treat its findings as leads only, or change the models and compare again: the models used are
recorded with each judgment. Only questions a human decided count; a disputed verdict or a
question with only model verdicts does not. `agreement_with_human` shows how often a model
verdict (for example `ragchecker.f1` with a pass mark) matched the human one.

## Evidence level

Everything here is model-judged. It is evidence to review, not a measurement: it depends on
the extractor and checker models you chose, which are recorded with each judgment
(`evaluator_version`).
