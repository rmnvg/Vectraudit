# Projecting recall to the whole corpus

Re-embedding a large corpus with a new model takes days and costs money, so the decision is
usually made on a sample. Recall measured on a sample is too high, because fewer documents
compete with each relevant one. `vectraudit experiment project` measures each candidate on
nested samples and projects its recall to the full corpus, with an interval. It ranks the
candidates at full size and gives the chance that the sample's leader is still the leader.

```bash
vectraudit experiment project experiment.json --out projection.json
```

It uses the same [experiment manifest](experiments.md) as `experiment run`: the corpus, the
labelled questions, `k`, and the candidates (embedder and chunking). On neural-bridge's
rag-dataset-12000 (11,997 passages, 300 questions), with only 1,200 passages embedded per
model:

```text
Candidate   recall@10 at 480   at 720   at 960   at 1200   projected at 11997 [95%]   P(best)
bge-small   0.947              0.940    0.937    0.930     0.876 [0.860, 0.890]       100%
minilm      0.940              0.933    0.927    0.910     0.837 [0.807, 0.870]       1%
```

Measured on the whole corpus, recall@10 for these questions was 0.893 for bge-small and 0.823
for MiniLM. MiniLM's interval held its measured value. bge-small's interval missed by 0.003,
which a 95% interval is expected to do about one draw in twenty; the next five seeds covered
both models. The sample overstated recall by 0.05 and 0.07.

## How it works

- **Samples.** The command draws nested, seeded samples of 4, 6, 8 and 10% of the corpus
  (`--fractions`). Every sample always holds every labelled document. Only the largest
  sample is embedded, and the smaller ones are nested inside it. Search is exact.
- **What is counted.** For each question and each of its relevant documents, it counts the
  documents in the sample that score above the relevant one, kept apart into labelled
  documents (present at every size) and the others. Each other document of the full corpus
  is in the sample with a known chance, so the count in the sample is a thinned draw of the
  full count.
- **The projection.** The full counts across questions are modelled as a Poisson mixture with
  a mixing distribution estimated without a parametric form, by EM. Each relevant document's
  chance of staying in the top `k` at full size follows from its posterior. The projected
  recall is their mean, with questions weighted as the release check weights them (equal per
  family, then per question).
- **The interval.** The mixing distribution is refitted on resampled question families, and
  each relevant document's outcome is drawn, 2,000 times. The interval is for the
  full-corpus recall of **these** questions.
- **Ranking.** The draws are paired across candidates, with the same resampled families and
  the same random numbers, so `P(best)` is each candidate's chance of having the highest
  recall at full size.

On the Phase 48.0 spike (`scripts/corpus_projection_spike.py`, `docs/DECISIONS.md`), the
interval from a 10% sample held the measured full-corpus recall@10 in 18 to 20 of 20 sample
draws. That was on two synthetic corpora of 60,000 documents and on the corpus above, with two
embedders each. Recall measured on the sample overstated it by 0.04 to 0.21 and never fell in
the interval. A line fitted to recall against log corpus size was biased upwards and twice as
wide.

## A corpus larger than the file

When the corpus file is itself a sample of a larger corpus, pass the full size in documents:

```bash
vectraudit experiment project experiment.json --full-size 400000 --out projection.json
```

This assumes the file is a uniform random sample of the full corpus, apart from the labelled
documents. The command refuses to project beyond `--max-multiple` (default 10) times the
largest measured sample, which is the range the spike validated. The refusal says how many
documents would have to be measured. The command also refuses sample sizes that would hold
nothing but the labelled documents.

## What it will not say

- **Only recall@k is projected.** The other metrics depend jointly on several relevant
  documents' ranks, which the model does not capture.
- **Search is exact.** An approximate index (IVF, HNSW) loses more recall, which is not
  projected. Measure that at its real size with `vectraudit tune` or `record`.
- **The documents are the ones sampled.** Documents added later, or a full corpus unlike the
  sample, are not modelled.
- **The projection is for these questions.** It is not for questions you did not label.
