# Judge-corrected estimates

A team that grades answers with a model usually has a verdict on every question from the model
and human verdicts on a few. The model's rate is cheap but biased by the model's own mistakes
(a lenient judge overstates quality); the human rate is unbiased but noisy. This combines them
into one rate with an honest 95% interval, using prediction-powered inference. Vectraudit never
runs a judge: it only combines judgments you already have.

It is **exploratory evidence about a rate**. It never changes a question's verdict or
diagnosis, and no release policy reads it.

## The round trip

```bash
# 1. Draw the random sample of question families people will review (100+ is best).
vectraudit workbench review-sample current.run.json --dataset benchmark.jsonl \
  --size 150 --seed 0 --out review-sample.json

# 2. Have people give a verdict on every question of exactly those families, as judgments
#    of kind "human". The model judged every question with its own evaluator name.

# 3. Attach everything, naming the sample.
vectraudit workbench attach-answers current.run.json judgments.jsonl \
  --sample review-sample.json --threshold ragas.answer_correctness=0.6 --out answered.run.json
```

For each model evaluator the summary shows three numbers side by side: the judge only (biased
by its errors), people only (with an interval), and corrected (with an interval, narrower than
people only when the judge is informative), plus how many families people reviewed and how often
the judge agreed with them. The workbench run view shows the same table.

## What makes it valid, and what is checked

The interval holds however wrong the judge is, **if the reviewed questions are a uniform random
sample of the judged questions**. A review queue, a list of disputed answers or hand-picked
examples is not one. So the sample is drawn by `review-sample`, by a rule anyone can re-run
(families ranked by `sha256(seed:family)`, first N taken), and checked when attached: a sample
whose chosen families differ from what its seed draws, whose frame was shrunk, or that belongs to
another dataset or run is refused.

The unit is the question family (paraphrases share a `family_id`; a question alone is its own
family), so a family is reviewed together and counts once. Held-out questions and families with
an errored or missing question are never in the frame.

A corrected estimate is **absent, with the reason**, when:

- no sample was given, or it is not this run's;
- the judge did not give a usable verdict for every question of every frame family (judging only
  some would break the randomness);
- people did not review every question of every sampled family, or reviewers disagreed on one
  (a partial sample is not a random one);
- fewer than 30 families were reviewed (the sample is refused when drawn); below 100 the interval
  is marked approximate.

Human verdicts outside the sample still decide those questions' verdicts but never enter the
estimate.

## What the rate is

The share of question families whose questions were all judged `correct`, averaged over families.
Partial and incorrect count as not correct. A model evaluator needs a verdict: give a
`--threshold` for score-only judgments (there is no default).

## Comparing two runs

When baseline and candidate were attached with the **same** sample, `vectraudit check` adds the
judge-corrected *change* in correct rate to its Markdown, job-summary and HTML reports, from the
per-family differences. It is information beside the p-values and the verdict does not use it. If
the two runs used different samples, the report says the change is not estimated.

## What it cannot fix

A judge that fails differently on the reviewed families than on the rest is guarded against only by
the sample being random. A judge changed between the two runs (different model or prompt) makes
the comparison meaningless: use the same evaluator version on both. And a corrected rate says
nothing about any single answer.

Credit: ARES (Saad-Falcon, Khattab, Potts and Zaharia, NAACL 2024) applies prediction-powered
inference to RAG evaluation; the estimator is from Angelopoulos et al., "Prediction-powered
inference" (Science, 2023) and Angelopoulos, Duchi and Zrnic, "PPI++" (2023). See
[credits](credits.md).
