# Judging what can still change a release check

A release check counts every unjudged document as not relevant. When the candidate returns
documents nobody has judged, the check may be wrong, but judging the whole pool is the most
expensive part of an evaluation, and most of those judgments cannot change the verdict. The
decision queue orders the unjudged documents by how much they could move **this check**, and
says when no remaining judgment can change its verdict.

```bash
vectraudit workbench decision-queue base.run.json candidate.run.json \
  --dataset benchmark.jsonl --policy retrieval/policy.json --out decision-queue.json
```

```text
Verdict: fail. About 38 more judgments could still change the verdict.
  q-114 / kb-2031 (baseline rank -, candidate rank 1)
  q-207 / kb-0457 (baseline rank 2, candidate rank -)
  ...
```

Judge the first documents (in the workbench's Guided workflow, or in your own labelling tool),
save the labels as a new benchmark version, measure both runs again and run the queue again.
When it says **"No remaining judgment of an exploratory question can change this verdict"**,
further judging of the pool cannot change this check.

## In the workbench

In the Guided workflow, **Build passage review queue** now shows the pair's stop rule above the
queue ("About 4 more judgments could still change the verdict", or "No remaining judgment
… can change this verdict"), names the questions with the most documents that could still
change it, and lists this question's passages in decision order. Source and rank stay hidden
until a passage is graded, as before, so the order is the only hint. The workbench uses the
default policy here; run `workbench decision-queue --policy` for a saved release policy.

## The order

A document's **swing** is how far the difference between the two runs could move, on a rule
that is not yet settled, if that document alone were judged relevant at the highest grade:
documents only one run returns near the top come first, and documents both runs return at the
same rank cannot move the difference and come last. A rule that can no longer change stops
counting.

How much this saves was measured on synthetic runs (`scripts/decision_queue_spike.py`): the
verdict stopped changing after 56 to 68% of the judgments a random order needed. That is
**about a third fewer judgments, not half**; near the policy's threshold a verdict keeps
changing until most documents that could matter are judged, whatever the order.

## The stop rule

For every question, Vectraudit bounds each metric over every way its unjudged documents could be
judged (any grade from 0 to `--max-grade`, default 3). The bounds are exact for recall,
precision, MRR and hit rate and safe for nDCG. It then runs the check itself with every
question at its lowest and at its highest. The policy's metric rules (with or without
`require_significance` or `non_inferiority`), `min` floors, `max_regressed_queries` and
cohort rules without control can only move one way as values rise, and the bootstrap draws the
same resamples each time, so a rule that is the same at both extremes is the same for every
way the documents could be judged. A pass is settled when every such rule is; a fail is
settled when one rule fails at both extremes, or a failure labels cannot touch (latency, a
critical document) is present.

The rule is conservative: it can say "could still change" when no actual set of judgments
would, but it never says "settled" while one could. The tests check this against every
possible judgment on small cases.

What it does not certify, and says so (`state: not_certifiable`; the order is still given):

- rules on MAP, which has no tight bound here;
- cohort rules under `bonferroni` or `holm` control, whose normal approximation uses the
  spread of the changes and does not move one way.

An exploratory question with no relevant label yet keeps the verdict open: judging one of its
documents relevant adds it to the check.

## Scope

- Exploratory questions only. Held-out questions are never queued, and their unjudged
  documents stay not relevant, as every metric counts them.
- Documents already judged, marked not relevant in a workbench draft, labelled through a span,
  or forbidden are never queued.
- Only documents a run returned in its top `k` are in the pool; a relevant document neither
  run returned changes recall's denominator but no judgment queue can find it.
- Nothing is judged for you, and the queue never edits the benchmark.
