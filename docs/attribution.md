# Which changed setting caused the regression?

A release often bundles several changes, such as a new embedding model, a chunk size, a
reranker version and a prompt. When the release check fails, the usual choices are to bisect
by hand or to revert everything. `workbench attribute` runs the combinations of the changed
settings on your exploratory questions. It reports how much of the metric difference each
setting accounts for, which settings interact, and which single revert recovers the most.

```bash
vectraudit workbench attribute retrieval/manifest.json \
  --baseline current --candidate release --metric recall --out attribution/
```

```text
Plan: 4 settings differ (chunk_size, embedding, prompt, reranker); every combination, 16 runs, on 200 exploratory questions, 1 repeat: up to 3200 calls.
recall@10 changes by -0.210 (95% interval -0.241 to -0.180) from the baseline to the candidate over 200 exploratory questions. embedding contributes the most: -0.096 (46% of the change). chunk_size and reranker interact: together they change it by -0.167 beyond their separate effects, split equally between their contributions.
  embedding: -0.096 (-0.119 to -0.074); alone -0.090; reverting it from the candidate +0.117
  reranker: -0.057 (-0.067 to -0.049); alone +0.025; reverting it from the candidate +0.155
  chunk_size: -0.054 (-0.062 to -0.046); alone +0.031; reverting it from the candidate +0.155
  prompt: -0.003 (-0.009 to +0.002); alone +0.000; reverting it from the candidate +0.010
Best single revert: chunk_size to {"chunk_size": 256}, recovering +0.155 (+0.131 to +0.180).
Reverting reranker instead recovers about as much on these questions; validate both.
```

The two configurations are candidates in one manifest. They must use the same adapter and
pipeline version, because only parameters can be mixed. Each parameter that differs is a
setting. A parameter present in only one configuration is removed when the combination takes
the other side's value.

## What it reports

- **Contribution.** Each setting's Shapley value: its effect averaged over every order in
  which the changes could have been made. It is computed per question family and then
  averaged, so the contributions add up to the measured difference exactly. Each comes with a
  95% paired bootstrap interval over families. "Contributes the most" is said only when the
  top setting is shown to be ahead of the next one. Otherwise the report says the two cannot
  be told apart.
- **Interactions.** A pair is reported when its interaction is shown to be larger than
  `--min-effect` (default 0.01), not merely nonzero. The interaction is the effect of changing
  both settings minus the effect of changing each, averaged over the others. The test is a
  sign-flip randomization test, Holm-adjusted across the pairs. A Shapley value splits an
  interaction equally between its two settings, so in the example the chunk size and the
  reranker are each a gain alone, yet each carries half of the loss they cause together. The
  report always names the pair beside those shares.
- **Alone and reverted.** Each setting's effect when changed alone from the baseline, and
  when reverted alone from the candidate.
- **Best single revert.** The setting whose revert from the candidate recovers the most,
  measured directly, with an interval. This answers a different question from the
  contribution. In the example the embedding model contributes the most, but reverting either
  half of the interacting pair recovers more, because the revert also removes the
  interaction. A close second is named, and nothing is suggested when no revert is shown to
  help.

## Confirm the revert on held-out questions

```bash
vectraudit workbench attribute retrieval/manifest.json --baseline current --candidate release \
  --validate --policy retrieval/policy.json --out attribution/
```

`--validate` runs the baseline and the candidate with the best single revert applied,
interleaved, on the **held-out** questions, and judges the result with the release policy,
as `try-fix --validate` does. Only a pass reads "approved under the release policy". The
command exits 3 when the revert is not approved. `--revert SETTING` validates a setting you
choose, for example the close second. In the example a single revert is not enough: undoing
the chunk size leaves the embedding model's loss, so the check fails. That is the answer to
take back to the release: revert two settings, or fix the embedding model's weak domain. The
plan line includes the held-out runs in its call count, and the command refuses when the
benchmark has no held-out questions. It never validates on the questions it attributed on.
The command writes `validation.json` (format `vectraudit.revert_validation`) and both runs.

## Budget and designs

The default budget is 16 runs (decision D16), which is every combination of up to four
settings (`2^n` runs). More settings are refused unless you pass `--max-runs`:

- when `2^n` fits the budget, every combination is run;
- otherwise a **screen** of `2n + 2` runs is used: the baseline, the candidate, each setting
  alone and each setting reverted. A setting's contribution is the average of its two
  effects, which equals its Shapley value exactly when no three settings interact.
  Three-way interactions are not tested. A setting whose two effects differ interacts with
  something, and each pair of such settings gets one more run, while the budget lasts. Pairs
  the budget did not allow are listed as untested;
- when neither design fits, the command refuses and gives both sizes.

Every combination runs interleaved, question by question, and the order rotates, so drift in
the endpoint or caches falls on all combinations alike. The plan line gives the number of
calls before anything runs, and `--max-calls` refuses a larger plan.

## Settings that change together

Some parameters cannot vary independently. For example, an embedding model is only valid with
the index built from it. Declare them with `--together embedding_model,index_name`. They are
attributed as one setting, named `embedding_model+index_name`, and listed as changed together.
They are never split. A combination the pipeline cannot run fails its questions, and questions
that any combination failed to measure are left out and listed in the report. If every
question fails, the command refuses and names the failed combinations.

## What it will not say

- It uses **exploratory questions only**. Held-out questions are never run, so they stay
  unconsulted for validating the revert under the release policy.
- It is not a release decision. The report is marked exploratory.
- Contributions describe these questions, this corpus and these two configurations. Another
  baseline would split the difference differently.
- On the Phase 47.0 spike's synthetic pipeline (`scripts/attribution_spike.py`), with 200
  questions, the planted culprit was ranked first and the planted pair reported, with no other
  pair, in 100 of 100 seeds the rules were not tuned on. The culprit's interval held its true
  contribution in 92 of 100. With 80 questions the screen ranked the culprit first in 98 of
  100, which is why ties are reported as ties. Picking the larger of two truly equal shares
  and testing it wrongly says "clear" about one time in twenty.

The command writes `attribution.json` (format `vectraudit.attribution`) and one run file per
combination into `--out`.
