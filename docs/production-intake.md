# Production intake

Keep the benchmark in step with production. Run the intake on each new trace export, for
example nightly after your trace exporter writes a file:

```bash
vectraudit workbench intake traces-2026-10-18.jsonl questions.jsonl \
  --state retrieval/intake --context-field tenant \
  --feedback feedback.jsonl --routes reviewers.json \
  --workspace workbench-runs --export-review review.jsonl
```

Traces are read like `workbench import-traces` (OpenInference OTLP JSON or Phoenix span
records, LangSmith runs, Langfuse observations).
Everything stays local: the state folder holds a cursor and the review queue, including
question text, and nothing is uploaded.

## What each run does

1. **Continues from the cursor.** Only traces newer than the last run are taken, so
   exports may overlap. The cursor also remembers the trace IDs at its exact instant, so
   two traces starting at the same moment are never dropped or counted twice.
2. **Matches traces to benchmark questions** by normalised question text. A matched
   question whose production retrieval delivered **none of its labelled relevant
   documents** is reported as a *likely recurrence*. The labels decide this, never the
   application's own output. With `--workspace`, the report names the question's
   regression cases and asks to reopen any verified ones. `--fail-on-recurrence` exits 3,
   so a scheduled job can alert.
3. **Queues new questions for review.** Questions not in the benchmark are de-duplicated by
   text and the context fields you name, with counts, first and last seen times and the
   latest 20 trace IDs. A reviewer turns an incident into a test without re-entering it.
4. **Prioritises without promoting.** Feedback per trace (`{"trace_id": "…", "signal":
   "negative"}`, also `escalated` or `positive`) and frequency raise or lower priority.
   **They never become labels**: the exported review file has no relevance grades.
5. **Keeps an independent sample.** A seeded share of new questions (`--sample-rate`,
   default 10%) is chosen from the question's identity alone, before any signal. Review
   therefore also covers questions nobody complained about. A capped export keeps the
   sample's share.
6. **Routes to reviewers.** `reviewers.json` lists rules such as
   `[{"when": {"tenant": "acme"}, "reviewer": "ana"}]`. The first matching rule assigns
   the reviewer, and unassigned questions are counted. `--reviewer ana` exports only that
   reviewer's questions.
7. **Marks promotions.** Once a queued question appears in the benchmark, the next run
   marks it `promoted`.

**Not judged.** Some traces are counted but not judged: those without question text,
those whose final ranking can't be read, and those whose text matches several benchmark
questions.

## From incident to protected release

**Production trace** → `intake` queues it → a reviewer labels it from `review.jsonl` in
passage review → it joins the benchmark (`promoted`) → a failing release creates a
regression case → a fix is verified → `intake` reports a *likely recurrence* if production
misses its labelled documents again, and asks to reopen the verified case.

## Limits

- **Matching:** by question text, not by meaning. Paraphrases are new questions until a
  reviewer links them, for example as a query family.
- **Recurrence:** "Likely" because a production ranking can legitimately differ (other
  filters, a newer corpus). Check the trace before reopening.
- **Effort:** reviewer minutes per accepted question aren't measured here. Record them in
  the pilot log.
