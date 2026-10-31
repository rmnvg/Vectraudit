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

## Pulling directly from Langfuse

Teams on Langfuse or LangSmith can skip the export step. `workbench langfuse-pull` reads a time window
from the project's public API and writes the same observation records an export holds, plus
the project's scores as answer judgments:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your region or self-hosted URL
vectraudit workbench langfuse-pull --since 2026-10-19T00:00:00Z --until 2026-10-20T00:00:00Z \
  --traces traces-2026-10-19.jsonl --judgments judgments-2026-10-19.jsonl
vectraudit workbench intake traces-2026-10-19.jsonl questions.jsonl --state retrieval/intake
```

- **Read-only.** Nothing is written to Langfuse. Keys go only over https (plain http only
  to a local server) and are never written to files.
- **Overlapping windows are safe:** intake's cursor skips traces it has already taken.
- **Scores:** a boolean score becomes `correct` or `incorrect`; a categorical score only
  when its category is `correct`, `partial` or `incorrect`; a numeric score stays a score
  and needs `--threshold` in `attach-answers`. Text and correction scores, other
  categories, and session or experiment scores are skipped and counted. Annotation scores
  are human review (change with `--human-source`); API and evaluator scores are model
  judgments. Comments are never requested.
- **Freshness:** Langfuse says scores can lag by about ten minutes, so end score windows a
  little in the past, or pull again.
- **Bounds:** at most 200,000 records per window; a larger window is refused, so pull a
  shorter one.
- **Window edges:** Langfuse filters observations by their own start time, so a trace
  that begins just before `--until` can lose its later observations to the next window.
  It then imports as a gap (no retriever, or context unknown), never as a wrong result.
  End windows at a quiet time, or overlap them; intake skips traces it has already taken.

## Pulling directly from LangSmith

```bash
export LANGSMITH_API_KEY=lsv2_...   # LANGSMITH_ENDPOINT for the EU region or self-hosted
vectraudit workbench langsmith-pull --project support-bot \
  --since 2026-10-19T00:00:00Z --until 2026-10-20T00:00:00Z \
  --traces runs-2026-10-19.jsonl --judgments judgments-2026-10-19.jsonl
```

- **Whole traces:** the window applies to each trace's root run, so a trace that started
  in the window comes back with all of its runs, even those that ended after it.
- **Feedback:** a `value` of `correct`, `partial` or `incorrect` is that verdict; otherwise
  a boolean `score` is a verdict and a numeric `score` stays a score that needs
  `--threshold`. Other values, feedback without a score, and feedback on runs outside the
  pulled traces are skipped and counted. Feedback from the LangSmith app (UI and
  annotation queues, source type `app`) is human review (change with `--human-source`);
  `api` and `model` feedback are model judgments. Feedback on a child run counts for its
  trace.
- **Read-only.** Nothing is written to LangSmith. The key goes only over https (plain http
  only to a local server). LangSmith returns comments and corrections with feedback; they
  are never written to the output files.

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
