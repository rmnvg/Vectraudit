# Production intake

Keep the benchmark in step with production. Run the intake on each new trace export, for
example nightly after your trace exporter writes a file:

```bash
vectraudit workbench intake traces-2026-10-18.jsonl questions.jsonl \
  --state retrieval/intake --context-field tenant \
  --feedback feedback.jsonl --routes reviewers.json \
  --workspace workbench-runs --export-review review.jsonl
```

Traces are read like `workbench import-traces` (OpenInference, OpenLLMetry or OpenTelemetry GenAI OTLP JSON,
Phoenix span records, LangSmith runs, Langfuse observations, Opik pulls).
Everything stays local: the state folder holds a cursor and the review queue, including
question text, and nothing is uploaded.

## Pulling directly from Langfuse

Teams on Langfuse, LangSmith, Opik or OpenObserve can skip the export step. `workbench langfuse-pull` reads a time window
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

## Pulling directly from Opik

```bash
export OPIK_API_KEY=... OPIK_WORKSPACE=acme   # Opik Cloud; a self-hosted Opik may need neither
export OPIK_URL_OVERRIDE=http://localhost:5173/api   # only for a self-hosted Opik
vectraudit workbench opik-pull --project support-bot \
  --since 2026-10-19T00:00:00Z --until 2026-10-20T00:00:00Z \
  --traces spans-2026-10-19.jsonl --judgments judgments-2026-10-19.jsonl
vectraudit workbench intake spans-2026-10-19.jsonl questions.jsonl --state retrieval/intake \
  --stage-kind search_kb=vector
```

- **Name the retriever.** Opik's span types are `general`, `tool`, `llm` and `guardrail`,
  with no retriever type, so pass `--stage-kind SPAN_NAME=KIND` for the span that returns
  documents. It is never guessed from a name.
- **The question is on the trace.** Each Opik trace is written as its trace's root span,
  with top-level spans under it, so the trace's input is where the question is found.
- **Window:** Opik filters by creation time. Traces created in the window are taken, and
  their spans are read up to `--grace-minutes` (default 60) after it, because spans can be
  created after their trace. Spans later than that leave the trace incomplete, which
  imports as a gap, never as a result.
- **Scores** come inline on traces and spans; a span's score counts for its trace. A
  `category_name` of `correct`, `partial` or `incorrect` is that verdict, and other
  categories are skipped, never read from their numeric value. Uncategorised values stay
  scores that need `--threshold`. `ui` scores (the Opik interface and annotation queues)
  are human review (change with `--human-source`); `sdk` and `online_scoring` are model
  judgments. Opik's averaged `span_feedback_scores` are not used. Reasons are never
  written out.
- **Read-only.** An API key goes only over https (plain http only to a local server). A
  keyless self-hosted Opik can be read over http, since no secret is sent.

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

## Pulling directly from OpenObserve

Teams that send their application's OpenTelemetry traces (OpenInference or OpenLLMetry) to
OpenObserve can pull them as OTLP JSON:

```bash
export OPENOBSERVE_URL=https://o2.example.com OPENOBSERVE_ORG=acme
export OPENOBSERVE_USER=dev@example.com OPENOBSERVE_PASSWORD=...   # or OPENOBSERVE_TOKEN
vectraudit workbench openobserve-pull --stream default \
  --since 2026-10-19T00:00:00Z --until 2026-10-20T00:00:00Z --traces otlp-2026-10-19.json
vectraudit workbench intake otlp-2026-10-19.json questions.jsonl --state retrieval/intake
```

- **Whole traces:** traces are taken by start time from OpenObserve's latest-traces
  listing, then every span of each is read up to `--grace-minutes` (default 60) after the
  window.
- **Restored keys:** OpenObserve stores attribute keys flattened (lower case, dots as
  underscores, `attr_` before a key that collides with a span column). That cannot be
  undone in general, so the keys Vectraudit reads (OpenInference retrieval, reranker,
  message, prompt and session keys, and the OpenLLMetry keys) are restored by explicit
  patterns. Other attributes keep their stored names, where nothing reads them. Span
  events are stored with their original keys and come back as recorded.
- **Read-only.** Credentials go only over https (plain http only to a local server); trace
  IDs are checked to be 32 hex characters before they appear in a query; error bodies are
  not echoed.
- OpenObserve is AGPL-3.0. Vectraudit only calls its HTTP API, as any client does, and
  contains none of its code.

## Is the benchmark still representative?

`workbench drift` compares production questions from traces with the benchmark's, in the
manner of Evidently's data drift reports, so you learn when a passing release check has
started to say less about production:

```bash
vectraudit workbench drift traces-2026-10-19.jsonl questions.jsonl \
  --context-field tenant --context-field language --json drift.json
```

- **Descriptors** per question: word count, script (Latin, Cyrillic, CJK, ...), digits,
  question form, whether it uses a term the benchmark never uses, and each
  `--context-field`. For the benchmark, an unseen term is one no *other* benchmark
  question uses, so both sides are measured the same way.
- **Tests:** the ones Evidently selects by default. With up to 1,000 benchmark questions,
  the Kolmogorov-Smirnov test for numbers with more than five distinct values and
  otherwise chi-square (a Z-test for two values), drift at p < 0.05. With more, the normed
  Wasserstein distance for numbers and the Jensen-Shannon distance otherwise, drift at
  0.1. `--method psi` uses the population stability index instead. At its usual 0.1
  threshold it raises false alarms on samples of a few hundred (about 90% of the time on
  stable data at 200 questions), so it is not the default.
- **What moved:** new values (a script or tenant the benchmark lacks), the most frequent
  unseen terms, and the share of production questions that are benchmark questions word
  for word.
- `--fail-on-drift` exits 3, for a scheduled job. Below about 30 questions on either side,
  the report warns that the tests have little power.

### Did the retriever change?

`workbench score-drift EARLIER LATER` compares two windows of production traces: the
retriever's first-document score, mean score, spread from first to last, and how many
documents came back, with the same tests. It compares the questions of the two windows
too, so it can tell the causes apart:

```bash
vectraudit workbench score-drift traces-2026-10-12.jsonl traces-2026-10-19.jsonl --fail-on-drift
```

- **Retriever shift likely:** scores moved while the questions did not. A new embedding
  model, a re-index, or a similarity or score-scale change are the usual causes. Recheck
  older baselines before trusting them. `--fail-on-drift` exits 3 only in this case.
- **Drift with question drift:** both moved, so the questions may explain the shift.
  Compare recorded runs on the fixed benchmark to isolate the retriever.
- **Which stage:** the first retriever span is used; `--stage NAME` picks another. Traces
  whose retriever recorded no scores are counted and left out, never filled in.
- **Local only:** the report holds production terms and context values and is never
  uploaded.
- **Next step after drift:** run intake and review its queue, then add or relabel
  questions before trusting a pass.

## Limits

- **Matching:** by question text, not by meaning. Paraphrases are new questions until a
  reviewer links them, for example as a query family.
- **Recurrence:** "Likely" because a production ranking can legitimately differ (other
  filters, a newer corpus). Check the trace before reopening.
- **Effort:** reviewer minutes per accepted question aren't measured here. Record them in
  the pilot log.
