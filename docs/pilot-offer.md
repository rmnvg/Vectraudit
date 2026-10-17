# Vectraudit pilot: explain a retrieval failure, verify the fix, protect the next release

**Draft for owner review.** Items in *[brackets]* are commercial decisions the owner has not
made yet; everything else describes what the product does today.

## What you get

A guided pilot on **one application and one real change** your team cares about: a recent
retrieval incident, or a planned reranker, embedding or database change.

1. **A reviewed benchmark** of your own questions (typically 30–50 to start, up to 1,000),
   graded by your domain owner, frozen with your corpus inventory.
2. **An investigation** of the failure or change: which questions got worse, which
   documents were lost and at which pipeline stage, and what is uncertain.
3. **A measured fix**: the candidate compared with your current setup under a release
   policy you agree, with intervals, per-cohort results and critical questions that must not
   regress.
4. **A release check in your CI** that fails when the same failure comes back. Deliberate
   trade-offs can be accepted through an audited, expiring exception, and the decision
   history outlives raw data retention.

## Supported today

| Area | Supported |
|---|---|
| Your application | A Python callable or an HTTP endpoint that answers a question with ranked document IDs (optionally per-stage evidence and the documents the model received); LangChain and LlamaIndex retrievers through wrappers |
| Traces you already record | OpenInference exports (OTLP JSON or Phoenix span exports) imported as a run, with a report of what the traces can and cannot show. Other trace formats are not supported yet |
| Beyond ranking | Which relevant documents reached the model in full, were left out or were cut short (reported by your pipeline, or inferred from recorded prompts); evidence groups ("A and B are both needed"); answer judgments imported from your reviewers or tools such as Ragas, with model-judge agreement against human review |
| Investigation and repair | Per-question investigation of what was lost and where, fixes your pipeline declares tried on exploratory questions and validated on held-out ones under the policy, and the repaired question protected in the release policy |
| Vector stores, directly | FAISS, Qdrant, pgvector, Pinecone, Weaviate |
| Runtime | Python 3.10–3.13 on Linux and macOS. Windows is not verified |
| CI | GitHub Actions templates; any other CI can run the same command line |
| Scale | Up to 1,000 questions per benchmark in the workbench; concurrent runs with rate limits and automatic retries |
| Permissions | Tenant/user context and filters per question; forbidden-document and no-answer assertions |
| Team view (optional) | A hosted project with release timeline, plain-language summaries for the lead, trends, shared cases and notifications (signed webhook or Slack) |

**Not part of this pilot:** importing Langfuse or LangSmith traces; evaluating answers
ourselves (we import your judgments); citation checking; automatic repairs or fixes your
pipeline does not declare. We decide these with design partners from real inputs; they are
not promised features. Context inferred from recorded prompts and model-judge results are
checked against your own review before anyone relies on them.

## What we need from you

- **An engineer** who can register the application (a callable or an HTTP endpoint) and
  run the workbench locally or on a CI runner.
- **A domain owner** who grades which documents answer each question, and later owns the
  benchmark when your content changes.
- **One application, one real change** and a representative question set, or the incident
  that prompted the pilot.
- **A decision on where results live**: entirely on your machines, or with release
  decisions shared in a hosted project (see the [trust package](trust-package.md)).
- **About 2–4 hours of engineer time and 1–2 hours of domain-owner time** across the
  sessions below. *[Owner to confirm from the first pilots; these are estimates.]*

## How it runs

| Session | Outcome |
|---|---|
| 1. Connect | Your application answers a sample question in the workbench; gaps in what it reports are listed |
| 2. Establish evidence | Questions graded, cohorts and critical questions chosen, benchmark frozen |
| 3. Investigate and fix | Baseline and candidate measured; each failed question investigated (accepted or disputed); a fix tried and validated under the agreed policy |
| 4. Protect | The repaired question added to the release policy; the release check runs in your CI, calibrated against your own run-to-run noise |
| Follow-up | Your team runs the check on its next real change with less help; we record what was needed |

## How we will judge success

Agreed with you before the first session:

- Your team understands the verdict and the evidence behind it, and a second engineer can
  reproduce it.
- The fix is verified on your benchmark, and the release check runs in your CI.
- False alarms on your setup are measured, not assumed.
- Your team runs the check again on a later change. We record time to the first useful
  result, time to an accepted diagnosis and a verified fix (the workbench records these),
  and where manual help was needed.

## Your data

By default everything runs on your machines; nothing is uploaded. If you use the hosted
project, it receives a reviewed, allowlisted projection (IDs, metrics, verdicts), never
documents, passages, request context or credentials. Question text is sent only if you opt
in, and you can preview exactly what would be sent before sending it. Details:
[trust package](trust-package.md).

## Commercial terms

*[Owner to decide: price and payment, pilot length, included sessions and support hours,
support channel and response times, what continues after the pilot, cancellation.]* The
software is proprietary. It is installed as a pinned, checksummed package from a private
channel with credentials per customer, which grant no access to our source code *[channel
being provisioned before the first outside installation]*.
