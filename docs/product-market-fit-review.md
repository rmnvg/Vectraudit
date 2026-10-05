# Vectraudit: product–market fit review

Reviewed October 5, 2026. This is a product recommendation based on repository inspection
and public competitor documentation, not evidence of customer demand. No customer usage,
revenue, interviews or retention data was supplied. The priorities below are hypotheses
to validate with prospective buyers. Application code was not changed or retested for this review.

## Recommendation

Position Vectraudit around one recurring job:

> Catch retrieval regressions before release, show where relevant documents were lost,
> and verify the fix on your own infrastructure.

The project already has enough technical breadth for a focused pilot. The next investment
should reduce the work needed to create a credible benchmark and act on findings. More
backends, charts or general-purpose AI features are unlikely to establish fit on their own.

The strongest initial customer hypothesis is a small AI/search engineering team operating
a production B2B knowledge assistant with tenant permissions, frequent retrieval changes,
and an identifiable engineer responsible for search quality. The user is the retrieval
engineer; the likely buyer is the engineering lead. Qualify both rather than assuming they
are the same person. Begin with whichever existing backend the first qualified customers use.

A concrete buying trigger: a team is about to change its reranker, chunking, embedding model
or filters after a costly search failure, and cannot confidently determine whether the
change breaks existing questions. Ask for the actual incident and upcoming release.

## What is already present

| Capability | Repository evidence | Product implication |
|---|---|---|
| Baseline/candidate release verdicts and GitHub checks | `evaluation/compare.py`, `evaluation/policy.py`, `integrations/github.py` | Sell a release decision; do not propose CI gates as a new feature. |
| Query findings and stage-level document loss | `evaluation/triage.py`, `frontend/src/views/Findings.tsx` | A useful technical differentiator, especially when evidence is recorded. |
| Guided pipeline execution and benchmark editing | `pipelines/guided.py`, `frontend/src/views/GuidedWorkflow.tsx` | A starting workflow exists; improve it rather than create another entrypoint. |
| Local annotation and independent review | `labelling/`, `docs/labelling.md` | Reuse this work to make labeling accessible from the workbench. |
| Dataset imports and framework adapters | `evaluation/dataset_formats.py`, `evaluation/dataset_tools.py`, `integrations/retrievers.py` | Integration foundations exist; validate specific customer exports and adapters. |
| No-answer and forbidden-document checks | `pipelines/runner.py`, `pipelines/models.py` | Permission-sensitive retrieval is a promising focus; passing sampled cases is not an access-control guarantee. |
| Benchmark coverage and judgment queues | `pipelines/benchmark.py` | Build on existing auditing, rather than add a second benchmark system. |
| Optimization and frozen-benchmark monitoring | `pipelines/optimize.py`, `pipelines/monitor.py` | Cost constraints and monitoring are not missing wholesale. Cost is adapter-reported; monitoring is not a production traffic analytics platform. |
| Hosted history, pinned baselines and governance | `cloud/`, `docs/cloud-operations.md` | Enough infrastructure to explore team value; production operations still need validation. |

The repository's `docs/workbench-validation.md` records extensive previous local checks.
Those are engineering evidence, not evidence of adoption, supported live deployments or
market fit. `docs/launch-readiness.md` already recognizes the need for repeat customer use.

## Market context

Public documentation checked for this review:

| Alternative | Documented overlap | Implication for Vectraudit |
|---|---|---|
| [LangSmith evaluation](https://www.langchain.com/langsmith/evaluation) | Evaluations, production traces feeding datasets and human review | A generic evaluation dashboard is insufficient positioning. |
| [Braintrust CI/CD](https://www.braintrust.dev/learn/ci-cd/v0) | Experiment comparisons and CI gates | “Run evaluations in CI” alone is not a defensible distinction. |
| [Arize Phoenix](https://arize.com/docs/phoenix/) | Open-source observability, evaluation and experimentation | Expect comparison with tools customers can already adopt. |
| [Langfuse evaluation](https://langfuse.com/docs/evaluation/overview) | Datasets, experiments, annotation and regression checks in CI | Fit into existing evaluation workflows and preserve portable data. |
| [Qdrant retrieval relevance](https://qdrant.tech/documentation/search-evaluation/retrieval-relevance/) | Building labeled sets and computing relevance metrics | A backend plus a small evaluation script is also an alternative. |

These sources establish feature overlap, not comparative product quality or willingness
to pay. They do not prove competitors lack any particular feature. The proposed advantage
to test is the combined workflow: retrieval-specific evidence, permission/no-answer tests,
private execution, and an understandable release decision with little setup.

## Ranked additions

### 1. A connected, document-based benchmark workflow

**Problem:** the guided editor accepts JSONL and asks for relevance grades as a JSON map of
document IDs. A separate local labeling application already has richer review functionality.
Users must understand IDs and move between workflows before reaching a trustworthy verdict.

**Add:** an “Import questions → review candidate passages → freeze benchmark → compare” flow.
Offer CSV column mapping and existing supported import formats in the UI. Show local document
titles/passages and grade buttons; connect existing pooling, second-pass review and export.
Keep raw JSON as an advanced option. Where a remote adapter returns only IDs, require a
customer-controlled document lookup or corpus import rather than inventing passage content.

**Boundaries:** clicks and negative feedback identify cases to investigate, not relevance
labels. Preserve unjudged versus irrelevant versus no-answer states. Keep human-reviewed
labels, dataset provenance and held-out protection. Do not send source text to hosted
storage by default.

**Acceptance:** a new engineer imports their questions, reviews retrieved passages, freezes
a dataset and compares a real change without manually editing document-ID JSON. Record
setup time and labeling time separately. “First result” must mean an interpretable customer
comparison, not the synthetic example or a baseline compared with itself.

**Existing seams:** `GuidedWorkflow.tsx`, `pipelines/guided.py`, `labelling/`,
`evaluation/dataset_tools.py`, `evaluation/dataset_formats.py`.

### 2. Findings that can be resolved and verified

**Problem:** the existing findings view locates losses and links evidence, but the reviewed
workflow does not maintain a resolution history across follow-up runs.

**Add:** a small case record linked to the benchmark query, dataset version, baseline,
candidate and recorded stage evidence. Track open/investigating/fix proposed/verified,
an optional owner and notes. Let an engineer rerun the affected exploratory cases, then
require the agreed full suite before declaring release readiness. Reopen a case when its
assertion fails again. Start with a downloadable issue summary; defer issue-tracker APIs.

**Acceptance:** an engineer can follow one real regression from its first finding to a
verified candidate and reproduce the decision later. A manual “resolved” action must not
override the release verdict. Missing traces must remain “cause unknown.”

**Existing seams:** `evaluation/triage.py`, `Findings.tsx`, guided run artifacts.
Use stable case identity separately from immutable run IDs and version the case schema.

### 3. Permission and no-answer test-suite builder

**Problem:** assertions exist, but constructing representative tenant/user/filter cases
requires manual context and forbidden-ID entries.

**Add:** a matrix editor for customer-provided identities, contexts and expected allowed or
forbidden documents. Include public/private documents, cross-tenant attempts, revoked access
and explicit no-answer examples. Clone a reviewed question across contexts and show the
observed result for every row.

**Acceptance:** a deliberately introduced filter regression fails a reproducible test;
restoring the filter passes the same test. The adapter must actually forward the context.
Report missing identity coverage. Do not call a passing suite a security certification.

**Sequencing:** move this ahead of item 2 if interviews show permission regressions are the
purchase trigger. Otherwise it remains a focused extension after onboarding.

### 4. Production incidents into reviewed regression cases

**Problem:** importers and judgment queues exist, but a maintained path from operational
failures to newly reviewed benchmark cases is not established in the inspected workflow.

**Add:** start with customer-exported CSV/JSON logs, not an always-on ingestion service.
Preview fields locally, retain source references, deduplicate, sample both failures and
ordinary traffic, then route selected cases through human review into a new benchmark version.
Keep source-system adapters demand-driven. Feedback, empty results and high latency should
be selection signals rather than automatic “bad retrieval” labels.

**Acceptance:** a customer can turn an actual incident into a reviewed test, verify a fix,
and catch an intentional recurrence. Benchmark changes must not silently rewrite an existing
baseline; remeasure comparable runs on the agreed new version.

### 5. Benchmark freshness and label-maintenance checks

**Problem:** frozen benchmarks can become misleading after document updates. Existing
monitoring reports supplied corpus-version changes and changed result IDs, but explicitly
does not establish a corpus inventory.

**Add:** optional document inventory snapshots containing stable IDs and content hashes,
plus a review queue for labels referencing changed or deleted documents. Preserve old
versions. Flag possible stale labels rather than automatically changing judgments or
declaring that corpus changes caused a regression.

**Acceptance:** a controlled document edit/deletion produces a specific review item; the
reviewer can accept the new label version and rerun comparisons without losing the old
evidence. If inventory is unavailable, state that freshness is unknown.

Build this after customers demonstrate benchmark-maintenance pain. It is not a prerequisite
for the first pilot.

## First implementation slice

Select item 1 unless customer evidence favors the permission suite. Keep its first slice
small: import preview, document lookup, grade controls, and export/freeze using the existing
labeling/data models. Exclude live trace ingestion, hosted annotation and new backends.

Before implementation, observe one prospective user performing the current flow and record
where they stop. After implementation, repeat with a fresh user. Functional verification
should cover format conversion, label-state preservation, corpus-ID mapping, held-out
separation and a browser journey ending in a real baseline/candidate comparison.

## Six-week validation plan

This is a proposed sequence, not a delivery estimate for all five features.

| Period | Work | Decision evidence |
|---|---|---|
| Week 1 | Interview 8–10 qualified engineers/leads; ask for a recent failure, current process and upcoming change. Show the existing workflow. | At least three teams share a repeated problem and can provide an evaluation owner and data. |
| Week 2 | Run guided evaluations with three design partners on their own infrastructure. Record every manual intervention. | Identify whether labels, connection setup, diagnosis or trust is the dominant obstacle. |
| Weeks 3–4 | Build one measured obstacle away; install the existing release check in willing pilot projects. | Customers complete a second evaluation themselves on a real change. |
| Weeks 5–6 | Observe subsequent releases, investigate false alarms and propose paid continuation. | Repeated use and actual purchase decisions, including reasons for declining. |

Interview prompts: “Show me the last retrieval change that broke something”; “How did you
detect it?”; “What did fixing it cost in engineering effort?”; “Who approves the next
release?”; “Why would your existing tool or a script not be sufficient?”; “Who can approve
a paid pilot?” Avoid asking only whether the idea sounds useful.

Suggested pilot decision thresholds: three teams complete a real comparison; at least two
repeat independently within 14 days and use the result in a release decision; at least two
accept paid continuation. These are founder decision rules, not universal PMF benchmarks.
Three pilots cannot establish broad market fit; look for sustained use across later
releases, renewals and repeatable acquisition among similar customers.

Capture time to first useful result, setup/labeling effort, manual support, accepted and
disputed findings, repeat evaluations, CI usage and paid conversion. Record why a finding
was disputed: noisy timing, insufficient labels, changed corpus or an implementation defect.
Initially use a pilot log with customer agreement rather than add product telemetry.

## Offer and distribution

Offer a scoped retrieval-release pilot: one application, one backend, a reviewed benchmark,
one real candidate comparison, a CI check and an agreed support period. Tie success to a
useful release decision and independent repeat use. Do not promise a percentage quality gain.

Test a fixed paid pilot with qualified buyers; set the amount through budget conversations
and the support effort required. No price has been validated here. Use manual invoicing
before building subscription billing. A later recurring offer could cover shared history,
baseline governance and ongoing support; validate whether customers value that separately
from a one-off diagnostic engagement.

Use the existing technical case studies and an installable private trial in targeted
conversations. After obtaining permission, publish a customer case study with the real
problem, what the evidence revealed, the fix and repeat use. Existing synthetic examples
must stay labeled as demonstrations. Do not change the proprietary distribution policy.

## Defer

- More vector databases until a qualified pilot is blocked by one.
- A general agent observability platform, prompt studio or autonomous repair system.
- More FAISS visualizations as the main commercial investment; retain them as a specialist tool.
- Broad enterprise feature expansion before validating existing deployment and SSO paths.
- Automatic ground-truth generation marketed as trustworthy labels.
- Large-scale production ingestion, billing automation and marketplace integrations before repeat demand.

The central uncertainty is whether retrieval engineers will adopt a dedicated release tool
alongside their current stack. Resolve that with real evaluations, independent reuse and
paid continuation. If users only value occasional diagnosis, reconsider a diagnostic/service
offer; if they consistently prefer extending existing tools, narrow the differentiation or
consider an integration-led product before funding a larger platform.
