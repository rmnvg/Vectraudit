# Vectraudit: product strategy and evidence required for launch

*Updated October 8, 2026 against checkout `63da2c4`. No launch deadline. Priorities are
hypotheses to validate, not a commitment to build every feature. A second pass the same day
ran the full test suite and verified further gaps in code (section 2, new items 1.9 and 1.10).*

**October 9 implementation audit:** the historical gaps below are not all still open.
Typed errors, bounded retries/concurrency, 1,000-question bounds, local question text and
query-family statistics have landed. Distribution foundations are now prepared locally.
See [current status and remaining work](startup-implementation-2026-10-08.md#october-9-code-audit)
before selecting the next implementation phase; customer acceptance gates remain unverified.

## 1. Recommendation

**Build a retrieval reliability product whose strongest workflow is: explain a real failure,
test a repair, and protect every later release.**

The project already has a substantial engineering foundation. Its biggest limitation is
not the number of databases or charts: it is the lack of evidence that a particular customer
repeatedly needs the workflow, can adopt it economically, and will pay for it alongside
existing tools. Invest in five outcomes:

1. **Explain and verify a repair.** Extend existing findings and case verification with
   evidence completeness, controlled experiments and useful next actions.
2. **Connect and maintain evidence cheaply.** Reuse customer traces and labels; expose gaps
   in instrumentation; reduce benchmark upkeep as documents and permissions change.
3. **Follow evidence through to the answer.** A correct document can be retrieved while
   the required passage never reaches the model. Current results stop at ranked hits.
4. **Make release decisions trustworthy.** Complete optimizer-policy integration and expose
   uncertainty, intermittent failures and insufficient evidence consistently.
5. **Support independent team use.** A lead understands the decision and a second engineer
   reproduces it without the founder.

The preferred deeper product bet is **source-to-context lineage and knowledge freshness**,
with permission lifecycle assurance as an alternative if buyers pull harder there. Having
no deadline gives us time to prove that value. It should not mean building indefinitely
before customers use the product. Private discovery and design partnerships are learning,
not a public launch.

**Current judgment:** enough foundation to pursue a scoped design partnership; insufficient
evidence to declare a repeatable commercial product or product–market fit.

This document is a strategy, not a declaration of product–market fit. No customer usage,
interviews, revenue or retention data was available. Customer profiles, priorities and
thresholds are hypotheses to test.

### Short answer: what must exist before I would call it launch-ready

No feature list makes a product fit its market; only customers returning and paying show
that (section 9). But the product side of the launch bar is concrete. It is complete when
these ten things are true. Everything else in this document stays optional until customer
behaviour promotes it.

| # | Product capability | Plan item |
|---|---|---|
| 1 | Customers install and upgrade a pinned release without gaining access to the source | 1.9 *(blocks paid pilots)* |
| 2 | A fresh engineer connects a real application without founder-written code, and sees what cannot be observed | 1.0, 1.1 |
| 3 | A failure is explained from source through ranking to model context, with unknowns kept explicit | 1.2, 1.3 |
| 4 | One release policy applies on every surface, calibrated to the customer's measured noise | 1.4, 1.10 |
| 5 | A team can deliberately accept a regression through an audited, expiring exception instead of bypassing the gate | 1.10 |
| 6 | A 1,000-question suite runs reliably: concurrency, typed errors and retries | 1.5 |
| 7 | Cases, benchmarks and release decisions are shared by the team and outlive raw-run retention | 1.6 |
| 8 | Results reach the team's CI and chosen channel | 1.7 |
| 9 | Benchmark upkeep costs are low enough to sustain across releases | 1.8 |
| 10 | One deeper advantage customers pay for works on real systems | Tier 2 |

A commercial launch additionally needs the customer and commercial evidence in section 9.

## 2. Review scope and evidence

- This was a broad repository scan with selected implementation reads, not a line-by-line
  audit or live UX study. It covered the README, `PLAN.md`, the product requirements DOCX,
  earlier market-fit and readiness documents, frontend workflow views, pipeline contracts and
  runner, optimizer, triage, policies, framework adapters, experiments, connectors and the
  cloud control plane.
- Tests exercised in this review: `test_pipelines.py`, `test_workbench.py`,
  `test_critical_queries.py`, `test_eval_compare.py`, `test_exp_labels.py`,
  `test_framework_adapters.py`, `test_market_fit_workflows.py` and `test_reference_search.py`.
  **All 126 selected tests ultimately passed.** The initial sandbox run passed 125 and
  blocked one localhost HTTP test at socket binding; that test passed when rerun with
  loopback access. The full suite, browser journeys, live backend tests and hosted
  deployment checks were not rerun. Only this strategy document was edited.
- **Second pass:** the full Python suite (`uv run pytest`) gave **1,239 passed, 22 skipped,
  1 deselected**. All 22 skips are pgvector, Pinecone and Weaviate integration/acceptance
  tests whose live services were not configured locally; CI provisions them. Frontend
  tests, browser journeys and hosted deployment checks were still not rerun.
- The October 5 and 6 proposals were since implemented. Do not describe these as missing:
  passage review, incident import, permission scenarios, benchmark freshness, regression
  cases, saved release policies, release exports, connection diagnostics, document lookup
  and resumable evaluations.

Gaps confirmed in the code during this review:

| Gap | Evidence |
|---|---|
| No trace ingestion | No OpenTelemetry/OpenInference, Langfuse or LangSmith trace readers; only dataset-export importers |
| No answer layer | `pipelines/models.py` has no context, answer or citation fields |
| Serial execution | Run metadata records `"concurrency": 1`; the pipeline runner executes queries one at a time |
| No delivery channels | No Slack, email or webhook delivery; `pipelines/monitor.py` emits machine-readable alerts only |
| GitHub-only CI | `integrations/github.py` uses Actions' `GITHUB_TOKEN`; no GitLab or other CI publisher |
| Optimizer has separate feasibility rules | `pipelines/optimize.py` takes no saved `Policy`; it already emits `requires_review`, so this is not evidence that CI accepts optimizer output |
| Benchmark cap | The guided editor caps benchmarks at 500 questions; the PRD's pilot target is 1,000 |
| No self-serve billing | PRD FR14 is unimplemented (acceptable for pilots) |

Additional findings from implementation reads:

| Finding | Evidence and implication |
|---|---|
| Quality variation is not fully measured | `pipelines/runner.py` stores repeats but uses `results[0]` for relevance metrics and primary stage evidence. Critical/permission checks already inspect repeats. |
| Document presence is weaker than evidence sufficiency | Framework wrappers deduplicate document IDs; `experiments/labels.py` grades any overlap with a labelled span, not completeness of required evidence. |
| Richer evidence needs an export boundary | `integrations/cloud.py` uploads the provided run record; `cloud/routes/runs.py` stores it after validation. Adding text/answers changes what upload sends. |
| Significance policy is not proof of non-inferiority | `evaluation/policy.py` and `compare.py`: `require_significance` detects sufficiently clear harm. Failure to detect harm does not establish safety. |

Gaps verified in the second pass:

| Gap | Evidence | Why it matters |
|---|---|---|
| Installing requires source access | The `init` workflow templates run `gh release download --repo rmnvg/Vectraudit` with a `VECTRAUDIT_DOWNLOAD_TOKEN`; `release.yml` uploads `dist/*`, which includes the sdist. On GitHub, read access to a private repository's releases is read access to its code. | Every customer CI installation needs source access, which conflicts with the proprietary model. Fix before paid pilots (1.9). |
| Team records are local only | `cloud/models.py` stores users, workspaces, projects, tokens, runs, baselines, checks, audit events, sessions and invites. Regression cases, benchmark drafts and snapshots, passage judgments and saved workbench policies exist only in `.workbench/` on one machine. | A second engineer or a lead cannot see cases or benchmark history without file copying (1.6). |
| Release history expires with raw runs | `cloud/quotas.py` `purge_expired` deletes runs past retention (30 days on the pilot plan, 90 on team) except pinned baselines, and their checks are deleted with them. Pin audit events remain. | On a pilot plan the release timeline cannot cover even one quarter (1.6). |
| No audited way to accept a known regression | Policies, baselines and critical rules exist; nothing records "we accept this failure, for this reason, until this date". | When a team makes a deliberate trade-off, it must loosen the policy, re-pin or bypass the required check, and the evidence trail is lost (1.10). |
| Run errors lose their cause | `HTTPPipeline` raises typed `PipelineCallError` codes (`http_auth`, `http_rate_limit`, `timeout`, `invalid_response`…) and the connection probe keeps them, but `pipelines/runner.py` records every failure as `pipeline_error`. Nothing retries automatically. | Incomplete runs never pass, so one transient failure in a 1,000-question CI run blocks a release without a stated cause (1.5). |
| Request contract is scalar-only | `PipelineRequest.filters` and `context` accept only `str/int/float/bool`. There is no list-valued group membership, no range and no conversation history. | Common permission models (a user in several groups) and conversational query rewriting must be flattened into strings by each adapter (1.0). |
| Evidence is shown by ID | `Findings.tsx`, `RegressionCases.tsx` and the exported case summary name questions and documents only by ID. Run records omit query text by design, but the local draft holds it. | A lead cannot read a finding without a lookup. A local display join changes nothing that is uploaded (1.6). |
| Effective sample size is overstated | `workflows.permission_rows` clones one question across contexts under new IDs with positive `relevant` grades. The clones count toward `min_queries`, aggregate metrics and the query-level bootstrap as independent queries. Each cohort is judged separately, with no multiplicity control. | Intervals look narrower, and cohort false alarms are more frequent, than the numbers suggest (1.4). |
| Recurring work leaves the browser | The workbench's Experiments, Benchmark and Monitoring tabs display JSON produced by CLI commands. `workbench monitor` writes to a local `--history` folder and uploads nothing. | The browser covers one comparison; repeated checks and monitoring happen elsewhere (1.6, 1.7, 2.5). |

Corrections from the second pass: the hosted service has no case history to "reuse", so 1.6
must create one (local cases are the model). Tier 3's LLM label suggestions are partly built:
`labelling/suggestions.py` already stores judge suggestions separately and tracks agreement,
with local Transformers models through `[label-llm]`. An API-backed judge would be an adapter,
not a new subsystem.

**Corrections to the earlier plan:** `Diagnosis.tsx` is the FAISS index guidance screen,
not the pipeline investigation workspace. The actual workflow spans `GuidedWorkflow.tsx`,
`Findings.tsx`, `RegressionCases.tsx` and related views. Component line counts do not establish
UX quality. Likewise, server-rendered HTML is not inherently inadequate: observe whether
leads can make decisions before requiring a React rewrite.

Existing sign-in is a fixed-provider OAuth code/userinfo integration, not demonstrated
general SAML/OIDC compatibility. See [deployment boundaries](enterprise-deployment.md).
Existing operations code and historical validation are foundations, not live deployment proof.

## 3. What already exists

| Area | Present foundation | Open question |
|---|---|---|
| FAISS inspection | Miss explanations, graph replay, quantization analysis, tuning, comparisons | How often does a paying customer's problem need this depth? |
| Pipeline evaluation | Python/HTTP adapters, framework wrappers, stage traces, relevance metrics | Can a customer connect their real application without custom work? |
| Benchmark creation | Passage review, pooling, imports, reviewer workflows, snapshots, freshness | Can teams maintain representative evidence at reasonable cost? |
| Release protection | Policies, cohorts, critical queries, repeated assertions, exports, CI | Are verdicts understandable, stable and useful in real decisions? |
| Experimentation | Embedding/chunk/index comparisons, bounded optimization | Does the product guide the next experiment and verify the fix? |
| Backends | FAISS, Qdrant, pgvector, Pinecone, Weaviate | Which backend blocks the next qualified customer? |
| Team infrastructure | Workspaces, roles, hosted run/check history, baseline governance, SSO, operations tooling | Has the deployment been exercised with real users and workload? Cases, benchmarks and judgments are still local-only. |

## 4. Customer, buying event and promise

**First customer hypothesis:** an engineering team maintaining a B2B support or knowledge
assistant, with tenant-specific documents, frequent retrieval changes and a named quality
owner. The user is a retrieval, AI or backend engineer; the buyer is the engineering lead.

**Buying events:** a recent retrieval incident, an upcoming reranker or embedding migration,
or changed permission logic. Also test forced changes the team did not choose: an embedding
model retirement, a vector database version upgrade or a provider change that requires
re-indexing. These arrive on someone else's schedule and cannot be postponed.

| Segment | Why attractive | Main drawback | Decision |
|---|---|---|---|
| B2B knowledge/support AI teams | Recurring releases, costly regressions, permission context | Must prove value beside existing eval tools | **First focus** |
| AI consultancies | Reusable setup across client projects | Bespoke service work can dominate | Test as a channel |
| Enterprise internal search | Complex permissions and document lifecycles | Long sales cycle, deployment burden | Follow demonstrated pull |
| E-commerce relevance teams | Clear search-quality ownership | Different evaluation and backend needs | Later expansion |
| FAISS researchers and individuals | Strong fit for the debugger | Little recurring budget | Technical audience, not buyer |

**Promise:** *"Find why your knowledge assistant missed the right information, measure the
repair, and catch the same failure before the next release."*

Sell one application and one important change first. Backend independence helps even when a
customer uses only one database.

Three possible entry offers are worth testing:

| Entry offer | Immediate value | Recurring reason to stay |
|---|---|---|
| Failure investigation | Explain a costly incident and verify its repair | Protect the repaired case across releases. |
| Migration assurance | Compare current and proposed retrieval configurations | Reuse the benchmark and policy for subsequent changes. |
| Permission/freshness reliability | Reproduce a revocation, stale-answer or cache defect | Retest as content and access change. |

Start with failure investigation plus release protection. Promote permission/freshness to
the primary proposition only if several buyers independently value it more. A migration fee
can be useful revenue without proving subscription demand.

Identify the engineer, domain judgment owner, budget owner and deployment approver separately.
Defer prospects with no real application, no evidence access, no judgment owner or no recurring
change. Do not interpret every lost prospect as a missing feature.

## 5. Competition and differentiation

Official documentation checked for this review on October 8, 2026:

- LangSmith documents offline/online evaluation, annotation and production traces feeding
  datasets. [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation)
- Braintrust documents experiments, CI/CD and online scoring. A general experiment dashboard
  faces established alternatives. [Braintrust evaluation](https://www.braintrust.dev/docs/evaluate)
- Langfuse documents critical-case and approved-baseline release policies, so CI gates and
  critical assertions are not exclusive. [Langfuse CI/CD](https://langfuse.com/docs/evaluation/experiments/experiments-ci-cd)
- OpenSearch has a relevance workbench; Quepid supports relevance tuning. Competition also
  comes from search tooling. [OpenSearch workbench](https://docs.opensearch.org/latest/search-plugins/search-relevance/using-search-relevance-workbench/), [Quepid](https://quepidapp.com/docs/)
- Qdrant documents labelled evaluation with standard metrics, so a database plus an internal
  script is a credible alternative. [Qdrant retrieval evaluation](https://qdrant.tech/documentation/search-evaluation/retrieval-relevance/)
- Phoenix documents RAG tracing and retrieval/generation evaluation. Trace import and RAG
  metrics alone will not distinguish us. [Phoenix RAG evaluation](https://arize.com/docs/phoenix/cookbook/evaluation/evaluate-rag)
- Ragas supplies context, answer and faithfulness metrics. Import customer evaluator results
  before building another evaluator suite. [Ragas metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)

These sources show overlap, not a full feature comparison. **The differentiation to prove
is the depth and convenience of the full diagnosis → verified fix → protected release loop
on customer infrastructure**, plus stage-level loss attribution, ingestion lineage and
permission lifecycle testing. Private execution alone is not a moat.

Strategy implication: **integrate with these tools rather than replace them.** Reading
Langfuse, LangSmith and OpenTelemetry traces turns competitors' data into Vectraudit's input.

**Competitive proof required:** compare the same real investigation using Vectraudit and the
customer's current tool/script. Measure setup effort, time to a correct explanation, expert
review effort, reproduction and recurring maintenance. Ask why they would add another product.
Public documentation does not establish comparative quality, willingness to pay, market share
or that competitors lack a proposed feature.

Potential defensibility comes from reliable evidence contracts, reusable diagnostic methods,
integration quality and customer-owned case history embedded in recurring work. None is yet
a moat. Do not assume customer data can be pooled or claim an unproven data network effect.

## 6. The plan

Tier 1 completes the core workflow; Tier 2 deepens a proven use case; Tier 3 waits for
repeated demand. These are investment priorities, not an instruction to build every item
before customer learning. Promote whichever capability drives a demonstrated purchase.

### Tier 1: foundations and the core customer workflow

#### 1.0 Evidence contracts and the local/hosted boundary *(new prerequisite)*

Before richer trace/answer fields, version the evidence contract: source ID/revision, chunk
ID/span, retrieval invocation, stage identity/edges, actual context and judgment provenance.
Distinguish observed empty output, unrecorded output, partial/sampled traces and unsupported
capabilities. Preserve backward compatibility with existing run files.

Widen the request contract in the same version, because the first segment needs it:
list-valued context (a user's groups or entitlements), list-valued filter conditions, and
optional conversation history for assistants that rewrite queries from earlier turns. A
hit should be able to carry its chunk ID and its source document ID separately, so adapters
stop choosing one in advance. Keep portable filters simple. Mark backend-native expressions
as nonportable, as the PRD already requires.

Separate raw local evidence from an allowlisted hosted projection. Add a transfer preview,
explicit text-sharing settings and limits; IDs and labels can themselves be sensitive.
Keep raw artifacts and exported projections separately identified so export settings do not
rewrite historical evidence. Apply the boundary to reports and notifications too.

**Acceptance:** old files still load; partial evidence remains explicit; seeded private text
in new evidence fields stays out of default cloud payloads and alerts; permitted summaries
still reproduce the intended comparison. This is a design requirement for richer evidence,
not a claim that customer data has leaked.

#### 1.1 Easier connection: trace ingestion and capability reporting

Start with one trace format used by design partners; an OpenInference export is a reasonable
candidate, then a Langfuse or LangSmith source if needed. OpenInference defines retrieval
documents and reranker input/output fields; it does not ensure every application records
complete document flow. [OpenInference conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)

- Map available spans to stages. Distinguish observed/reported data from inferred mappings;
  a timing source is not proof of stage semantics. Trace nesting alone does not establish
  document-flow edges. Request mappings when ambiguous; do not fabricate filter/fusion stages.
- Report missing or unmappable observations as gaps, never as empty successful stages.
- Retrospective import and retesting are different capabilities. Retesting requires a
  registered callable/HTTP endpoint, local credentials and appropriate corpus/version context;
  a trace file alone cannot rerun an application.
- Keep raw trace text local by default and apply the export boundary in 1.0.

Show a capability report: stable IDs, observed stages, filters/tenant context, actual model
context, missing fields and replay availability. Offer useful result comparison with ranked
IDs alone, observed loss attribution when stages exist, and deeper explanation only when
source/context evidence supports it. Keep existing adapters and guided ID mapping as
fallbacks; avoid a universal “no-code connection” promise.

**Acceptance:** a supported instrumented application imports without a custom Vectraudit
adapter; evidence gaps are visible; a second engineer independently follows the documented
connection path. Measure setup time separately from judgment creation.

#### 1.2 Answer and context evidence *(promoted from Priority 8)*

Add optional actual-context, answer, citation and answer-judgment records, including source
revision, chunk/span references, context order and truncation when available. Keep raw text
local by default. Expose these observations; **allow mixed causes and unknowns**:

1. **Not observed:** relevant evidence absent from recorded stages. Missing instrumentation
   or an absent/stale source is still possible; do not automatically blame retrieval.
2. **Lost downstream:** retrieved, then dropped by a filter, reranker or truncation. Identify
   intended permission filtering separately from a defect.
3. **Lost in context packing:** useful retrieval evidence omitted from actual model context,
   including trimming within a chunk.
4. **Answered wrong despite evidence:** sufficient reviewed evidence in context, but the
   answer is incorrect under a human/imported judgment. This does not by itself prove a
   particular model or prompt is the root cause.

Add simple evidence groups: **A and B are both needed**, or **either equivalent source is
sufficient**. Document-level recall and any-overlap span grades remain valid but separate
metrics; they do not establish answer sufficiency. A chunk can overlap a required clause
while omitting its essential exception. Citation existence is separate from support for
the cited claim. Distinguish retrieval abstention, generated-answer abstention and an
appropriate human handoff.

Accept answer judgments from existing tools (Ragas, DeepEval, Langfuse scores, human review)
before building any answer evaluator. Keep retrieval relevance, evidence sufficiency and
answer correctness as separate metrics. Model judgments must be calibrated against human
review and shown with their agreement rate.

**Acceptance:** detect a missing exception even when document recall passes; identify a
planted context omission and answer failure separately; preserve mixed/unknown outcomes.
Record evaluator/rubric versions and disagreements rather than hiding them in one score.

#### 1.3 Investigation workspace: from finding to verified fix *(Priority 1)*

One screen per failed query or regression: what was lost, at which stage, observed facts,
possible causes, a proposed experiment, measured results and remaining uncertainty.

- Capture, when adapters or traces supply them: query rewrites, candidate depth, filter
  decisions and reason codes, fusion contribution, reranker scores, truncation and
  component versions.
- **"Try this fix"** buttons for bounded interventions: raise candidate depth, swap
  reranker, change hybrid weight, change k. Run on exploratory cases, show which failures
  recover and which regress, then promote the chosen candidate to a full policy check on the
  protected suite.
- Offer only interventions the registered pipeline supports. A recorded-candidate reranking
  or packing replay can be a cheap first experiment, but cannot discover documents outside
  the captured pool. Label that limit and validate chosen changes with a full run. Show
  budgets and keep other inputs controlled where possible.
- Start with the two most frequent failure classes seen in pilots, not an open-ended
  troubleshooting agent.
- If a document never entered retrieval, ask for corpus/index evidence before blaming
  embeddings. Disabling permissions is never a deployable repair.

**Acceptance:** an engineer explains and verifies a real fix with linked evidence, with less
effort than their current workflow. Track investigation time, accepted and disputed
diagnoses, and repairs reproduced by a second engineer.

#### 1.4 One release policy everywhere, with honest uncertainty

Give `optimize()` the saved `Policy`, critical assertions and comparability rules. Label
every candidate as either **"best exploratory candidate"** or **"approved under policy"**,
in the UI, CLI, reports and CI alike.

**Acceptance:** no path can present an optimized or exploratory result as release-approved
without the full policy check.

Promote evaluation confidence from optional depth to a foundation for release claims:

- Expose quality variation and relevant stage evidence across executions, not only the first
  ranking. Preserve existing all-repeat critical and permission checks.
- Define paired execution, cache/warm-up and endpoint/version protocols. Matching OS and
  CPU-count metadata alone does not establish identical hardware or remote conditions.
- Group query families across splits. Repeatedly consulted held-out data becomes regression
  data; reserve fresh evidence when claiming generalization.
- Count effective sample size, not rows. Permission scenarios cloned from one question are
  one family for `min_queries`, aggregate metrics and the bootstrap. Resample by family,
  and show rows and families side by side.
- Control cohort false alarms. With many judged cohorts, a few will fail by chance alone;
  report how many cohorts were judged and calibrate cohort rules against unchanged-system
  repeats (1.10). Offer cohorts derived from declared context fields (tenant, language,
  plan) as well as tags, so a change that helps large tenants and hurts small ones is visible.
- If required, add an explicit versioned non-inferiority policy: for candidate-minus-baseline
  deltas, the interval's lower bound must exceed the negative allowed-drop boundary to
  support “within tolerance.” An interval spanning that boundary is inconclusive. Current
  `require_significance` instead detects sufficiently clear harm. Do not silently change
  semantics or call absence of significant harm proof of safety.
- Explain how inconclusive evidence maps to CI. Keep hard critical/permission failures
  separate from statistical aggregate decisions; report independent query-family/cohort counts.

**Additional acceptance:** unchanged-system repeats meet an agreed false-alarm budget;
planted intermittent defects are visible; identical artifacts/policy yield identical decisions
on every surface; insufficient evidence cannot masquerade as demonstrated safety.

#### 1.5 Concurrent, bounded execution *(from Priority 7)*

Add bounded worker concurrency, per-pipeline rate limits, retry/backoff and timeouts to the
pipeline runner, keeping the existing checkpoints and resume behaviour. Record concurrency in
the timing protocol, and report latency under concurrency separately from serial latency.
Raise the guided editor's 500-question cap to the PRD's 1,000-query pilot target.

Existing HTTP timeouts and retry-on-resume are foundations. Do the error work first; it is
small and needs no customer evidence. Keep the typed `PipelineCallError` code on each failed
query instead of the generic `pipeline_error`, and summarize failures by code. Then retry
retryable codes (timeout, rate limit, connection failure) automatically within a declared
budget, honouring `Retry-After`, and record every attempt. A query that still fails keeps
the run incomplete; the gate's meaning does not change. Handle non-thread-safe adapters
and isolate per-query tenant context. Extend UI browsing, artifact bounds and progress with
the larger suite; do not just raise one input limit. Preview total calls/spend from candidates
× questions × repeats; distinguish total experiment spend, per-production-query cost and
unknown costs. Use lookup rather than requiring the whole corpus in the browser.

**Acceptance:** a declared 1,000-query scenario fits the agreed CI window with cancel/resume
working. Deterministic fixtures agree across serial/concurrent runs apart from timing;
stochastic services satisfy a declared repeat protocol, not a promise of identical outputs.
Validate the PRD's two concurrent workspace runs separately from per-query concurrency.

#### 1.6 A hosted decision view built for the buyer

Improve the existing hosted UI around decisions. Choose incremental changes or React after
observing usability friction; rendering technology is not a launch requirement. Show:

- quality and latency trends per project across releases
- a release timeline: verdicts, baseline changes, who approved what
- findings that recur across runs, with links to investigations
- a plain-language "what changed and why it matters" summary for the engineering lead
- the authenticated path to approve policy and baseline changes

This view needs records the hosted service does not yet hold (section 2):

- **Shared cases and benchmark identities.** Sync regression cases, snapshot identities
  (dataset hash, inventory hashes), judgment metadata and saved policies from the workbench
  to the project. Upload IDs, hashes and status by default; question and passage text
  require the explicit sharing setting from 1.0. Build on the local case model and its
  history rather than designing a second one.
- **A decision ledger that outlives raw evidence.** Keep a small record of each verdict,
  policy hash, run digests, exceptions (1.10) and approvals after full run payloads expire
  under retention. The trend and timeline views read the ledger, so a pilot plan can still
  show a quarter of release decisions.
- **Monitor results in history.** Let `workbench monitor` upload its result like `record
  --upload`, so scheduled checks appear beside release checks.
- **Readable evidence.** Locally, show question text and document titles next to IDs by
  joining the saved draft and document lookup. Hosted pages show the same only when the
  sharing setting allows; otherwise they show IDs and say why.

Apply the sharing boundary in 1.0. Preserve authorization and session/form protections if
the frontend changes; reuse the existing governance and audit records rather than parallel
decision records.

**Acceptance:** an engineering lead understands a release decision without learning every
ranking metric, and a second engineer reproduces it without the founder.

#### 1.7 Delivery channels and customer CI *(PRD FR13)*

- The first customer's preferred channel for monitor/check results, with retries,
  deduplication, sharing controls and resolution state. Signed webhooks can be a reusable
  starting point; add Slack/email according to use.
- GitLab publishing when a qualified customer needs it; existing GitHub support may suffice.
- Optionally, a GitHub App for organisations that do not want workflow tokens.

**Acceptance:** a scheduled monitor regression reaches the owning team's channel once, and
resolves when the next run passes.

#### 1.8 Production-to-benchmark upkeep *(promoted from Tier 2)*

Extend existing incident import, deduplication, passage review and freshness checks with one
maintained source integration, import cursors, reviewer routing and links from incident to
reviewed test, verified fix and later recurrence. Preserve filters and tenant context.

Keep representative production samples, deliberately difficult incident cases and protected
validation data distinct. Feedback/clicks select cases for review; they are not ground truth.
Prioritize by business importance, disagreement, novelty and stale evidence, while retaining
an independently sampled review slice beyond what current retrievers already find.

Reuse annotation guidance and review tools. Show coverage by intent/language/tenant/freshness;
distinguish unjudged, irrelevant and explicitly unanswerable. Source changes should propose
affected labels for review, not silently carry them forward.

**Acceptance:** a real incident becomes a reviewed test without repeated data entry, catches
a recurrence, and is flagged correctly when its source changes. Measure expert minutes per
accepted case and upkeep per release, not simply number of labels.

#### 1.9 Commercial delivery foundations *(new; precedes paid pilots)*

A buyer's security review and a customer's CI both meet these before any feature does.

- **Distribution without source access.** Publish wheels, and optionally a runner container
  image, to a channel with per-customer credentials. A private package index, a container
  registry or a releases-only repository would all work. Stop attaching the sdist to
  customer-visible releases. Revoking one customer's credential must stop their downloads
  without affecting others. Keep the existing no-PyPI rule.
- **A reusable CI step.** Wrap install, version and checksum pinning, record and check in
  one reusable GitHub step that the `init` templates call. Add the equivalent GitLab
  template only when 1.7 needs it.
- **Entitlement by contract and credentials.** Hosted features are already gated by tokens.
  Avoid phone-home licence checks that would break private runners or offline CI.
- **Support and compatibility policy.** State the supported versions and platforms, the
  compatibility promise for run, check, policy and benchmark formats (they already carry
  `format_version`), the deprecation window, upgrade notes and a security advisory channel.
- **Trust package.** For each deployment mode: a data-flow diagram, what leaves the
  customer's environment by default, subprocessors, retention and deletion timing, a DPA
  template and questionnaire answers drawn from [the threat model](threat-model.md). Make no
  compliance claim until an independent audit supports it.

**Acceptance:** a design partner installs a pinned release in their CI with credentials that
grant no source access; revoking them stops downloads. The buyer's security reviewer gets
their questions answered from the trust package without a founder call.

#### 1.10 Make the gate livable: calibration and audited exceptions *(new)*

A gate that raises false alarms gets switched off, and a gate with no legitimate override
gets bypassed. Both outcomes lose the evidence trail this product exists to keep.

- **Calibrate against measured noise.** Record the unchanged baseline several times on the
  customer's own runner and endpoint (an A/A comparison). Report observed metric and latency
  variation per metric and cohort, and how often the current policy would have failed an
  unchanged system. Propose thresholds (`max_drop`, `min_increase_ms`, cohort minimums) as
  a reviewable policy draft; never apply them automatically. Re-run after infrastructure
  changes, and show the cost preview from 1.5 first.
- **Accept a known regression explicitly.** Let an approver record an exception for named
  queries or metrics on one specific check: reason, owner, linked case and an expiry (a
  date or a number of releases). The check reads "fail, accepted by X", never "pass". The
  policy declares which rules allow exceptions and which role may grant them. Permission
  and no-answer failures should need a higher role or not be waivable at all. When an
  exception expires, its case reopens. Exceptions live in the decision ledger (1.6).

**Acceptance:** an exception unblocks exactly the named failure on exactly that check, and
any new failure still blocks. An engineer cannot waive a permission failure. Expired
exceptions resurface. CLI, CI, workbench and hosted views all show the exception. A/A
calibration on a declared setup yields a policy whose false-alarm rate is reported, not
assumed.

### Tier 2: deepen the advantage customers actually buy

#### 2.1 Source-to-context lineage and knowledge freshness *(preferred deeper bet)*

Trace source version → extracted text → chunk spans → embedding revision → indexed record →
retrieved evidence → actual context, using a customer-exported lineage manifest and the
context contract in 1.2. Detect empty extraction, lost
table content, evidence split across chunk boundaries, stale indexed revisions and duplicate
passages filling retrieval slots. Show original evidence beside its chunked form and name the
first observable loss. Do not build a general ingestion platform.

**Acceptance:** a known chunk-boundary or extraction defect produces a reproducible finding,
and repairing that stage recovers the evidence. The product sometimes recommends fixing
ingestion instead of switching embedding models.

Extend lineage into **knowledge freshness**: after an update, replacement, deletion or
archive, does the intended source version become retrievable within the customer's propagation
window, and does the old one disappear from retrieval, caches and context? Distinguish
intentionally historical queries from stale answers; customers define version precedence.
Handle conflicting old/new evidence explicitly. This could create a recurring job even when
application code changes infrequently. Verify a real stale-answer repair before building
broad ingestion connectors. Existing span/chunking experiments are the starting point.

#### 2.2 Permission lifecycle tests *(Priority 6; move up if it is the purchase trigger)*

Extend the permission matrix to change over time: group membership updates, revocation,
document moves, archived versions, cache isolation and reindex delay. Assert where a
forbidden passage travelled (reranker input, LLM context), not only whether it appeared in
final results. Traces must respect the same boundaries.

Derive expected access from independently reviewed entitlements or fixtures; copying the
application's own decision into the expected result tests little. Use isolated scenarios,
customer-controlled principals and declared propagation windows. Keep permission-test
content out of broadly shared case histories.

**Acceptance:** a customer-built revocation or cache scenario catches a planted lifecycle
defect and passes after correction. A passing suite is evidence about those cases, not proof
of authorization correctness.

#### 2.3 and 2.4: promoted into the core workflow

Production-to-benchmark upkeep is now **1.8**; evaluation confidence is part of **1.4**.
Keep these identifiers for continuity with the previous plan, not as duplicate workstreams.

#### 2.5 Optimization workspace and migration reports *(Priorities 5 and 7)*

Configure, budget, run, resume and compare controlled experiments in the browser: retrieval
depth, hybrid weights, reranking, embedding and chunking. Show several quality, latency and
cost tradeoffs, not one unexplained winner, and keep measured and estimated costs apart.
Add a migration report comparing two customer-configured systems (for example, pgvector to
Qdrant) on the same labelled suite, disclosing environment differences. Copying or
reindexing production data stays out of scope.

Validate logical corpus equivalence, ID mapping and filter semantics. Use source-span labels
for chunking comparisons; use relevance, not cross-space ANN recall, to choose embeddings.
Disclose hardware, region, cache/load and reindexing costs separately from query costs.

This doubles as a **one-off paid offering** ("migration assurance") and a natural entry point
to the subscription.

#### 2.6 Team workflow *(Priority 9)*

Reviewer assignment and status, annotation guidance with examples, owner summaries, and
handoffs between local workbench and hosted projects. Extend the existing roles and history.
Design the first session around "investigate this failure" or "compare this planned change";
keep FAISS internals available but optional.

### Tier 3: build only on repeated customer demand

| Idea | Build when |
|---|---|
| Elasticsearch/OpenSearch connector (vector + hybrid) | A qualified support-search customer is blocked; this is the most likely next backend |
| Postgres full-text + pgvector hybrid | pgvector customers run hybrid search |
| Milvus, LanceDB, Turbopuffer, Chroma | A qualified customer is blocked |
| Multi-step and agent retrieval (several calls, conversation state) | The first segment's applications need it |
| TypeScript helper SDK | Node teams are blocked by the HTTP contract |
| API-backed judge for existing label suggestions (e.g. Claude) | Labelling cost blocks adoption. Extend the existing judge interface and agreement tracking in `labelling/suggestions.py`; suggestions only, human-confirmed; customers still own labels |
| Reviewed typo, paraphrase and multilingual scenarios | Customers repeatedly miss these; keep query families together |
| PDF, table and image evidence inspection | Lineage work proves valuable on these source types |
| Consultant portfolio reports | Consultancies run several client projects |
| Issue tracker integrations (Jira, Linear) | Findings and alerts are used enough to justify upkeep |
| Self-serve billing (PRD FR14) | Manual invoicing becomes a bottleneck |
| Scoped investigation assistant | Structured diagnostics are trusted; assistant cites artifacts and cannot invent causes or approve releases |
| Knowledge coverage map | Customers supply reviewed intents/source inventory; distinguish missing knowledge from poor retrieval |
| Cost-reduction experiments | Buyer supplies a cost baseline; compare smaller context or cheaper rerankers under the same quality policy |
| Shadow candidate evaluation | Offline checks are adopted; sampled traffic, budgets and data boundaries are controlled; no automatic traffic switching |
| Retrieval-specific prompt-injection cases | Customers need it and answer/trace support exists; integrate specialist evaluators rather than become a broad security platform |
| Reviewed robustness packs | Real failures justify distractors, conflicting sources or adversarial phrasing; synthetic cases remain separately reported |
| Release evidence pack for audit reviewers | A regulated buyer asks for retained, signed release evidence; build on exports and the decision ledger; claim no regulatory compliance |
| Forced-migration checklist | Customers repeatedly buy migration assurance around embedding retirements or database upgrades; start as a section of the 2.5 migration report |
| Windows-native runners | A qualified customer needs them; task locking is implemented but unverified on Windows ([launch readiness](launch-readiness.md)) |

### Do not build

A broad prompt playground, general LLM tracing or agent orchestration, hosting customer
vector databases, automatic production repair, every vector backend, or more graph
visualizations as a substitute for the core workflow.

## 7. Build sequence

Work through separately reviewable implementation phases. Pilot conversations run alongside
engineering. This document proposes work; it does not itself implement or release features.

| Phase | Build / product work | Customer work alongside it | Evidence to continue |
|---|---|---|---|
| 0. Pick the job | Package the existing workflow as a demo and one-page pilot offer; 1.9 distribution channel, support policy and trust package; typed run errors from 1.5 | 10–15 qualified conversations about recent incidents and planned changes | Repeated costly job, named owner, several teams willing to supply real inputs. |
| 1. Connect one real system | 1.0 evidence boundary; smallest useful 1.1 adapter/import improvement | Assisted discovery using current capabilities | Correct IDs/context and one useful comparison; setup friction recorded. |
| 2. Trust the decision | 1.4 policy/uncertainty work, 1.10 calibration and exceptions, and evidence fields for selected failures | Agree labels, risk tolerance and repeat protocol | Customer understands the decision; missing evidence explicit; second engineer reproduces it. |
| 3. Complete the repair loop | 1.2/1.3 for the two observed failure classes | Resolve real incidents beside the current workflow | Verified fixes, reduced investigation effort and a reason to run again. |
| 4. Make repeat use economical | 1.8 benchmark upkeep; 1.5 execution as needed; 1.6 shared records and decision ledger; 1.7 delivery | Observe customer-initiated release cycles | Repeat decisions, manageable label/support costs and paid continuation. |
| 5. Deepen the buying reason | Tier 2 capability with strongest pull, preferably lineage/freshness | Prove value on another team/application | Similar reason to buy, less bespoke integration. |
| 6. Qualify commercial release | Selected deployment, supported modes, packaging, operations and usability | Confirm buyer, price, support and case-study permission | Launch scorecard in section 9 passes with linked evidence. |

Dependencies matter: evidence/sharing contracts precede richer artifacts; a callable pipeline
precedes live repair experiments; a common policy precedes optimizer approval; representative
labels precede meaningful quality claims. A full dashboard rewrite, every CI vendor and every
alert channel are not prerequisites for customer learning.

Avoid precise implementation dates before seeing representative traces, lineage manifests and
customer review effort. Those are major discovery risks. Scope each phase after inspecting
real inputs, keep one primary product bet active, and change the order when a customer blocker
provides evidence. No launch deadline is needed.

## 8. Commercial offer and distribution

**Offer:** an assisted retrieval reliability engagement covering one application, a reviewed
benchmark, one real candidate change, a reproducible investigation and an installed release
check. Agree upfront who owns labels, infrastructure, results and maintenance. A team
subscription becomes credible once checks, benchmark upkeep and shared decision history are
used repeatedly.

**Packaging and pricing hypotheses:**

| Offer | What the customer buys | Pricing approach to test |
|---|---|---|
| Diagnostic/migration engagement | A bounded decision and reproducible evidence report | Fixed scope and fee; separate setup labor from recurring software. |
| Team subscription | Repeated checks, maintained benchmarks, case history and collaboration | Team/workspace price with clear active-project and usage allowances. |
| Private deployment/support | Specific operational requirements | Quote against validated scope and support effort. |

The PRD's USD 99–249/workspace/month is an untested hypothesis. Do not let it anchor high-touch
pilots before measuring effort. Test a small number of explicit paid offers with comparable
scope; competitor prices and “sounds reasonable” are not willingness-to-pay evidence. Avoid
charging per vector or making every additional quality check feel like a penalty.

Measure subscription revenue less hosted costs, included evaluator costs and realistically
costed recurring support. Track one-time onboarding separately; disclose customer-side model
and database costs. Keep service revenue separate from software revenue when assessing
repeatability. Do not promise unlimited handholding under a low monthly plan.

**Distribution:** keep the proprietary/private-package boundary; 1.9 covers how paying
customers install without source access. A synthetic interactive demo, short video, sample
evidence report or controlled evaluation license can reduce trial friction without
publishing source or the package. A free read-only debugger is an optional
later decision, not a prerequisite or an assumption. Use Vectraudit consistently in customer
materials; name/domain clearance remains an owner follow-up, not established by this review.

**Acquisition:** a short reproducible failure → fix demonstration; detailed technical writeups
(the existing nprobe and experiment case studies are a start); direct conversations with
teams planning retrieval changes; consultancies as a secondary channel. Seek permission for
real case studies. Synthetic demos stay labelled as demos.

**Metrics to track from the first pilot:**

- time to first useful comparison
- setup effort versus labelling effort
- accepted fixes and disputed findings
- customer-initiated repeat runs
- release decisions influenced
- support hours per customer
- paid conversion and renewal reasons

For customers who use the hosted service, count runs, checks, baseline pins and active
projects per week from metadata it already stores (the admin usage export is a start)
rather than adding content telemetry. Local-only customers stay on the pilot log.

The primary value measure is **customer-confirmed useful decisions per active project**.
Do not promise revenue gains or prevented incidents without evidence.

## 9. When I would say “good enough to launch”

There are three different decisions: private design partnership, focused commercial launch,
and product–market fit. Do not collapse them into a feature checklist.

### Ready for a private design partnership

- [ ] Named team has a real failure/change, judgment owner and permission to use the inputs.
- [ ] One integration/deployment is reviewed for the engagement.
- [ ] Existing workflow produces interpretable evidence with incomplete/error states preserved.
- [ ] Scope, data handling, assistance and success criteria are understood.
- [ ] The partner installs the release without access to the source repository (1.9).

The existing foundation supports pursuing this stage now, once installation no longer needs
source access. Customer agreement and target acceptance remain necessary. This does not
require all proposed Tier 1/Tier 2 work or a public launch.

### Ready for a focused commercial launch

Both bars below must pass. These are founder decision gates, not industry benchmarks. Record
an evidence link, owner and date for every checked item. If a gate changes, record why.

#### Product and operations bar

- [ ] A fresh engineer connects through a documented supported path without founder-written
      bespoke code; setup and label-creation time are measured separately (1.1).
- [ ] The chosen failure classes are explained with linked evidence, including mixed/unknown
      outcomes. Retrieval, context and answer judgments remain separate where advertised (1.2).
- [ ] An engineer verifies a real repair and turns it into a repeatable release check with
      less effort than their previous workflow (1.3).
- [ ] Optimizer, workbench, CLI and CI all apply one release policy, and exploratory results
      can never look approved (1.4).
- [ ] Benchmark review has an owner, representative coverage, source-change handling and a
      protected validation strategy. Noise/false alarms and upkeep effort are measured (1.4, 1.8).
- [ ] The agreed workload fits its declared window with retry/cancel/resume evidence. Claim
      the PRD's 1,000-query/two-run capacity only after end-to-end validation (1.5).
- [ ] Every failed query reports its cause, and transient endpoint failures retry within a
      declared budget before a run is called incomplete (1.5).
- [ ] A lead understands the decision and a second engineer reproduces it. The target team's
      CI and delivery route work; every vendor/channel is unnecessary (1.6, 1.7).
- [ ] Cases, benchmark identities and release decisions are shared within the project and
      remain available for the agreed history window, independent of raw-run retention (1.6).
- [ ] Policy thresholds come from measured A/A noise on the customer's setup. A deliberate
      regression can be accepted only through an audited, expiring exception that never
      reads as a pass, and safety failures need a higher role or cannot be waived (1.10).
- [ ] A specific deeper advantage customers pay for works on real systems—likely lineage,
      freshness or permission lifecycle. A different repeated advantage can satisfy this gate
      if documented; a predetermined feature cannot substitute for customer value.
- [ ] No known correctness defect can turn incomplete or incomparable evidence into an
      approved release.
- [ ] The release artifact, supported backends, declared workload and chosen deployment have
      current end-to-end evidence. For the hosted service this includes tenant isolation,
      restore, revocation, sharing boundaries and deletion in the target environment.
- [ ] Installation, upgrades, rollback/recovery, support ownership and limits are documented;
      mocked sign-in and local restore tests are not presented as live deployment proof.
- [ ] Customers install and upgrade through revocable credentials that grant no source
      access. Supported versions, artifact-format compatibility and the trust package are
      published to buyers (1.9).

#### Customer and commercial evidence bar

- [ ] At least five independent teams in the chosen segment complete meaningful work on their
      own application/data; failed and abandoned pilots are recorded too.
- [ ] At least three use it without founder prompting across several real changes, ideally
      6–8 weeks or an equivalent number of release cycles, with declining founder assistance.
- [ ] Several customers pay, and at least two choose paid continuation after using the
      workflow. If annual contracts prevent observing renewal, say so rather than infer
      retention.
- [ ] Users explain why Vectraudit is worth adding to their existing tool or script, and the
      reason is similar across teams.
- [ ] A fresh engineer can connect, interpret and repeat a result from the documentation;
      support effort is measured and sustainable.
- [ ] Scope, customer-side costs, support responsibility and price are understood by a named
      buyer.
- [ ] Measured onboarding, hosting and recurring support costs make the offer sustainable.

When both bars are met, launch to the proven segment with the validated scope, even if many
brainstorming ideas remain unbuilt. Waiting for unrelated breadth adds little.

### Evidence of product–market fit

After launch, look for continued retention, paid renewals/expansion, new teams buying for the
same reason, referrals and acquisition that repeats without exceptional founder effort.
Track cohorts and why customers stay or leave. Five pilots and a polished product can justify
a launch; they cannot establish durable product–market fit. Finishing a feature list cannot
supply that evidence.

## 10. Customer discovery and explicit decisions to change direction

### Discovery that can change the roadmap

Ask about behavior and artifacts rather than whether someone likes our feature ideas:

1. Describe the last retrieval-related failure: who noticed, what was affected, how was the
   cause determined? Ask to inspect a sanitized trace or incident record.
2. What did you try, and how much engineering/domain-expert time did it take?
3. What change is coming next, and what currently decides whether it ships?
4. Show the existing benchmark/script/tool. What still requires manual work?
5. Who owns labels, source updates, false alarms and release approval?
6. What can run locally, what may be shared, and which deployment is acceptable?
7. Who approves spending, and what would need to happen for paid continuation?

Keep an evidence ledger: segment, trigger, alternative, customer-reported effort/impact,
integration/label blockers, buyer and next commitment. Separate observed behavior, statements
and our interpretations. Customer content should not enter this repository by default.

### Design-partner experiment

Recruit 3–5 initial partners with the same job. Agree one application, a real change,
benchmark responsibility, data boundary and success criteria. A reviewed 30–50-question suite
can start the workflow; it is not statistical sufficiency for every decision.

Record their current process first. Connect, investigate, verify a repair and install a
repeatable check with Vectraudit. Observe the next change with less founder help. Keep failed
pilots in the results; demos and friendly testimonials do not replace behavior.

| Experiment | Evidence to collect | Decision if it fails |
|---|---|---|
| Can teams connect? | Time to valid IDs/context and first result; custom code/support needed | Fix the dominant mapping/adapter issue before adding backends. |
| Is diagnosis better? | Time to an accepted explanation and verified repair versus current workflow | Narrow to a class we can explain; drop unsupported root-cause claims. |
| Is benchmark upkeep affordable? | Expert review time, disputed/stale labels and upkeep per release | Improve review/lineage or price assisted setup explicitly. |
| Does use become routine? | Customer-initiated checks on later changes, cases revisited | Investigate recurring value; consider a diagnostic service if that is the actual job. |
| Is a separate purchase justified? | Paid offer accepted by budget owner, with reason | Revisit buyer, scope or proposition; integration requests alone are not demand. |

Reassess after the first 10–15 qualified conversations and after the first few completed
pilots. These are learning checkpoints, not statistically validated thresholds. If existing
tools consistently solve the job or teams cannot identify recurring value, pause feature
expansion and change the proposition. If lineage/permissions repeatedly drive payment, narrow
around that job. A service-led business is legitimate but has different economics from SaaS.

### Measure behavior through the funnel

Track qualified conversation → usable real input → accepted decision → independent repeat
use → paid continuation. Define activation as a useful customer-accepted result on their own
application that they can rerun, not opening a demo.

The primary value measure remains customer-confirmed useful decisions per active project,
with the actual decision and evidence recorded. Track setup, labeling/upkeep and investigation
time separately, plus false alarms, disputed findings, support hours and continuation reasons.
At small sample sizes report counts and stories, not percentages without denominators. Measure
retention by opportunities to make changes: no release this week differs from bypassing the tool.

## 11. The experience we should build toward

A customer should eventually be able to tell this story:

> “Our assistant answered an account-recovery question incorrectly. The required exception
> was in a retrieved source but missing from the model context. Vectraudit showed the loss.
> We tested a packing change, checked the other important questions and permissions, and
> saved the verified configuration. The next release now checks this case automatically.”

This is a proposed experience, not a customer result or a claim of current answer support.

The main journey should expose five decisions:

1. **Connect:** start from a real incident or planned change; map identities and see what
   Vectraudit can and cannot observe.
2. **Establish evidence:** review required passages, expected behavior and comparison context;
   show unjudged cases and sampling limits before presenting a verdict.
3. **Investigate:** show expected evidence beside its observed path, facts separate from
   hypotheses, with a way to dispute a diagnosis or request missing evidence.
4. **Test:** compare a controlled change, recovered cases, new failures and budgets; validate
   the selected candidate under the agreed full release policy.
5. **Protect and return:** create the regression case, reviewed configuration and delivery
   route; reopen cases when recurrence or stale evidence appears.

Keep advanced FAISS inspection accessible when it helps an investigation. The lead's view
should answer: what changed, which important cases worsened, what is uncertain, who reviewed
it, and what action is proposed. Improve navigation and defaults before adding more surfaces.

## 12. Next concrete work

1. Move customer installation off the source repository (1.9) before any outside team
   installs a release.
2. Package the existing fail/fix/release demonstration into a short pilot offer with exact
   supported scope and customer prerequisites.
3. Start the discovery ledger and obtain representative failures, traces and upcoming-change
   examples through permissioned design partnerships.
4. Meanwhile, make the small fixes from the second pass that need no customer evidence:
   keep typed error codes on failed queries, show question text beside IDs in the local
   workbench, and count permission clones as one query family. Each is a separately
   reviewable phase.
5. Define the evidence capability/identity contract (including list-valued context and
   conversation history) and the hosted export boundary before richer answer/trace fields.
   Extend the existing architecture rather than restart the product.
6. Choose the first import path and two diagnostic failure classes from those examples.
7. Design policy/uncertainty changes, A/A calibration, exceptions and the repair journey
   together, with customer acceptance cases; implement one reviewed phase at a time.

The ambition is a product teams depend on to keep retrieval working as their applications
and knowledge change. Each addition should make that recurring job more trustworthy, easier
or cheaper—and earn its place through observed use.
