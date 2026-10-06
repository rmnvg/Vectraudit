# Vectraudit: next product–market fit priorities

Review date: October 6, 2026. Reviewed `main` at `cb26019`, which includes the five
market-fit workflows merged in PR #28. This document supersedes the implementation
priorities in the October 5 review; that earlier document remains a historical proposal.
This review inspected code and current public product documentation. It did not rerun
engineering tests or observe customers. Customer stage, paid usage and retention remain
unknown; the commercial recommendations provisionally assume an early pilot stage.

## Recommendation

Complete one reliable journey from a customer's real retriever to a repeatable release
check. Start selling guided pilots while improving that journey. A broader feature list
cannot establish whether customers have a frequent, expensive problem and will pay for
Vectraudit to solve it.

Initial customer hypothesis: an engineering team shipping a production B2B knowledge
assistant, changing retrieval regularly, with an owner accountable for search failures.
Start conversations around an upcoming embedding/reranker/filter change or a recent
retrieval incident. Qualify willingness to provide questions, source-document judgments,
read-only access and a second evaluation session. An engineering lead is a plausible buyer,
not an established buyer profile.

Suggested promise: **“Know which customer questions a retrieval change breaks, inspect the
recorded evidence, and apply the same release rules locally and in CI.”**

## What changed since the last review

Passage grading, incident-file import, permission scenarios, immutable benchmark snapshots,
freshness review and persistent regression cases now exist. Recommending these again as
missing features would be wrong. The remaining opportunities lie in integration and depth:

| Observed implementation | Remaining customer friction |
|---|---|
| Guided comparison and case verification call `compare_runs(baseline, candidate)` without a selected policy (`pipelines/guided.py`, `pipelines/workflows.py`). | A customer's CLI/CI policy is not carried through the browser workflow. A default-policy verdict and a custom-policy verdict can differ legitimately but confusingly. |
| `Policy.min_queries` defaults to 1; the scaffolded policy requires 30. Small-sample warnings already exist. | Exploratory demonstration results need clearer separation from an agreed release decision. Neither 1 nor 30 is a universal statistical sufficiency threshold. |
| Snapshots capture dataset/revision/inventory; jobs select drafts and configured adapters. The existing generated GitHub workflow records backend connections. | There is no guided export tying the chosen full pipeline, frozen query split and policy into a reusable CI package. Existing CLI and GitHub integrations are foundations, not missing capabilities. |
| The passage grader browses local corpus order, ten documents per page. Pooling and run-disagreement queues exist elsewhere. | Engineers still have to find the right passages to judge. The richer labeling and workbench workflows remain partly separate. |
| Guided corpora are capped at 5,000 documents and benchmarks at 500 questions; source text needs local import. | A production customer with a large corpus needs a small, relevant document lookup/pool, not necessarily a larger browser upload limit. |
| Permission/no-answer assertions are absolute gates; positive relevance is evaluated through quality metrics and policies. | Teams may need an explicit “this particular question must still retrieve this document” rule even when average quality remains acceptable. |
| Doctor is offline, and real adapters are registered at startup. | Users still need to establish that their actual pipeline forwards context, returns stable IDs and exposes useful stage evidence. |

## Build order

### 1. One saved release configuration, used everywhere

This is the next recommended engineering milestone. A release configuration should bind:

- Frozen dataset version and exact exploratory or held-out query set.
- Selected baseline/candidate adapter configuration and k.
- Reviewed policy version and its hash.
- Corpus/version evidence when available, runner settings and package version.

Add a policy form using the existing schema: required metrics, tolerances, minimum measured
coverage, cohort rules and latency constraints. Explain each setting in customer language.
Use the same explicit policy in guided comparison, case verification, exported CLI commands
and CI. Keep a clearly labeled exploratory mode; do not silently turn demo defaults into a
customer's release contract. Changes to approved policy must produce a new configuration
version and must not rewrite historical case verification evidence.

From a successful comparison, offer **Export release check**: the selected benchmark split,
policy, pipeline manifest, setup instructions and a GitHub workflow. Include required adapter
source/dependencies or identify them explicitly as customer-owned prerequisites. Reference
secrets by environment variable. Report nonportable local paths and missing dependencies
before export. Let the customer review/apply the files; do not automatically push them.

For Python pipelines, ensure baseline and candidate actually execute their intended code
revisions. Changing only a connection file is insufficient to compare application changes.
For HTTP pipelines, identify the two deployments being compared. Do not copy whole corpora
or secret values into exports implicitly.

**Acceptance:** on fixed run artifacts, browser, CLI, case verification and CI produce the
same verdict under the same policy. A downloaded configuration runs from a clean customer
checkout with documented prerequisites. A stricter policy changes every surface consistently.
Exploration and held-out execution remain separate. Missing artifacts fail clearly.

**Why first:** this turns an occasional local evaluation into a repeatable release practice,
and makes the meaning of “verified” understandable to the team.

### 2. Review the passages that matter

Connect the existing benchmark judgment queue and local annotation engine to the guided
passage grader. Pool documents from actual exploratory baseline/candidate runs, prioritize
unjudged disagreements, and allow corpus browsing to inspect outside the pool. Conceal
rank/source during grading where appropriate; reveal the evidence afterwards.

Retain separate irrelevant judgments so reviewers do not repeatedly grade the same document.
Preserve context/filter-specific queries rather than treating every repeated question text
as redundant. Exclude held-out cases from exploratory prioritization. Offer independent
review and disagreement resolution using the existing annotation machinery rather than
building a second competing review model.

Support a bounded document lookup registered locally at startup, or a customer-exported
pool, for remote retrievers. Stable document/chunk/source mappings must be checked. Browser
input must not become a way to register arbitrary executable adapters or outbound targets.

**Acceptance:** a pilot engineer judges the candidate passages for their actual failing
questions without scanning the full corpus. Measure time per reviewed question, manual
lookup work and reviewer disagreement before and after. Do not promise a fixed time saving
without measurement. Missing document text remains explicit.

### 3. Protect business-critical questions explicitly

Add an optional per-query positive assertion such as “document X appears in the top 5,”
or “at least one document from this approved set appears in the top 3.” Keep these distinct
from graded relevance and from forbidden/no-answer assertions. Every required assertion
needs measured results; missing output cannot count as a pass. Show the assertion beside
its evidence in the report and CI result.

An example is a cancellation or account-recovery question: average nDCG can remain acceptable
while that one important intent breaks. Existing metric/cohort policies remain useful;
this addition makes an individual requirement direct and reviewable.

**Acceptance:** with 100 otherwise unchanged questions, a required question's lost document
blocks the release even when aggregate metrics stay inside tolerance. Repairing it clears
that assertion. Assertions carry through imports, snapshots and release exports.

**Build condition:** prioritize this when a pilot names specific intents whose failure is
unacceptable. It should not become an unvalidated catalog of business rules.

### 4. A real-pipeline connection check

After a user registers an adapter, provide a small, explicitly initiated diagnostic run:
validate the response contract, document lookup, context/filter forwarding where testable,
timeouts and available stage traces. Reuse offline doctor for applicable configuration
checks; distinguish static validity from successful live execution. A single live query
cannot establish permission correctness or deployment capacity.

Show a short capability result: final-ranking evidence available, stage traces available,
corpus version known/unknown, and which checks were actually exercised. Keep credentials
local and make the destination visible before sending a query.

**Acceptance:** a new user can diagnose a missing environment variable, unmapped document
ID or unsupported context without reading an opaque exception. Measure time to first useful
customer comparison separately from the time spent labeling.

**Build condition:** move this ahead of milestone 2 if onboarding observation shows
connection setup is the main point where customers stop.

## Position against existing alternatives

Current official sources confirm significant overlap:

- [Langfuse's CI documentation](https://langfuse.com/docs/evaluation/experiments/experiments-ci-cd)
  describes release policies, experiment execution and regression gates in GitHub Actions.
- [LangSmith evaluation](https://www.langchain.com/langsmith/evaluation) includes automated
  evaluators and human review through annotation queues.
- [Braintrust datasets](https://www.braintrust.dev/docs/annotate/datasets) connect datasets to
  evaluation and production/staging workflows.

These facts do not prove that competitors lack the additions proposed here. The strategic
inference is that another generic evaluation dashboard is hard to differentiate. Test
whether private execution, retrieval-specific document-loss evidence, and a convenient
release workflow together justify adding Vectraudit to a customer's existing stack. Treat
an internal Python script as a competing alternative as well.

## Commercial work to do alongside development

Recruit three qualified design partners using the current product. Do not wait for all four
milestones. Offer one application, one supported backend, a reviewed benchmark, one real
change comparison and a repeat run owned by the customer. Establish a budget conversation
and a named purchase decision-maker early; do not mistake demo interest for paid demand.

Use a simple pilot log, initially outside product telemetry:

| Signal | Record |
|---|---|
| Activation | Customer data connected; first interpretable real comparison; elapsed setup and labeling time separately. |
| Value | Specific release decision affected; which finding the engineer accepted or disputed. |
| Retention | Whether the customer initiates another evaluation for a later change without prompting. |
| Operational cost | Founder/support time, failed setup steps and nuisance failures. |
| Commercial demand | Paid pilot/continuation accepted or declined, buyer and reason. |

A useful early decision rule is two of three pilots independently returning for a second
release and agreeing to paid continuation. This is a proposed founder checkpoint, not a
universal definition of product–market fit. Subsequent renewals, sustained use and repeatable
acquisition among similar teams matter more than one successful pilot cohort.

Package the first offer as an assisted retrieval release-readiness pilot. Test pricing with
qualified buyers and support costs; no amount is validated here. If buyers only need an
occasional diagnosis, evaluate a service/one-off offer instead of assuming subscription
retention. If they will not adopt a separate tool, consider a narrow integration with their
existing evaluation platform.

For acquisition, prepare a short demonstration of a plausible retrieval regression, a
reviewable sample report, installation instructions and a clear pilot request. Existing
synthetic case studies can support the demonstration if labeled as such. Customer claims
need actual customer evidence and permission. This review did not contact anyone, publish
anything, change pricing or deploy a service.

## Defer until evidence supports it

More databases, a general LLM tracing platform, agent dashboards, automated fixes, payment
infrastructure, and broader enterprise features. Do not add answer-generation evaluation
merely because competitors have it: first establish that buyers want Vectraudit to own that
problem rather than integrate with an existing answer evaluator.

The immediate decision is to validate whether the current product helps a team make a real
release decision, then remove the biggest obstacle to repeating that decision. Milestone 1
is the strongest code-grounded next bet; customer observation can change its priority.
