# Launch scorecard

**Status as of October 11, 2026, after Step 43.** This maps every gate in section 9 of the
[strategy](startup-strategy-2026-10-08.md) to what exists in the repository and what is
still needed to check it. Steps refer to the
[implementation log](startup-implementation-2026-10-08.md).

Section 9 asks for an evidence link, owner and date for every checked item. **No box is
checked here.** Code, tests and documents can show a capability exists; a gate is met only
when the evidence the gate names exists, which for most gates means a real team on a real
system. Update the "Evidence" column with links (ledger rows, pilot reports, CI runs) as
they come in, and tick a box only then.

Status meanings:

- **Built:** implemented and tested in this repository, not yet shown on a customer's system.
- **Partly built:** some of the gate exists; the rest is listed.
- **Not built:** deliberately waiting for real inputs (the strategy says to choose formats and
  failure classes from design partners' data).
- **Owner action:** needs a decision or provisioning only the founder can do.
- **Customer evidence:** cannot be produced by building; it comes from pilots.

## Ready for a private design partnership

| Gate | Status | What exists | Still needed | Owner |
|---|---|---|---|---|
| Named team with a real failure or change, judgment owner and permission to use inputs | Customer evidence | [Discovery ledger](discovery-ledger.md) with qualification rules; [pilot offer](pilot-offer.md); [sample evidence report](pilot-release-validation.md) (Step 36) | Qualified conversations recorded in the ledger | Founder |
| One integration or deployment reviewed for the engagement | Partly built | Python/HTTP adapters, framework wrappers, five vector stores, connection test and diagnostics; local-only or hosted modes described in the [trust package](trust-package.md) | Review against the partner's actual application and chosen mode | Founder with partner |
| Workflow produces interpretable evidence with incomplete and error states preserved | Built | Typed failure causes and retries (Steps 1, 4); incomplete runs never pass; plain-language summaries (Step 16); readable questions and titles (Steps 2, 24) | Confirmation that the partner's engineer and lead can read it | Partner |
| Scope, data handling, assistance and success criteria understood | Owner action | Pilot offer and trust package drafts (Step 26) with bracketed owner decisions | Fill the brackets (price, terms, support hours, hosting region, subprocessors, retention, advisory channel); agree success criteria per pilot | Founder |
| Partner installs without access to the source repository (1.9) | Owner action | Wheel-only distribution channel design, pinned installer, revocation checklist (Step 8, [RELEASING](RELEASING.md)); one reusable CI step (Step 27) | Make the source repository private; provision the distribution channel and per-customer credentials; run the CI step once on GitHub-hosted runners; test a revocation | Founder |

**Blocking now:** the last two rows. The first needs conversations. These are owner and
customer actions, not code.

## Product and operations bar

| Gate | Status | What exists | Still needed |
|---|---|---|---|
| Fresh engineer connects without founder-written code; setup and labelling time measured separately (1.1) | Partly built | Adapters, wrappers, `init`, connection test, document lookup; OpenInference trace import with a capability report (Step 39) | A real instrumented application imported (or another reader if partners use Langfuse or LangSmith); measured setup by an engineer who is not the founder |
| Failure classes explained with linked evidence, mixed and unknown outcomes kept; retrieval, context and answer judgments separate (1.2) | Partly built | Stage traces and loss attribution for retrieval; evidence coverage contract (Step 9); model-context records and omitted/truncated findings, from pipelines or trace prompts (Step 40); required-evidence groups, imported answer judgments with human agreement, and per-question diagnosis separating context loss from wrong answers (Step 41) | Citation support; the chosen failure classes explained on a real application; checking text matching and model judges against the customer's human review |
| Engineer verifies a real repair and turns it into a release check with less effort (1.3) | Partly built | Findings, regression cases, verification, optimizer under the release policy (Step 13), exports to CI; investigation of a failed question, declared fixes, paired exploratory trials and held-out validation under the policy, in the CLI (Step 42) and the workbench (Step 43) | Fixes for the two failure classes seen in pilots, declared for the partner's pipeline; measured effort against the team's current workflow |
| Optimizer, workbench, CLI and CI apply one release policy; exploratory results never look approved (1.4) | Built | Steps 13, 28, 31; one comparison function on every surface | Observation that teams read the labels as intended |
| Benchmark review owner, representative coverage, source-change handling, protected validation; noise and upkeep measured (1.4, 1.8) | Partly built | Held-out splits and query families (Step 3), freshness and snapshots, A/A calibration (Steps 12, 25), cohort controls (Steps 20, 33) | A maintained production source integration (1.8); measured upkeep on a real benchmark |
| Agreed workload fits its declared window with retry, cancel and resume (1.5) | Partly built | Concurrency, rate limits, retries, checkpoints, 1,000-question suites, two concurrent runs (Steps 4–6, 32), spend preview and usage (Steps 19, 30) | A declared customer workload measured end to end in their CI window |
| Every failed query reports its cause; transient failures retry within a budget (1.5) | Built | Steps 1, 4 | None beyond customer use |
| Lead understands the decision; second engineer reproduces it; CI and delivery route work (1.6, 1.7) | Partly built | Lead summaries and trends (Step 16), shared records (Step 15), webhooks and Slack (Step 18), monitor staleness (Step 29), CI step (Step 27), paired runs (Steps 34, 37) | A lead and a second engineer on a real team; the first customer's actual channel |
| Cases, benchmark identities and decisions shared, kept for the agreed history window regardless of run retention (1.6) | Built | Decision ledger (Step 10), shared records (Step 15), recurring problems (Step 22) | Retention window agreed with the customer |
| Thresholds from measured A/A noise; regressions accepted only through audited, expiring exceptions; safety failures need a higher role or cannot be waived (1.10) | Built | Steps 11, 12, 25, 28 | A/A calibration on the customer's own runner and endpoint |
| A deeper advantage customers pay for works on real systems (Tier 2) | Not built | Chunking and span experiments, permission scenarios as starting points | Choice of lineage, freshness or permission lifecycle from repeated customer pull, then proof on a real system |
| No known correctness defect can turn incomplete or incomparable evidence into an approved release | Built (no known defect) | Incomplete and errored runs never pass; errors and incomparable runs cannot be accepted (Step 11); comparability checks on query sets, families, repeats, concurrency, environment, overlap and pairing; adversarial audit fixed two defects (Step 38) | A review by someone other than the author before claiming it |
| Release artifact, backends, workload and deployment have current end-to-end evidence, including hosted tenant isolation, restore, revocation, sharing and deletion in the target environment | Partly built | Route-level tenant audit, deletion and retention tests, PostgreSQL runs; live pgvector, Pinecone and Weaviate tests run only where CI provisions them | Runs in the chosen target environment; live backend runs for the backends a customer uses |
| Installation, upgrades, rollback, support ownership and limits documented; no mocked proof presented as live | Partly built | [RELEASING](RELEASING.md), `init --ci-only` upgrades (Step 27), [cloud operations](cloud-operations.md), trust package draft | Owner-approved support policy; a real upgrade and rollback on a customer install |
| Revocable credentials without source access; supported versions, format compatibility and trust package published to buyers (1.9) | Owner action | Distribution design (Step 8); compatibility policy draft in the trust package (Step 26) | Provisioned channel, approved compatibility policy, published trust package |

## Customer and commercial evidence bar

Every gate in this part of section 9 is **customer evidence**: five independent teams doing
meaningful work, three using it unprompted across several changes, paid continuation, a
repeated reason to add Vectraudit to existing tools, documentation-only onboarding,
understood costs and sustainable support economics. Record them in the
[discovery ledger](discovery-ledger.md) and summarise each checkpoint here with links.

## What to do next, in order

1. **Make the source repository private** and provision the distribution channel (1.9).
2. **Fill the owner decisions** in the pilot offer and trust package.
3. **Hold 10–15 qualified conversations**, recorded in the ledger, using the sample evidence
   report.
4. **Run the CI step once on GitHub-hosted runners** in a staging or partner repository.
5. **Start the first design partnership.** Then confirm the trace format (1.1), failure
   classes (1.2, 1.3) and deeper bet (Tier 2) from what that team actually has.

Building more features before step 5 would be guessing at requirements, which the strategy
warns against.
