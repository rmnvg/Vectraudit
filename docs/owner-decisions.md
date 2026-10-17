# Owner decisions before the first pilot

**For the founder.** Every open *[bracket]* in the [pilot offer](pilot-offer.md) and the
[trust package](trust-package.md), plus the owner actions in the
[launch scorecard](launch-scorecard.md), with options, a recommendation and the reason.
**The recommendations are proposals, not decisions.** Nothing here has been applied to the
customer-facing documents. Commercial numbers are starting hypotheses to test, as section 8
of the [strategy](startup-strategy-2026-10-08.md) asks, not market evidence.

Decide in the order below. Each group unblocks a stage.

## Before any outside conversation

These cost nothing to change later, but a prospect will ask.

| Decision | Options | Recommendation | Why |
|---|---|---|---|
| Repository visibility | Public · private | **Private, now** | The product is proprietary, and the strategy's distribution model assumes no source access. A public repository makes the pilot offer's "no access to our source" untrue. |
| Name and domain | Use "Vectraudit" as is · clear it first | **Clear the name and register a domain before printing materials** | Section 8 leaves name and domain clearance open. A security contact and advisory channel (below) need a domain you control. |
| Which leave-behind to send | Sample evidence report · context-story page · both | **Both, the context story first** | The context story ([`examples/context_story_demo.py`](../examples/context_story_demo.py)) shows the differentiating loop: retrieval looked fine, the evidence never reached the model, the fix was verified and protected. The sample report shows the release gate. Both are labelled synthetic. |

## Before the first design partnership (unpaid or nominally paid)

| Decision | Options | Recommendation | Why |
|---|---|---|---|
| Deployment for the first pilots | Local only · self-hosted control plane · operated by us | **Local, plus the customer's own self-hosted control plane if they want the team view. No service operated by us yet** | Removes most open trust-package brackets at once (region, hosting provider, subprocessors, our encryption, our backups). Nothing leaves the customer's environment, which is the easiest security review to pass. Offer operated hosting once a partner asks for it. |
| Distribution channel | Private package index · container registry · releases-only repository | **Releases-only repository or a private index with per-customer tokens, as in [RELEASING.md](RELEASING.md)** | Satisfies 1.9 without new infrastructure. Test one revocation before the first outside install: the scorecard's blocking gate. |
| Pilot length | 4 · 6 · 8 weeks | **6 weeks, with the follow-up session in week 5 or 6** | The strategy wants to observe the team running the check on a *later* change without help. Four weeks rarely contains a second real change; eight weeks slows learning across the first 3–5 partners. |
| Sessions and support | Unlimited · fixed sessions plus async | **Four 60–90 minute sessions (the pilot offer's table) plus an async channel answered within one business day** | Section 8 warns against unlimited hand-holding and asks to measure support effort. A fixed shape makes support hours comparable across pilots. |
| Engineer and domain-owner time | Keep the estimate · remove it | **Keep "2–4 hours engineer, 1–2 hours domain owner", labelled as an estimate** | Customers need a number to say yes. Investigation records and the discovery ledger will replace it with measured time after the first pilots. |
| Success criteria | Generic · agreed per pilot | **Agree them per pilot in writing before session 1, from the pilot offer's list** | The strategy defines activation as a customer-accepted result they can rerun. Writing it down first stops later reinterpretation. |

## Before charging for a pilot

| Decision | Options | Recommendation | Why |
|---|---|---|---|
| Pilot price structure | Free · fixed fee · fee credited to a subscription | **A fixed fee for the assisted pilot, credited against the first subscription period if they continue** | Section 8: fixed scope and fee for the engagement; keep service and software revenue apart. A credit tests continuation without discounting the software. |
| Pilot price level | Any number | **Pick two price points and offer each to comparable prospects; record acceptance and reasons in the ledger** | Section 8 says the PRD's USD 99–249/month is untested and must not anchor high-touch pilots. Only offers accepted or refused by a budget owner are evidence. I have no market data to recommend a number. |
| Payment and cancellation | Upfront · on milestones | **Half at start, half after session 3 (the verified fix); either side can stop after session 1 with the second half waived** | Ties payment to the moment the value is demonstrated, and lets a team that cannot connect leave cheaply. That is a learning outcome the ledger should record. |
| What continues after the pilot | Nothing · subscription · paid support | **Offer the team subscription; the release check keeps working locally even if they decline** | Local-first means stopping does not break their CI. That reduces the buyer's risk, and the strategy wants continuation to be a real choice we can observe. |

## Before a security review asks (trust package)

| Bracket | Recommendation | Note |
|---|---|---|
| Hosted project, operated by us *[if offered]* | **Mark "not offered during pilots"** | Follows from the deployment decision above. |
| Region and hosting provider | **Not applicable while not offered**; decide when you offer it | Choose where the first paying partners are, and list the provider as a subprocessor. |
| Subprocessors | **"None" for local and self-hosted** (already stated) | Add hosting and email providers only when operated hosting exists. |
| Encryption in transit (ours) | **Not applicable while not offered** | Self-hosted relies on the customer's TLS proxy, which the document already says. |
| Encryption at rest | **"Provided by the customer's database or volume encryption (self-hosted)"** | A fact per deployment, not a promise we can make for them. |
| Backup retention | **Customer-controlled for self-hosted** | Set our own figure (for example 30 days) only when operating hosting. |
| Penetration test or certification | **"None yet"; plan a third-party penetration test before the first operated-hosting customer** | Never imply a certification. Local and self-hosted pilots do not expose a service of ours to test. |
| Incident notification timelines | **Commit to notifying affected customers without undue delay, and within 72 hours of confirming an incident that affects their data. Have this reviewed by a lawyer before signing contracts** | 72 hours matches common regulatory expectations for notifying authorities, but your contractual obligations are a legal question, not one I can settle. |
| Licensing and distribution *[channel being provisioned]* | **Replace with the channel's name once one revocation has been tested** | The bracket should disappear only with evidence, as the scorecard requires. |
| Deprecation window *[N months]* | **6 months, and at least one minor release with a warning** | Long enough for a customer's release cycle; formats already keep old files loading. |
| Security fixes *[latest minor version]* | **The latest minor release only, during pilots** | One supported line is sustainable for a single maintainer; widen it for team contracts. |
| Security contact and advisories | **A dedicated mailbox on your domain (for example security@…), and advisories emailed to each pilot's named contact** | Needs the domain decision first. A mailing list is enough at pilot scale. |

## Owner actions only you can take

From the [launch scorecard](launch-scorecard.md) and the implementation log's
"Next increments":

1. Make the repository private, then provision the distribution channel and test one
   credential revocation ([RELEASING.md](RELEASING.md)).
2. Run the reusable CI step once on GitHub-hosted runners in a staging repository.
3. Apply the decisions above to the pilot offer and the trust package, and remove each
   bracket you have decided.
4. Hold 10–15 qualified conversations, recorded in the [discovery ledger](discovery-ledger.md)
   outside this repository.
5. Start the first design partnership. Their real traces, incidents and changes then decide
   the next product work: the trace format, the two failure classes and the Tier 2 bet.
