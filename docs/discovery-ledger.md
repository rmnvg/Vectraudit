# Discovery ledger

**Template for the founder.** It turns sections 4, 8 and 10 of the
[strategy](startup-strategy-2026-10-08.md) into a record kept after every qualified
conversation and pilot session. The template lives here; **the filled-in ledger does not.**
Keep it outside this repository (a private spreadsheet or notes store with access limited to
the people running pilots). Customer names, incidents, traces and quotes never enter the
repository, its issues or its test fixtures.

A blank spreadsheet header is in [templates/discovery-ledger.csv](templates/discovery-ledger.csv).

## Why keep it

No feature list shows product–market fit; customers' behaviour does. The ledger is where
that behaviour is recorded consistently enough to make the strategy's decisions:

- **After 10–15 qualified conversations:** is there a repeated, costly job with a named
  owner and teams willing to supply real inputs? If not, change the proposition before
  building more.
- **After the first few completed pilots:** did teams connect, accept a diagnosis, verify a
  fix and run the check again on their own? If not, use the "decision if it fails" column
  below.
- **Before the next roadmap choice:** which trace format, failure classes and deeper
  advantage (lineage, freshness, permissions) did real inputs point to?

## Rules for every entry

1. **Separate three things:** what we *observed* (they showed us, they ran it), what they
   *said*, and what *we interpret*. Interpretations are useful but never counted as
   evidence.
2. **Counts with denominators.** At this sample size write "3 of 5 teams", never "60%".
3. **Record failures and drop-outs.** An abandoned pilot is a result, not a gap.
4. **One row per conversation or session.** Link rows for the same team with the same
   team code.
5. **Minimum detail.** Record what decides the roadmap, not everything that was said. Use a
   team code (`T03`), not a company name, wherever the ledger may be seen by more people.

## Columns

| Column | What to write | Strategy reference |
|---|---|---|
| `date` | Date of the conversation or session | |
| `team` | Team code (`T01`); the key to identities lives elsewhere | |
| `stage` | `conversation`, `qualified`, `pilot-1-connect`, `pilot-2-evidence`, `pilot-3-fix`, `pilot-4-protect`, `follow-up`, `paid`, `dropped` | §10 funnel |
| `segment` | `b2b-support`, `consultancy`, `enterprise-search`, `ecommerce`, `individual`, `other` | §4 segments |
| `trigger` | `incident`, `planned-change`, `forced-migration`, `permission-change`, `none` | §4 buying events |
| `current_alternative` | Their tool, script or manual process today | §5 competitive proof |
| `observed` | What we saw them do or showed us (artifacts inspected, commands run) | §10 rule 1 |
| `said` | What they told us, briefly and without quotes that identify them | |
| `interpreted` | Our reading, marked as such | |
| `effort_reported` | Their stated engineering and domain-expert time on the last failure | §10 question 2 |
| `owners` | Who owns labels, source updates, false alarms and release approval (roles only) | §10 question 5 |
| `data_boundary` | What may run locally, what may be shared, acceptable deployment | §10 question 6 |
| `blockers` | Integration or labelling blockers met, in their words where possible | |
| `setup_minutes` | Time from start to the first useful comparison, measured | §8 metrics |
| `labelling_minutes` | Domain-owner time to grade the benchmark, measured separately | §8 metrics |
| `investigation_minutes` | Time to an accepted explanation, measured | §8 metrics |
| `diagnosis` | `accepted`, `disputed`, `inconclusive`, `n/a` | §8 metrics |
| `fix_verified` | `yes`, `no`, `n/a`; who verified it | §8 metrics |
| `repeat_runs` | Checks the team ran without prompting since the last row | §8 metrics |
| `decision_influenced` | The release decision this changed, if any | §8 primary value measure |
| `support_minutes` | Our time spent helping since the last row | §8 metrics |
| `buyer` | Budget owner's role and what paid continuation would require | §10 question 7 |
| `next_commitment` | The concrete next step they agreed to, with a date | |
| `product_signal` | The roadmap item this row is evidence for or against (`1.1`, `1.2`, `2.1`, …) | §6 plan |

## Qualifying a prospect

Before a conversation becomes a pilot, check the defer rules from §4. Defer, and record why,
when any of these is missing:

- a real application in production or close to it;
- access to evidence (questions, documents, ideally traces or incident records);
- a named judgment owner who can grade answers;
- a recurring change (releases, content updates, migrations) that makes the check worth
  running again.

## Conversation guide

Ask about behaviour and artifacts, not about our feature ideas (§10):

1. Describe the last retrieval-related failure: who noticed, what was affected, how was the
   cause determined? Ask to see a sanitized trace or incident record.
2. What did you try, and how much engineering and domain-expert time did it take?
3. What change is coming next, and what currently decides whether it ships?
4. Show the existing benchmark, script or tool. What still requires manual work?
5. Who owns labels, source updates, false alarms and release approval?
6. What can run locally, what may be shared, and which deployment is acceptable?
7. Who approves spending, and what would need to happen for paid continuation?

## Reading the ledger at a checkpoint

Count, per stage, how many teams reached it and how many dropped, and why. Then answer each
experiment from the strategy with the evidence rows that support the answer:

| Experiment | Look at | Decision if it fails |
|---|---|---|
| Can teams connect? | `setup_minutes`, `blockers`, `support_minutes` at `pilot-1-connect` | Fix the dominant mapping or adapter issue before adding backends. |
| Is diagnosis better? | `investigation_minutes` versus `effort_reported`, `diagnosis` | Narrow to a class we can explain; drop unsupported root-cause claims. |
| Is benchmark upkeep affordable? | `labelling_minutes`, disputed or stale labels in `blockers` | Improve review and lineage, or price assisted setup explicitly. |
| Does use become routine? | `repeat_runs`, `decision_influenced` at `follow-up` | Investigate recurring value; consider a diagnostic service if that is the actual job. |
| Is a separate purchase justified? | `buyer`, `stage = paid`, reasons in `said` | Revisit buyer, scope or proposition; integration requests alone are not demand. |

Record the checkpoint's conclusion, and any change of direction with its reason, in the
strategy's progress document rather than in this file.

## Product data that helps (and what not to collect)

For teams using the hosted service, two existing records answer `repeat_runs` without
asking:

- the administrator usage export, `vectraudit-cloud usage --month YYYY-MM`, which counts
  runs, queries, checks, stored runs and members per workspace for one month;
- the project's release timeline (dashboard or API), which lists each check and monitor run
  with its date, so checks between two sessions can be counted.

Do not add content telemetry to the product to fill the ledger. Record local-only teams from
the sessions themselves.
