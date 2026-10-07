# Pilot release validation — October 7, 2026

The next milestone is a reviewed private release that a pilot engineer can install,
connect to their application, and use again for a later change. Release configurations,
passage review, connection diagnostics, critical assertions, document lookup and resumable
evaluations are implemented. Customer acceptance remains to be demonstrated.

## Reproduce the release demonstration

From a development checkout with dependencies and the frontend built:

```bash
uv run python examples/pilot_release_demo.py --out /tmp/vectraudit-pilot-demo
```

Choose a new output directory each time; the script refuses to overwrite an existing one.
For a private wheel installation, use that environment's Python to run the same example
script; it needs `httpx` alongside Vectraudit. No model download or external service is used.

This is a **synthetic behavioral fixture**, with 40 deliberately duplicated questions under
distinct IDs, deterministic rankings and simulated stage timings. It is not a representative
benchmark, a real customer application, or statistical evidence. The 30-question policy
minimum here exercises configuration propagation; it does not establish sample adequacy.

The script registers three Python adapters, saves and freezes a benchmark, and evaluates
each twice through the guided API used by the browser. The baseline ranks the relevant
document second for ordinary queries. The candidate improves those rankings but loses
`account-guide` for `account-recovery` on its second execution. The fixed adapter retains it.

| Comparison | First-ranking mean nDCG@2 | Critical executions | Verdict |
|---|---|---|---|
| Baseline | 0.6402 | Both retrieve the required document | Reference |
| Candidate | 1.0000 | Repeat 2 loses the required document | Fail, exit 3 |
| Fixed | 1.0000 | Both retrieve the required document | Pass, exit 0 |

Quality metrics use each question's first ranking; critical assertions inspect **every
recorded execution**. This explains why the candidate's perfect first-ranking nDCG does
not clear the gate. Synthetic latency values carry no performance claim.

The passing fixed configuration is exported into three independent directories representing
application checkouts. Each contains the reviewed release files and an explicitly supplied
adapter implementation. Isolated Python processes run their pipelines, then judge both
fresh runs and the exact guided artifacts against the approved policy. Assertions check
policy hashes, verdicts, critical evidence and CLI exit codes. Removing the benchmark must
make validation fail; the script restores it afterward.

## Short walkthrough and reports

1. Open `candidate/report.html` in the output directory. Point out the improved nDCG and
   the failing critical-question row: repeat 2 returned only `unrelated`.
2. Inspect `candidate/run.json` for both rankings and the recorded retrieve/rerank stages.
   The first critical execution passes, so inspecting only it would miss the defect.
3. Open `fixed/report.html`. The policy is unchanged and both critical executions pass.
4. Review `baseline/retrieval/release/release.json`, `policy.json` and `benchmark.jsonl`.
   The exported GitHub workflow is a reviewable artifact; the demonstration does not run
   GitHub Actions or publish anything.

Each candidate directory also contains `report.json` and `report.md`. This stable excerpt
from the generated candidate report is the sample result:

> Critical query account-recovery: document account-guide must appear in the top 1 on
> every execution; failed on 1 of 2 executions; first at repeat 2: document was not retrieved.

| Query | Document | Execution | Observed rank | Observed ranking |
|---|---|---|---|---|
| account-recovery | account-guide | repeat 2 | Not observed | unrelated |

## Engineering checks

The main workflow now runs recovery after killing and restarting the workbench server.
The private tagged-release workflow reuses those checks. Both package jobs run
`tests/package_smoke.py`, which invokes the demonstration using the installed wheel with
`python -I` from a temporary working directory, preventing checkout imports from satisfying
the package test. The example is included in the source distribution.

Existing assertion tests cover later-repeat misses, rank limits, missing/contradictory
repeat evidence, failed-attempt history, report escaping and connector recording.
Checkpoint tests cover durable recovery and explicit retries; the browser release journey
covers policy editing, comparisons, exports and case verification.

Local validation on October 7 (macOS, Python 3.12):

| Check | Result |
|---|---|
| Default Python suite | 1,232 passed; 22 external-service tests skipped; 1 slow test deselected |
| Ruff lint/format and mypy | Passed |
| Frontend TypeScript/production build | Passed |
| Recovery browser journey | Passed across killed/restarted server |
| Release browser journey | Passed, including policy changes and exported CLI execution |
| Wheel/source distribution build | Passed; source distribution includes the rehearsal |
| Isolated installed-wheel smoke | Passed, including guided/exported fail and fix checks |

The first sandboxed suite could not bind loopback ports; the complete rerun with local
socket access produced the result above. External pgvector, Pinecone and Weaviate services
were not configured. This is local evidence, not a live GitHub CI result or validation on
every supported Python/platform combination.

## Next: one real application and repeat usage

Use the [pilot acceptance criteria](launch-readiness.md#guided-pilot-acceptance) with one
supported customer application. Agree representative questions, source judgments and a
policy, then compare an actual retrieval change. Have a second engineer reproduce the
exported check from their clean checkout. Record friction before choosing another feature.

Keep one customer-controlled log per pilot:

| Evidence | Record |
|---|---|
| Installation | Reviewed commit/tag, wheel checksum, target platform, backend, CI result and skips |
| Inputs | Benchmark hash, corpus version/freeze, policy hash, baseline/candidate code or deployment |
| Activation | Setup time and labeling time separately; manual help and failed steps |
| Value | Release decision affected; accepted/disputed findings and false alarms |
| Reproduction | Second engineer's verdict and whether it matches the guided result |
| Return use | Next evaluation initiated by the customer, date and reason |
| Continuation | Customer decision, support owner, ongoing scope and support effort |

Customer data, production connectivity, hosted capacity, restore procedures, willingness
to pay and repeat usage have not been established by this rehearsal. Collect those records
before representing a pilot as accepted. Outreach and customer sessions are separate work.
