# Permission lifecycle

A benchmark's `forbidden` lists are a snapshot of who may see what. Real access changes:
someone leaves a group, a document moves to another space, an old version is archived. The
application has to catch up: its index, its caches and anything it hands to a reranker or
the model. These checks test that catch-up. They don't prove the authorization is correct.
A passing check is evidence about the principals and documents tested.

Three pieces work together:

1. **Where forbidden documents travelled** (every permission check): see
   [release checks](release-checks.md#where-forbidden-documents-travelled). A forbidden
   document in the model context fails even when the results are clean, and a policy's
   `forbidden_stages` can forbid stages such as `rerank`.
2. **Access changes over time** (this page): an access change log from your identity or
   content system becomes the expected access, judged after each change's propagation
   window.
3. **Cache isolation** (below): one user's question must not warm another user's answer.

## The access change log

One change per line:

```json
{"change": "revoke", "principal": {"user": "u-17"}, "doc_ids": ["payroll-2026"], "changed_at": "2026-10-18T09:00:00Z", "window_minutes": 15}
{"change": "grant", "principal": {"groups": "finance"}, "doc_ids": ["budget"], "changed_at": "2026-10-18T09:00:00Z"}
{"change": "move", "doc_ids": ["roadmap"], "from": {"tenant": "acme"}, "to": {"tenant": "beta"}, "changed_at": "2026-10-18T09:00:00Z"}
{"change": "archive", "doc_ids": ["manual-v1"], "changed_at": "2026-10-17T12:00:00Z"}
```

| Change | Effect |
|---|---|
| `revoke` | The principal may no longer see the documents |
| `grant` | The principal may now see them; they stop being forbidden |
| `move` | Revoked for `from`, granted for `to` |
| `archive` | Revoked for everyone |

**Who a change applies to.** A principal names context fields. It applies to a benchmark
question when every field it names is in the question's `context` with that value. For
list-valued context such as `"groups": ["finance", "staff"]`, it applies when the list
contains the value. So the benchmark's per-question context is the set of principals under
test: use test users your team controls.

**Order and windows.** For each principal and document, the latest change in effect
decides; a later grant restores a revoked document. Each change has a propagation window
(`window_minutes`, default `--window-minutes` 60). Inside the window a change is
**pending**: the earlier access still stands and nothing fails.

Take expected access from this log or from reviewed entitlements. Never derive it from the
application's own decisions: copying what the application did into the expected result
tests nothing.

## Apply the log to a benchmark

```bash
vectraudit workbench access access.jsonl questions.jsonl \
  --write-dataset questions.access.jsonl --as-of 2026-10-18T12:00:00Z
```

- **Forbidden lists:** revoked documents join each affected question's `forbidden` list,
  and granted ones leave it.
- **Labels:** a revoked document that was labelled relevant is removed from `relevant`,
  `relevant_spans` and `required_evidence`, and listed for review. A label is never carried
  silently past a revocation.
- **Tags:** affected questions are tagged `access-change`, which makes them a cohort.

Evaluate with the written benchmark. Release checks then enforce the new access on every
repeat, including the model context and any `forbidden_stages`.

## Check a run recorded after the changes

```bash
vectraudit workbench access access.jsonl questions.jsonl --run after.run.json --fail-on-leak
```

The check is judged at the run's recorded time unless you pass `--as-of`.

- **Violations:** revoked documents the run still served, and where each one appeared:
  a recorded stage, the results or the model context. Documents match by ID or by source
  ID. `--fail-on-leak` exits 3 when there is any violation, so the check can run on a
  schedule after access changes.
- **Pending:** changes still inside their window, with any old access the run still showed.
  These are not failures yet.
- **Granted but not delivered:** documents labelled relevant that were granted and never
  delivered. This may be a grant that hasn't propagated, or just ranking, so it's
  reported, not failed.

`--write-dataset` and `--run` can be combined. `--json` writes both reports.

## Cache isolation

A cache keyed on the question text alone serves one user's documents to the next user who
asks the same thing. That cache might hold results, filtered candidates, reranked lists or
packed prompts. A normal evaluation catches this only by accident of order.

```bash
vectraudit workbench cache-probe pipelines.json --candidate current --fail-on-leak
```

**Which questions are probed.** The probe uses questions the benchmark already has: the
same question (same `family_id`, otherwise the same text) asked under different contexts,
where at least one of them has `forbidden` documents. Permission scenarios create exactly
these.

**How each pair runs.** For each such *target* and each *primer* asked under another
context, the probe calls the primer first, then the target, serially, using the manifest's
pipeline. The primer must go first: asking the target first would cache its own clean
answer, and the defect would hide.

**What counts as a leak.** Any forbidden document the target receives, in the results,
the model context or a recorded stage. Each leak also records what the probe observed:

- **Identical to the primer's answer:** what a shared cache entry produces, though an
  access check that ignores the user looks the same.
- **Different from the primer's answer:** the access check itself let the document through.

**Exit codes.** `--fail-on-leak` exits 3 when there is any leak, or when calls failed and
the probe is incomplete. `--max-pairs` (default 500) bounds the number of calls (2 per
pair).

**Run it on a fresh deployment.** The probe cannot clear your caches, and an answer
cached before the probe can mask a leak. Probe a fresh deployment, or one whose cache has
expired.

## See it run

`examples/reliability_story_demo.py` plants a revocation the assistant never received and a
cache shared between users, and verifies both corrections (see
[pilot release validation](pilot-release-validation.md#third-demonstration-content-and-access-that-changed)).

## Limits

- **Cached answers before the probe:** they can mask a cache leak (see above).
- **Time to propagate:** not tracked across a series of runs.
- **Customer change logs:** the check has not yet been run on one.
- **Data stays local:** nothing here uploads the change log or the derived benchmark. Keep
  them out of broadly shared case histories.
