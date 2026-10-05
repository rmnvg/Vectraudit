# Build a labelled retrieval dataset locally

Start with 40 representative real questions and the **source documents**, before chunking.
A useful first pass can take about 30 minutes with shallow pools and short passages; long
sources, larger pools and careful review need more time. This is a workflow target, not a
measured annotation-speed guarantee.

## Pool, label, export

`questions.csv` needs a `text` column and optionally `query_id` (otherwise row ids are
assigned). Quote questions containing commas. JSONL dataset queries also work. The corpus
is JSONL: `{"doc_id":"guide-1","text":"Full source document…"}`.

```bash
vectraudit label pool questions.csv corpus.jsonl --k 5 --out task.json
vectraudit label serve task.json
# Open http://127.0.0.1:8766; enter your name and make relevance decisions.
# Stop the server with Ctrl-C when finished.
vectraudit label export task.json --out dataset-v1.jsonl
vectraudit dataset validate dataset-v1.jsonl
```

The default pool uses the existing hashing lexical embedder, entirely offline and without
FAISS. Each question's pool has a randomized order, with no system names, scores or ranks
in the annotation page. Pool files retain source provenance for audit. They require the
corpus even when pooling runs: run records do not contain passages, and corpus membership
prevents accidentally labelling chunk ids. The pool covers retrieved documents, not a
random sample of the entire corpus.

Add multiple runs with repeated `--run candidate.json`, use `--experiment results/` for
an experiment output directory, or `--manifest experiment.json` to execute candidates
against the supplied questions and corpus. Manifest pooling uses the candidate specs;
its queries path need not exist yet. Use `--no-lexical` to exclude the default baseline.
Direct local execution accepts hashing candidates only; precomputed runs support any
retriever. Each run must be complete and contain exactly the question IDs. Verify those
IDs refer to the same question texts; ordinary run files do not store text for comparison.

Grades are **0 not relevant, 1 partly, 2 relevant, 3 perfect**. A click autosaves a human
decision atomically. Focus a passage and press `0`–`3`, or `S` to skip. Select passage text
to populate optional start/end offsets, then choose a positive grade. Offsets are Unicode
character positions, end exclusive, on the full source document. You may enter them
manually. Reopening a task restores decisions; selecting a new grade replaces your prior
judgment. Skip is not grade zero and can be revisited.

## Review and agreement

Stop the server before CLI commands that modify the task. Only one server process or
writer may use a task file at a time. Several browser tabs can use the same local server.
Names identify annotators, not authenticated accounts; this is a trusted local workflow.

```bash
vectraudit label sample task.json Bob --fraction 0.2 --seed 7
vectraudit label serve task.json
# Bob enters their name and sees the assigned questions, without Alice's grades.
# Stop the server again.
vectraudit label agreement task.json
vectraudit label resolve task.json q1 guide-1 Reviewer 2
# Optional: --start 10 --end 80
```

Sampling selects whole questions, so the reviewer evaluates the same pool. The JSON
agreement report lists every annotator pair, its shared judgment count, binary Cohen's
kappa (`grade > 0`), and unweighted graded Cohen's kappa (0–3). Skips and pairs judged by
only one person are excluded. No shared pairs or constant marginals give `null`, not a
misleading perfect score. Kappa can be negative and small overlap is weak evidence.

The report also lists disagreements, including different spans. A strict majority on
**both grade and span** resolves a pair. Otherwise use the named reviewer's decision;
export blocks unresolved conflicts. A new judgment invalidates the previous reviewer
resolution. Reviewer overrides do not alter original annotations or their agreement.

## Dataset versions and evaluation

The exported header includes the task and corpus hashes, pooling sources, annotators,
agreement and suggestion provenance. The dataset's normal content hash is its version;
no existing dataset hashing rules changed. Preserve earlier exports for comparison:

```bash
vectraudit label export task.json --out dataset-v2.jsonl
vectraudit dataset diff dataset-v1.jsonl dataset-v2.jsonl
```

Only human-decided labels are exported. **Unjudged pooled documents remain unjudged; the
metric contract counts missing relevance as non-relevant when evaluating a labelled
question.** This convention is not evidence that those documents are irrelevant. The
existing metrics require at least one positive label per labelled question: questions
with none remain unlabelled and are listed in provenance, so they contribute no semantic
quality metrics. All zero judgments remain in the task. Export refuses a wholly
unlabelled set. Expand the pool if it contains no answer; do not invent a positive label.

Use the dataset as the `queries` path in an experiment manifest:

```bash
vectraudit experiment run experiment.json --out results
vectraudit check results/runs/baseline.json results/runs/candidate.json
```

`record` also accepts the dataset unchanged, with its usual query embedder configuration
(`--embedder MODEL`) or after `dataset embed` with the same embedding model as the backend.
Source-document labels require source-document retrieval IDs; a raw chunk index must map
to source documents first (the experiment runner already does this).

## Suggestions are optional

Nothing invokes a judge by default. The Python API accepts a trusted callable returning
an integer grade; it fills `task.suggestions`, never human judgments:

```python
from pathlib import Path
from vectraudit.labelling.task import load_task, save_task
from vectraudit.labelling.suggestions import suggest

path = Path("task.json")
task = load_task(path)
suggest(task, my_judge, name="my-local-judge-v1")
save_task(task, path)
```

For a local causal LLM, install the optional `label-llm` extra and supply an already
downloaded compatible model directory (including its tokenizer and safetensors weights):

```bash
vectraudit label suggest task.json --llm-model /path/to/local/model
```

This explicitly opts into inference. Model loading has no downloads, disallows remote
code and uses safetensors. Model quality, memory requirements and its licence remain
model-specific. A response other than one grade is rejected. The page marks the result
as a machine suggestion; a human must click **Accept suggestion** or choose their own
grade. Export records explicit acceptances and the suggestion/human grade agreement rate
among decisions made with suggestions visible. That rate includes accepted suggestions
and is therefore not an independent judge-quality evaluation. Custom callables are trusted
code: review their behavior before supplying them, especially any network access.

## Privacy and limits

Task files contain question and document text. They are local, content-hashed, written
with private permissions and atomic replacement; they are not encrypted. Keep them in
customer-controlled storage. The server binds to `127.0.0.1` only, with Host validation,
CSRF-safe forms, escaped HTML, no external assets and strict CSP. Local pooling, UI and
export make no outbound network requests. Deleting the task, exported datasets and any
copies deletes those local artifacts; manage backups under your own retention policy.

Optional hosted labelling is **deferred**. No upload routes, cloud tables, hosted consent
flow or cloud retention changes are introduced. A future hosted variant must first add
per-label-set consent, tenant isolation, role checks, export/deletion/retention coverage
and route-audit tests. Do not expose the local server through a public proxy.

Start with 30–50 questions spanning real intents and known failures, then grow to 100+
for more useful cohort coverage and smaller effects. These are planning heuristics:
statistical power depends on effect size and variance; inspect confidence intervals.
Double-label roughly 20% initially to discover ambiguous guidance, review disagreements,
and keep an untouched evaluation set to avoid tuning to labels.

Pooling several genuinely different retrievers reduces single-system bias but cannot
find documents missed by every system. A lexical pool can miss paraphrases. Different
chunk sizes of the same embedder offer less diversity than lexical and semantic systems.
Randomized presentation hides retrieval provenance, not this coverage bias. Increase
pool depth, add varied systems and independently inspect documents outside the pool as
time permits; compare releases on the same frozen dataset version.

## Three-candidate demo

The example reuses the existing RAG corpus loader; `--rag` explicitly permits downloading
the public example dataset if absent (no customer documents are uploaded). It discards
its supplied labels and pools three hashing candidates with 80/150/300-word chunks.

```bash
uv run python examples/labelling_demo.py prepare --out /tmp/labels-demo --rag
uv run vectraudit label serve /tmp/labels-demo/task.json
# Label, then stop the server.
uv run python examples/labelling_demo.py finish --out /tmp/labels-demo
```

`finish` exports and validates labels, runs the three-candidate experiment, and checks a
run against itself as an interoperability smoke test. Its report is not evidence of a
retrieval improvement. Omit `--rag` for synthetic offline inputs. No demo code silently
converts retrieval scores or suggestions into human labels.
