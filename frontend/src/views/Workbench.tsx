import { useEffect, useMemo, useState } from 'react'
import RepeatSelector from '../components/RepeatSelector'
import { executionView } from '../lib/repeats'
import GuidedWorkflow from './GuidedWorkflow'
import CheckReports from './CheckReports'
import { isCheckReport } from '../lib/checks'
import type { CheckReport } from '../lib/checks'
import { parseArtifact } from '../lib/workbench'
import type { Answer, Artifact, Benchmark, ContextEvidence, Monitor, Optimization, Run } from '../lib/workbench'

const tabs = ['Guided workflow', 'Pipeline runs', 'Experiments', 'Benchmark', 'Monitoring', 'Checks', 'Setup'] as const
type Tab = typeof tabs[number]
const number = (value: number | null | undefined, digits = 2) => value == null ? '—' : value.toFixed(digits)
const panel = 'rounded-2xl border border-line bg-surface p-5'
const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-teal-500'


export default function Workbench() {
  const [tab, setTab] = useState<Tab>('Guided workflow')
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [checks, setChecks] = useState<{ report: CheckReport; name: string }[]>([])
  const [files, setFiles] = useState<string[]>([])
  const [error, setError] = useState('')
  const [focus, setFocus] = useState<{ run: string; query: string; doc: string } | null>(null)
  const [active, setActive] = useState(0)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    fetch('api/workbench').then(r => r.ok ? r.json() : null).then(data => {
      if (data?.artifacts) setFiles(data.artifacts.map((f: { name: string }) => f.name))
    }).catch(() => { /* File import also works in the index UI. */ })
  }, [])
  const accept = (raw: unknown, name = '') => {
    if (isCheckReport(raw)) {
      setChecks(items => [...items, { report: raw, name }])
      setTab('Checks')
      setError('')
      return
    }
    const data = parseArtifact(raw)
    if (data.format === "vectraudit.run") {
      const existing = runs.findIndex(r => r.run_id === data.run_id)
      setActive(existing < 0 ? runs.length : existing)
      if (existing < 0) setArtifacts(items => [...items, data])
    } else setArtifacts(items => [...items, data])
    setTab(data.format === 'vectraudit.run' ? 'Pipeline runs' : data.format === 'vectraudit.optimization' ? 'Experiments' : data.format === 'vectraudit.benchmark' ? 'Benchmark' : 'Monitoring')
    setError('')
  }
  const runs = artifacts.filter((a): a is Run => a.format === 'vectraudit.run')
  const optimization = artifacts.filter((a): a is Optimization => a.format === 'vectraudit.optimization').at(-1)
  const benchmark = artifacts.filter((a): a is Benchmark => a.format === 'vectraudit.benchmark').at(-1)
  const monitors = artifacts.filter((a): a is Monitor => a.format === 'vectraudit.monitor').sort((a, b) => a.created_at.localeCompare(b.created_at))
  const selected = runs[Math.min(active, runs.length - 1)]
  async function load(name: string) {
    if (!name) return
    setBusy(true)
    try {
      const response = await fetch(`api/workbench/${encodeURIComponent(name)}`)
      if (!response.ok) throw new Error('Could not read this artifact.')
      accept(await response.json(), name)
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  return <div className="mx-auto max-w-[1500px] space-y-6 p-5 md:p-9">
    <header className="flex flex-wrap items-start justify-between gap-5">
      <div><div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-teal-600"><span className="h-2 w-2 rounded-full bg-teal-500" />Retrieval engineering</div>
        <h1 className="text-3xl font-semibold tracking-tight">Your next release. With evidence.</h1>
        <p className="mt-2 text-sm text-muted">Test the whole pipeline. Find the loss. Measure the fix.</p></div>
      <div className="flex flex-wrap items-center gap-3"><span className="rounded-full bg-teal-500/10 px-3 py-1.5 text-xs text-teal-600">Local workspace</span>
        <label className={`${button} cursor-pointer bg-ink text-page`}>Import artifacts<input aria-label="Import artifacts" type="file" accept=".json" multiple className="sr-only" onChange={async e => {
          for (const file of Array.from(e.target.files || [])) {
            try { if (file.size > 64 * 1024 * 1024) throw new Error('File exceeds 64 MiB'); accept(JSON.parse(await file.text()), file.name) } catch (err) { setError(String(err)) }
          }
          e.target.value = ''
        }} /></label></div>
    </header>
    <div className="rounded-2xl bg-gradient-to-r from-teal-950 to-slate-900 px-6 py-5 text-white">
      <div className="grid gap-4 text-sm md:grid-cols-4">{['01  Define a benchmark', '02  Trace your pipeline', '03  Compare measured fixes', '04  Guard every release'].map((step, i) => <button key={step} onClick={() => setTab((['Guided workflow', 'Pipeline runs', 'Experiments', 'Monitoring'] as Tab[])[i])} className="rounded-lg px-2 py-2 text-left hover:bg-white/10">{step}<span className="mt-1 block text-xs text-teal-100/60">{['Labels, permissions, no-answer', 'Every stage, every lost document', 'Quality within your latency budget', 'Frozen checks and change history'][i]}</span></button>)}</div>
    </div>
    <nav aria-label="Workbench" className="flex gap-2 overflow-x-auto border-b border-line">{tabs.map(t => <button key={t} onClick={() => setTab(t)} className={`whitespace-nowrap border-b-2 px-4 py-3 text-sm ${tab === t ? 'border-teal-500 font-semibold text-teal-600' : 'border-transparent text-muted hover:text-ink'}`}>{t}</button>)}</nav>
    {error && <div role="alert" className="rounded-lg bg-red-500/10 p-3 text-sm text-red-600">{error}</div>}
    {files.length > 0 && <label className="flex flex-col items-start gap-2 text-sm text-muted sm:flex-row sm:items-center">Open local artifact<select aria-label="Open local artifact" className={`${button} w-full min-w-0 max-w-md bg-surface`} disabled={busy} value="" onChange={e => void load(e.target.value)}><option value="">{busy ? 'Loading…' : 'Choose a recorded result'}</option>{files.map(name => <option key={name}>{name}</option>)}</select></label>}
    <div hidden={tab !== 'Guided workflow'}><GuidedWorkflow onInspect={(run, at) => { accept(run); setFocus({ run: run.run_id, ...at }) }} /></div>
    {tab === 'Pipeline runs' && (selected ? <>
      <div className="flex flex-wrap gap-2">{runs.map((run, i) => <button className={`${button} ${active === i ? 'border-teal-500 text-teal-600' : ''}`} key={`${run.run_id}-${i}`} onClick={() => setActive(i)}>{run.label || run.run_id.slice(0, 12)}</button>)}</div>
      <RunView run={selected} key={`${selected.run_id}-${focus?.run === selected.run_id ? `${focus.query}-${focus.doc}` : ''}`} focus={focus?.run === selected.run_id ? focus : null} />
    </> : <Empty title="Bring your real retrieval pipeline." text="Import a recorded run to inspect quality, stage latency, document losses and permission violations. Python and HTTP adapters run on your machine." command="vectraudit workbench run manifest.json --out workbench-runs" />)}
    {tab === 'Experiments' && (optimization ? <div className="space-y-5">
      <div className={panel}><h2 className="text-lg font-semibold">Measured candidates</h2><p className={`mt-2 text-sm font-semibold ${optimization.approved ? 'text-teal-600' : 'text-amber-600'}`}>{optimization.status_label || 'Best exploratory candidate; not checked under a release policy'}{optimization.selected ? `: ${optimization.selected}` : ''}</p><p className="mt-1 text-sm text-muted">Held-out constraints: {optimization.eligible ? 'met' : 'not met'}{optimization.policy ? ` · Policy ${optimization.policy.sha256.slice(0, 12)} against baseline ${optimization.policy.baseline}` : ''}. Review before release.</p>{optimization.heldout?.check && <div className="mt-3 text-sm"><b>Held-out release check: {optimization.heldout.check.verdict}</b>{optimization.heldout.check.reasons.map((r, i) => <p key={i} className="text-muted">{r}</p>)}</div>}</div>
      <div className={`${panel} overflow-auto`}><table className="w-full text-left text-sm"><thead><tr>{['Candidate', 'nDCG', 'p95 ms', 'Cost / query', 'Memory bytes', 'Policy screen', 'Constraints'].map(h => <th className="pb-3 pr-5 text-muted" key={h}>{h}</th>)}</tr></thead><tbody>{optimization.candidates.map(c => <tr key={c.name} className="border-t border-line"><td className="py-4 pr-5 font-medium">{c.name}{optimization.selected === c.name && ' ↗'}</td><td>{number(c.ndcg, 3)}</td><td>{number(c.p95_ms)}</td><td>{number(c.cost_usd, 5)}</td><td>{number(c.memory_bytes, 0)}</td><td>{c.policy_screen ? c.policy_screen.verdict : optimization.policy?.baseline === c.name ? 'baseline' : '—'}</td><td className={c.feasible ? 'text-teal-600' : 'text-amber-600'}>{c.feasible ? 'Met' : c.reasons.join(', ')}</td></tr>)}</tbody></table><p className="mt-4 text-xs text-muted">Latency: client wall clock, p95 of per-query medians. Cost and memory: adapter-reported; — means unmeasured.</p></div>
      {optimization.heldout && <div className={panel}><h2 className="mb-4 font-semibold">Held-out validation · {optimization.selected}</h2><RunView run={optimization.heldout.run} /></div>}
    </div> : <Empty title="Optimize against your budget." text="Compare retrieval depth, hybrid weights and rerankers. Only feasible candidates are selected; one selected candidate is validated on held-out queries." command="vectraudit workbench optimize manifest.json --out optimization.json" />)}
    {tab === 'Benchmark' && (benchmark ? <div className="grid gap-5 lg:grid-cols-2"><div className={panel}><h2 className="text-lg font-semibold">Benchmark health</h2><p className="my-4 text-3xl font-semibold">{benchmark.n_labelled}<span className="text-base font-normal text-muted"> / {benchmark.n_queries} annotated</span></p><p className="text-sm text-muted">{benchmark.duplicates.length} duplicate groups · {benchmark.unannotated.length} awaiting judgment</p><h3 className="mb-3 mt-6 font-medium">Coverage to add</h3>{benchmark.suggestions.map(s => <div key={s.category} className="border-t border-line py-3"><div className="text-sm font-semibold">{s.category.replaceAll('_', ' ')}</div><p className="mt-1 text-sm text-muted">{s.instruction}</p></div>)}</div><div className={panel}><h2 className="font-semibold">Human judgment queue</h2><p className="my-3 text-sm text-muted">Exploratory queries only. Suggestions never become labels automatically.</p>{benchmark.judgment_queue.slice(0, 100).map((q, i) => <div key={i} className="border-t border-line py-3 text-sm"><span className="font-mono">{q.query_id} → {q.doc_id}</span><p className="mt-1 text-muted">{q.reason}</p></div>)}{!benchmark.judgment_queue.length && <p className="text-sm text-muted">No pooled documents awaiting judgment. Supply candidate runs to build a review queue.</p>}</div></div> : <Empty title="A benchmark you can trust." text="Find duplicate questions, missing test categories and documents that need a human judgment. Keep exploratory work separate from the final held-out benchmark." command="vectraudit workbench benchmark queries.jsonl --runs workbench-runs" />)}
    {tab === 'Monitoring' && (monitors.length ? <div className="space-y-4">{monitors.map((m, i) => <div key={i} className={panel}><div className="flex flex-wrap justify-between gap-3"><h2 className="font-semibold">{new Date(m.created_at).toLocaleString()}</h2><Badge good={m.verdict === 'pass'}>{m.verdict}</Badge></div><div className="mt-4 grid gap-4 text-sm md:grid-cols-4"><span>Corpus: <b>{m.drift.corpus}</b></span><span>Configuration: <b>{m.drift.configuration}</b></span><span>Changed queries: <b>{m.drift.results.length}</b></span><span>p95: <b>{number(m.summary.latency_ms?.p95)} ms</b></span></div>{m.alerts.map((a, j) => <p key={j} className="mt-3 text-sm text-red-600">{a.message}</p>)}</div>)}</div> : <Empty title="Make regressions visible." text="Schedule checks against a frozen baseline. Track result changes and distinguish corpus changes from configuration changes. Import multiple monitor artifacts to review trends." command="vectraudit workbench monitor manifest.json baseline.run.json" />)}
    {tab === 'Checks' && (checks.length ? <CheckReports reports={checks.map(c => c.report)} names={checks.map(c => c.name)} /> : <Empty title="Lineage, freshness, permissions, migrations and intake." text="Import the JSON these checks write to read them here: where labelled passages were lost, stale content served, revoked documents still reaching users, cache leaks between users, migration readiness and production recurrences. The checks run from the command line against your exports." command="vectraudit workbench lineage lineage.jsonl questions.jsonl --json lineage.json" />)}
    {tab === 'Setup' && <div className="grid gap-5 lg:grid-cols-2"><div className={panel}><h2 className="text-lg font-semibold">One local workflow</h2><ol className="mt-4 space-y-4 text-sm"><li>1. Import questions with <code>vectraudit dataset import</code>.</li><li>2. Judge candidate documents with <code>vectraudit label</code>.</li><li>3. Define Python, HTTP or hybrid candidates in a manifest.</li><li>4. Run, inspect and optimize in this workbench.</li><li>5. Use <code>vectraudit check</code> for release comparisons.</li><li>6. Schedule <code>vectraudit workbench monitor</code>.</li></ol></div><div className={panel}><h2 className="text-lg font-semibold">Your data stays with you</h2><p className="mt-4 text-sm leading-7 text-muted">Imported artifacts are read in this browser. Saved benchmarks and evaluation results stay in the configured workspace directory. Pipeline execution happens in your local process. An HTTP adapter sends query text, vectors, filters and user context to the endpoint you configure. Cloud upload is a separate, explicit action.</p><p className="mt-4 text-sm text-muted">Start with the runnable example in <code>examples/workbench/</code> and the workbench guide in <code>docs/retrieval-workbench.md</code>.</p></div></div>}
  </div>
}

function Badge({ good, children }: { good: boolean; children: React.ReactNode }) { return <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${good ? 'bg-teal-500/10 text-teal-600' : 'bg-amber-500/10 text-amber-600'}`}>{children}</span> }
function Empty({ title, text, command }: { title: string; text: string; command: string }) { return <div className={`${panel} py-14 text-center`}><div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-500/10 text-2xl text-teal-600">⌘</div><h2 className="text-xl font-semibold">{title}</h2><p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-muted">{text}</p><code className="mt-6 inline-block max-w-full overflow-x-auto rounded-lg bg-surface-2 px-4 py-3 text-xs">{command}</code></div> }
function RunView({ run, focus = null }: { run: Run; focus?: { query: string; doc: string } | null }) {
  const [repeat, setRepeat] = useState(0)
  const [queryId, setQueryId] = useState(focus?.query || run.queries[0]?.query_id || '')
  const [search, setSearch] = useState('')
  const [doc, setDoc] = useState(focus?.doc || '')
  const queries = useMemo(() => run.queries.filter(q => `${q.query_id} ${q.tags.join(' ')}`.toLowerCase().includes(search.toLowerCase())), [run, search])
  const query = run.queries.find(q => q.query_id === queryId)
  const evidence = executionView(query, repeat).evidence
  const stats = [ ['nDCG', number(run.summary.metrics?.ndcg, 3)], ['Pipeline p95', `${number(run.summary.latency_ms?.p95)} ms`], ['Successful queries', `${run.summary.n_ok} / ${run.summary.n_queries}`], ['Policy violations', String(run.summary.permissions?.violations ?? '—')] ]
  return <div className="space-y-5"><div className="grid grid-cols-2 gap-4 xl:grid-cols-4">{stats.map(([label, value]) => <div className={panel} key={label}><div className="text-xs text-muted">{label}</div><div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div></div>)}</div>
    <div className="flex items-center gap-3 text-sm text-muted"><Badge good={run.status === 'complete'}>{run.status}</Badge><span>{run.summary.no_answer?.n || 0} no-answer checks · accuracy {number(run.summary.no_answer?.accuracy, 3)}</span></div>
    <div className="grid gap-5 xl:grid-cols-[280px_1fr]"><section className={panel}><h2 className="mb-3 font-semibold">Query explorer</h2><input aria-label="Filter queries" placeholder="Search ID or cohort…" value={search} onChange={e => setSearch(e.target.value)} className="mb-3 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm" /><div className="max-h-[550px] space-y-1 overflow-auto">{queries.length > 500 && <p className="mb-2 text-xs text-muted">Showing 500 of {queries.length} questions. Search by ID or cohort to find the rest.</p>}{queries.slice(0, 500).map(q => <button key={q.query_id} onClick={() => { setQueryId(q.query_id); setDoc(''); setRepeat(0) }} className={`w-full rounded-lg p-3 text-left text-sm ${q.query_id === queryId ? 'bg-teal-500/10 text-teal-600' : 'hover:bg-surface-2'}`}><div className="flex justify-between gap-3"><span className="truncate font-medium">{q.query_id}</span><span>{q.error ? 'Error' : number(q.metrics?.ndcg, 2)}</span></div><div className="mt-1 text-xs text-muted">{q.tags.join(' · ') || 'No cohort'} · {number(q.latency_ms)} ms</div></button>)}</div></section>
      <section className={`${panel} min-w-0`}><div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><h2 className="font-semibold">Stage evidence</h2><p className="mt-1 text-xs text-muted">{query?.query_id} · recorded document outputs</p></div><input aria-label="Trace document ID" value={doc} onChange={e => setDoc(e.target.value)} placeholder="Trace a document ID…" className="rounded-lg border border-line bg-page px-3 py-2 text-sm" /></div>
      {query?.error ? <p role="alert" className="text-sm text-red-600">This query failed. No quality score is assigned.</p> : <>
      {query && <RepeatSelector query={query} index={repeat} onChange={setRepeat} />}
      {evidence?.stages?.length ? <div className="space-y-3">{evidence.stages.map((s, i) => <div key={s.name} className={`rounded-xl border p-4 ${doc && s.hits.some(h => h.id === doc) ? 'border-teal-500 bg-teal-500/5' : 'border-line'}`}><div className="flex flex-wrap items-center justify-between gap-2"><div className="flex items-center gap-3"><span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, '0')}</span><b className="text-sm">{s.name}</b><span className="text-xs text-muted">{s.kind}</span></div><span className="text-xs text-muted">{s.hits.length} documents · {number(s.latency_ms)} ms · {s.source}</span></div><p className="mt-2 text-xs text-muted">Input: {s.parents.join(' + ') || 'query'}</p><div className="mt-3 flex max-h-24 flex-wrap gap-1 overflow-auto">{s.hits.slice(0, 100).map(h => <button onClick={() => setDoc(h.id)} key={h.id} className={`rounded px-2 py-1 font-mono text-[11px] ${doc === h.id ? 'bg-teal-600 text-white' : 'bg-surface-2 text-ink-2'}`}>{h.id}</button>)}</div>{s.hits.length > 100 && <p className="mt-2 text-xs text-muted">Showing the first 100 IDs. Document lookup checks every recorded ID.</p>}</div>)}</div> : <p className="text-sm text-muted">This adapter did not provide stage traces. Internal causes are unknown.</p>}
      {!!evidence?.failures?.length && <div className="mt-5 rounded-xl bg-amber-500/10 p-4"><h3 className="text-sm font-semibold text-amber-600">Where relevant documents disappeared</h3>{evidence!.failures!.map((f, i) => <button key={i} onClick={() => setDoc(f.doc_id)} className="mt-2 block text-left text-sm"><b>{f.doc_id}</b> · {f.stage || 'Not observed'}<span className="block text-xs text-muted">{f.reason}</span></button>)}</div>}
      {evidence?.context && <ModelContextPanel context={evidence.context} onDoc={setDoc} />}
      {query?.answer && <AnswerPanel answer={query.answer} />}
      {!!query?.forbidden_ids?.length && <p role="alert" className="mt-4 text-sm text-red-600">Forbidden documents delivered: {query.forbidden_ids.join(', ')}</p>}
      {query?.no_answer_correct != null && <p className="mt-4 text-sm">No-answer assertion: {query.no_answer_correct ? 'passed' : 'failed'}</p>}
      </>}</section></div></div>
}

/** What reached the model: losses after retrieval, kept apart from retrieval quality. */
function ModelContextPanel({ context, onDoc }: { context: ContextEvidence; onDoc: (doc: string) => void }) {
  const inferred = context.method === 'text_match' ? ' Inferred by finding each document\'s recorded text in the recorded prompt.' : ''
  return <div className={`mt-5 rounded-xl p-4 ${context.losses.length ? 'bg-amber-500/10' : 'bg-surface-2'}`} aria-label="Model context">
    <h3 className={`text-sm font-semibold ${context.losses.length ? 'text-amber-600' : ''}`}>Model context</h3>
    <p className="mt-1 text-xs text-muted">{context.reached.length} relevant document(s) reached the model in full{context.coverage === 'observed' ? '' : `; context recorded ${context.coverage}, so absences are not proof of omission`}.{inferred}</p>
    {context.losses.map(l => <button key={l.doc_id} onClick={() => onDoc(l.doc_id)} className="mt-2 block text-left text-sm"><b>{l.doc_id}</b> · {l.kind === 'omitted' ? 'retrieved but left out of the prompt' : 'cut short in the prompt'}</button>)}
    {!!context.unknown.length && <p className="mt-2 text-xs text-muted">Unknown whether these reached the model: {context.unknown.join(', ')}</p>}
  </div>
}

/** An imported answer judgment and the first observable loss; never a root cause. */
function AnswerPanel({ answer }: { answer: Answer }) {
  const label = (v: string) => v.replaceAll('_', ' ')
  const verdict = answer.outcome === 'answered' ? answer.verdict ?? 'unjudged' : answer.outcome
  return <div className={`mt-5 rounded-xl p-4 ${verdict === 'correct' ? 'bg-surface-2' : 'bg-amber-500/10'}`} aria-label="Answer judgment">
    <h3 className="text-sm font-semibold">Answer: {label(verdict)}</h3>
    <p className="mt-1 text-xs text-muted">{answer.decided_by ? `Decided by ${answer.decided_by} review` : 'No deciding verdict'} · {answer.judgments.map(j => `${j.evaluator} (${j.kind}): ${j.verdict ?? 'score only'}`).join(' · ')}</p>
    {answer.diagnosis && <><p className="mt-2 text-sm">First observable loss: <b>{label(answer.diagnosis.class)}</b></p><ul className="mt-1 list-disc pl-5 text-xs text-muted">{answer.diagnosis.facts.map(f => <li key={f}>{f}</li>)}</ul></>}
  </div>
}
