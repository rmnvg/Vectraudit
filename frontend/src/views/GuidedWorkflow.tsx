import { useEffect, useRef, useState } from 'react'
import { demoCorpus, demoQueries, guidedRequest, parseQueries, serializeQueries } from '../lib/guided'
import type { Comparison, Draft, GuidedSession, JobState, QueryRow } from '../lib/guided'
import type { Run } from '../lib/workbench'

const panel = 'rounded-2xl border border-line bg-surface p-5'
const button = 'rounded-lg border border-line px-4 py-2 text-sm hover:bg-surface-2 disabled:opacity-40 disabled:cursor-not-allowed'
const primary = `${button} bg-teal-600 text-white hover:bg-teal-700`
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'
const format = (value: number | null | undefined) => value == null ? '—' : value.toFixed(3)

export default function GuidedWorkflow({ onInspect }: { onInspect: (run: Run) => void }) {
  const [session, setSession] = useState<GuidedSession | null>(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const [pipeline, setPipeline] = useState('builtin')
  const [corpus, setCorpus] = useState('')
  const [rows, setRows] = useState<QueryRow[]>([])
  const [header, setHeader] = useState<Record<string, unknown> | undefined>()
  const [name, setName] = useState('My retrieval benchmark')
  const [draft, setDraft] = useState<Draft | null>(null)
  const [dirty, setDirty] = useState(true)
  const [selected, setSelected] = useState(0)
  const [saving, setSaving] = useState(false)
  const [invalidFields, setInvalidFields] = useState<Record<string, boolean>>({})
  const [editorVersion, setEditorVersion] = useState(0)
  const invalid = Object.values(invalidFields).some(Boolean)
  function fieldValidity(field: string, bad: boolean) {
    setInvalidFields(previous => ({ ...previous, [field]: bad }))
    setDirty(true)
  }
  const [label, setLabel] = useState('Baseline')
  const [k, setK] = useState(2)
  const [depth, setDepth] = useState(50)
  const [weight, setWeight] = useState(0)
  const [reranker, setReranker] = useState('none')
  const [mode, setMode] = useState('explore')
  const [remote, setRemote] = useState(false)
  const [heldout, setHeldout] = useState(false)
  const [jobId, setJobId] = useState('')
  const [job, setJob] = useState<JobState | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [baseId, setBaseId] = useState('')
  const [candidateId, setCandidateId] = useState('')
  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [compareQuery, setCompareQuery] = useState('')
  const [focusDoc, setFocusDoc] = useState('')
  const imported = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    guidedRequest<GuidedSession>('session', '', undefined, controller.signal).then(setSession).catch(e => {
      if (!controller.signal.aborted) setError(String(e))
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!jobId || !session) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async () => {
      try {
        const state = await guidedRequest<JobState>(`jobs/${jobId}`, session.token, undefined, controller.signal)
        if (stopped) return
        setJob(state)
        if (state.status === 'running') timer = setTimeout(poll, 500)
        else {
          setJobId('')
          if (state.run) {
            const run = state.run
            setRuns(previous => previous.some(r => r.run_id === run.run_id) ? previous : [...previous, run])
            setBaseId(previous => previous || run.run_id)
            setCandidateId(run.run_id)
          }
        }
      } catch (e) { if (!stopped) { setError(String(e)); setJobId('') } }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer); controller.abort() }
  }, [jobId, session])

  const choice = session?.pipelines.find(p => p.id === pipeline)
  const current = rows[selected]
  const running = !!jobId || submitting
  const exploreCount = rows.filter(r => r.split !== 'heldout').length
  const heldoutCount = rows.length - exploreCount

  function changeRow(row: QueryRow) {
    setRows(previous => previous.map((r, i) => i === selected ? row : r))
    setDirty(true)
  }
  function startDemo() {
    setPipeline('builtin'); setCorpus(demoCorpus); setRows(demoQueries.map(r => ({ ...r })))
    setName('Retrieval starter benchmark'); setDraft(null); setDirty(true); setInvalidFields({}); setEditorVersion(v => v + 1); setSelected(0); setHeader(undefined)
    setStep(1); setError('')
  }
  async function readFile(file: File) {
    if (file.size > 4 * 1024 * 1024) throw new Error('Choose a file smaller than 4 MiB.')
    return file.text()
  }
  async function save() {
    if (!session || invalid) return
    setSaving(true); setError('')
    try {
      const saved = await guidedRequest<Draft>('drafts', session.token, {
        id: draft?.id, revision: draft?.revision, name, dataset: serializeQueries(rows, header),
      })
      setDraft(saved); setDirty(false)
      setSession(previous => previous && ({ ...previous, drafts: [...previous.drafts.filter(d => d.id !== saved.id), saved] }))
    } catch (e) { setError(String(e)) } finally { setSaving(false) }
  }
  async function restore(id: string) {
    if (!session || !id) return
    if (dirty && rows.length && !window.confirm('Discard unsaved benchmark edits and open this saved draft?')) return
    try {
      const saved = await guidedRequest<Draft>(`drafts/${id}`, session.token)
      const content = parseQueries(saved.dataset)
      setDraft(saved); setRows(content.rows); setHeader(content.header); setName(saved.name)
      setDirty(false); setInvalidFields({}); setEditorVersion(v => v + 1); setSelected(0); setError(''); setStep(1)
    } catch (e) { setError(String(e)) }
  }
  async function evaluate() {
    if (!session || !draft || dirty || invalid || running) return
    setError(''); setSubmitting(true)
    try {
      const started = await guidedRequest<{ id: string }>('jobs', session.token, {
        draft_id: draft.id, revision: draft.revision, pipeline, label, k, depth,
        vector_weight: weight, reranker, corpus: pipeline === 'builtin' ? corpus : '',
        mode, confirm_remote: remote, confirm_heldout: heldout,
      })
      setJob({ status: 'running', progress: 0, message: 'Preparing evaluation' }); setJobId(started.id)
    } catch (e) { setError(String(e)) } finally { setSubmitting(false) }
  }
  async function compare() {
    if (!session) return
    setComparing(true); setError('')
    try {
      const result = await guidedRequest<Comparison>('compare', session.token, { baseline: baseId, candidate: candidateId })
      setComparison(result); setCompareQuery(result.baseline.queries[0]?.query_id || '')
    } catch (e) { setError(String(e)) } finally { setComparing(false) }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([serializeQueries(rows, header)], { type: 'application/x-ndjson' }))
    const link = document.createElement('a'); link.href = url; link.download = 'benchmark.jsonl'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  if (!session) return <div className={panel}><h2 className="text-lg font-semibold">Guided evaluation</h2>{error ? <><p className="mt-3 text-sm text-muted">This server has no guided evaluation workspace.</p><p className="mt-3 text-sm">Run <code>vectraudit workbench serve DIR</code> to build benchmarks and run evaluations here. Artifact inspection remains available in the other tabs.</p></> : <p className="mt-3 text-sm text-muted">Connecting to the local workspace…</p>}</div>

  return <div className="space-y-5">
    <nav aria-label="Evaluation steps" className="grid grid-cols-2 gap-2 md:grid-cols-4">
      {['Connect pipeline', 'Build benchmark', 'Evaluate & inspect', 'Compare a fix'].map((title, index) => <button key={title} disabled={invalid || saving} onClick={() => setStep(index)} aria-current={step === index ? 'step' : undefined} className={`${button} text-left ${step === index ? 'border-teal-500 bg-teal-500/10 text-teal-600' : 'text-muted'}`}><span className="mr-2 font-mono text-xs">0{index + 1}</span>{title}</button>)}
    </nav>
    {error && <div role="alert" className="rounded-lg bg-red-500/10 p-3 text-sm text-red-600">{error}</div>}
    {step === 0 && <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
      <section className={panel}><h2 className="text-xl font-semibold">Connect your retrieval pipeline</h2><p className="mt-2 text-sm text-muted">Choose a local baseline or an adapter registered for this workspace.</p>
        <label className="mt-5 block text-sm">Pipeline<select aria-label="Pipeline" value={pipeline} onChange={e => { setPipeline(e.target.value); setRemote(false) }} className={input}>{session.pipelines.map(p => <option value={p.id} key={p.id}>{p.name}</option>)}</select></label>
        {pipeline === 'builtin' ? <><p className="mt-4 text-sm text-muted">Add source documents as JSONL. Each document needs <code>doc_id</code> and <code>text</code>; metadata and vectors are optional.</p><label className="mt-3 block text-sm">Corpus file<input aria-label="Corpus file" type="file" accept=".jsonl,.ndjson" className="mt-2 block text-sm" onChange={async e => { const file = e.target.files?.[0]; if (file) { try { setCorpus(await readFile(file)); setError('') } catch (err) { setError(String(err)) } } }} /></label><label className="mt-4 block text-sm">Corpus JSONL<textarea aria-label="Corpus JSONL" rows={7} value={corpus} onChange={e => setCorpus(e.target.value)} className={`${input} font-mono text-xs`} placeholder={'{"doc_id":"guide","text":"Your source document"}'} /></label></> : <div className="mt-5 rounded-xl bg-surface-2 p-4 text-sm"><p>{choice?.kind === 'http' ? `Queries will be sent to ${choice.destination}.` : 'This adapter runs locally with its registered configuration.'}</p><p className="mt-2 text-muted">To register another Python or HTTP adapter, restart the workbench with <code>--manifest pipelines.json</code>.</p></div>}
        <button className={`${primary} mt-5`} disabled={pipeline === 'builtin' && !corpus.trim()} onClick={() => setStep(1)}>Continue to benchmark</button>
      </section>
      <aside className={panel}><div className="text-xs font-semibold uppercase tracking-widest text-teal-600">First evaluation</div><h2 className="mt-3 text-xl font-semibold">Start with a working example</h2><p className="mt-3 text-sm leading-6 text-muted">Four documents and four questions show relevance, tenant permissions, outdated content, no-answer behavior and a separate held-out query.</p><button className={`${button} mt-5`} onClick={startDemo}>Use starter example</button><hr className="my-6 border-line" /><h3 className="font-semibold">Data stays under your control</h3><p className="mt-2 text-sm leading-6 text-muted">Benchmarks are saved to this local workspace. HTTP adapters require an explicit query-transfer confirmation before evaluation. Browser input cannot register executable code or a new endpoint.</p></aside>
    </div>}
    {step === 1 && <fieldset disabled={saving} className={panel}>
      <div className="flex flex-wrap justify-between gap-4"><div><h2 className="text-xl font-semibold">Build a trustworthy benchmark</h2><p className="mt-2 text-sm text-muted">{exploreCount} exploratory · {heldoutCount} held-out · {dirty ? 'Unsaved changes' : 'Saved locally'}</p></div><div className="flex flex-wrap gap-2"><button disabled={invalid} className={button} onClick={() => imported.current?.click()}>Import queries</button><button className={button} disabled={!rows.length || invalid} onClick={download}>Download JSONL</button><button className={primary} disabled={!rows.length || saving || !dirty || invalid} onClick={() => void save()}>{saving ? 'Saving…' : 'Save benchmark'}</button></div></div>
      <input ref={imported} aria-label="Query file" type="file" accept=".jsonl,.ndjson" className="sr-only" onChange={async e => { const file = e.target.files?.[0]; if (file) { try { const parsed = parseQueries(await readFile(file)); setRows(parsed.rows); setHeader(parsed.header); setDraft(null); setDirty(true); setInvalidFields({}); setEditorVersion(v => v + 1); setSelected(0); setError('') } catch (err) { setError(String(err)) } } e.target.value = '' }} />
      <div className="my-5 grid gap-4 md:grid-cols-2"><label className="text-sm">Benchmark name<input aria-label="Benchmark name" value={name} onChange={e => { setName(e.target.value); setDirty(true) }} className={input} /></label><label className="text-sm">Resume saved benchmark<select aria-label="Resume saved benchmark" disabled={invalid} className={input} value="" onChange={e => void restore(e.target.value)}><option value="">Choose a saved draft</option>{session.drafts.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}</select></label></div>
      <div className="grid gap-5 lg:grid-cols-[240px_1fr]"><div><div className="max-h-96 space-y-1 overflow-auto">{rows.map((r, i) => <button className={`w-full rounded-lg p-3 text-left text-sm ${i === selected ? 'bg-teal-500/10 text-teal-600' : 'hover:bg-surface-2'}`} key={`${i}-${r.query_id}`} disabled={invalid} onClick={() => setSelected(i)}><b className="block truncate">{r.query_id}</b><span className="text-xs text-muted">{r.split === 'heldout' ? 'Held-out' : 'Explore'} · {r.no_answer ? 'No answer' : Object.keys(r.relevant || {}).length ? 'Judged' : 'Unannotated'}</span></button>)}</div><button className={`${button} mt-3 w-full`} disabled={invalid} onClick={() => { setSelected(rows.length); setRows(previous => [...previous, { query_id: `q-${crypto.randomUUID().slice(0, 8)}`, text: '' }]); setDirty(true) }}>Add question</button></div>
        {current ? <div className="space-y-4"><div className="grid gap-4 sm:grid-cols-2"><label className="text-sm">Query ID<input aria-label="Query ID" className={input} value={current.query_id} onChange={e => changeRow({ ...current, query_id: e.target.value })} /></label><label className="text-sm">Split<select aria-label="Query split" className={input} value={current.split || 'explore'} onChange={e => changeRow({ ...current, split: e.target.value as 'explore' | 'heldout' })}><option value="explore">Exploratory</option><option value="heldout">Held-out</option></select></label></div>
          <label className="block text-sm">Question<textarea aria-label="Question" className={input} rows={2} value={current.text || ''} onChange={e => changeRow({ ...current, text: e.target.value })} /></label>
          <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={current.no_answer || false} onChange={e => { setInvalidFields(previous => ({ ...previous, relevant: false })); const row = { ...current, no_answer: e.target.checked }; if (e.target.checked) { delete row.relevant; delete row.relevant_spans } changeRow(row) }} /><span>No relevant document exists<span className="mt-1 block text-xs text-muted">An explicit no-answer judgment. Unchecked with no labels means unannotated.</span></span></label>
          {!current.no_answer && <JsonField key={`${editorVersion}-${selected}-relevant`} label="Relevance grades" onInvalid={bad => fieldValidity("relevant", bad)} value={current.relevant || {}} hint={'Document IDs mapped to grades, e.g. {"guide":2}. Existing span labels are preserved.'} onValue={v => changeRow({ ...current, relevant: v as Record<string, number> })} />}
          <div className="grid gap-4 sm:grid-cols-2"><JsonField key={`${editorVersion}-${selected}-context`} label="User or tenant context" onInvalid={bad => fieldValidity("context", bad)} value={current.context || {}} hint={'For example {"tenant":"acme"}'} onValue={v => changeRow({ ...current, context: v as QueryRow['context'] })} /><JsonField key={`${editorVersion}-${selected}-filter`} label="Document filters" onInvalid={bad => fieldValidity("filter", bad)} value={current.filter || {}} hint={'For example {"current":true}'} onValue={v => changeRow({ ...current, filter: v as QueryRow['filter'] })} /></div>
          <ListField key={`${editorVersion}-${selected}-forbidden`} label="Forbidden document IDs" value={current.forbidden || []} separator={"\n"} onValue={value => changeRow({ ...current, forbidden: value })} />
          <ListField key={`${editorVersion}-${selected}-tags`} label="Cohort tags" value={current.tags || []} separator="," onValue={value => changeRow({ ...current, tags: value })} />
          <button className={`${button} text-red-600`} onClick={() => { setRows(previous => previous.filter((_, i) => i !== selected)); setSelected(0); setInvalidFields({}); setDirty(true) }}>Remove question</button>
        </div> : <p className="text-sm text-muted">Import a JSONL dataset or add your first question.</p>}
      </div>
      {draft && !dirty && <div className="mt-5 rounded-xl bg-surface-2 p-4 text-sm"><b>{draft.audit.n_labelled}/{draft.audit.n_queries} annotated</b> · {draft.audit.duplicates.length} duplicate groups<p className="mt-2 text-muted">Coverage to review: {draft.audit.missing_categories.join(', ') || 'All suggested categories represented'}</p></div>}
      <button className={`${primary} mt-5`} disabled={!draft || dirty || invalid} onClick={() => setStep(2)}>Continue to evaluation</button>
    </fieldset>}
    {step === 2 && <section className={panel}><h2 className="text-xl font-semibold">Evaluate and inspect the evidence</h2><p className="mt-2 text-sm text-muted">{choice?.name} · {draft?.name || 'Save a benchmark first'} · {exploreCount} exploratory queries</p>
      <div className="mt-5 grid gap-4 md:grid-cols-3"><label className="text-sm">Run label<input aria-label="Run label" className={input} value={label} onChange={e => setLabel(e.target.value)} /></label><label className="text-sm">Results per query (k)<input aria-label="Results per query" type="number" min={1} max={100} className={input} value={k} onChange={e => setK(Number(e.target.value))} /></label><label className="text-sm">Evaluation split<select aria-label="Evaluation split" className={input} value={mode} onChange={e => { setMode(e.target.value); setHeldout(false) }}><option value="explore">Exploratory queries only</option><option value="heldout">Final held-out validation</option></select></label></div>
      {pipeline === 'builtin' && <div className="mt-4 grid gap-4 md:grid-cols-3"><label className="text-sm">Retrieval depth<input aria-label="Retrieval depth" className={input} type="number" min={k} max={10000} value={depth} onChange={e => setDepth(Number(e.target.value))} /></label><label className="text-sm">Vector weight<input aria-label="Vector weight" className={input} type="number" min={0} max={1} step={0.1} value={weight} onChange={e => setWeight(Number(e.target.value))} /><span className="text-xs text-muted">Above zero requires query and corpus vectors.</span></label><label className="text-sm">Reranker<select aria-label="Reranker" className={input} value={reranker} onChange={e => setReranker(e.target.value)}><option value="none">None</option><option value="phrase">Exact phrase baseline</option></select></label></div>}
      {mode === 'heldout' && <label className="mt-5 flex gap-2 rounded-lg bg-amber-500/10 p-4 text-sm"><input type="checkbox" checked={heldout} onChange={e => setHeldout(e.target.checked)} /><span>I am validating a chosen configuration on {heldoutCount} held-out queries. Repeated tuning against these results contaminates the benchmark.</span></label>}
      {choice?.kind === 'http' && <label className="mt-5 flex gap-2 rounded-lg bg-amber-500/10 p-4 text-sm"><input type="checkbox" checked={remote} onChange={e => setRemote(e.target.checked)} /><span>Send query text, vectors, filters and user context to {choice.destination} for this evaluation.</span></label>}
      {dirty && <p className="mt-4 text-sm text-amber-600">Save benchmark changes before running.</p>}
      <div className="mt-5 flex flex-wrap gap-3"><button className={primary} disabled={!draft || dirty || running || (mode === 'heldout' && !heldout) || (choice?.kind === 'http' && !remote)} onClick={() => void evaluate()}>{running ? 'Evaluation running…' : 'Run evaluation'}</button>{jobId && <button className={button} onClick={() => void guidedRequest(`jobs/${jobId}/cancel`, session.token, {}).catch(e => setError(String(e)))}>Cancel evaluation</button>}</div>
      {job && <div className="mt-5" role="status"><p className="text-sm">{job.status === 'done' ? `Evaluation ${job.run?.status || 'complete'}` : job.status} · {job.message}</p><progress aria-label="Evaluation progress" className="mt-2 w-full accent-teal-600" value={job.progress} max={1} />{job.error && <p className="mt-2 text-sm text-red-600">{job.error}</p>}<p className="mt-1 text-xs text-muted">Cancellation takes effect between queries. An active adapter call finishes first.</p></div>}
      {runs.length > 0 && <div className="mt-6 space-y-3"><h3 className="font-semibold">Recorded evaluations</h3>{runs.map(run => <div key={run.run_id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line p-4"><div><b className="text-sm">{run.label}</b><p className="mt-1 text-xs text-muted">{run.status} · {run.summary.n_queries} queries · nDCG {format(run.summary.metrics?.ndcg)} · p95 {format(run.summary.latency_ms?.p95)} ms</p></div><button className={button} onClick={() => onInspect(run)}>Inspect failures</button></div>)}<button className={button} onClick={() => { setLabel('Candidate fix'); setStep(3) }}>Compare a fix</button></div>}
    </section>}
    {step === 3 && <section className={`${panel} space-y-5`}><div><h2 className="text-xl font-semibold">Compare a measured fix</h2><p className="mt-2 text-sm text-muted">Keep the benchmark and k unchanged. Change pipeline configuration, record another run, then inspect both results.</p></div><button className={button} onClick={() => { setLabel('Candidate fix'); setStep(2) }}>Evaluate another configuration</button>
      {runs.length < 2 ? <p className="text-sm text-muted">Record a baseline and a candidate to compare measured results.</p> : <><div className="grid gap-4 sm:grid-cols-2"><label className="text-sm">Baseline<select aria-label="Baseline run" value={baseId} className={input} onChange={e => setBaseId(e.target.value)}>{runs.map(r => <option key={r.run_id} value={r.run_id}>{r.label} · {r.run_id.slice(0, 8)}</option>)}</select></label><label className="text-sm">Candidate<select aria-label="Candidate run" value={candidateId} className={input} onChange={e => setCandidateId(e.target.value)}>{runs.map(r => <option key={r.run_id} value={r.run_id}>{r.label} · {r.run_id.slice(0, 8)}</option>)}</select></label></div><button className={primary} disabled={baseId === candidateId || comparing} onClick={() => void compare()}>{comparing ? 'Comparing…' : 'Compare runs'}</button></>}
      {comparison && <><div role="status" className="rounded-xl bg-surface-2 p-4"><b className={comparison.check.verdict === 'pass' ? 'text-teal-600' : 'text-amber-600'}>Release comparison: {comparison.check.verdict}</b>{comparison.check.reasons.map((r, i) => <p className="mt-2 text-sm" key={i}>{r.message}</p>)}{comparison.check.warnings.map((w, i) => <p className="mt-2 text-xs text-muted" key={i}>{w}</p>)}</div>
        <div className="grid gap-3 md:grid-cols-2"><label className="text-sm">Compare query<select aria-label="Compare query" value={compareQuery} className={input} onChange={e => setCompareQuery(e.target.value)}>{comparison.baseline.queries.map(q => <option key={q.query_id}>{q.query_id}</option>)}</select></label><label className="text-sm">Highlight document ID<input aria-label="Highlight document ID" className={input} value={focusDoc} onChange={e => setFocusDoc(e.target.value)} /></label></div>
        <div className="grid gap-4 lg:grid-cols-2">{[comparison.baseline, comparison.candidate].map((run, i) => <TraceComparison key={run.run_id} title={i === 0 ? 'Baseline' : 'Candidate'} run={run} queryId={compareQuery} document={focusDoc} onDocument={setFocusDoc} />)}</div>
      </>}
    </section>}
  </div>
}

function JsonField({ label, value, hint, onValue, onInvalid }: { label: string; value: object; hint: string; onValue: (value: object) => void; onInvalid: (bad: boolean) => void }) {
  const [text, setText] = useState(JSON.stringify(value))
  const [invalid, setInvalid] = useState(false)
  return <label className="block text-sm">{label}<input aria-label={label} aria-invalid={invalid} className={`${input} font-mono text-xs`} value={text} onChange={e => {
    setText(e.target.value)
    try { const parsed: unknown = JSON.parse(e.target.value); if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error(); onValue(parsed); setInvalid(false); onInvalid(false) } catch { setInvalid(true); onInvalid(true) }
  }} /><span className={`text-xs ${invalid ? 'text-red-600' : 'text-muted'}`}>{invalid ? 'Enter a JSON object. Fix this field before saving or continuing.' : hint}</span></label>
}

function ListField({ label, value, separator, onValue }: { label: string; value: string[]; separator: string; onValue: (value: string[]) => void }) {
  const [text, setText] = useState(value.join(separator))
  return <label className="block text-sm">{label}<textarea aria-label={label} className={input} rows={2} value={text} onChange={e => { setText(e.target.value); onValue(e.target.value.split(separator).map(v => v.trim()).filter(Boolean)) }} /><span className="text-xs text-muted">{separator === ',' ? 'Separate tags with commas.' : 'One ID per line. Returning any of these fails a permission assertion.'}</span></label>
}

function TraceComparison({ title, run, queryId, document, onDocument }: { title: string; run: Run; queryId: string; document: string; onDocument: (id: string) => void }) {
  const query = run.queries.find(q => q.query_id === queryId)
  return <div className="min-w-0 rounded-xl border border-line p-4"><h3 className="font-semibold">{title} · {run.label}</h3><p className="mt-2 text-sm text-muted">nDCG {format(query?.metrics?.ndcg)} · {format(query?.latency_ms)} ms</p>{!!query?.error && <p className="mt-3 text-sm text-red-600">Query failed; no quality score.</p>}
    {!!query?.forbidden_ids?.length && <p className="mt-3 text-sm text-red-600">Forbidden: {query.forbidden_ids.join(', ')}</p>}
    {query?.no_answer_correct != null && <p className="mt-3 text-sm">No-answer check: {query.no_answer_correct ? 'passed' : 'failed'}</p>}
    {query?.evidence?.stages?.map(stage => <div key={stage.name} className={`mt-3 rounded-lg border p-3 ${document && stage.hits.some(h => h.id === document) ? 'border-teal-500 bg-teal-500/5' : 'border-line'}`}><b className="text-sm">{stage.name}</b><p className="text-xs text-muted">{stage.hits.length} documents · {format(stage.latency_ms)} ms · {stage.source}</p><div className="mt-2 flex max-h-24 flex-wrap gap-1 overflow-auto">{stage.hits.slice(0, 50).map(h => <button className={`rounded px-2 py-1 font-mono text-xs ${document === h.id ? 'bg-teal-600 text-white' : 'bg-surface-2'}`} key={h.id} onClick={() => onDocument(h.id)}>{h.id}</button>)}</div></div>)}
    {!query?.evidence?.stages?.length && <p className="mt-3 text-sm text-muted">No recorded stage evidence.</p>}
    {query?.evidence?.failures?.map((failure, i) => <button key={i} className="mt-3 block text-left text-sm text-amber-600" onClick={() => onDocument(failure.doc_id)}>{failure.doc_id} · {failure.stage || 'Not observed'}<span className="block text-xs text-muted">{failure.reason}</span></button>)}
  </div>
}
