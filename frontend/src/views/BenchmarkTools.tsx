import { useEffect, useMemo, useState } from 'react'
import { guidedRequest, parseQueries } from '../lib/guided'
import type { Draft, QueryRow } from '../lib/guided'
import { documentGrade, gradeDocument, pendingReview, readDocuments } from '../lib/benchmarkTools'
import type { Irrelevant, PassageLookup, ReviewPool } from '../lib/benchmarkTools'
import { ShareRecord } from '../components/ShareRecord'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'
export function PassageGrader({ corpus, row, irrelevant, onGrade, pool, lookup }: { corpus: string; row: QueryRow; irrelevant: string[]; onGrade: (row: QueryRow, irrelevant: string[]) => void; pool?: ReviewPool; lookup?: PassageLookup }) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const parsed = useMemo(() => { try { return { docs: readDocuments(corpus), error: '' } } catch (e) { return { docs: [], error: String(e) } } }, [corpus])
  const byId = new Map((lookup ? Object.values(lookup.documents).filter(d => d.status === 'found').map(d => ({ doc_id: d.doc_id, text: d.text || '', metadata: { title: d.title || d.doc_id } })) : parsed.docs).map(d => [d.doc_id, d]))
  const documents = pool ? pool.items.filter(item => !row.forbidden?.includes(item.doc_id)).map(item => byId.get(item.doc_id) || { doc_id: item.doc_id, text: '', metadata: undefined }) : parsed.docs
  const matches = documents.filter(d => `${d.doc_id} ${d.text}`.toLowerCase().includes(search.toLowerCase()))
  const currentPage = Math.min(page, Math.max(0, Math.ceil(matches.length / 10) - 1))
  const missing = [...new Set([...Object.keys(row.relevant || {}), ...(row.relevant_spans || []).map(s => s.doc_id), ...irrelevant])].filter(id => !byId.has(id))
  return <section aria-label="Passage grading" className="rounded-xl border border-line p-4 space-y-3">
    <h3 className="font-semibold">{pool ? 'Review retrieved passages' : 'Read and grade passages'}</h3>
    <p className="text-xs text-muted">Grades are human judgments. Irrelevant is distinct from unjudged; neither means no answer exists. Passage grades replace span grades for that document.</p>
    {!corpus && !lookup && <p className="text-sm">Attach a local corpus below to look up document text, including for registered pipelines.</p>}
    {parsed.error && <p role="alert">{parsed.error}</p>}
    {missing.length > 0 && (lookup
      ? <p className="text-xs text-muted">Already labeled, not fetched from the registered source: {missing.join(', ')}. Existing judgments are preserved.</p>
      : <p className="text-sm text-amber-600">Text unavailable for labeled documents: {missing.join(', ')}. Existing judgments are preserved.</p>)}
    <label className="block text-sm">Find a passage<input className={input} value={search} onChange={e => { setSearch(e.target.value); setPage(0) }} /></label>
    <p className="text-xs text-muted">{pool ? `${pendingReview(pool, row, irrelevant).length} passages awaiting judgment · unjudged disagreements were queued first` : `${matches.length} matching documents · local corpus order, no retrieval ranks`}</p>
    {pool && <p className="text-xs text-muted">Source and rank stay hidden until you grade a passage. {lookup ? 'Text comes from the registered document source for the selected run.' : 'Text comes from your attached local corpus.'} Save the reviewed benchmark, then rerun both pipelines before comparing the new labels.</p>}
    {pool && (!pool.baseline_measured || !pool.candidate_measured) && <p role="status" className="text-sm text-amber-600">One run failed this question. Its ranking is unknown; the available results are pooled without claiming a disagreement.</p>}
    {pool && !pool.items.length && <p className="text-sm">No unjudged retrieved passages for this question. You can still browse outside the pool; this does not establish that no answer exists.</p>}
    {lookup && <button className={button} disabled={lookup.busy || !lookup.confirmed || !matches.length} onClick={() => lookup.fetch(matches.slice(currentPage * 10, currentPage * 10 + 10).map(d => d.doc_id))}>{lookup.busy ? 'Fetching passages…' : 'Fetch visible passages'}</button>}
    {matches.slice(currentPage * 10, currentPage * 10 + 10).map(doc => {
      const spans = row.relevant_spans?.filter(s => s.doc_id === doc.doc_id) || []
      const grade = documentGrade(row, doc.doc_id, irrelevant)
      const evidence = pool?.items.find(item => item.doc_id === doc.doc_id)
      const forbidden = row.forbidden?.includes(doc.doc_id)
      return <article key={doc.doc_id} className="rounded-lg bg-surface-2 p-3 space-y-2">
        <b className="break-all text-sm">{String(doc.metadata?.title || doc.doc_id)}</b><p className="text-xs text-muted break-all">{doc.doc_id}{spans.length ? ' · Span-labeled' : ''}{forbidden ? ' · Forbidden' : ''}</p>
        <p className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-sm">{doc.text || (lookup ? ({ missing: 'Document not found in this corpus version.', forbidden: 'Document inaccessible in this context.', version_unavailable: 'Requested document version is unavailable.', found: '' }[lookup.documents[doc.doc_id]?.status] || 'Passage not fetched. Fetch visible passages before grading.') : 'Passage text unavailable. Attach a local corpus containing this document before grading.')}</p>
        {lookup?.documents[doc.doc_id]?.status === 'found' && <p className="text-xs text-muted break-all">Content version: {lookup.documents[doc.doc_id].version} · SHA-256: {lookup.documents[doc.doc_id].sha256}</p>}
        <div className="flex flex-wrap gap-2">{[{ label: 'Unjudged', value: null }, { label: 'Irrelevant', value: 0 }, { label: 'Relevant', value: 1 }, { label: 'Highly relevant', value: 2 }].map(option => <button key={option.label} className={button} aria-label={`${option.label}: ${doc.doc_id}`} aria-pressed={option.value === null ? grade === undefined : grade === option.value} disabled={!doc.text || (!!forbidden && (option.value || 0) > 0)} onClick={() => { const next = gradeDocument(row, doc.doc_id, option.value, irrelevant); onGrade(next.row, next.irrelevant) }}>{option.label}{(option.value === null ? grade === undefined : grade === option.value) ? ' ✓' : ''}</button>)}</div>
        {evidence && grade !== undefined && <details className="text-xs text-muted"><summary className="cursor-pointer">Show retrieval evidence for {doc.doc_id}</summary><p>{evidence.reason}</p><p>Baseline rank: {pool?.baseline_measured ? evidence.baseline_rank ?? 'Not retrieved' : 'Unknown (query failed)'}</p><p>Candidate rank: {pool?.candidate_measured ? evidence.candidate_rank ?? 'Not retrieved' : 'Unknown (query failed)'}</p></details>}
      </article>
    })}
    <div className="flex gap-2"><button className={button} disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>Previous passages</button><button className={button} disabled={(currentPage + 1) * 10 >= matches.length} onClick={() => setPage(currentPage + 1)}>Next passages</button></div>
  </section>
}

type Preview = { dataset: string; columns: string[]; count: number; duplicates: number }
export function ImportQuestions({ token, onImport }: { token: string; onImport: (dataset: string) => void }) {
  const [content, setContent] = useState('')
  const [format, setFormat] = useState('csv')
  const [source, setSource] = useState('Local export')
  const [textColumn, setTextColumn] = useState('text')
  const [idColumn, setIdColumn] = useState('query_id')
  const [relevantColumn, setRelevantColumn] = useState('relevant')
  const [incident, setIncident] = useState(false)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function prepare() {
    setBusy(true); setError(''); setPreview(null)
    try {
      const result = await guidedRequest<Preview>('import', token, { content, format, source, text_column: textColumn, id_column: idColumn, relevant_column: relevantColumn, incident })
      setPreview(result); setSelected(parseQueries(result.dataset).rows.map(r => r.query_id))
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  function accept() {
    if (!preview) return
    const parsed = parseQueries(preview.dataset)
    onImport([...(parsed.header ? [JSON.stringify(parsed.header)] : []), ...parsed.rows.filter(r => selected.includes(r.query_id)).map(r => JSON.stringify(r))].join('\n'))
    setPreview(null)
  }
  return <details className="my-4 rounded-xl border border-line p-4"><summary className="cursor-pointer font-semibold">Import questions or production incidents</summary>
    <fieldset disabled={busy} onChange={() => setPreview(null)} className="mt-3 space-y-3">
      <p className="text-sm text-muted">Preview local CSV/JSON exports. Select both incidents and ordinary traffic for coverage. Feedback never becomes a relevance judgment.</p>
      <label className="block text-sm">Source file<input type="file" accept=".csv,.json,.jsonl,.ndjson" className={input} onChange={async e => { const f = e.target.files?.[0]; if (!f) return; if (f.size > 4 * 1024 * 1024) { setError('Choose a file smaller than 4 MiB.'); return } setContent(await f.text()); setSource(f.name); setFormat(f.name.endsWith('.csv') ? 'csv' : f.name.endsWith('.json') ? 'json' : 'jsonl'); setPreview(null) }} /></label>
      <div className="grid gap-3 sm:grid-cols-3"><label className="text-sm">Question column<input className={input} value={textColumn} onChange={e => setTextColumn(e.target.value)} /></label><label className="text-sm">ID column (optional)<input className={input} value={idColumn} onChange={e => setIdColumn(e.target.value)} /></label><label className="text-sm">Relevance column<input disabled={incident} className={input} value={relevantColumn} onChange={e => setRelevantColumn(e.target.value)} /></label></div>
      <label className="flex gap-2 text-sm"><input type="checkbox" checked={incident} onChange={e => setIncident(e.target.checked)} />Production incidents: discard labels, deduplicate questions within context, queue for human review</label>
      <button className={button} disabled={!content} onClick={() => void prepare()}>{busy ? 'Preparing…' : 'Preview import'}</button>
    </fieldset>
    {error && <p role="alert" className="mt-3 text-red-600">{error}</p>}
    {preview && <div className="mt-3 space-y-3"><p className="text-sm">{preview.count} questions · {preview.duplicates} duplicates removed · source columns: {preview.columns.join(', ')}</p><div className="max-h-64 overflow-auto">{parseQueries(preview.dataset).rows.map(row => <label key={row.query_id} className="flex gap-2 p-2 text-sm"><input type="checkbox" checked={selected.includes(row.query_id)} onChange={e => setSelected(previous => e.target.checked ? [...previous, row.query_id] : previous.filter(id => id !== row.query_id))} /><span>{row.text} <span className="text-muted">({row.query_id})</span></span></label>)}</div><button className={button} disabled={!selected.length} onClick={accept}>Add {selected.length} reviewed selections</button><p className="text-xs text-muted">Selecting a question does not label its answers. Existing questions and held-out splits are retained.</p></div>}
  </details>
}

type Scenario = { name: string; context: Record<string, string>; filters: Record<string, string>; allowed: string[]; forbidden: string[]; no_answer: boolean }
export function PermissionBuilder({ token, dataset, row, onAdd }: { token: string; dataset: string; row: QueryRow; onAdd: (rows: QueryRow[]) => void }) {
  const [name, setName] = useState('Tenant scenario')
  const [key, setKey] = useState('tenant')
  const [value, setValue] = useState('')
  const [filterKey, setFilterKey] = useState('')
  const [filterValue, setFilterValue] = useState('')
  const [allowed, setAllowed] = useState('')
  const [forbidden, setForbidden] = useState('')
  const [noAnswer, setNoAnswer] = useState(false)
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const ids = (text: string) => text.split('\n').map(s => s.trim()).filter(Boolean)
  async function create() {
    setBusy(true); setError('')
    try { const result = await guidedRequest<{ rows: QueryRow[] }>('permissions', token, { dataset, query_id: row.query_id, scenarios }); onAdd(result.rows); setScenarios([]) } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  return <details className="rounded-xl border border-line p-4"><summary className="cursor-pointer font-semibold">Build permission scenarios</summary><fieldset disabled={busy || row.split === 'heldout'} className="mt-3 space-y-3">
    <p className="text-xs text-muted">Repeat this exploratory question under explicit contexts. Allowed IDs become expected relevant documents. Passing cases only cover the identities and documents tested. Filters here use string values; typed filters remain available in the query editor.</p>
    <div className="grid gap-3 sm:grid-cols-3">{[{ label: 'Scenario name', value: name, set: setName }, { label: 'Context key', value: key, set: setKey }, { label: 'Context value', value, set: setValue }, { label: 'Filter key (optional)', value: filterKey, set: setFilterKey }, { label: 'Filter value', value: filterValue, set: setFilterValue }].map(f => <label key={f.label} className="text-sm">{f.label}<input className={input} value={f.value} onChange={e => f.set(e.target.value)} /></label>)}</div>
    <div className="grid gap-3 sm:grid-cols-2"><label className="text-sm">Allowed document IDs<textarea className={input} disabled={noAnswer} value={allowed} onChange={e => setAllowed(e.target.value)} /></label><label className="text-sm">Forbidden scenario IDs<textarea className={input} value={forbidden} onChange={e => setForbidden(e.target.value)} /></label></div>
    <label className="flex gap-2 text-sm"><input type="checkbox" checked={noAnswer} onChange={e => setNoAnswer(e.target.checked)} />Expect no answer in this context</label>
    <button className={button} disabled={!name.trim() || !key.trim() || !value.trim() || scenarios.length >= 50} onClick={() => setScenarios(previous => [...previous, { name, context: { [key]: value }, filters: filterKey ? { [filterKey]: filterValue } : {}, allowed: noAnswer ? [] : ids(allowed), forbidden: ids(forbidden), no_answer: noAnswer }])}>Queue scenario</button>
    {scenarios.length > 0 && <><ul className="space-y-2">{scenarios.map((s, i) => <li key={i} className="text-sm">{s.name} · {Object.values(s.context).join(', ')} · {s.allowed.length} allowed / {s.forbidden.length} forbidden{s.no_answer ? ' · no answer' : ''} <button className={button} onClick={() => setScenarios(previous => previous.filter((_, n) => n !== i))}>Remove</button></li>)}</ul><button className={button} onClick={() => void create()}>Add permission cases</button></>}
  </fieldset>{row.split === 'heldout' && <p className="mt-2 text-sm">Choose an exploratory question to generate scenarios.</p>}{error && <p role="alert" className="text-red-600">{error}</p>}</details>
}

type Snapshot = { id: string; name: string; created_at: string; dataset_sha256: string; draft_id: string; dataset?: string; irrelevant?: Irrelevant }
type Freshness = { status: string; changed: string[]; deleted: string[]; added: string[]; items: { query_id: string; doc_id: string | null; reason: string }[] }
export function BenchmarkSnapshots({ token, draft, dirty, corpus, onRestore }: { token: string; draft: Draft | null; dirty: boolean; corpus: string; onRestore: (dataset: string, irrelevant: Irrelevant) => void }) {
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [selected, setSelected] = useState('')
  const [checked, setChecked] = useState<{ corpus: string; snapshot: string; result: Freshness } | null>(null)
  const result = checked?.corpus === corpus && checked.snapshot === selected ? checked.result : null
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => { let cancelled = false; guidedRequest<Snapshot[]>('snapshots', token).then(s => { if (!cancelled) setSnapshots(s) }).catch(e => { if (!cancelled) setError(String(e)) }); return () => { cancelled = true } }, [token])
  async function action(kind: 'freeze' | 'check' | 'restore') {
    setBusy(true); setError('')
    try {
      if (kind === 'freeze' && draft) { const saved = await guidedRequest<Snapshot>('snapshots', token, { draft_id: draft.id, revision: draft.revision }); setSnapshots(previous => [saved, ...previous]); setSelected(saved.id) }
      if (kind === 'check') setChecked({ corpus, snapshot: selected, result: await guidedRequest<Freshness>('freshness', token, { snapshot: selected, corpus }) })
      if (kind === 'restore') { const snapshot = await guidedRequest<Snapshot>(`snapshots/${selected}`, token); if (snapshot.dataset) onRestore(snapshot.dataset, snapshot.irrelevant || {}) }
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  return <section className="mt-4 rounded-xl border border-line p-4 space-y-3"><h3 className="font-semibold">Frozen versions and label freshness</h3><p className="text-xs text-muted">Freeze the saved benchmark and corpus inventory. Later edits do not overwrite it. Changed content needs human review; checking freshness never changes labels. Restoring creates a new draft and keeps your currently attached corpus.</p><button className={button} disabled={!draft || dirty || busy} onClick={() => void action('freeze')}>Freeze saved benchmark</button>
    <label className="block text-sm">Frozen benchmark<select className={input} value={selected} onChange={e => setSelected(e.target.value)}><option value="">Choose a version</option>{snapshots.map(s => <option key={s.id} value={s.id}>{s.name} · {s.created_at} · {s.dataset_sha256.slice(0, 8)}</option>)}</select></label>
    <div className="flex flex-wrap gap-2"><button className={button} disabled={!selected || busy} onClick={() => void action('check')}>Check current corpus</button><button className={button} disabled={!selected || busy} onClick={() => void action('restore')}>Restore as new draft</button></div>
    {selected && <ShareRecord key={selected} token={token} kind="benchmark" id={selected} label="this frozen benchmark" />}
    {result && <div role="status" className="text-sm"><p>Freshness: {result.status}{result.status === 'unknown' ? ' — attach inventories for both versions to compare.' : ` · ${result.changed.length} changed, ${result.deleted.length} deleted, ${result.added.length} added`}</p><ul className="mt-2 max-h-64 overflow-auto">{result.items.map((r, i) => <li key={i}>{r.query_id} · {r.doc_id || 'No-answer judgment'} · {r.reason}</li>)}</ul>{result.status === 'review' && <p className="mt-2">Review affected passages, save and freeze a new version, then remeasure both baseline and candidate. No labels were changed.</p>}</div>}
    {error && <p role="alert" className="text-red-600">{error}</p>}
  </section>
}
