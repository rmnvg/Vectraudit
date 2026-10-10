import { useEffect, useState } from 'react'
import { documentLabel, documentNote, guidedRequest, questionLabel } from '../lib/guided'
import type { Comparison, DocumentDisplay } from '../lib/guided'
import type { RunListing } from '../lib/workbench'
import { policyLabel } from '../lib/release'
import { ShareRecord } from '../components/ShareRecord'
import type { PolicyRef } from '../lib/release'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'
export interface RegressionCase {
  id: string; revision: string; query_id: string; dataset_sha256: string
  baseline: string; candidate: string; status: string; owner: string
  finding: { findings: { message: string; doc_ids?: string[] }[] }; policy?: PolicyRef
  question_display?: { text: string | null; reason: string | null }
  document_display?: DocumentDisplay
  history: { at: string; kind: string; status?: string; owner?: string; note?: string; candidate?: string; verdict?: string; verified?: boolean; policy?: PolicyRef }[]
}
export default function RegressionCases({ token, comparison, runs, policy, policyRef }: { token: string; comparison: Comparison | null; runs: RunListing[]; policy: string | null; policyRef: PolicyRef | null }) {
  const [cases, setCases] = useState<RegressionCase[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [query, setQuery] = useState('')
  const unresolved = comparison?.triage.items.filter(i => ['blocking', 'error', 'regression', 'info'].includes(i.severity)) || []
  const selectedQuery = unresolved.some(q => q.query_id === query) ? query : unresolved[0]?.query_id || ''
  useEffect(() => { let cancelled = false; guidedRequest<RegressionCase[]>('cases', token).then(c => { if (!cancelled) setCases(c) }).catch(e => { if (!cancelled) setError(String(e)) }); return () => { cancelled = true } }, [token])
  async function perform(path: string, body: unknown) {
    setError(''); setBusy(true)
    try { const saved = await guidedRequest<RegressionCase>(path, token, body); setCases(previous => [saved, ...previous.filter(c => c.id !== saved.id)]) } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  async function reload() { setError(''); try { setCases(await guidedRequest<RegressionCase[]>('cases', token)) } catch (e) { setError(String(e)) } }
  return <section aria-label="Regression cases" className="space-y-3 rounded-xl border border-line p-4"><div className="flex flex-wrap justify-between gap-3"><h3 className="font-semibold">Track findings through verified fixes</h3><button className={button} disabled={busy} onClick={() => void reload()}>Reload cases</button></div>
    <p className="text-sm text-muted">Cases persist locally. Verification requires a new, complete, comparable run that passes the selected release policy ({policyLabel(policyRef)}) and has no unresolved finding for this question. Each verification records the policy version it used. Manual status changes never change release verdicts. Verify later runs again to detect recurrence.</p>
    {!!unresolved.length && comparison && <div className="flex flex-wrap items-end gap-3"><label className="min-w-0 flex-1 text-sm">Finding to track<select className={input} value={selectedQuery} onChange={e => setQuery(e.target.value)}>{unresolved.map(q => <option key={q.query_id} value={q.query_id}>{questionLabel(comparison.question_display?.baseline, q.query_id)} · {q.severity}</option>)}</select></label><button className={button} disabled={busy} onClick={() => void perform('cases', { baseline: comparison.baseline.run_id, candidate: comparison.candidate.run_id, query_id: selectedQuery, policy: comparison.policy?.id ?? null })}>Track finding</button></div>}
    {error && <p role="alert" className="text-red-600">{error}</p>}
    {!cases.length && <p className="text-sm text-muted">Compare a baseline and candidate, then track a finding here.</p>}
    {cases.map(c => <CaseEditor key={`${c.id}-${c.revision}`} token={token} record={c} runs={runs} busy={busy} onAction={perform} policy={policy} policyRef={policyRef} />)}
  </section>
}
/** The documents a case's finding cites, with titles from the attached corpus when known. */
function CaseDocuments({ record }: { record: RegressionCase }) {
  const docs = [...new Set(record.finding.findings.flatMap(f => f.doc_ids ?? []))]
  if (!docs.length) return null
  const note = documentNote(record.document_display)
  return <div className="text-xs">
    <p className="text-muted">Documents in this finding</p>
    <ul className="mt-1 space-y-0.5">{docs.slice(0, 20).map(d => <li key={d} className="break-words font-mono">{documentLabel(record.document_display, d, 120)}</li>)}</ul>
    {note && <p className="mt-1 text-muted">{note}</p>}
  </div>
}

function CaseEditor({ token, record, runs, busy, onAction, policy, policyRef }: { token: string; record: RegressionCase; runs: RunListing[]; busy: boolean; onAction: (path: string, body: unknown) => Promise<void>; policy: string | null; policyRef: PolicyRef | null }) {
  const [owner, setOwner] = useState(record.owner)
  const [note, setNote] = useState('')
  const [status, setStatus] = useState(record.status === 'verified' ? 'investigating' : record.status)
  const [candidate, setCandidate] = useState('')
  const choices = runs.filter(r => r.run_id !== record.baseline && r.run_id !== record.candidate)
  function download() {
    const text = [`# Retrieval regression: ${record.query_id}`, `Status: ${record.status}`, `Owner: ${record.owner || 'Unassigned'}`, `Benchmark: ${record.dataset_sha256}`, `Baseline evidence: ${record.baseline}.run.json`, `Original candidate: ${record.candidate}.run.json`, `Found under: ${policyLabel(record.policy)}`, ...record.finding.findings.map(f => f.message), '', '## History', ...record.history.map(h => JSON.stringify(h))].join('\n')
    const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown' })); const link = document.createElement('a'); link.href = url; link.download = `case-${record.id}.md`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <details className="rounded-lg bg-surface-2 p-3" open><summary className="cursor-pointer text-sm font-semibold">{record.query_id} · {record.status} · {record.owner || 'Unassigned'}</summary><fieldset disabled={busy} className="mt-3 space-y-3">
    <p className="whitespace-pre-wrap break-words text-sm">{record.question_display?.text ?? record.question_display?.reason ?? 'Question text unavailable for this query.'}</p>
    <CaseDocuments record={record} />
    <p className="text-xs text-muted break-all">Evidence: {record.baseline}.run.json → {record.candidate}.run.json</p>
    <div className="grid gap-3 sm:grid-cols-2"><label className="text-sm">Case owner<input className={input} value={owner} maxLength={128} onChange={e => setOwner(e.target.value)} /></label><label className="text-sm">Case status<select className={input} value={status} onChange={e => setStatus(e.target.value)}><option value="open">Open</option><option value="investigating">Investigating</option><option value="fix_proposed">Fix proposed</option></select></label></div>
    <label className="block text-sm">Investigation note<textarea className={input} maxLength={4000} value={note} onChange={e => setNote(e.target.value)} /></label>
    <button className={button} onClick={() => void onAction(`cases/${record.id}`, { revision: record.revision, owner, note, status })}>Save case update</button>
    <div className="flex flex-wrap items-end gap-3"><label className="min-w-0 flex-1 text-sm">Verification run<select className={input} value={candidate} onChange={e => setCandidate(e.target.value)}><option value="">Choose a new candidate</option>{choices.map(r => <option key={r.run_id} value={r.run_id}>{r.label} · {r.run_id.slice(0, 8)}</option>)}</select></label><button className={button} disabled={!candidate} onClick={() => void onAction(`cases/${record.id}/verify`, { revision: record.revision, candidate, policy })}>Verify under {policyLabel(policyRef)}</button></div>
    <ul className="max-h-48 space-y-2 overflow-auto text-xs">{record.history.map((h, i) => <li key={i}>{h.at} · {h.kind}{h.status ? ` · ${h.status}` : ''}{h.candidate ? ` · run ${h.candidate.slice(0, 8)}` : ''}{h.verdict ? ` · release ${h.verdict} under ${policyLabel(h.policy)} · ${h.verified ? 'verified' : 'reopened'}` : ''}{h.note ? ` · ${h.note}` : ''}</li>)}</ul>
    <button className={button} onClick={download}>Download issue summary</button>
    <ShareRecord token={token} kind="case" id={record.id} label={`case ${record.query_id}`} />
  </fieldset></details>
}
