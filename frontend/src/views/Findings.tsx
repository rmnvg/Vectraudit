import RepeatSelector from '../components/RepeatSelector'
import { executionView } from '../lib/repeats'
import { useMemo, useState } from 'react'
import { documentLabel, documentNote, questionLabel, questionText } from '../lib/guided'
import type { DocumentDisplay } from '../lib/guided'
import type { Comparison, Finding, QuestionDisplay, Severity, StageLoss, TriageItem } from '../lib/guided'
import type { Run } from '../lib/workbench'

const button = 'rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface-2'
const format = (value: number | null | undefined) => value == null ? '—' : value.toFixed(3)
const signed = (value: number | null) => value == null ? '' : `${value > 0 ? '+' : ''}${value.toFixed(3)}`
const severities: { key: Severity; label: string; tone: string }[] = [
  { key: 'blocking', label: 'Blocking', tone: 'bg-red-500/10 text-red-600' },
  { key: 'error', label: 'Errors', tone: 'bg-amber-500/10 text-amber-600' },
  { key: 'regression', label: 'Regressed', tone: 'bg-amber-500/10 text-amber-600' },
  { key: 'info', label: 'Missing in both', tone: 'bg-surface-2 text-muted' },
  { key: 'improvement', label: 'Improved', tone: 'bg-teal-500/10 text-teal-600' },
]
const tone = (severity: Severity) => severities.find(s => s.key === severity)!.tone
const located = (stages: StageLoss[] | string[]): stages is StageLoss[] => stages.length > 0 && typeof stages[0] !== 'string'

export type Inspect = (run: Run, focus: { query: string; doc: string }) => void

/** Ranked, evidence-linked findings for a baseline/candidate comparison. */
export default function Findings({ comparison, onInspect, onReview, reviewableQueries = [] }: { comparison: Comparison; onInspect: Inspect; onReview?: (query: string) => void; reviewableQueries?: string[] }) {
  const { triage, files, baseline, candidate, question_display: display } = comparison
  const [filter, setFilter] = useState<Severity | 'all'>('all')
  const [selected, setSelected] = useState(triage.items[0]?.query_id || baseline.queries[0]?.query_id || '')
  const [doc, setDoc] = useState(triage.items[0]?.findings[0]?.doc_ids[0] || '')
  const items = useMemo(() => triage.items.filter(i => filter === 'all' || i.severity === filter), [triage, filter])
  const item = triage.items.find(i => i.query_id === selected)
  const diverged = item?.findings.flatMap(f => f.documents || []).find(d => d.doc_id === doc)?.diverged_at || null
  const choose = (next: TriageItem) => { setSelected(next.query_id); setDoc(next.findings.find(f => f.doc_ids.length)?.doc_ids[0] || '') }

  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Filter findings">
      <button className={`${button} ${filter === 'all' ? 'border-teal-500 text-teal-600' : ''}`} onClick={() => setFilter('all')}>All findings · {triage.items.length}</button>
      {severities.filter(s => triage.counts[s.key]).map(s => <button key={s.key} aria-pressed={filter === s.key} className={`${button} ${filter === s.key ? 'border-teal-500' : ''}`} onClick={() => setFilter(s.key)}><span className={`mr-2 rounded px-1.5 py-0.5 text-xs ${s.tone}`}>{triage.counts[s.key]}</span>{s.label}</button>)}
      <span className="text-xs text-muted">{triage.counts.unchanged} unchanged of {triage.n_paired} paired queries{triage.primary_metric ? ` · ranked by ${triage.primary_metric}` : ''}</span>
    </div>
    {(triage.unpaired.baseline_only.length > 0 || triage.unpaired.candidate_only.length > 0) && <p className="text-xs text-amber-600">Queries recorded in only one run are not paired: {[...triage.unpaired.baseline_only, ...triage.unpaired.candidate_only].join(', ')}</p>}
    {triage.truncated && <p className="text-xs text-muted">Showing the 500 most severe findings.</p>}
    {!triage.items.length ? <p className="rounded-xl bg-surface-2 p-4 text-sm">No query changed between these runs, and no relevant document is missing from either.</p> :
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <ul aria-label="Findings" className="max-h-[560px] space-y-1 overflow-auto">{items.map(i => <li key={i.query_id}><button aria-current={i.query_id === selected ? 'true' : undefined} onClick={() => choose(i)} className={`w-full rounded-lg border p-3 text-left text-sm ${i.query_id === selected ? 'border-teal-500 bg-teal-500/5' : 'border-transparent hover:bg-surface-2'}`}>
        <div className="flex items-center justify-between gap-2"><b className="truncate">{i.query_id}</b><span className={`shrink-0 rounded px-1.5 py-0.5 text-xs ${tone(i.severity)}`}>{severities.find(s => s.key === i.severity)!.label}</span></div>
        {questionText(display?.candidate, i.query_id) && <p className="mt-1 line-clamp-2 break-words">{questionText(display?.candidate, i.query_id)}</p>}
        <p className="mt-1 line-clamp-2 text-xs text-muted">{i.findings[0].message}</p>
        {i.delta != null && <p className="mt-1 font-mono text-xs">{i.metric} {format(i.baseline)} → {format(i.candidate)} ({signed(i.delta)})</p>}
      </button></li>)}</ul>
      <div className="min-w-0 space-y-4">
        {item ? <section aria-label={`Finding ${item.query_id}`} className="space-y-3">{item.findings.map((f, i) => <FindingCard key={i} finding={f} doc={doc} onDocument={setDoc} file={files.candidate} titles={comparison.document_display} />)}
          {!item.stages_recorded && <p className="text-xs text-muted">This pipeline recorded no stage traces, so losses cannot be located inside it.</p>}
          <div className="flex flex-wrap gap-2"><button className={button} onClick={() => onInspect(baseline, { query: item.query_id, doc })}>Open in baseline run</button><button className={button} onClick={() => onInspect(candidate, { query: item.query_id, doc })}>Open in candidate run</button></div>
          {onReview && reviewableQueries.includes(item.query_id) && <button className={button} onClick={() => onReview(item.query_id)}>Review passages for this question</button>}
          <p className="text-xs text-muted">Evidence: <code>{files.baseline}</code> · <code>{files.candidate}</code> · verdict <code>{files.check}</code></p>
        </section> : <p className="text-sm text-muted">Choose a finding.</p>}
      </div>
    </div>}
    <div className="grid gap-3 md:grid-cols-2"><label className="text-sm">Compare query<select aria-label="Compare query" value={selected} className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm" onChange={e => setSelected(e.target.value)}>{baseline.queries.map(q => <option key={q.query_id} value={q.query_id}>{questionLabel(display?.baseline, q.query_id)}</option>)}</select></label><label className="text-sm">Highlight document ID<input aria-label="Highlight document ID" className="mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm" value={doc} onChange={e => setDoc(e.target.value)} /></label></div>
    <div className="grid gap-4 xl:grid-cols-2">{[baseline, candidate].map((run, i) => <TraceComparison key={`${run.run_id}-${selected}`} title={i === 0 ? 'Baseline' : 'Candidate'} run={run} queryId={selected} document={doc} onDocument={setDoc} divergedAt={i === 1 ? diverged : null} display={i === 0 ? display?.baseline : display?.candidate} />)}</div>
  </div>
}

function FindingCard({ finding, doc, onDocument, file, titles }: { finding: Finding; doc: string; onDocument: (id: string) => void; file: string; titles?: DocumentDisplay }) {
  const note = finding.doc_ids.length ? documentNote(titles) : null
  return <div className={`rounded-xl border p-4 ${finding.severity === 'blocking' ? 'border-red-500/40 bg-red-500/5' : 'border-line'}`}>
    <div className="flex flex-wrap items-center gap-2"><span className={`rounded px-1.5 py-0.5 text-xs ${tone(finding.severity)}`}>{finding.kind.replaceAll('_', ' ')}</span><p className="text-sm">{finding.message}</p></div>
    {finding.doc_ids.length > 0 && <div className="mt-3 flex flex-wrap gap-1">{finding.doc_ids.slice(0, 50).map(id => <button key={id} onClick={() => onDocument(id)} aria-pressed={doc === id} title={titles?.titles[id] ?? id} className={`rounded px-2 py-1 font-mono text-xs ${doc === id ? 'bg-teal-600 text-white' : finding.severity === 'blocking' ? 'bg-red-500/10 text-red-600' : 'bg-surface-2'}`}>{documentLabel(titles, id)}</button>)}</div>}
    {note && <p className="mt-1 text-xs text-muted">{note}</p>}
    {finding.documents?.filter(d => d.doc_id === doc).map(d => <div key={d.doc_id} className="mt-3 text-xs text-muted">{located(d.stages)
      ? d.stages.map((s, i) => <p key={i}>Dropped at <b className="text-ink">{s.stage}</b> (input: {s.parents.join(' + ') || 'query'}) · {s.reason} · {s.evidence || 'unknown'} evidence in <code>{file}</code></p>)
      : d.stages.length ? <p>Present in candidate stages: {(d.stages as string[]).join(' → ')} · recorded in <code>{file}</code></p> : !d.diverged_at && <p>{d.reason || 'Not present in recorded stages; internal cause unknown.'}</p>}
      {d.diverged_at && <p className="mt-1">First missing at <b className="text-ink">{d.diverged_at}</b> in the candidate. Baseline kept it through {d.baseline_stages?.join(' → ') || '—'}; candidate {d.candidate_stages?.length ? `kept it through ${d.candidate_stages.join(' → ')}` : 'never recorded it'} · <code>{file}</code></p>}</div>)}
  </div>
}

function TraceComparison({ title, run, queryId, document, onDocument, divergedAt, display }: { title: string; run: Run; queryId: string; document: string; onDocument: (id: string) => void; divergedAt: string | null; display?: QuestionDisplay }) {
  const [repeat, setRepeat] = useState(0)
  const query = run.queries.find(q => q.query_id === queryId)
  const evidence = executionView(query, repeat).evidence
  const dropped = new Set(evidence?.failures?.filter(f => f.doc_id === document && f.stage).map(f => f.stage) || [])
  if (divergedAt && repeat === 0) dropped.add(divergedAt)
  const forbidden = new Set(query?.forbidden_ids || [])
  const text = questionText(display, queryId)
  return <div className="min-w-0 rounded-xl border border-line p-4"><h3 className="font-semibold">{title} · {run.label}</h3><p className="mt-2 text-sm text-muted">nDCG {query?.repeat_metrics ? 'mean ' : ''}{format(query?.metrics?.ndcg)} · {format(query?.latency_ms)} ms</p>{!!query?.error && <p className="mt-3 text-sm text-red-600">Query failed; no quality score.</p>}
    <p className="mt-3 text-xs text-muted">Question · {queryId}</p>
    <p className="mt-1 whitespace-pre-wrap break-words text-sm">{text ?? display?.reason ?? 'Question text unavailable for this query.'}</p>
    {query && <RepeatSelector query={query} index={repeat} onChange={setRepeat} />}
    {!!query?.forbidden_ids?.length && <p className="mt-3 text-sm text-red-600">Forbidden delivered: {query.forbidden_ids.join(', ')}</p>}
    {query?.no_answer_correct != null && <p className={`mt-3 text-sm ${query.no_answer_correct ? '' : 'text-red-600'}`}>No-answer check: {query.no_answer_correct ? 'passed' : 'failed'}</p>}
    {evidence?.stages?.map(stage => {
      const has = !!document && stage.hits.some(h => h.id === document)
      const lost = dropped.has(stage.name)
      return <div key={stage.name} data-dropped={lost || undefined} className={`mt-3 rounded-lg border p-3 ${lost ? 'border-red-500 bg-red-500/5' : has ? 'border-teal-500 bg-teal-500/5' : 'border-line'}`}><div className="flex flex-wrap items-center justify-between gap-2"><b className="text-sm">{stage.name}</b>{lost && <span className="text-xs font-semibold text-red-600">{document} dropped here</span>}</div><p className="text-xs text-muted">{stage.hits.length} documents · {format(stage.latency_ms)} ms · {stage.source}</p><div className="mt-2 flex max-h-24 flex-wrap gap-1 overflow-auto">{stage.hits.slice(0, 50).map(h => <button className={`rounded px-2 py-1 font-mono text-xs ${document === h.id ? 'bg-teal-600 text-white' : forbidden.has(h.id) ? 'bg-red-500/10 text-red-600' : 'bg-surface-2'}`} key={h.id} onClick={() => onDocument(h.id)}>{h.id}</button>)}</div></div>
    })}
    {!evidence?.stages?.length && <p className="mt-3 text-sm text-muted">No recorded stage evidence.</p>}
  </div>
}
