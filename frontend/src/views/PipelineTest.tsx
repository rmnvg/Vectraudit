import { useEffect, useRef, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import type { PipelineChoice, QueryRow } from '../lib/guided'
import { diagnosticBody, diagnosticVerdict, sampleFromRows, statusLabel } from '../lib/diagnostics'
import type { DiagnosticReport, DiagnosticResponse, DiagnosticSample, FindingStatus } from '../lib/diagnostics'

const button = 'rounded-lg border border-line px-4 py-2 text-sm hover:bg-surface-2 disabled:opacity-40 disabled:cursor-not-allowed'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'
const tone: Record<FindingStatus, string> = {
  ok: 'bg-teal-500/15 text-teal-600', warning: 'bg-amber-500/15 text-amber-600', error: 'bg-red-500/15 text-red-600',
  unknown: 'bg-surface-2 text-muted', skipped: 'bg-surface-2 text-muted',
}
const verdictText = {
  connected: 'Connected. Every check that could run on this sample passed.',
  gaps: 'Connected, with gaps. The call worked, but some evidence is missing or could not be verified.',
  failed: 'The pipeline test failed. Fix the first failed check and test again.',
}

function parseObject(text: string): Record<string, string | number | boolean> | null {
  try {
    const value = JSON.parse(text || '{}')
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return Object.values(value).every(v => ['string', 'number', 'boolean'].includes(typeof v)) ? value : null
  } catch { return null }
}

/** One sample call to the selected pipeline; writes no run and never shows adapter exception text. */
export default function PipelineTest({ token, choice, corpus, rows }: {
  token: string; choice: PipelineChoice; corpus: string; rows: QueryRow[]
}) {
  const initial = sampleFromRows(rows, 5)
  const [text, setText] = useState(initial.text)
  const [k, setK] = useState(initial.k)
  const [filters, setFilters] = useState(JSON.stringify(initial.filters))
  const [context, setContext] = useState(JSON.stringify(initial.context))
  const [forbidden, setForbidden] = useState(initial.forbidden.join(', '))
  const [noAnswer, setNoAnswer] = useState(initial.no_answer)
  const [remote, setRemote] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<{ pipeline: string; report: DiagnosticReport } | null>(null)
  const request = useRef<AbortController | null>(null)
  useEffect(() => () => request.current?.abort(), [])
  const parsedFilters = parseObject(filters)
  const parsedContext = parseObject(context)
  const http = choice.kind === 'http'
  const report = result?.pipeline === choice.id ? result.report : null

  function loadBenchmarkQuestion() {
    const sample = sampleFromRows(rows, k)
    setText(sample.text); setFilters(JSON.stringify(sample.filters)); setContext(JSON.stringify(sample.context))
    setForbidden(sample.forbidden.join(', ')); setNoAnswer(sample.no_answer)
  }

  async function run() {
    if (!parsedFilters || !parsedContext) return
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setBusy(true); setError(''); setResult(null)
    const sample: DiagnosticSample = {
      text, k, filters: parsedFilters, context: parsedContext, no_answer: noAnswer,
      forbidden: forbidden.split(',').map(s => s.trim()).filter(Boolean),
    }
    try {
      const response = await guidedRequest<DiagnosticResponse>('diagnostics', token, diagnosticBody(sample, choice.id, corpus, remote), controller.signal)
      setResult(response)
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (request.current === controller) setBusy(false)
    }
  }

  async function cancel() {
    try { await guidedRequest('diagnostics/cancel', token, {}) } catch (err) { setError(String(err)) }
  }

  const verdict = report ? diagnosticVerdict(report) : null
  return <section aria-label="Test pipeline" className="mt-6 rounded-xl border border-line p-4">
    <h3 className="font-semibold">Test pipeline</h3>
    <p className="mt-1 text-sm text-muted">Send one sample question to this pipeline and check the response, document IDs, stage evidence and context handling. No run is saved.</p>
    <div className="mt-4 grid gap-4 md:grid-cols-[1fr_8rem]">
      <label className="text-sm">Sample question<input aria-label="Sample question" className={input} value={text} onChange={e => setText(e.target.value)} placeholder="A question your users ask" /></label>
      <label className="text-sm">k<input aria-label="Sample k" className={input} type="number" min={1} max={100} value={k} onChange={e => setK(Number(e.target.value))} /></label>
    </div>
    <details className="mt-3 text-sm"><summary className="cursor-pointer text-muted">Filters, user context and assertions</summary>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <label>Filters (JSON)<input aria-label="Sample filters" aria-invalid={!parsedFilters} className={`${input} font-mono text-xs`} value={filters} onChange={e => setFilters(e.target.value)} /></label>
        <label>User or tenant context (JSON)<input aria-label="Sample context" aria-invalid={!parsedContext} className={`${input} font-mono text-xs`} value={context} onChange={e => setContext(e.target.value)} /></label>
        <label>Forbidden document IDs<input aria-label="Sample forbidden IDs" className={input} value={forbidden} onChange={e => setForbidden(e.target.value)} placeholder="Comma separated" /></label>
        <label className="flex items-center gap-2 self-end"><input type="checkbox" checked={noAnswer} onChange={e => setNoAnswer(e.target.checked)} />Expect no documents</label>
      </div>
      {(!parsedFilters || !parsedContext) && <p className="mt-2 text-xs text-red-600">Filters and context must be JSON objects of strings, numbers or booleans.</p>}
    </details>
    {rows.some(r => r.text) && <button className="mt-3 text-sm text-teal-600 underline" onClick={loadBenchmarkQuestion}>Use the first benchmark question</button>}
    {http && <label className="mt-4 flex gap-2 rounded-lg bg-amber-500/10 p-3 text-sm"><input type="checkbox" checked={remote} onChange={e => setRemote(e.target.checked)} /><span>Send this sample question, filters and context to {choice.destination}.</span></label>}
    <div className="mt-4 flex flex-wrap gap-3">
      <button className={button} disabled={busy || !text.trim() || !parsedFilters || !parsedContext || (http && !remote) || (choice.id === 'builtin' && !corpus.trim())} onClick={() => void run()}>{busy ? 'Testing…' : 'Test pipeline'}</button>
      {busy && <button className={button} onClick={() => void cancel()}>Cancel test</button>}
    </div>
    {error && <p role="alert" className="mt-3 text-sm text-red-600">{error}</p>}
    {report && verdict && <div className="mt-4 space-y-3" aria-live="polite">
      <p className={`rounded-lg p-3 text-sm font-medium ${verdict === 'failed' ? tone.error : verdict === 'gaps' ? tone.warning : tone.ok}`}>{verdictText[verdict]}{report.latency_ms != null && ` Sample call took ${report.latency_ms.toFixed(0)} ms.`}</p>
      <ul className="space-y-2">{report.findings.map(f => <li key={f.code} className="rounded-lg border border-line p-3 text-sm">
        <div className="flex items-start gap-2"><span className={`shrink-0 rounded px-2 py-0.5 text-xs font-medium ${tone[f.status]}`}>{statusLabel[f.status]}</span><span>{f.message}</span></div>
        {f.hint && <p className="mt-1 pl-1 text-xs text-muted">{f.hint}</p>}
      </li>)}</ul>
      {report.returned_ids.length > 0 && <p className="break-words text-xs text-muted">Returned: <span className="font-mono">{report.returned_ids.join(', ')}</span>{report.missing_ids.length > 0 && <> · Not in local corpus: <span className="font-mono">{report.missing_ids.join(', ')}</span></>}</p>}
    </div>}
  </section>
}
