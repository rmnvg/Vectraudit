import { useEffect, useRef, useState } from 'react'
import { guidedRequest, parseQueries } from '../lib/guided'
import type { Draft, QueryRow } from '../lib/guided'
import type { LookupResponse, ReviewPool } from '../lib/benchmarkTools'
import { decisionSummary, reviewRequestKey } from '../lib/benchmarkTools'
import type { RunListing } from '../lib/workbench'
import { PassageGrader } from './BenchmarkTools'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'

/** The parent keys this session by saved revision, corpus and retrieval request, not labels. */
export default function PassageReview({ token, draft, dirty, runs, baseline, candidate, onBaseline, onCandidate, corpus, row, irrelevant, onGrade }: {
  token: string; draft: Draft | null; dirty: boolean; runs: RunListing[]
  baseline: string; candidate: string; onBaseline: (id: string) => void; onCandidate: (id: string) => void
  corpus: string; row: QueryRow; irrelevant: string[]; onGrade: (row: QueryRow, irrelevant: string[]) => void
}) {
  const [pool, setPool] = useState<ReviewPool | null>(null)
  const [poolCorpus, setPoolCorpus] = useState('')
  const [browse, setBrowse] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const request = useRef<AbortController | null>(null)
  const lookupRequest = useRef<AbortController | null>(null)
  const [lookupSource, setLookupSource] = useState('candidate')
  const [confirmed, setConfirmed] = useState(false)
  const [lookupBusy, setLookupBusy] = useState(false)
  const [lookupError, setLookupError] = useState('')
  const [lookedUp, setLookedUp] = useState<{ pool: ReviewPool; result: LookupResponse } | null>(null)
  useEffect(() => () => { request.current?.abort(); lookupRequest.current?.abort() }, [])
  // Selecting another source never displays the previous pair's queue as its evidence.
  const savedRow = draft ? parseQueries(draft.dataset).rows.find(q => q.query_id === row.query_id) : undefined
  const sourceMatches = !!savedRow && reviewRequestKey(savedRow) === reviewRequestKey(row) && corpus === (draft?.corpus || '')
  const active = sourceMatches && poolCorpus === corpus && pool?.baseline === baseline && pool.candidate === candidate ? pool : null
  const sourceRun = lookupSource === 'baseline' ? baseline : candidate
  const remoteLookup = !!active?.document_lookup && lookupSource !== 'local' && !browse
  const lookupResult = lookedUp?.pool === active && lookedUp?.result.source_run === sourceRun ? lookedUp.result : null
  useEffect(() => () => { lookupRequest.current?.abort(); setLookupBusy(false) }, [active, sourceRun, lookupSource])
  function resetLookup() {
    lookupRequest.current?.abort(); setLookedUp(null); setLookupBusy(false); setLookupError(''); setConfirmed(false)
  }
  async function fetchDocuments(ids: string[]) {
    if (!active || !confirmed || !ids.length) return
    lookupRequest.current?.abort()
    const controller = new AbortController()
    lookupRequest.current = controller
    setLookupBusy(true); setLookupError('')
    setLookedUp(previous => previous && ({ ...previous, result: { ...previous.result, documents: previous.result.documents.filter(d => !ids.includes(d.doc_id)) } }))
    try {
      const result = await guidedRequest<LookupResponse>('review-documents', token, {
        draft_id: active.draft_id, revision: active.revision, baseline, candidate,
        query_id: row.query_id, source_run: sourceRun, doc_ids: ids, confirm_lookup: true,
      }, controller.signal)
      if (!controller.signal.aborted) setLookedUp(previous => ({ pool: active, result: {
        ...result, documents: [...(previous?.pool === active && previous.result.source_run === sourceRun && previous.result.corpus_version === result.corpus_version ? previous.result.documents.filter(d => !ids.includes(d.doc_id)) : []), ...result.documents],
      } }))
    } catch (e) { if (!controller.signal.aborted) setLookupError(String(e)) }
    finally { if (!controller.signal.aborted) setLookupBusy(false) }
  }
  async function build() {
    if (!draft || !sourceMatches) return
    request.current?.abort()
    resetLookup()
    const controller = new AbortController()
    request.current = controller
    setBusy(true); setError(''); setPool(null)
    try {
      const result = await guidedRequest<ReviewPool>('review-pool', token, {
        draft_id: draft.id, revision: draft.revision, baseline, candidate, query_id: row.query_id,
      }, controller.signal)
      if (!controller.signal.aborted) { setPool(result); setPoolCorpus(corpus); setBrowse(false) }
    } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  return <div className="space-y-3">
    {row.split === 'heldout'
      ? <p className="text-sm text-muted">Held-out questions are excluded from retrieval-driven review queues.</p>
      : <section aria-label="Passage review queue" className="space-y-3 rounded-xl border border-line p-4">
        <h3 className="font-semibold">Review passages from evaluation runs</h3>
        <p className="text-xs text-muted">Pool unjudged results for this question from two exploratory runs of the saved benchmark. Relevant, irrelevant, span-labeled and forbidden documents are excluded.</p>
        <fieldset disabled={busy} className="grid gap-3 sm:grid-cols-2">
          <label className="min-w-0 text-sm">Review baseline<select aria-label="Review baseline" className={input} value={baseline} onChange={e => { resetLookup(); onBaseline(e.target.value) }}><option value="">Choose a run</option>{runs.map(run => <option value={run.run_id} key={run.run_id}>{run.label || run.run_id}</option>)}</select></label>
          <label className="min-w-0 text-sm">Review candidate<select aria-label="Review candidate" className={input} value={candidate} onChange={e => { resetLookup(); onCandidate(e.target.value) }}><option value="">Choose a run</option>{runs.map(run => <option value={run.run_id} key={run.run_id}>{run.label || run.run_id}</option>)}</select></label>
        </fieldset>
        <button className={button} disabled={busy || !sourceMatches || !baseline || !candidate || baseline === candidate} onClick={() => void build()}>{busy ? 'Loading passages…' : 'Build passage review queue'}</button>
        {!sourceMatches && <p className="text-xs text-muted">Save your question and corpus before building a queue. Changed questions need new evaluation runs.</p>}
        {sourceMatches && dirty && <p className="text-xs text-muted">Review uses the last saved benchmark's results. You can grade multiple questions before saving; new relevance labels require new evaluation runs.</p>}
        {!!baseline && baseline === candidate && <p className="text-xs text-muted">Choose two different evaluation runs.</p>}
        {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
        {active?.decision && <div role="status" aria-label="Release check decision" className="space-y-1 rounded-lg bg-surface-2 p-3 text-sm">
          <p>{decisionSummary(active.decision).status}</p>
          {decisionSummary(active.decision).next && <p className="text-xs">{decisionSummary(active.decision).next}</p>}
          <p className="text-xs text-muted">Passages are ordered by how much they could move this check. {active.decision.saving}</p>
        </div>}
        {active && <div className="flex flex-wrap gap-2"><button className={button} aria-pressed={!browse} onClick={() => setBrowse(false)}>Review pooled passages</button><button className={button} aria-pressed={browse} onClick={() => setBrowse(true)}>Browse all passages</button></div>}
        {active?.document_lookup && !browse && <div className="space-y-2">
          <label className="block text-sm">Passage source<select aria-label="Passage source" className={input} value={lookupSource} onChange={e => { resetLookup(); setLookupSource(e.target.value) }}><option value="candidate">Registered source · candidate corpus version</option><option value="baseline">Registered source · baseline corpus version</option><option value="local">Attached local corpus</option></select></label>
          {remoteLookup && <><label className="flex gap-2 text-sm"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />Send document IDs, saved context and filters to the registered document source</label>
            <p className="text-xs text-muted">Fetch up to ten visible passages. Text stays in this browser session and is not added to the saved corpus or release export. Content versions are shown for review; saving grades does not freeze remote documents.</p>
            {lookupResult && <p className="text-xs text-muted">{lookupResult.version_verified ? `Source reports the requested corpus version: ${lookupResult.corpus_version}` : 'Run corpus version unknown: fetched text cannot be verified against this run.'}</p>}
            {lookupError && <p role="alert" className="text-sm text-red-600">{lookupError}</p>}
          </>}
        </div>}
      </section>}
    <PassageGrader key={active && !browse ? 'pool' : 'corpus'} corpus={corpus} row={row} irrelevant={irrelevant} onGrade={onGrade} pool={row.split !== 'heldout' && active && !browse ? active : undefined} lookup={remoteLookup ? { documents: Object.fromEntries((lookupResult?.documents || []).map(d => [d.doc_id, d])), busy: lookupBusy, confirmed, fetch: ids => void fetchDocuments(ids) } : undefined} />
  </div>
}
