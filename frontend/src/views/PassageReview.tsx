import { useEffect, useRef, useState } from 'react'
import { guidedRequest, parseQueries } from '../lib/guided'
import type { Draft, QueryRow } from '../lib/guided'
import type { ReviewPool } from '../lib/benchmarkTools'
import { reviewRequestKey } from '../lib/benchmarkTools'
import type { Run } from '../lib/workbench'
import { PassageGrader } from './BenchmarkTools'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'

/** The parent keys this session by saved revision, corpus and retrieval request, not labels. */
export default function PassageReview({ token, draft, dirty, runs, baseline, candidate, onBaseline, onCandidate, corpus, row, irrelevant, onGrade }: {
  token: string; draft: Draft | null; dirty: boolean; runs: Run[]
  baseline: string; candidate: string; onBaseline: (id: string) => void; onCandidate: (id: string) => void
  corpus: string; row: QueryRow; irrelevant: string[]; onGrade: (row: QueryRow, irrelevant: string[]) => void
}) {
  const [pool, setPool] = useState<ReviewPool | null>(null)
  const [poolCorpus, setPoolCorpus] = useState('')
  const [browse, setBrowse] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const request = useRef<AbortController | null>(null)
  useEffect(() => () => request.current?.abort(), [])
  // Selecting another source never displays the previous pair's queue as its evidence.
  const savedRow = draft ? parseQueries(draft.dataset).rows.find(q => q.query_id === row.query_id) : undefined
  const sourceMatches = !!savedRow && reviewRequestKey(savedRow) === reviewRequestKey(row) && corpus === (draft?.corpus || '')
  const active = sourceMatches && poolCorpus === corpus && pool?.baseline === baseline && pool.candidate === candidate ? pool : null
  async function build() {
    if (!draft || !sourceMatches) return
    request.current?.abort()
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
          <label className="min-w-0 text-sm">Review baseline<select aria-label="Review baseline" className={input} value={baseline} onChange={e => onBaseline(e.target.value)}><option value="">Choose a run</option>{runs.map(run => <option value={run.run_id} key={run.run_id}>{run.label || run.run_id}</option>)}</select></label>
          <label className="min-w-0 text-sm">Review candidate<select aria-label="Review candidate" className={input} value={candidate} onChange={e => onCandidate(e.target.value)}><option value="">Choose a run</option>{runs.map(run => <option value={run.run_id} key={run.run_id}>{run.label || run.run_id}</option>)}</select></label>
        </fieldset>
        <button className={button} disabled={busy || !sourceMatches || !baseline || !candidate || baseline === candidate} onClick={() => void build()}>{busy ? 'Loading passages…' : 'Build passage review queue'}</button>
        {!sourceMatches && <p className="text-xs text-muted">Save your question and corpus before building a queue. Changed questions need new evaluation runs.</p>}
        {sourceMatches && dirty && <p className="text-xs text-muted">Review uses the last saved benchmark's results. You can grade multiple questions before saving; new relevance labels require new evaluation runs.</p>}
        {!!baseline && baseline === candidate && <p className="text-xs text-muted">Choose two different evaluation runs.</p>}
        {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
        {active && <div className="flex flex-wrap gap-2"><button className={button} aria-pressed={!browse} onClick={() => setBrowse(false)}>Review pooled passages</button><button className={button} aria-pressed={browse} onClick={() => setBrowse(true)}>Browse all passages</button></div>}
      </section>}
    <PassageGrader key={active && !browse ? 'pool' : 'corpus'} corpus={corpus} row={row} irrelevant={irrelevant} onGrade={onGrade} pool={row.split !== 'heldout' && active && !browse ? active : undefined} />
  </div>
}
