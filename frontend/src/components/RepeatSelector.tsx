import type { Query } from '../lib/workbench'
import { executionView } from '../lib/repeats'

export default function RepeatSelector({ query, index, onChange }: { query: Query; index: number; onChange: (n: number) => void }) {
  const view = executionView(query, index)
  if (!view.count) return null
  const values = query.repeat_metrics?.flatMap(m => m?.ndcg == null ? [] : [m.ndcg]) || []
  return <div className="my-4 space-y-2 rounded-lg bg-surface-2 p-3 text-xs">
    <label className="flex flex-wrap items-center gap-2">Execution
      <select aria-label="Execution" value={view.selected} onChange={e => onChange(Number(e.target.value))} className="rounded border border-line bg-page px-2 py-1">
        {Array.from({ length: view.count }, (_, i) => <option key={i} value={i}>Repeat {i + 1} of {view.count}</option>)}
      </select>
      <span>nDCG this repeat: {view.metrics?.ndcg?.toFixed(3) ?? 'not recorded'}</span>
    </label>
    {!!values.length && <p>Mean nDCG {query.metrics?.ndcg?.toFixed(3)} · observed range {Math.min(...values).toFixed(3)}–{Math.max(...values).toFixed(3)}. Repeats remain one query for sampling.</p>}
    <p className="break-words">Delivered IDs: {view.ids?.join(', ') || 'None'}</p>
    {view.sameOutputs && <p className="text-muted">Stage outputs match repeat 1; per-stage timings for this repeat were not retained.</p>}
    {view.selected > 0 && !view.evidence && <p className="text-muted">Stage evidence for this repeat was not recorded or was withheld from sharing.</p>}
  </div>
}
