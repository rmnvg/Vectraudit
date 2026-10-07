import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'

interface SavedEvaluation {
  id: string; job_id: string; label: string; status: string; running: boolean
  resume_blocker?: string | null
  finished: number; failed: number; total: number; remote: boolean; heldout: boolean
}
const button = 'rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-40'

export default function EvaluationRecovery({ token, busy, onStart }: {
  token: string; busy: boolean; onStart: (id: string) => void
}) {
  const [evaluations, setEvaluations] = useState<SavedEvaluation[]>([])
  const [selected, setSelected] = useState('')
  const [retry, setRetry] = useState(false)
  const [remote, setRemote] = useState(false)
  const [heldout, setHeldout] = useState(false)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function refresh() {
      try {
        const saved = await guidedRequest<SavedEvaluation[]>('evaluations', token, undefined, controller.signal)
        if (!controller.signal.aborted) setEvaluations(saved)
      } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
      if (!controller.signal.aborted) timer = setTimeout(refresh, 2000)
    }
    void refresh()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [token, busy])
  async function resume(item: SavedEvaluation) {
    setSubmitting(true); setError('')
    try {
      const result = await guidedRequest<{ id: string }>(`evaluations/${item.id}/resume`, token, {
        retry_errors: retry, confirm_remote: remote, confirm_heldout: heldout,
      })
      onStart(result.id); setSelected('')
    } catch (e) { setError(String(e)) } finally { setSubmitting(false) }
  }
  const pending = evaluations.filter(e => e.running || e.status !== 'complete')
  if (!pending.length && !error) return null
  return <div className="mt-6 space-y-3">
    <h3 className="font-semibold">Saved progress</h3>
    <p className="text-sm text-muted">Resume the saved benchmark and settings after a restart. Completed queries are skipped. An interrupted query may run again.</p>
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {pending.map(item => <div key={item.id} className="rounded-xl border border-line p-4 space-y-3">
      <p className="text-sm"><b>{item.label}</b> · {item.finished}/{item.total} saved · {item.failed} failed · {item.running ? 'running' : item.status}</p>
      {item.running ? <button className={button} disabled={busy} onClick={() => onStart(item.job_id)}>View progress</button> : <button className={button} disabled={busy || submitting || !!item.resume_blocker} onClick={() => { setSelected(item.id); setRetry(false); setRemote(false); setHeldout(false) }}>Resume evaluation</button>}
      {item.resume_blocker && <p className="text-sm text-muted">{item.resume_blocker}</p>}
      {selected === item.id && !item.running && <div className="space-y-3 text-sm">
        {item.failed > 0 && <label className="block"><input type="checkbox" checked={retry} onChange={e => setRetry(e.target.checked)} /> Retry failed queries too (previous attempts are retained)</label>}
        {item.remote && <label className="block"><input type="checkbox" checked={remote} onChange={e => setRemote(e.target.checked)} /> Allow remaining queries to be sent to the configured endpoint</label>}
        {item.heldout && <label className="block"><input type="checkbox" checked={heldout} onChange={e => setHeldout(e.target.checked)} /> Confirm continued held-out evaluation</label>}
        <button className={button} disabled={busy || submitting || (item.remote && !remote) || (item.heldout && !heldout)} onClick={() => void resume(item)}>{submitting ? 'Resuming…' : 'Continue saved evaluation'}</button>
      </div>}
    </div>)}
  </div>
}
