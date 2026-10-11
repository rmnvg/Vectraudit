import { useEffect } from 'react'
import { guidedRequest } from '../lib/guided'
import type { JobState } from '../lib/guided'
import type { TrackedJob } from '../lib/evaluations'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'

/** One evaluation's progress: polls its job while running and reports every change upward. */
export function EvaluationProgress({ token, job, onUpdate, onError }: {
  token: string
  job: TrackedJob
  onUpdate: (id: string, state: JobState) => void
  onError: (message: string) => void
}) {
  const running = job.state.status === 'running'
  useEffect(() => {
    if (!running) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async () => {
      try {
        const state = await guidedRequest<JobState>(`jobs/${job.id}`, token, undefined, controller.signal)
        if (stopped) return
        onUpdate(job.id, state)
        if (state.status === 'running') timer = setTimeout(poll, 500)
      } catch (e) {
        if (stopped) return
        onError(String(e instanceof Error ? e.message : e))
        onUpdate(job.id, { ...job.state, status: 'failed', message: 'Progress is no longer available' })
      }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer); controller.abort() }
    // Poll once per job; later state changes come from this loop itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, running, token])

  const state = job.state
  return <div className="mt-5" role="status" aria-label={`Evaluation ${job.label}`}>
    <div className="flex flex-wrap items-center justify-between gap-3">
      <p className="text-sm"><b>{job.label}</b> · <span>{state.status === 'done' ? `Evaluation ${state.run?.status || 'complete'}` : state.status} · {state.message}</span></p>
      {running && <button className={button} onClick={() => void guidedRequest(`jobs/${job.id}/cancel`, token, {}).catch(e => onError(String(e)))}>Cancel {job.label}</button>}
    </div>
    <progress aria-label={`Progress of ${job.label}`} className="mt-2 w-full accent-teal-600" value={state.progress} max={1} />
    {state.error && <p className="mt-2 text-sm text-red-600">{state.error}</p>}
  </div>
}
