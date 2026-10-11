/** Evaluations running in this workspace: the server runs two at once, one per registered pipeline. */
import type { JobState } from './guided'

/** Mirrors the server's limit (MAX_EVALUATIONS in pipelines/guided.py). */
export const MAX_EVALUATIONS = 2

export interface TrackedJob { id: string; label: string; pipeline: string; state: JobState }

export function activeJobs(jobs: TrackedJob[]): TrackedJob[] {
  return jobs.filter(j => j.state.status === 'running')
}

/** Why another evaluation of ``pipeline`` cannot start now, or null when it can. */
export function startBlocker(jobs: TrackedJob[], pipeline: string): string | null {
  const active = activeJobs(jobs)
  if (active.length >= MAX_EVALUATIONS) return `${MAX_EVALUATIONS} evaluations are running. Wait for one to finish or cancel it.`
  if (pipeline !== 'builtin' && active.some(j => j.pipeline === pipeline)) {
    return 'This pipeline already has an evaluation running; its concurrency and rate limit apply to one evaluation at a time.'
  }
  return null
}

/** Finished jobs stay visible until the next start; at most the newest few are kept. */
export function withStarted(jobs: TrackedJob[], job: TrackedJob, keepFinished = 2): TrackedJob[] {
  const finished = jobs.filter(j => j.state.status !== 'running').slice(-keepFinished)
  return [...activeJobs(jobs), ...finished.filter(j => j.id !== job.id), job]
}
