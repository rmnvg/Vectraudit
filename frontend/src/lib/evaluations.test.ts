import { describe, expect, it } from 'vitest'
import { MAX_EVALUATIONS, startBlocker, withStarted } from './evaluations'
import type { TrackedJob } from './evaluations'

const job = (id: string, pipeline: string, status: TrackedJob['state']['status'] = 'running'): TrackedJob =>
  ({ id, label: id, pipeline, state: { status, progress: 0, message: '' } })

describe('evaluations', () => {
  it('allows two evaluations, one per registered pipeline', () => {
    expect(MAX_EVALUATIONS).toBe(2)
    expect(startBlocker([], 'configured-0')).toBeNull()
    expect(startBlocker([job('a', 'configured-0')], 'configured-1')).toBeNull()
    expect(startBlocker([job('a', 'configured-0')], 'configured-0')).toContain('already has an evaluation running')
    expect(startBlocker([job('a', 'builtin')], 'builtin')).toBeNull()
    expect(startBlocker([job('a', 'builtin'), job('b', 'configured-1')], 'builtin')).toContain('2 evaluations are running')
    expect(startBlocker([job('a', 'configured-0', 'done')], 'configured-0')).toBeNull()
  })

  it('keeps running jobs and the newest finished ones', () => {
    const jobs = [job('old', 'builtin', 'done'), job('x', 'builtin', 'failed'), job('y', 'builtin', 'cancelled'), job('r', 'builtin')]
    expect(withStarted(jobs, job('new', 'builtin')).map(j => j.id)).toEqual(['r', 'x', 'y', 'new'])
  })
})
