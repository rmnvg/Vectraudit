import { describe, expect, it } from 'vitest'
import { criticalSummary, executionLabel } from './guided'

describe('critical results across repeats', () => {
  const rule = { query_id: 'invoice', doc_id: 'guide', max_rank: 2, rank: null, verdict: 'fail' }

  it('summarises executions and treats missing repeat evidence as insufficient', () => {
    expect(criticalSummary(rule)).toBe('no per-repeat evidence')
    expect(criticalSummary({ ...rule, executions: 2, failed_executions: 1 })).toBe('failed on 1 of 2 executions')
    expect(criticalSummary({ ...rule, rank: 2, verdict: 'pass', executions: 3, failed_executions: 0 })).toBe('held on all 3 executions, worst rank 2')
  })

  it('names the attempt only for retried queries', () => {
    expect(executionLabel({ attempt: null, repeat: 2, rank: null, ids: [] })).toBe('Repeat 2')
    expect(executionLabel({ attempt: 1, repeat: 2, rank: 5, ids: ['a'] })).toBe('Attempt 1, repeat 2')
  })
})
