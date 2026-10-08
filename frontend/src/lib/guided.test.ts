import { describe, expect, it } from 'vitest'
import { criticalSummary, executionLabel, questionLabel, questionText } from './guided'

describe('local question labels', () => {
  it('keeps the ID beside text and falls back when matching text is unavailable', () => {
    expect(questionLabel({ texts: { q: 'How do I export invoices?' }, reason: null }, 'q')).toBe('How do I export invoices? · q')
    expect(questionLabel(undefined, 'q')).toBe('q')
    expect(questionText({ texts: {}, reason: 'Benchmark changed' }, 'q')).toBeNull()
  })

  it('does not mistake inherited object properties for question text', () => {
    expect(questionText({ texts: {}, reason: null }, 'toString')).toBeNull()
    expect(questionText({ texts: JSON.parse('{"__proto__":"A real question"}'), reason: null }, '__proto__')).toBe('A real question')
  })
})

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
