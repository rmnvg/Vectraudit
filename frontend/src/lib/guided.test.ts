import { describe, expect, it } from 'vitest'
import { criticalSummary, documentLabel, documentNote, executionLabel, questionLabel, questionText } from './guided'

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

describe('document titles', () => {
  const display = { titles: { 'kb-1': 'Refund policy for enterprise customers (2026 edition)' }, reason: null, verified: true }
  it('labels IDs with shortened titles and keeps unknown IDs as they are', () => {
    expect(documentLabel(display, 'kb-1')).toBe('kb-1 · Refund policy for enterprise customers (2026 ed…')
    expect(documentLabel(display, 'kb-1', 200)).toBe('kb-1 · Refund policy for enterprise customers (2026 edition)')
    expect(documentLabel(display, 'kb-2')).toBe('kb-2')
    expect(documentLabel(undefined, 'kb-1')).toBe('kb-1')
  })
  it('explains missing or unverified titles', () => {
    expect(documentNote(display)).toBeNull()
    expect(documentNote({ ...display, verified: false })).toMatch(/did not record a corpus version/)
    expect(documentNote({ titles: {}, reason: 'Document titles unavailable: the attached corpus is not the one this run used.', code: 'mismatch', verified: false })).toMatch(/not the one this run used/)
    expect(documentNote({ titles: {}, reason: 'Document titles unavailable: the attached corpus has no titles.', code: 'no_titles', verified: false })).toBeNull()
  })
})
