import { describe, expect, it } from 'vitest'
import { diagnosticBody, diagnosticVerdict, sampleFromRows } from './diagnostics'
import type { DiagnosticFinding, DiagnosticReport } from './diagnostics'

const report = (executed: boolean, ...statuses: DiagnosticFinding['status'][]): DiagnosticReport => ({
  executed, latency_ms: 1, returned_ids: [], missing_ids: [], stage_count: 0, reported_stage_count: 0,
  corpus_version_known: false,
  findings: statuses.map((status, i) => ({ code: `c${i}`, status, message: '', hint: '' })),
})

describe('pipeline connection diagnostics', () => {
  it('samples the first exploratory question and maps its filter to the request contract', () => {
    const sample = sampleFromRows([
      { query_id: 'final', text: 'held out', split: 'heldout' },
      { query_id: 'empty' },
      { query_id: 'q', text: 'invoice', filter: { current: true }, context: { tenant: 'a' }, forbidden: ['p'] },
    ], 3)
    expect(sample).toEqual({ text: 'invoice', k: 3, filters: { current: true }, context: { tenant: 'a' }, forbidden: ['p'], no_answer: false })
    expect(diagnosticBody(sample, 'configured-0', '', true)).toMatchObject({ pipeline: 'configured-0', confirm_remote: true, filters: { current: true } })
  })
  it('falls back to an empty sample without a benchmark', () => {
    expect(sampleFromRows([], 5).text).toBe('')
  })
  it('separates failures and unverified evidence from a clean connection', () => {
    expect(diagnosticVerdict(report(false))).toBe('failed')
    expect(diagnosticVerdict(report(true, 'ok', 'error'))).toBe('failed')
    expect(diagnosticVerdict(report(true, 'ok', 'unknown'))).toBe('gaps')
    expect(diagnosticVerdict(report(true, 'ok', 'skipped'))).toBe('connected')
  })
})
