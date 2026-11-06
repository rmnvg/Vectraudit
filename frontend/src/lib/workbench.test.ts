import { describe, expect, it } from 'vitest'
import { claimFindingText, parseArtifact } from './workbench'
import type { ClaimFinding } from './workbench'

const run = (evidence: unknown) => ({
  format: 'vectraudit.run', run_id: 'r', status: 'complete', created_at: '2026-10-12',
  summary: { n_queries: 1, n_ok: 1, metrics: null, latency_ms: null },
  queries: [{ query_id: 'q', tags: [], ids: ['a'], latency_ms: 1, metrics: null, evidence }],
})

describe('parseArtifact model context', () => {
  it('accepts recorded context evidence', () => {
    const context = { coverage: 'observed', method: 'text_match', reached: [], unknown: [], losses: [{ doc_id: 'a', kind: 'truncated', reason: 'cut' }] }
    expect(() => parseArtifact(run({ context }))).not.toThrow()
  })
  it('rejects malformed context evidence', () => {
    const context = { coverage: 'observed', method: 'reported', reached: [], unknown: [], losses: [{ doc_id: 'a', kind: 'lost' }] }
    expect(() => parseArtifact(run({ context }))).toThrow()
  })
})

describe('claim findings', () => {
  const base: ClaimFinding = { finding: 'misled_by_irrelevant_context', count: 1, points_to: 'noise', evidence: 'supported', disputed: false }
  it('names the finding, the count and whether it is claims or documents', () => {
    expect(claimFindingText(base)).toBe('misled by irrelevant context: 1 claim')
    expect(claimFindingText({ ...base, finding: 'unsupported_correct', count: 6 })).toBe('unsupported correct: 6 claims')
    expect(claimFindingText({ ...base, finding: 'contradicting_document', count: 2 })).toBe('contradicting document: 2 documents')
    expect(claimFindingText({ ...base, finding: 'contradicting_document' })).toBe('contradicting document: 1 document')
  })
})
