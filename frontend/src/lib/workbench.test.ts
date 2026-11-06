import { describe, expect, it } from 'vitest'
import { claimFindingText, parseArtifact, proposalText } from './workbench'
import type { ClaimFinding, EvidenceProposal } from './workbench'

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

describe('proposed required evidence', () => {
  const base: EvidenceProposal = { query_id: 'q1', docs: ['a', 'b'], claims_covered: 5, reference_claims: 6, coverage: 5 / 6, status: 'proposed', needs_judgment: ['b'], already_relevant: ['a'] }
  it('says what the group covers and what is still to judge', () => {
    expect(proposalText(base)).toBe('q1: a, b covers 5 of 6 reference claims (83%), model-judged; judge b first')
    expect(proposalText({ ...base, needs_judgment: [] })).toContain('; ready to accept')
  })
})
