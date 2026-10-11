import { describe, expect, it } from 'vitest'
import { parseArtifact } from './workbench'

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
