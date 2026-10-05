import { describe, expect, it } from 'vitest'
import { parseArtifact } from '../lib/workbench'

describe('workbench import boundary', () => {
  it('rejects a malformed query before it can break the rendered workbench', () => {
    expect(() => parseArtifact({ format: 'vectraudit.run', run_id: 'x', summary: {}, queries: [{ tags: 4 }] })).toThrow('Malformed artifact')
  })
  it('rejects unsupported files', () => {
    expect(() => parseArtifact({ format: 'secrets' })).toThrow('Unsupported')
    expect(() => parseArtifact(null)).toThrow('Choose a Vectraudit')
  })
  it('accepts an empty complete run artifact', () => {
    const run = { format: 'vectraudit.run', run_id: 'x', status: 'complete', summary: { n_queries: 0, n_ok: 0, metrics: null, latency_ms: null }, queries: [] }
    expect(parseArtifact(run)).toEqual(run)
  })
})
