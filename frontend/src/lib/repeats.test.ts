import { expect, it } from 'vitest'
import { executionView } from './repeats'
import { parseArtifact, type Query } from './workbench'

const query: Query = {
  query_id: 'q', ids: ['a'], latency_ms: 2, tags: [], metrics: { ndcg: 0.5 },
  repeat_ids: [['a'], ['b']], repeat_metrics: [{ ndcg: 1 }, { ndcg: 0 }],
  evidence: { stages: [{ name: 'final', kind: 'truncate', parents: [], hits: [{ id: 'a' }], latency_ms: 2, source: 'measured' }] },
}

it('selects the later ranking and its measured failure', () => {
  const failure = { doc_id: 'a', stage: 'final', reason: 'missing' }
  const view = executionView({ ...query, repeat_evidence: { '2': { failures: [failure] } } }, 1)
  expect(view.ids).toEqual(['b'])
  expect(view.metrics?.ndcg).toBe(0)
  expect(view.evidence?.failures).toEqual([failure])
})

it('shares unchanged outputs without inventing timings or mutating repeat one', () => {
  const view = executionView({ ...query, repeat_ids: [['a'], ['a']], repeat_evidence: {} }, 1)
  expect(view.sameOutputs).toBe(true)
  expect(view.evidence?.stages?.[0].latency_ms).toBeNull()
  expect(query.evidence?.stages?.[0].latency_ms).toBe(2)
})

it('never reuses first-stage evidence when later evidence is unavailable', () => {
  expect(executionView(query, 1).evidence).toBeUndefined()
  expect(executionView(query, 0).evidence).toEqual(query.evidence)
  expect(executionView(query, 99).selected).toBe(1)
  expect(executionView({ ...query, repeat_ids: [['a']] }, 1).selected).toBe(0)
})

it('rejects malformed repeat evidence at the import boundary', () => {
  expect(() => parseArtifact({ format: 'vectraudit.run', run_id: 'x', summary: {}, queries: [{ ...query, repeat_evidence: { '2': { stages: 42 } } }] })).toThrow()
})
