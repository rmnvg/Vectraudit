import { describe, expect, it } from 'vitest'
import { STARTER_POLICY, formFromPolicy, policyFromForm, policyLabel } from './release'

describe('release policy form', () => {
  it('preserves critical rules through editing and accepts assertion-only policies', () => {
    const raw = { critical_queries: [{ query_id: 'invoice', doc_id: 'guide', max_rank: 3 }] }
    const { policy, errors } = policyFromForm(formFromPolicy(raw))
    expect(errors).toEqual([])
    expect(policy?.critical_queries).toEqual(raw.critical_queries)
  })
  it('rejects malformed, duplicate and incomplete critical assertions', () => {
    expect(() => formFromPolicy({ critical_queries: {} })).toThrow()
    expect(() => formFromPolicy({ critical_queries: [{ typo: 'query' }] })).toThrow()
    const form = formFromPolicy(STARTER_POLICY)
    form.criticalQueries = [{ queryId: 'q', docId: 'd', maxRank: '0' }, { queryId: 'q', docId: 'd', maxRank: '' }]
    expect(policyFromForm(form).policy).toBeUndefined()
    expect(policyFromForm(form).errors.join(' ')).toMatch(/duplicates/)
  })
  it('round-trips the starter policy without changing its rules', () => {
    const { policy, errors } = policyFromForm(formFromPolicy(STARTER_POLICY))
    expect(errors).toEqual([])
    expect(policy).toEqual({ ...STARTER_POLICY, latency: { p95_max_increase_pct: 30, min_increase_ms: 2 } })
  })
  it('keeps minimums, latency ceilings and environment rules', () => {
    const raw = { metrics: { ann_recall: { min: 0.9 } }, latency: { p95_max_ms: 80, require_same_environment: false }, min_queries: 5, primary_metric: 'recall', require_significance: true }
    const { policy } = policyFromForm(formFromPolicy(raw))
    expect(policy?.metrics).toEqual({ ann_recall: { min: 0.9 } })
    expect(policy?.latency).toEqual({ p95_max_ms: 80, min_increase_ms: 2, require_same_environment: false })
    expect(policy?.primary_metric).toBe('recall')
    expect(policy?.require_significance).toBe(true)
  })
  it('keeps cohort control and audited exception rules through editing', () => {
    const exceptions = { allow: ['metric'], approver: 'engineer' }
    const raw = { metrics: { ndcg: { max_drop: 0.02 } }, cohorts: { min_queries: 5, metrics: { ndcg: { max_drop: 0.05 } }, control: 'bonferroni' }, exceptions }
    const form = formFromPolicy(raw)
    expect(form.cohortControl).toBe(true)
    const { policy } = policyFromForm(form)
    expect(policy?.cohorts).toEqual(raw.cohorts)
    expect(policy?.exceptions).toEqual(exceptions)
    form.cohortControl = false
    expect(policyFromForm(form).policy?.cohorts).toEqual({ min_queries: 5, metrics: { ndcg: { max_drop: 0.05 } } })
  })
  it('chooses one way to judge allowed drops', () => {
    expect(formFromPolicy({ metrics: { ndcg: { max_drop: 0.02 } } }).dropTest).toBe('measured')
    const form = formFromPolicy({ metrics: { ndcg: { max_drop: 0.02 } }, non_inferiority: true })
    expect(form.dropTest).toBe('non_inferiority')
    expect(policyFromForm(form).policy).toMatchObject({ non_inferiority: true, require_significance: false })
    form.dropTest = 'significance'
    const { policy } = policyFromForm(form)
    expect(policy?.require_significance).toBe(true)
    expect(policy).not.toHaveProperty('non_inferiority')
  })
  it('explains what to fix instead of producing an invalid policy', () => {
    const form = formFromPolicy({ metrics: {} })
    expect(policyFromForm(form).errors[0]).toMatch(/at least one quality rule/)
    form.metrics.ndcg = { enabled: true, maxDrop: '', min: '' }
    expect(policyFromForm(form).errors).toContain('Set an allowed drop or a minimum for ndcg, or turn it off.')
    form.metrics.ndcg = { enabled: true, maxDrop: '-1', min: '2' }
    const { errors, policy } = policyFromForm(form)
    expect(policy).toBeUndefined()
    expect(errors).toEqual(expect.arrayContaining(['ndcg allowed drop must be a number of at least 0.', 'ndcg minimum must be at most 1.']))
  })
  it('labels exploratory defaults distinctly from an agreed policy', () => {
    expect(policyLabel(null)).toBe('Exploratory defaults')
    expect(policyLabel({ id: 'x', name: 'Release', version: 2, sha256: 'a'.repeat(64), source: 'workspace' })).toBe('Release v2 · aaaaaaaaaaaa')
  })
})

it('keeps context assertions through an edit instead of dropping them', () => {
  const raw = { version: 1, metrics: { ndcg: { max_drop: 0.02 } }, critical_context: [{ query_id: 'account-recovery' }] }
  const { policy, errors } = policyFromForm(formFromPolicy(raw))
  expect(errors).toEqual([])
  expect(policy?.critical_context).toEqual([{ query_id: 'account-recovery' }])
  expect(() => formFromPolicy({ critical_context: [{ query_id: 'q', doc_id: 'd' }] })).toThrow()
  const repeated = formFromPolicy({ ...raw, critical_context: [{ query_id: 'q' }, { query_id: 'q' }] })
  expect(policyFromForm(repeated).errors).toContain('Context assertion 2 repeats a question.')
  const onlyContext = policyFromForm({ ...formFromPolicy({ critical_context: [{ query_id: 'q' }] }) })
  expect(onlyContext.errors).toEqual([])
})
