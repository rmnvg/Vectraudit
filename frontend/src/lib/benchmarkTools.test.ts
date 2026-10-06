import { describe, expect, it } from 'vitest'
import { documentGrade, gradeDocument, pendingReview, readDocuments, reviewRequestKey } from './benchmarkTools'
import type { ReviewPool } from './benchmarkTools'
import type { QueryRow } from './guided'

describe('passage review semantics', () => {
  const row: QueryRow = { query_id: 'q', text: 'invoice', split: 'heldout', context: { tenant: 'a' }, forbidden: ['b'], relevant: { a: 2 }, relevant_spans: [{ doc_id: 'c', start: 0, end: 5, grade: 1 }] }
  it('keeps irrelevant distinct from unjudged without fabricating no-answer', () => {
    const negative = gradeDocument(row, 'a', 0, [])
    expect(negative.row.relevant).toEqual({})
    expect(negative.irrelevant).toEqual(['a'])
    expect(negative.row.no_answer).toBeUndefined()
    const unjudged = gradeDocument(negative.row, 'a', null, negative.irrelevant)
    expect(unjudged.irrelevant).toEqual([])
    expect(unjudged.row.relevant).toEqual({})
  })
  it('preserves other spans and permissions when changing a document grade', () => {
    const next = gradeDocument(row, 'a', 1, [])
    expect(next.row.relevant_spans).toEqual(row.relevant_spans)
    expect(next.row.context).toEqual(row.context)
    expect(next.row.forbidden).toEqual(row.forbidden)
    expect(next.row.split).toBe('heldout')
    expect(gradeDocument(row, 'c', 0, []).row.relevant_spans).toEqual([])
  })
  it('clears a no-answer assertion when a human finds a relevant passage', () => {
    const next = gradeDocument({ query_id: 'q', no_answer: true }, 'a', 2, ['a'])
    expect(next.row.no_answer).toBe(false)
    expect(next.row.relevant).toEqual({ a: 2 })
    expect(next.irrelevant).toEqual([])
  })
  it('rejects malformed and duplicate documents before rendering', () => {
    expect(() => readDocuments('{"doc_id":"a","text":4}')).toThrow()
    expect(() => readDocuments('{"doc_id":"a","text":"a"}\n{"doc_id":"a","text":"b"}')).toThrow('unique')
    expect(readDocuments('')).toEqual([])
  })
})

describe('retrieved passage review', () => {
  const pool: ReviewPool = {
    draft_id: 'draft', revision: 'revision', dataset_sha256: 'hash', query_id: 'q',
    baseline: 'base', candidate: 'candidate', baseline_measured: true, candidate_measured: true,
    items: ['positive', 'negative', 'span', 'forbidden', 'unjudged'].map(doc_id => ({ query_id: 'q', doc_id, reason: 'candidate disagreement', priority: 2, baseline_rank: 1, candidate_rank: null })),
  }
  const row: QueryRow = { query_id: 'q', text: 'invoice', context: { tenant: 'a' }, relevant: { positive: 2 }, relevant_spans: [{ doc_id: 'span', start: 0, end: 2, grade: 1 }], forbidden: ['forbidden'] }
  it('counts only pending judgments and restores a cleared judgment', () => {
    expect(pendingReview(pool, row, ['negative']).map(i => i.doc_id)).toEqual(['unjudged'])
    const judged = gradeDocument(row, 'unjudged', 0, ['negative'])
    expect(pendingReview(pool, judged.row, judged.irrelevant)).toEqual([])
    const cleared = gradeDocument(judged.row, 'unjudged', null, judged.irrelevant)
    expect(pendingReview(pool, cleared.row, cleared.irrelevant).map(i => i.doc_id)).toEqual(['unjudged'])
    expect(documentGrade(row, 'span', [])).toBe(1)
  })
  it('never carries a pool into a held-out or different question', () => {
    expect(pendingReview(pool, { ...row, split: 'heldout' }, [])).toEqual([])
    expect(pendingReview(pool, { ...row, query_id: 'other-tenant' }, [])).toEqual([])
  })
  it('retains a review session for label edits but invalidates changed retrieval inputs', () => {
    expect(reviewRequestKey(gradeDocument(row, 'new', 2, []).row)).toBe(reviewRequestKey(row))
    for (const changed of [
      { ...row, text: 'password' }, { ...row, context: { tenant: 'b' } },
      { ...row, filter: { current: true } }, { ...row, vector: [1, 0] },
      { ...row, split: 'heldout' as const },
    ]) expect(reviewRequestKey(changed)).not.toBe(reviewRequestKey(row))
    expect(reviewRequestKey({ ...row, context: { a: 1, b: 2 } })).toBe(reviewRequestKey({ ...row, context: { b: 2, a: 1 }, split: 'explore' }))
    expect(reviewRequestKey({ ...row, vector: [0.1, 0.2] })).toBe(reviewRequestKey({ ...row, vector: [Math.fround(0.1), Math.fround(0.2)] }))
  })
})
