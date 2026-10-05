import { describe, expect, it } from 'vitest'
import { gradeDocument, readDocuments } from './benchmarkTools'
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
