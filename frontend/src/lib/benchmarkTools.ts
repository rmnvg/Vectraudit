import type { QueryRow } from './guided'

export type Irrelevant = Record<string, string[]>
type Doc = { doc_id: string; text: string; metadata?: Record<string, unknown> }
export function readDocuments(corpus: string): Doc[] {
  const docs: unknown[] = corpus.split('\n').filter(l => l.trim()).map(l => JSON.parse(l))
  if (docs.length > 5000 || docs.some(d => !d || typeof d !== 'object' || typeof (d as Doc).doc_id !== 'string' || typeof (d as Doc).text !== 'string')) throw new Error('Each corpus row needs a document ID and passage text.')
  if (new Set(docs.map(d => (d as Doc).doc_id)).size !== docs.length) throw new Error('Document IDs must be unique.')
  return docs as Doc[]
}

export function gradeDocument(row: QueryRow, doc: string, grade: number | null, irrelevant: string[]): { row: QueryRow; irrelevant: string[] } {
  const relevant = { ...row.relevant }
  delete relevant[doc]
  if (grade != null && grade > 0) relevant[doc] = grade
  const next = { ...row, relevant, relevant_spans: row.relevant_spans?.filter(s => s.doc_id !== doc) }
  if (grade != null && grade > 0) next.no_answer = false
  return { row: next, irrelevant: [...irrelevant.filter(d => d !== doc), ...(grade === 0 ? [doc] : [])] }
}
