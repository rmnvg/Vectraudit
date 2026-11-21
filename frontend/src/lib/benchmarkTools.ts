import type { QueryRow } from './guided'

export type Irrelevant = Record<string, string[]>
type Doc = { doc_id: string; text: string; metadata?: Record<string, unknown> }
export interface LookupDocument {
  doc_id: string; status: 'found' | 'missing' | 'forbidden' | 'version_unavailable'
  text: string | null; title: string | null; version: string | null; sha256: string | null
}
export interface LookupResponse {
  source_run: string; query_id: string; corpus_version: string | null; version_verified: boolean
  documents: LookupDocument[]
}
export interface PassageLookup {
  documents: Record<string, LookupDocument>; busy: boolean; confirmed: boolean
  fetch: (ids: string[]) => void
}
export interface ReviewItem {
  query_id: string; doc_id: string; reason: string; priority: number
  baseline_rank: number | null; candidate_rank: number | null
  /** How far this document could move a release rule not yet settled (decision order). */
  swing?: number
}
/** The pair's stop rule: whether any remaining judgment can still change the verdict. */
export interface ReviewDecision {
  state: 'settled' | 'open' | 'not_certifiable' | 'not_applicable'
  verdict: string; message: string; n_could_change: number; saving: string
  next_questions: { query_id: string; documents: number }[]
}
export interface ReviewPool {
  draft_id: string; revision: string; dataset_sha256: string; query_id: string
  baseline: string; candidate: string; baseline_measured: boolean; candidate_measured: boolean
  items: ReviewItem[]
  decision?: ReviewDecision | null
  document_lookup?: boolean
}

/** Labels may change during review; changing the retrieval request invalidates its pool. */
export function reviewRequestKey(row: QueryRow): string {
  const sorted = (value: Record<string, unknown> | undefined) => Object.entries(value || {}).sort(([a], [b]) => a.localeCompare(b))
  // Saved datasets canonicalize vectors to float32; match that representation after save.
  return JSON.stringify([row.query_id, row.text, row.vector?.map(Math.fround), sorted(row.context), sorted(row.filter), row.split || 'explore'])
}

/** One line for the stop rule's state, and the questions to review next. */
export function decisionSummary(decision: ReviewDecision): { status: string; next: string } {
  const next = decision.next_questions.length
    ? `Next: ${decision.next_questions.map(q => `${q.query_id} (${q.documents})`).join(', ')}`
    : ''
  return { status: decision.message, next: decision.state === 'settled' ? '' : next }
}

export function documentGrade(row: QueryRow, doc: string, irrelevant: string[]): number | undefined {
  const spans = row.relevant_spans?.filter(s => s.doc_id === doc) || []
  return row.relevant?.[doc] ?? (spans.length ? Math.max(...spans.map(s => s.grade)) : irrelevant.includes(doc) ? 0 : undefined)
}

export function pendingReview(pool: ReviewPool, row: QueryRow, irrelevant: string[]): ReviewItem[] {
  if (row.split === 'heldout' || row.query_id !== pool.query_id) return []
  return pool.items.filter(item => !row.forbidden?.includes(item.doc_id) && documentGrade(row, item.doc_id, irrelevant) === undefined)
}
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
