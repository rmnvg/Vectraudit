/** Local guided workflow contracts; all requests stay on the workbench origin. */
import type { PolicyRef } from './release'
import type { Run } from './workbench'

export interface PipelineChoice { id: string; name: string; kind: string; destination: string | null }
export interface DraftSummary { id: string; name: string; revision: string }
export interface GuidedSession { token: string; pipelines: PipelineChoice[]; drafts: DraftSummary[] }
export interface Draft extends DraftSummary {
  dataset: string
  corpus?: string
  irrelevant?: Record<string, string[]>
  audit: { n_queries: number; n_labelled: number; missing_categories: string[]; duplicates: string[][] }
}
export interface QueryRow {
  family_id?: string
  query_id: string; text?: string; vector?: number[]; relevant?: Record<string, number>
  no_answer?: boolean; forbidden?: string[]; context?: Record<string, string | number | boolean>
  filter?: Record<string, string | number | boolean>; tags?: string[]; split?: 'explore' | 'heldout'
  relevant_spans?: { doc_id: string; start: number; end: number; grade: number }[]
}
export interface JobState { status: 'running' | 'done' | 'failed' | 'cancelled'; progress: number; message: string; error?: string; run?: Run }
export type Severity = 'blocking' | 'error' | 'regression' | 'info' | 'improvement'
export interface StageLoss { stage: string; parents: string[]; evidence: string | null; reason: string }
export interface Finding {
  kind: 'error' | 'permission' | 'no_answer' | 'lost_relevant' | 'recovered_relevant' | 'still_missing' | 'ranking_regressed' | 'improved'
  severity: Severity; message: string; doc_ids: string[]; new_doc_ids?: string[]; runs?: string[]
  documents?: { doc_id: string; stages: StageLoss[] | string[]; reason?: string; diverged_at?: string | null; baseline_stages?: string[] | null; candidate_stages?: string[] }[]
}
export interface TriageItem {
  query_id: string; tags: string[]; severity: Severity; metric: string | null
  baseline: number | null; candidate: number | null; delta: number | null; stages_recorded: boolean; findings: Finding[]
}
export interface Triage {
  format: 'vectraudit.triage'; primary_metric: string | null; n_paired: number; truncated: boolean
  counts: Record<Severity | 'unchanged', number>; unpaired: { baseline_only: string[]; candidate_only: string[] }; items: TriageItem[]
}
export interface Comparison {
  baseline: Run; candidate: Run; triage: Triage; files: { baseline: string; candidate: string; check: string }; policy: PolicyRef
  check: { verdict: string; n_queries?: number; n_families?: number; reasons: { message: string }[]; warnings: string[]; metrics: Record<string, { baseline: number; candidate: number; delta: number }>; critical_queries?: CriticalResult[] }
  question_display?: { baseline: QuestionDisplay; candidate: QuestionDisplay }
}
export interface QuestionDisplay { texts: Record<string, string>; reason: string | null }

export function questionText(display: QuestionDisplay | undefined, queryId: string): string | null {
  return display && Object.hasOwn(display.texts, queryId) ? display.texts[queryId] : null
}

export function questionLabel(display: QuestionDisplay | undefined, queryId: string): string {
  const text = questionText(display, queryId)
  return text ? `${text} · ${queryId}` : queryId
}
export interface CriticalExecution { attempt: number | null; repeat: number; rank: number | null; ids: string[] }
export interface CriticalResult {
  query_id: string; doc_id: string; max_rank: number; rank: number | null; verdict: string
  executions?: number; failed_executions?: number; failures?: CriticalExecution[]
}

/** How a critical rule held across every recorded candidate execution. */
export function criticalSummary(rule: CriticalResult): string {
  const total = rule.executions ?? 0
  if (!total) return 'no per-repeat evidence'
  const failed = rule.failed_executions ?? 0
  return failed ? `failed on ${failed} of ${total} executions` : `held on all ${total} executions, worst rank ${rule.rank}`
}

export function executionLabel(execution: CriticalExecution): string {
  return execution.attempt === null ? `Repeat ${execution.repeat}` : `Attempt ${execution.attempt}, repeat ${execution.repeat}`
}

/** Guided benchmarks hold at most this many questions (the server enforces the same). */
export const MAX_QUESTIONS = 1000

export async function guidedRequest<T>(path: string, token = '', body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`api/guided/${path}`, {
    method: body === undefined ? 'GET' : 'POST', signal,
    headers: { 'X-Workbench-Token': token, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The local request failed.')
  return result as T
}

/** Preserve all supported query fields when a user edits a single annotation. */
export function parseQueries(text: string): { rows: QueryRow[]; header?: Record<string, unknown> } {
  const values = text.split('\n').filter(line => line.trim()).map(line => JSON.parse(line))
  let header: Record<string, unknown> | undefined
  if (values[0]?.dataset) header = values.shift()
  if (!values.length || values.some(v => !v || typeof v !== 'object' || typeof v.query_id !== 'string')) {
    throw new Error('Each query needs a query_id. Import a Vectraudit JSONL dataset.')
  }
  return { rows: values, header }
}

export function serializeQueries(rows: QueryRow[], header?: Record<string, unknown>): string {
  return [...(header ? [header] : []), ...rows].map(row => JSON.stringify(row)).join('\n') + '\n'
}

export const demoCorpus = [
  { doc_id: 'current-guide', text: 'Invoice export instructions. Download invoices from billing settings.', metadata: { tenant: 'acme', current: true } },
  { doc_id: 'old-guide', text: 'Invoice export invoice export. Old invoice export instructions.', metadata: { tenant: 'acme', current: false } },
  { doc_id: 'private-guide', text: 'Invoice export instructions for a different tenant.', metadata: { tenant: 'other', current: true } },
  { doc_id: 'account-guide', text: 'Reset password using account recovery.', metadata: { tenant: 'acme', current: true } },
].map(d => JSON.stringify(d)).join('\n')
export const demoQueries: QueryRow[] = [
  { query_id: 'invoice', text: 'invoice export', relevant: { 'current-guide': 2 }, context: { tenant: 'acme' }, filter: { current: true }, forbidden: ['private-guide'], tags: ['permission', 'outdated'] },
  { query_id: 'password', text: 'reset password', relevant: { 'account-guide': 1 }, tags: ['exact_identifier'] },
  { query_id: 'unknown', text: 'volcano weather', no_answer: true, tags: ['no_answer'] },
  { query_id: 'heldout', text: 'download invoices', relevant: { 'current-guide': 1 }, split: 'heldout', tags: ['paraphrase'] },
]
