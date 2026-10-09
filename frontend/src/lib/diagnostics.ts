/** Live pipeline connection check: one bounded sample call to a registered adapter. */
import type { ContextValue, FilterValue, ConversationTurn, QueryRow } from './guided'

export type FindingStatus = 'ok' | 'warning' | 'error' | 'unknown' | 'skipped'
export interface DiagnosticFinding { code: string; status: FindingStatus; message: string; hint: string }
export interface DiagnosticReport {
  executed: boolean; latency_ms: number | null; findings: DiagnosticFinding[]
  returned_ids: string[]; missing_ids: string[]; stage_count: number; reported_stage_count: number
  corpus_version_known: boolean
}
export interface DiagnosticResponse { pipeline: string; report: DiagnosticReport }
export interface DiagnosticSample {
  text: string; k: number; filters: Record<string, FilterValue>; context: Record<string, ContextValue>; history?: ConversationTurn[]
  forbidden: string[]; no_answer: boolean
}

/** Prefer an exploratory question with text; never borrow held-out questions for smoke tests. */
export function sampleFromRows(rows: QueryRow[], k: number): DiagnosticSample {
  const row = rows.find(r => r.text && r.split !== 'heldout')
  return {
    text: row?.text || '', k,
    filters: row?.filter || {}, context: row?.context || {},
    ...(row?.history?.length ? { history: row.history } : {}),
    forbidden: row?.forbidden || [], no_answer: !!row?.no_answer,
  }
}

export function diagnosticBody(sample: DiagnosticSample, pipeline: string, corpus: string, confirmRemote: boolean) {
  return { pipeline, confirm_remote: confirmRemote, corpus, ...sample }
}

/** Errors fail the check; warnings and unknowns pass with explicit gaps; skipped checks are neutral. */
export function diagnosticVerdict(report: DiagnosticReport): 'failed' | 'gaps' | 'connected' {
  if (!report.executed || report.findings.some(f => f.status === 'error')) return 'failed'
  if (report.findings.some(f => f.status === 'warning' || f.status === 'unknown')) return 'gaps'
  return 'connected'
}

export const statusLabel: Record<FindingStatus, string> = {
  ok: 'Passed', warning: 'Warning', error: 'Failed', unknown: 'Not verified', skipped: 'Not tested',
}
