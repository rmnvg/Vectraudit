/** Investigation screen contracts: one failed question, declared fixes, trials, validation. */

export interface Loss { class: string; doc_id?: string; stage?: string; stage_kind?: string | null; detail?: string; code?: string | null }
export interface FixSuggestion { name: string; params: Record<string, string | number | boolean>; addresses: [string, string][] }
export interface Investigation {
  query_id: string; question: string | null; metrics: Record<string, number> | null
  losses: Loss[]; hypotheses: string[]; hypotheses_note: string; cautions: string[]; uncertainty: string[]
  fixes: FixSuggestion[]; unaddressed: string[]; regression: { new_losses: Loss[] } | null
  pipeline: { name: string; remote: string | null; params: Record<string, unknown> }
  trial: { available: boolean; reason: string | null; explore_questions: number; heldout_questions: number; repeats: number; k: number }
}
export interface TrialRow {
  query_id: string; focus: boolean; ndcg_before: number | null; ndcg_after: number | null
  targeted_losses: Loss[]; recovered: Loss[]; moved: { from: Loss; to: Loss }[]; introduced: Loss[]; regressed: boolean
}
export interface Trial {
  id: string; kind: 'trial'; status_label: string; fix: { name: string; params: Record<string, unknown> }
  questions: number; focus: string[]; recovered_focus: number; regressed: string[]; untargeted_focus: string[]; rows: TrialRow[]
}
export interface Validation {
  id: string; kind: 'validation'; status: string; status_label: string; approved: boolean; questions: number
  check: { verdict: string; reasons: string[]; inconclusive: boolean } | null
  promote: { name: string; params: Record<string, unknown>; note: string }
}

const CLASSES: Record<string, string> = {
  not_observed: 'not in any recorded stage',
  lost_downstream: 'dropped',
  lost_in_context: 'did not reach the model intact',
  answered_wrong_despite_evidence: 'answered wrong despite the evidence reaching the model',
  permission: 'forbidden document delivered',
  no_answer: 'results delivered for a question with no answer',
  error: 'the question failed',
}

/** `kb-7 dropped at rerank (rerank)`; one line per recorded loss. */
export function lossLabel(loss: Loss): string {
  const what = CLASSES[loss.class] ?? loss.class.replaceAll('_', ' ')
  const where = loss.stage ? ` at ${loss.stage}${loss.stage_kind ? ` (${loss.stage_kind})` : ''}` : ''
  const detail = loss.detail ? `: ${loss.detail}` : loss.code ? `: ${loss.code}` : ''
  return `${loss.doc_id ? `${loss.doc_id} ` : ''}${what}${where}${detail}`
}

/** Pipeline calls a trial plans: both configurations, failing plus sampled questions, repeats. */
export function trialCalls(trial: Investigation['trial'], focus: number, sample: number): number {
  const others = Math.max(0, Math.min(sample, trial.explore_questions - focus))
  return 2 * (focus + others) * trial.repeats
}

/** One line per trial row: what the fix did to that question. */
export function trialOutcome(row: TrialRow): string {
  const parts: string[] = []
  if (row.recovered.length) parts.push(`recovered ${row.recovered.map(l => l.doc_id ?? l.class).join(', ')}`)
  if (row.moved.length) parts.push(`${row.moved.map(m => `${m.to.doc_id ?? ''} now ${lossLabel({ ...m.to, doc_id: undefined })}`.trim()).join('; ')} (moved, not recovered)`)
  if (row.introduced.length) parts.push(`new: ${row.introduced.map(lossLabel).join('; ')}`)
  if (row.focus && !row.targeted_losses.length) parts.push('the fix targets none of its losses')
  if (!parts.length) parts.push(row.regressed ? 'lower nDCG' : 'unchanged')
  return parts.join(' · ')
}

export interface InvestigationRecord {
  id: string; status: 'open' | 'closed'; opened_by: string; opened_at: string
  diagnosis: { status: 'pending' | 'accepted' | 'disputed'; by?: string; reason?: string | null }
  requests: { what: string; by: string; at: string }[]
  trials: { id: string; fix: string | null }[]; validations: { id: string; fix: string | null; approved: boolean }[]
  outcome: string | null
  protections?: { policy_id: string; name: string; version: number; rules: { kind: string }[]; by: string }[]
  measures?: { minutes_to_accepted_diagnosis: number | null; minutes_to_verified_fix: number | null; reproduced_by_second_engineer: boolean }
}
export const OUTCOMES = [
  ['fixed', 'Fixed (verified by an approved validation)'],
  ['not_fixed', 'Not fixed'],
  ['no_fix_needed', 'No fix needed'],
  ['abandoned', 'Abandoned'],
] as const

const NAME_KEY = 'vectraudit.reviewer'

/** The reviewer name this browser last used; recorded with decisions, never verified. */
export function savedName(): string {
  try { return localStorage.getItem(NAME_KEY) ?? '' } catch { return '' }
}

export function rememberName(name: string): void {
  try { localStorage.setItem(NAME_KEY, name) } catch { /* private window: the name is kept for this page only */ }
}

/** Whether closing as fixed is allowed: only an approved validation verifies a repair. */
export function canCloseFixed(record: InvestigationRecord): boolean {
  return record.validations.some(v => v.approved)
}
