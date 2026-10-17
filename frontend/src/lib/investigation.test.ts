import { expect, it } from 'vitest'
import { lossLabel, trialCalls, trialOutcome } from './investigation'

it('describes each loss with its stage', () => {
  expect(lossLabel({ class: 'lost_downstream', doc_id: 'b', stage: 'final', stage_kind: 'truncate' })).toBe('b dropped at final (truncate)')
  expect(lossLabel({ class: 'lost_in_context', doc_id: 'p', detail: 'truncated' })).toBe('p did not reach the model intact: truncated')
  expect(lossLabel({ class: 'permission', doc_id: 's' })).toBe('s forbidden document delivered')
})

it('plans calls for both configurations within the exploratory questions', () => {
  const trial = { available: true, reason: null, explore_questions: 5, heldout_questions: 1, repeats: 3, k: 2 }
  expect(trialCalls(trial, 1, 20)).toBe(2 * 5 * 3)
  expect(trialCalls(trial, 1, 2)).toBe(2 * 3 * 3)
})

it('says a moved loss is progress, not a recovery', () => {
  const lost = { class: 'not_observed', doc_id: 'rel' }
  const row = { query_id: 'q', focus: true, ndcg_before: 0, ndcg_after: 0, targeted_losses: [lost], recovered: [], introduced: [], regressed: false,
    moved: [{ from: lost, to: { class: 'lost_downstream', doc_id: 'rel', stage: 'final', stage_kind: 'truncate' } }] }
  expect(trialOutcome(row)).toBe('rel now dropped at final (truncate) (moved, not recovered)')
  expect(trialOutcome({ ...row, moved: [], recovered: [lost] })).toBe('recovered rel')
})

it('closes as fixed only after an approved validation', async () => {
  const { canCloseFixed } = await import('./investigation')
  const record = { id: 'a', status: 'open' as const, opened_by: 'A', opened_at: '', diagnosis: { status: 'accepted' as const }, requests: [], trials: [{ id: 't', fix: 'f' }], validations: [], outcome: null }
  expect(canCloseFixed(record)).toBe(false)
  expect(canCloseFixed({ ...record, validations: [{ id: 'v', fix: 'f', approved: false }] })).toBe(false)
  expect(canCloseFixed({ ...record, validations: [{ id: 'v', fix: 'f', approved: true }] })).toBe(true)
})
