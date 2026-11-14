import { expect, it } from 'vitest'
import { isCheckReport, summarizeCheck } from './checks'
import type { CheckReport } from './checks'
import lineage from './fixtures/lineage.json'
import freshness from './fixtures/freshness.json'
import access from './fixtures/access.json'
import cache from './fixtures/cache.json'
import migration from './fixtures/migration.json'
import intake from './fixtures/intake.json'
import fusionReplay from './fixtures/fusion_replay.json'
import corpusScan from './fixtures/corpus_scan.json'
import poisoningReport from './fixtures/poisoning.json'
import redteam from './fixtures/promptfoo_redteam.json'
import selectivity from './fixtures/selectivity_sweep.json'
import capacity from './fixtures/capacity.json'
import labelAudit from './fixtures/label_audit.json'

// Fixtures are real reports from the Python checks on synthetic test data
// (tests/test_check_fixtures.py keeps their shape in step with the code).
const summary = (report: unknown) => {
  expect(isCheckReport(report)).toBe(true)
  return summarizeCheck(report as CheckReport)
}

it('names the ingestion stage that lost a passage', () => {
  const s = summary(lineage)
  expect(s.verdict).toBe('fail')
  expect(s.headline).toBe('1 labelled passage followed; 1 lost during ingestion.')
  expect(s.items).toContain('1 first lost at split across chunks')
  expect(s.notes.some(n => n.includes('embedding model'))).toBe(true)
})

it('shows stale content served and conflicting versions', () => {
  const s = summary(freshness)
  expect(s.verdict).toBe('fail')
  expect(s.items).toContain('refund: stale policy#0@r8 reached the context')
  expect(s.items).toContain('refund: policy delivered in revisions r8, r9')
})

it('shows revoked documents still served and labels to review', () => {
  const s = summary(access)
  expect(s.verdict).toBe('fail')
  expect(s.items).toContain('payroll-u17: payroll-2026 reached results, model context')
  expect(s.items).toContain('Review payroll-u17: revoked label(s) payroll-2026 removed')
  expect(s.headline).toContain('Expected access applied to 1 question.')
})

it('explains a cache leak by what the probe observed', () => {
  const s = summary(cache)
  expect(s.verdict).toBe('fail')
  expect(s.items[0]).toBe("payroll-u17 after payroll-u2: payroll-2026 reached results, model context (identical to the primer's answer)")
})

it('reads a migration as not ready while a concern remains', () => {
  const s = summary(migration)
  expect(s.title).toBe('Migration: pgvector → qdrant')
  expect(s.verdict).toBe('fail')
  expect(s.headline).toBe('Not ready to migrate: release policy verdict pass, 1 concern.')
})

it('flags production recurrences for attention', () => {
  const s = summary(intake)
  expect(s.verdict).toBe('attention')
  expect(s.items[0]).toMatch(/^q2 at .* delivered none of its labelled documents$/)
})

it('refuses other or malformed JSON', () => {
  expect(isCheckReport({ format: 'vectraudit.run' })).toBe(false)
  expect(isCheckReport({ format: 'vectraudit.cache_probe', leaks: 'x', errors: [], complete: true })).toBe(false)
  expect(isCheckReport(null)).toBe(false)
})

it('shows a fusion replay as a hypothesis with its limits and next step', () => {
  const s = summary(fusionReplay)
  expect(s.title).toBe('Fusion replay')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toContain('a hypothesis, not a measurement')
  expect(s.headline).toContain('best wsum(norm=none)')
  expect(s.items.some(i => i.startsWith('wsum(norm=none) (exact): ndcg'))).toBe(true)
  expect(s.notes.some(n => n.includes('1 heldout'))).toBe(true)
  expect(s.notes.some(n => n.startsWith('Next: declare this fix'))).toBe(true)
})

it('does not read a replay with no comparable question as an improvement', () => {
  const s = summary({ ...fusionReplay, best: null, configurations: [] })
  expect(s.verdict).toBe('pass')
  expect(s.headline).toContain('none could be compared')
})

it('lists the documents a corpus scan flags as hypotheses to read, with what each detector means', () => {
  const s = summary(corpusScan)
  expect(s.title).toBe('Corpus injection scan')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toContain('5 documents scanned; 4 flagged')
  expect(s.headline).toContain('hypotheses for a person to read')
  expect(s.items.some(i => i.startsWith('1 document: instruction override.'))).toBe(true)
  expect(s.items[s.items.length - 1]).toMatch(/^Read page-/)
  expect(s.items.some(i => i.includes('Read page-chat_template_token (row 3, high severity'))).toBe(true)
})

it('does not read a clean scan as proof the corpus is safe', () => {
  const clean = { ...corpusScan, summary: { ...corpusScan.summary, flagged: 0, findings: 0, documents_by_detector: {}, by_detector: {} }, review: [], documents: [] }
  const s = summary(clean)
  expect(s.verdict).toBe('pass')
  expect(s.notes.some(n => n.includes('does not show the corpus is safe'))).toBe(true)
})

it('shows a failed poisoning check with its safety reason, rates and limits', () => {
  const s = summary(poisoningReport)
  expect(s.title).toBe('Poisoning check')
  expect(s.verdict).toBe('fail')
  expect(s.headline).toContain('exploratory, not a release approval')
  expect(s.items.some(i => i.includes('reached the final results'))).toBe(true)
  expect(s.items.some(i => i.startsWith('hijack: retrieved for 1 of 2 families'))).toBe(true)
  expect(s.items.some(i => i.startsWith('Poisoned documents on control questions: 1 of 1 family'))).toBe(true)
})

it('reads an inconclusive poisoning check as attention, never a pass', () => {
  const s = summary({ ...poisoningReport, verdict: 'inconclusive', failures: [], reasons_inconclusive: ['the overlay may not be loaded'], overlay: { warning: 'No poisoned document appears in any recorded stage.' } })
  expect(s.verdict).toBe('attention')
  expect(s.headline).toContain('Inconclusive')
  expect(s.notes.some(n => n.includes('overlay may not be loaded'))).toBe(true)
})

it('adds the rate of answers that followed an injected instruction when it was measured', () => {
  const followed = { questions_checked: 4, flagged: 1, families: 4, rate: 0.25, ci_low: 0.05, ci_high: 0.7 }
  const s = summary({ ...poisoningReport, followed_injection: followed })
  expect(s.items.some(i => i.startsWith('Answers that followed an injected instruction (canary found): 1 of 4 families (25%'))).toBe(true)
  expect(summary(poisoningReport).items.some(i => i.includes('followed an injected instruction'))).toBe(false)
})

it('summarises promptfoo red-team results per plugin as model-graded and exploratory', () => {
  const s = summary(redteam)
  expect(s.title).toBe('promptfoo red team (RAG plugins)')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toContain('Graded by a model and exploratory')
  expect(s.items.some(i => i.startsWith('rag-poisoning: 2 of 3 tests failed (67%'))).toBe(true)
  expect(s.items.some(i => i.includes('by strategy: base64 1/2, basic 1/1'))).toBe(true)
  expect(s.notes.some(n => n.includes('Skipped other plugins: harmful:hate (1)'))).toBe(true)
  expect(s.notes.some(n => n.includes('are counted here only'))).toBe(true)
})

it('reads a red-team run with no failures as a pass, not as proof', () => {
  const clean = { ...redteam, plugins: Object.fromEntries(Object.entries(redteam.plugins).map(([k, p]) => [k, { ...p, failed: 0, passed: p.tests, failure_rate: 0 }])) }
  const s = summary(clean)
  expect(s.verdict).toBe('pass')
  expect(s.headline).toContain('exploratory, not a release approval')
})

it('shows recall by pass rate, naming what was and was not measured', () => {
  const s = summary(selectivity)
  expect(s.title).toBe('Recall by filter pass rate')
  expect(s.verdict).toBe('attention')
  expect(s.items[0].startsWith('Unfiltered: 100% of the corpus passes (3,000 documents); efSearch 16:')).toBe(true)
  expect(s.items.some(i => i.startsWith('synthetic@0.01: 1.0% of the corpus passes (30 documents)'))).toBe(true)
  expect(s.items.some(i => i.includes('recall 0.95 not reached'))).toBe(true)
  expect(s.notes).toContain('Pass rate is the fraction of documents that satisfy the filter, not the fraction removed.')
  expect(s.notes.some(n => n.includes('was not reached in the sweep range'))).toBe(true)
  expect(s.notes[s.notes.length - 1]).toContain('brute-force')
})

it('reads a sweep where every filter reaches the target as a pass', () => {
  const easy = { ...selectivity, runs: selectivity.runs.map(r => ({ ...r, recommended: 16 })), suggestions: [] }
  expect(summary(easy).verdict).toBe('pass')
})

it('summarises a capacity curve per concurrency level, and says when the ceiling limited it', () => {
  const s = summary(capacity)
  expect(s.title).toBe('Capacity curve')
  expect(s.verdict).toBe('pass')
  expect(s.headline).toContain('1 concurrency level measured on qdrant target staging')
  expect(s.headline).toContain('a probe of staging, not a load test')
  expect(s.items[0]).toMatch(/^Concurrency 1: [\d.]+ requests\/s, p50 [\d.]+ ms, p95 [\d.]+ ms, p99 [\d.]+ ms, 0\.0% errors, recall 1\.000/)
  expect(s.notes.some(n => n.includes('runner environments match'))).toBe(true)
})

it('does not read a failed or erroring capacity run as a pass', () => {
  expect(summary({ ...capacity, status: 'failed', levels: [], error: { message: 'denied' } }).verdict).toBe('fail')
  const erroring = { ...capacity, levels: capacity.levels.map(l => ({ ...l, error_rate: 0.2 })) }
  expect(summary(erroring).verdict).toBe('attention')
  expect(summary({ ...capacity, status: 'cancelled' }).verdict).toBe('attention')
  const unlike = { ...capacity, comparison: { comparable: false, reasons: ['Runner environments differ in system'], levels: [] } }
  expect(summary(unlike).notes).toContain('Not compared with the baseline: Runner environments differ in system')
})

it('only recognises the two new formats when they carry the fields the summary reads', () => {
  expect(isCheckReport({ ...capacity, levels: undefined })).toBe(false)
  expect(isCheckReport({ ...selectivity, runs: undefined })).toBe(false)
})

it('lists suspect labels as hypotheses and says which signals were not measured', () => {
  const s = summary(labelAudit)
  expect(s.title).toBe('Label audit')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toBe('2 labels to review on 1 question, from rank. Suspects are hypotheses for a reviewer; nothing was changed.')
  expect(s.items).toContain('q1 / z: labelled relevant (grade 1), label may be wrong; rank')
  expect(s.items).toContain('q1 / m: unjudged, label may be missing; rank')
  expect(s.notes).toContain('near duplicate not measured: No corpus vectors given.')
})

it('does not read an audit without suspects as proof the labels are right', () => {
  const s = summary({ ...labelAudit, suspects: [], summary: { ...labelAudit.summary, suspects: 0, questions: 0 } })
  expect(s.verdict).toBe('pass')
  expect(s.notes.some(n => n.includes('does not show the labels are right'))).toBe(true)
})
