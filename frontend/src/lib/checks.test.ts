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
import embeddingDrift from './fixtures/embedding_drift.json'
import componentBoard from './fixtures/component_scoreboard.json'
import runHeader from './fixtures/run_header.json'
import tuneReport from './fixtures/tune.json'
import tuneComparison from './fixtures/tune_comparison.json'

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

it('reads embedding drift as a hypothesis to look into, with the rule that decided it', () => {
  const s = summary(embeddingDrift)
  expect(s.title).toBe('Embedding drift')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toMatch(/^Drift: 60 production questions against 60 benchmark questions; domain classifier ROC AUC [0-9.]+, drift when the ROC AUC is above the 95% quantile/)
  expect(s.notes.some(n => n.includes('not that retrieval got worse'))).toBe(true)
  expect(s.items[0]).toMatch(/production questions form a group no benchmark question is close to \(about: .*wallet.*\)\. \(hypothesis\)$/)
})

it('says when intake raised questions from a new topic', () => {
  const s = summary({ ...intake, queue: { ...intake.queue, new_topic: 3 } })
  expect(s.notes).toContain('3 pending questions from a new topic (embedding drift), raised for review.')
})

it('shows a calm comparison as no drift', () => {
  const s = summary({ ...embeddingDrift, status: 'no_drift', result: { ...embeddingDrift.result, drift: false } })
  expect(s.verdict).toBe('pass')
  expect(s.headline.startsWith('No drift: ')).toBe(true)
})

it('summarises a component scoreboard as information, worst component first', () => {
  const s = summary(componentBoard)
  expect(s.title).toBe('Component scoreboard')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toBe('8 observed questions in 7 families; 7 lost their required evidence, most at rerank (71.4%). Information only.')
  expect(s.items).toContain('Cohort multi_hop (meant to exercise retrieve, rerank): 1 of 2 observed questions lost their evidence there')
  expect(s.notes[0]).toBe('Not observed (never assigned): 2; not applicable: 1 unlabelled, 1 no-answer; errored: 0.')
})

it('summarises a run too large to open from its validated header', () => {
  const s = summary(runHeader)
  expect(s.title).toBe('Run summary: nightly')
  expect(s.verdict).toBe('pass')
  expect(s.headline).toMatch(/^large\.run\.json: complete, 12 questions \([0-9.]+ MiB\), every one validated without loading the run\./)
  expect(s.items.some(i => i.startsWith('ndcg: '))).toBe(true)
  expect(s.notes[0]).toContain('vectraudit runs table')
  expect(summary({ ...runHeader, run: { ...runHeader.run, status: 'partial' } }).verdict).toBe('attention')
})

it('summarises a tuning sweep: the recommended value, every point and missed neighbours without a cause', () => {
  const s = summary(tuneReport)
  expect(s.title).toBe('Tuning sweep: nprobe')
  expect(s.verdict).toBe('pass')
  expect(s.headline).toMatch(/^nprobe 8 is the smallest swept value whose mean ANN recall@10 reaches 0\.8 on faiss target IVF32_Flat\.index \(recall 0\.842, p95 [\d.]+ ms\)\.$/)
  expect(s.items.filter(i => i.startsWith('nprobe '))).toHaveLength(6)
  expect(s.items.find(i => i.startsWith('nprobe 8:'))).toContain('(recommended)')
  expect(s.items).toContain('q1 at nprobe 8: recall 0.60, missed 162 (from 16), 1273 (from 16), 97 (from 16), 1173 (from 16)')
  expect(s.notes).toContain('Recall is ANN recall against backend exact search.')
  expect(s.notes.some(n => n.startsWith('Latency: ') && n.includes('round trip included'))).toBe(true)
  expect(s.notes).toContain('A backend does not expose why a neighbour was missed, so no cause is given.')
})

it('never reads an unfinished or unmet tuning sweep as a pass', () => {
  const failed = summary({ ...tuneReport, status: 'failed', points: [], missed: [], error: { message: 'connection refused' } })
  expect(failed.verdict).toBe('fail')
  expect(failed.headline).toContain('connection refused')
  expect(summary({ ...tuneReport, status: 'cancelled' }).verdict).toBe('attention')
  const unmet = summary({ ...tuneReport, recommendation: { ...tuneReport.recommendation, status: 'recall', value: null } })
  expect(unmet.verdict).toBe('attention')
  expect(unmet.headline).toBe('No swept value reached recall 0.8 on faiss target IVF32_Flat.index (best 1.000 at nprobe 32).')
  const latency = summary({ ...tuneReport, max_p95_ms: 0.001, recommendation: { ...tuneReport.recommendation, status: 'latency', value: null, recall_needs: 8 } })
  expect(latency.headline).toBe('No swept value reached recall 0.8 within p95 0.001 ms on faiss target IVF32_Flat.index; recall needs nprobe 8.')
  const filtered = summary({ ...tuneReport, filtered: [{ filter: { lang: 'de' }, pass_rate: 0.5, recommendation: { value: 16 }, points: [{ value: 16, recall: 0.9 }] }] })
  expect(filtered.items).toContain('Filter {"lang":"de"} (pass rate 50.00%): recommended nprobe 16; recall 16: 0.900')
  expect(isCheckReport({ ...tuneReport, timing: undefined })).toBe(false)
})

it('compares two tuning sweeps per shared value and flags where B loses recall', () => {
  const s = summary(tuneComparison)
  expect(s.title).toBe('Tuning comparison')
  expect(s.verdict).toBe('attention')
  expect(s.headline).toBe('Compared at 6 shared values; B loses recall at 6.')
  expect(s.items[0]).toMatch(/^A: faiss target IVF32_Flat\.index, recommended nprobe 8 \(recall 0\.842, p95 [\d.]+ ms\)$/)
  expect(s.items[1]).toBe('B: faiss target IVF32_PQ4.index, no value reached its target')
  expect(s.items).toContain('Questions missing neighbours: q11, q8 only in A; q2, q5 only in B')
  const unlike = summary({ ...tuneComparison, comparable: false, reasons: ['The sweeps used k=10 and k=5.'], matched: [] })
  expect(unlike.headline).toBe('Not comparable: recall was not compared.')
  expect(unlike.notes).toContain('The sweeps used k=10 and k=5.')
})
