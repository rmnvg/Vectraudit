import { expect, it } from 'vitest'
import { isCheckReport, summarizeCheck } from './checks'
import type { CheckReport } from './checks'
import lineage from './fixtures/lineage.json'
import freshness from './fixtures/freshness.json'
import access from './fixtures/access.json'
import cache from './fixtures/cache.json'
import migration from './fixtures/migration.json'
import intake from './fixtures/intake.json'

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
