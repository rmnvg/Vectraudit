import { expect, it } from 'vitest'
import fixtures from './fixtures/answer_estimates.json'
import { estimateRows, readEstimates } from './estimates'

// answer_estimates.json is real output of answer_estimates.py (tests/test_answer_estimates.py
// keeps its shape in step with the code).
it('reads the estimates block the Python code writes, and nothing looser', () => {
  expect(readEstimates({ estimates: fixtures.ok })).not.toBeNull()
  expect(readEstimates({ estimates: { metric: 1 } })).toBeNull()
  expect(readEstimates({})).toBeNull()
  expect(readEstimates(null)).toBeNull()
})

it('shows the three numbers side by side, with the sample and the judge agreement', () => {
  const rows = estimateRows(readEstimates({ estimates: fixtures.ok })!)
  expect(rows).toHaveLength(1)
  const row = rows[0]
  expect(row.ok).toBe(true)
  expect(row.evaluator).toBe('ragas.correctness')
  expect(row.judgeOnly).toMatch(/^\d+%$/)
  expect(row.humanOnly).toMatch(/^\d+% \(\d+% to \d+%\)$/)
  expect(row.corrected).toMatch(/^\d+% \(\d+% to \d+%\)$/)
  expect(row.detail).toContain('40 of 100 families reviewed by people')
  expect(row.detail).toContain('the judge agreed with them on')
})

it('says why there is no corrected estimate rather than showing one', () => {
  const rows = estimateRows(readEstimates({ estimates: fixtures.absent })!)
  expect(rows[0].ok).toBe(false)
  expect(rows[0].corrected).toBeNull()
  expect(rows[0].humanOnly).toBeNull()
  expect(rows[0].judgeOnly).toMatch(/^\d+%$/)
  expect(rows[0].detail).toContain('workbench review-sample')
})

it('warns when the review was small', () => {
  const small = structuredClone(fixtures.ok) as typeof fixtures.ok
  small.by_evaluator['ragas.correctness'].corrected.warnings = ['Only 40 human-reviewed units: the interval is approximate below 100.']
  expect(estimateRows(readEstimates({ estimates: small })!)[0].detail).toContain('approximate below 100')
})
