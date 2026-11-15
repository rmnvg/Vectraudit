import { expect, it } from 'vitest'
import fixtures from './fixtures/components.json'
import { attribute, componentScoreboard, wilson } from './components'

// components.json holds compact runs and the scoreboard evaluation/components.py computes for
// them (tests/test_components.py keeps it current). The run view must say exactly the same.
const PYTHON_ONLY = ['format', 'format_version', 'run_id', 'label', 'evidence', 'note']
const expected = (board: Record<string, unknown>) => Object.fromEntries(Object.entries(board).filter(([k]) => !PYTHON_ONLY.includes(k)))

it.each(Object.entries(fixtures))('gives the same scoreboard as Python for %s', (_, { run, scoreboard }) => {
  expect(componentScoreboard(run)).toEqual(expected(scoreboard))
})

it('puts the broken reranker first and never assigns untraced questions', () => {
  const board = componentScoreboard(fixtures.broken_reranker.run)!
  const top = [...board.components].sort((a, b) => (b.rate ?? 0) - (a.rate ?? 0))[0]
  expect(top.component).toBe('rerank')
  expect(board.not_observed).toBe(2)
  expect(attribute({ metrics: { hit_rate: 0 }, evidence: { stages: [], failures: [{ doc_id: 'a', stage: null, evidence: 'unavailable' }] } })[0]).toBe('not_observed')
})

it('reads nothing that is not a run', () => {
  expect(componentScoreboard({ format: 'vectraudit.benchmark' })).toBeNull()
  expect(componentScoreboard(null)).toBeNull()
})

it('computes the Wilson interval as Python does, fractional successes included', () => {
  expect(wilson(0, 0)).toEqual([0, 1])
  // wilson_interval(2.5, 10) in evaluation/significance.py
  const [low, high] = wilson(2.5, 10)
  expect(low).toBeCloseTo(0.08094782420999441, 12)
  expect(high).toBeCloseTo(0.5578185757214502, 12)
})
