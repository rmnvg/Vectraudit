import { renderToStaticMarkup } from 'react-dom/server'
import { expect, it } from 'vitest'
import { FailingRules } from './FailingRules'
import type { Acceptance } from '../lib/exceptions'

const rule = (over: Partial<Acceptance>): Acceptance => ({
  rule: 'metrics.ndcg', kind: 'metric', message: 'ndcg dropped by 0.07', role: null, max_days: null, why: null, ...over,
})

const render = (options: Acceptance[]) =>
  renderToStaticMarkup(<FailingRules token="t" options={options} checkFile="pr-42.check.json" />)

it('shows nothing when no rule failed', () => {
  expect(render([])).toBe('')
})

it('lists each failing rule with who may accept it, and keeps the local verdict a fail', () => {
  const html = render([
    rule({ role: 'engineer', max_days: 14 }),
    rule({ rule: 'permissions.forbidden', kind: 'permission', why: 'never' }),
    rule({ rule: 'latency.p95', kind: 'latency', why: 'not_allowed' }),
  ])
  expect(html).toContain('<code class="text-xs">metrics.ndcg</code>')
  expect(html).toContain('Can be accepted by an engineer or owner in the hosted project, for up to 14 days.')
  expect(html).toContain('Cannot be accepted: this policy never waives permission or no-answer failures.')
  expect(html).toContain('Cannot be accepted: this policy does not open latency rules to exceptions.')
  expect(html).toContain('This comparison stays a fail on this machine.')
  // Nothing hosted is claimed before the user asks for it.
  expect(html).not.toContain('role="status"')
  expect(html).toContain('Check accepted regressions in the hosted project')
})

it('marks a rule nobody may accept in amber', () => {
  const html = render([rule({ why: 'no_exceptions' })])
  expect(html).toContain('text-amber-600">Cannot be accepted: this policy has no exceptions block.')
})
