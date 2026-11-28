import { renderToStaticMarkup } from 'react-dom/server'
import { expect, it } from 'vitest'
import CheckReports from './CheckReports'
import { summarizeCheck } from '../lib/checks'
import type { CheckReport } from '../lib/checks'
import lineage from '../lib/fixtures/lineage.json'
import intake from '../lib/fixtures/intake.json'
import tune from '../lib/fixtures/tune.json'

const reports = [lineage, intake, tune] as unknown as CheckReport[]
const names = ['lineage.json', 'intake.json', 'tune.json']
const LABEL = { pass: 'pass', fail: 'fail', attention: 'needs attention' }
const html = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;')

// Real reports from the Python checks (tests/test_check_fixtures.py keeps them current).
it('shows one card per report, newest import first, each with its verdict and file name', () => {
  const page = renderToStaticMarkup(<CheckReports reports={reports} names={names} />)
  const positions = reports.map(r => page.indexOf(`aria-label="${html(summarizeCheck(r).title)}"`))
  expect(positions.every(p => p >= 0)).toBe(true)
  expect([...positions].sort((a, b) => b - a)).toEqual(positions) // last imported renders first
  for (const [i, report] of reports.entries()) {
    const s = summarizeCheck(report)
    const card = page.slice(positions[i], i === 0 ? undefined : positions[i - 1])
    expect(card).toContain(`>${LABEL[s.verdict]}</span>`)
    expect(card).toContain(names[i])
    expect(card).toContain(html(s.headline))
  }
})

it('marks a failing report in red and never as a pass', () => {
  const s = summarizeCheck(reports[0])
  expect(s.verdict).toBe('fail')
  const page = renderToStaticMarkup(<CheckReports reports={[reports[0]]} names={['lineage.json']} />)
  expect(page).toContain('text-red-600')
  expect(page).not.toContain('>pass</span>')
})

it('renders nothing for no reports', () => {
  expect(renderToStaticMarkup(<CheckReports reports={[]} names={[]} />)).toBe('<div class="space-y-4"></div>')
})
