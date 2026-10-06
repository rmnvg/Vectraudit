// Browser journey through the guided workflow against a real local workbench server:
// starter benchmark -> save -> baseline run -> candidate run -> compare -> inspect failures.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, readdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const workspace = await mkdtemp(join(tmpdir(), 'vectraudit-guided-browser-'))
const cli = join(root, '.venv/bin/vectraudit')
const base = 'http://127.0.0.1:8808'
const server = spawn(cli, ['workbench', 'serve', workspace, '--port', '8808'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch(`${base}/api/workbench`)).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto(base)
  const steps = page.getByRole('navigation', { name: 'Evaluation steps' })
  await steps.waitFor()

  // 01 Connect: the starter example fills the local corpus and benchmark.
  await page.getByRole('button', { name: 'Use starter example' }).click()
  await page.getByRole('heading', { name: 'Build a trustworthy benchmark' }).waitFor()

  // 01 Connect: a pipeline test sends one sample question and saves no run.
  await steps.getByRole('button', { name: /Connect pipeline/ }).click()
  const check = page.getByRole('region', { name: 'Test pipeline' })
  assert.equal(await check.getByLabel('Sample question').inputValue(), 'invoice export')
  await check.getByRole('button', { name: 'Test pipeline' }).click()
  // Tenant context reaches the adapter but is never reported as verified authorization.
  await check.getByText(/^Connected, with gaps/).waitFor({ timeout: 20_000 })
  await check.getByText(/downstream use and authorization are unverified/).waitFor()
  assert.equal((await readdir(workspace)).filter(f => f.endsWith('.run.json')).length, 0)
  await steps.getByRole('button', { name: /Build benchmark/ }).click()

  // 02 Benchmark: invalid JSON blocks saving; a no-answer judgment is an explicit label.
  await page.getByRole('button', { name: /^unknown/ }).click()
  assert.ok(await page.getByRole('checkbox', { name: /No relevant document exists/ }).isChecked())
  await page.getByRole('button', { name: /^invoice/ }).click()
  const grades = page.getByLabel('Relevance grades')
  await grades.fill('{"current-guide":')
  assert.ok(await page.getByRole('button', { name: 'Save benchmark' }).isDisabled())
  await grades.fill('{"current-guide":2}')
  await page.getByRole('button', { name: 'Save benchmark' }).click()
  await page.getByText(/Saved locally/).waitFor()
  await page.getByText('4/4 annotated').waitFor()
  await page.screenshot({ path: join(workspace, 'guided-benchmark.png'), fullPage: true })
  await page.getByRole('button', { name: 'Continue to evaluation' }).click()

  // 03 Evaluate: the held-out split needs explicit confirmation.
  await page.getByLabel('Evaluation split').selectOption('heldout')
  assert.ok(await page.getByRole('button', { name: 'Run evaluation' }).isDisabled())
  await page.getByLabel('Evaluation split').selectOption('explore')
  const evaluate = async (label, reranker, depth) => {
    await page.getByLabel('Run label').fill(label)
    await page.getByLabel('Reranker').selectOption(reranker)
    await page.getByLabel('Retrieval depth').fill(String(depth))
    await page.getByRole('button', { name: 'Run evaluation' }).click()
    await page.getByRole('status').getByText(/^Evaluation (complete|partial)/).waitFor({ timeout: 20_000 })
    await page.getByRole('button', { name: 'Run evaluation' }).waitFor()
    // The previous run's status stays visible; wait for this run's recorded row instead.
    await page.getByText(label, { exact: true }).waitFor({ timeout: 20_000 })
  }
  await evaluate('Baseline', 'none', 50)
  // A shallower candidate pool: a plausible latency fix that loses a relevant document.
  await evaluate('Shallow depth', 'phrase', 2)
  const recorded = page.getByRole('button', { name: 'Inspect failures' })
  assert.equal(await recorded.count(), 2)
  assert.equal((await readdir(workspace)).filter(f => f.endsWith('.run.json')).length, 2)

  // 04 Compare: side-by-side traces from both recorded runs.
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()
  const runs = await page.getByLabel('Baseline run').locator('option').allTextContents()
  await page.getByLabel('Baseline run').selectOption({ label: runs.find(r => r.startsWith('Baseline')) })
  await page.getByLabel('Candidate run').selectOption({ label: runs.find(r => r.startsWith('Shallow depth')) })
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText('Release comparison: fail').waitFor()
  await page.getByRole('heading', { name: /^Baseline · / }).waitFor()
  await page.getByRole('heading', { name: /^Candidate · / }).waitFor()
  assert.ok((await readdir(workspace)).some(f => f.endsWith('.check.json')))

  // Findings: the lost relevant document is ranked, located at a stage, and linked to evidence.
  const findings = page.getByRole('list', { name: 'Findings' })
  await findings.getByRole('button', { name: /invoice/ }).waitFor()
  const finding = page.getByRole('region', { name: 'Finding invoice' })
  assert.match(await finding.textContent(), /lost relevant/)
  assert.match(await finding.textContent(), /First missing at keyword in the candidate/)
  assert.match(await finding.textContent(), /\.run\.json/)
  assert.equal(await page.getByLabel('Highlight document ID').inputValue(), 'current-guide')
  await page.getByText('current-guide dropped here').waitFor()
  await page.screenshot({ path: join(workspace, 'guided-compare.png'), fullPage: true })

  // Opening the evidence lands on the same query and document in the detailed trace view.
  await finding.getByRole('button', { name: 'Open in candidate run' }).click()
  await page.getByRole('heading', { name: 'Stage evidence' }).waitFor()
  assert.equal(await page.getByLabel('Trace document ID').inputValue(), 'current-guide')
  await page.getByText(/^invoice · recorded document outputs/).waitFor()
  await page.getByRole('navigation', { name: 'Workbench' }).getByRole('button', { name: 'Guided workflow' }).click()

  // Inspect hands the recorded run to the detailed trace view.
  await steps.getByRole('button', { name: /Evaluate & inspect/ }).click()
  await recorded.first().click()
  await page.getByRole('heading', { name: 'Stage evidence' }).waitFor()

  // A reload keeps the saved benchmark resumable.
  await page.reload()
  await steps.waitFor()
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  const drafts = await page.getByLabel('Resume saved benchmark').locator('option').allTextContents()
  assert.ok(drafts.includes('Retrieval starter benchmark'), drafts.join(', '))

  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: join(workspace, 'guided-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  assert.deepEqual(errors, [])
  console.log('Guided workflow browser checks passed. Screenshots: ' + workspace)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
