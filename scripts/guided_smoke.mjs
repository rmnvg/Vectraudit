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
  const evaluate = async (label, reranker) => {
    await page.getByLabel('Run label').fill(label)
    await page.getByLabel('Reranker').selectOption(reranker)
    await page.getByRole('button', { name: 'Run evaluation' }).click()
    await page.getByRole('status').getByText(/^Evaluation (complete|partial)/).waitFor({ timeout: 20_000 })
    await page.getByRole('button', { name: 'Run evaluation' }).waitFor()
  }
  await evaluate('Baseline', 'none')
  await evaluate('Phrase reranker', 'phrase')
  const recorded = page.getByRole('button', { name: 'Inspect failures' })
  assert.equal(await recorded.count(), 2)
  assert.equal((await readdir(workspace)).filter(f => f.endsWith('.run.json')).length, 2)

  // 04 Compare: side-by-side traces from both recorded runs.
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()
  const runs = await page.getByLabel('Baseline run').locator('option').allTextContents()
  await page.getByLabel('Baseline run').selectOption({ label: runs.find(r => r.startsWith('Baseline')) })
  await page.getByLabel('Candidate run').selectOption({ label: runs.find(r => r.startsWith('Phrase reranker')) })
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText(/Release comparison:/).waitFor()
  await page.getByRole('heading', { name: /^Baseline · / }).waitFor()
  await page.getByRole('heading', { name: /^Candidate · / }).waitFor()
  await page.getByLabel('Highlight document ID').fill('current-guide')
  assert.ok((await readdir(workspace)).some(f => f.endsWith('.check.json')))
  await page.screenshot({ path: join(workspace, 'guided-compare.png'), fullPage: true })

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
