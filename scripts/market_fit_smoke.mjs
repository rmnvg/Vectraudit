// Real browser journey for passage review, imports, permissions, freezing and cases.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const workspace = await mkdtemp(join(tmpdir(), 'vectraudit-market-fit-'))
const base = 'http://127.0.0.1:8818'
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', workspace, '--port', '8818'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let stderr = ''
server.stderr.on('data', b => { stderr += b })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(stderr)
    try { if ((await fetch(`${base}/api/workbench`)).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto(base)
  await page.getByRole('button', { name: 'Use starter example' }).click()
  const steps = page.getByRole('navigation', { name: 'Evaluation steps' })
  await page.getByRole('button', { name: 'Irrelevant: old-guide', exact: true }).click()
  await page.getByRole('button', { name: 'Relevant: current-guide', exact: true }).click()
  assert.equal(await page.getByRole('button', { name: 'Relevant: current-guide', exact: true }).getAttribute('aria-pressed'), 'true')
  await page.getByRole('button', { name: 'Save benchmark', exact: true }).click()
  await page.getByText(/Saved locally/).waitFor()
  await page.getByRole('button', { name: 'Freeze saved benchmark' }).click()
  await page.getByLabel('Frozen benchmark').locator('option').nth(1).waitFor({ state: 'attached' })
  await page.getByRole('button', { name: 'Check current corpus' }).click()
  await page.getByText(/Freshness: unchanged/).waitFor()

  // Both positive and negative judgments survive a browser reload and draft restore.
  await page.reload()
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  await page.getByLabel('Resume saved benchmark').selectOption({ label: 'Retrieval starter benchmark' })
  assert.equal(await page.getByRole('button', { name: 'Irrelevant: old-guide', exact: true }).getAttribute('aria-pressed'), 'true')
  await page.getByRole('button', { name: 'Continue to evaluation' }).click()
  async function evaluate(label, depth, reranker) {
    await page.getByLabel('Run label').fill(label)
    await page.getByLabel('Retrieval depth').fill(String(depth))
    await page.getByLabel('Reranker').selectOption(reranker)
    await page.getByRole('button', { name: 'Run evaluation', exact: true }).click()
    await page.getByText(label, { exact: true }).waitFor()
  }
  await evaluate('Baseline evidence', 50, 'none')
  await evaluate('Regression evidence', 2, 'phrase')
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByRole('button', { name: 'Track finding' }).click()
  await page.getByLabel('Case owner').fill('Pilot engineer')
  await page.getByLabel('Investigation note').fill('Restore candidate depth')
  await page.getByLabel('Case status').selectOption('fix_proposed')
  await page.getByRole('button', { name: 'Save case update' }).click()
  await page.getByText('invoice · fix_proposed · Pilot engineer', { exact: true }).waitFor()
  await page.reload()
  await steps.getByRole('button', { name: /Compare a fix/ }).click()
  await page.getByText('invoice · fix_proposed · Pilot engineer', { exact: true }).waitFor()
  assert.equal(await page.getByLabel('Baseline run').locator('option').count(), 2)

  // Build an explicit permission scenario without editing JSON.
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  await page.getByLabel('Resume saved benchmark').selectOption({ label: 'Retrieval starter benchmark' })
  await page.getByText('Build permission scenarios', { exact: true }).click()
  await page.getByLabel('Scenario name').fill('Revoked tenant')
  await page.getByLabel('Context value', { exact: true }).fill('revoked')
  await page.getByLabel('Expect no answer in this context').check()
  await page.getByRole('button', { name: 'Queue scenario' }).click()
  await page.getByRole('button', { name: 'Add permission cases' }).click()
  await page.getByRole('button', { name: /^permission-/ }).waitFor()
  await page.getByRole('button', { name: 'Save benchmark', exact: true }).click()
  await page.getByText(/Saved locally/).waitFor()
  await page.getByRole('button', { name: 'Continue to evaluation' }).click()
  await evaluate('Permission baseline', 50, 'none')
  await evaluate('Permission candidate', 50, 'none')
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()
  for (const [label, prefix] of [['Baseline run', 'Permission baseline'], ['Candidate run', 'Permission candidate']]) {
    const choices = await page.getByLabel(label).locator('option').allTextContents()
    await page.getByLabel(label).selectOption({ label: choices.find(value => value.startsWith(prefix)) })
  }
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText('4 query rows · 3 independent families', { exact: true }).waitFor()
  await steps.getByRole('button', { name: /Build benchmark/ }).click()

  // Import duplicate incident exports; do not import the supposed ground truth.
  await page.getByText('Import questions or production incidents', { exact: true }).click()
  await page.getByLabel('Source file', { exact: true }).setInputFiles({ name: 'incidents.csv', mimeType: 'text/csv', buffer: Buffer.from('query_id,text,relevant\ntrace-a,How to cancel?,old-guide:2\ntrace-b,How to cancel?,old-guide:2\n') })
  await page.getByRole('checkbox', { name: /Production incidents:/ }).check()
  await page.getByRole('button', { name: 'Preview import' }).click()
  await page.getByText(/1 questions · 1 duplicates removed/).waitFor()
  await page.getByRole('button', { name: 'Add 1 reviewed selections' }).click()
  const incident = page.getByRole('button', { name: /^incident-/ })
  await incident.click()
  assert.match(await incident.textContent(), /Unannotated/)
  assert.equal(await page.getByLabel('Relevance grades').inputValue(), '{}')

  // A changed local corpus invalidates prior freshness results and creates a review queue.
  await page.getByText('Local passage corpus', { exact: true }).click()
  await page.getByLabel('Passage corpus file').setInputFiles({ name: 'changed.jsonl', mimeType: 'application/x-ndjson', buffer: Buffer.from(JSON.stringify({ doc_id: 'current-guide', text: 'Changed invoice documentation' })) })
  await page.getByLabel('Frozen benchmark').selectOption({ index: 1 })
  await page.getByRole('button', { name: 'Check current corpus' }).click()
  await page.getByText(/Freshness: review/).waitFor()
  await page.getByText('invoice · current-guide · changed', { exact: true }).waitFor()
  await page.screenshot({ path: join(workspace, 'market-fit-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: join(workspace, 'market-fit-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  assert.deepEqual(errors, [])
  console.log('Market fit browser checks passed. Screenshots: ' + workspace)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
