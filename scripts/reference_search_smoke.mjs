// Actual BM25 case study through the browser, checked against exported CLI reports.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'

const root = resolve(import.meta.dirname, '..')
const output = process.env.VECTRAUDIT_REFERENCE_SMOKE_DIR
  ? resolve(process.env.VECTRAUDIT_REFERENCE_SMOKE_DIR)
  : await mkdtemp(join(tmpdir(), 'vectraudit-reference-browser-'))
await mkdir(output, { recursive: true })
const study = join(output, 'study')
const python = join(root, '.venv/bin/python')
const generated = spawnSync(python, ['-I', join(root, 'examples/real_pipeline_case_study.py'), '--out', study], {
  cwd: root, encoding: 'utf8', timeout: 120_000,
})
assert.equal(generated.status, 0, generated.error?.message || generated.stdout + generated.stderr)
const summary = JSON.parse(await readFile(join(study, 'summary.json'), 'utf8'))
const base = 'http://127.0.0.1:8821'
const server = spawn(python, ['-I', '-m', 'vectraudit', 'workbench', 'serve', join(study, 'workspace'),
  '--manifest', join(study, 'registered/manifest.json'), '--port', '8821'], {
  cwd: root, stdio: ['ignore', 'ignore', 'pipe'],
})
let browser, page, stderr = ''
server.stderr.on('data', chunk => { stderr += chunk })
async function choose(select, prefix) {
  await select.selectOption(await select.locator('option', { hasText: prefix }).first().getAttribute('value'))
}
try {
  let ready = false
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(stderr)
    try { if ((await fetch(`${base}/api/workbench`)).ok) { ready = true; break } } catch {}
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  assert.ok(ready, 'Reference workbench did not start: ' + stderr)
  browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true })
  await context.tracing.start({ screenshots: true, snapshots: true })
  page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(String(error)))
  await page.goto(base)
  await page.getByRole('navigation', { name: 'Evaluation steps' }).getByRole('button', { name: /Compare a fix/ }).click()
  await choose(page.getByLabel('Baseline run'), 'baseline ·')
  await choose(page.getByLabel('Release policy', { exact: true }), 'Reference search regression policy')
  async function compare(variant) {
    await choose(page.getByLabel('Candidate run'), variant + ' ·')
    const response = page.waitForResponse(r => r.url().endsWith('/api/guided/compare') && r.request().method() === 'POST')
    await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
    const result = await (await response).json()
    const expected = JSON.parse(await readFile(join(study, variant, 'report.json'), 'utf8'))
    assert.equal(result.check.verdict, expected.verdict)
    assert.equal(result.policy.sha256, summary.policy_sha256)
    assert.deepEqual(result.check.critical_queries, expected.critical_queries)
    assert.deepEqual(result.check.metrics, expected.metrics)
    await page.getByText(`Release comparison: ${expected.verdict}`, { exact: true }).waitFor()
    return result
  }
  await compare('candidate')
  const critical = page.getByRole('region', { name: 'Critical question results' })
  await critical.getByText(/fail: recover-account/).waitFor()
  assert.equal(await page.getByRole('list', { name: 'Failing executions for recover-account' }).getByRole('listitem').count(), 3)
  await page.getByLabel('Compare query', { exact: true }).selectOption('recover-account')
  await page.getByLabel('Highlight document ID').fill('account-recovery')
  await page.getByRole('region', { name: 'Finding recover-account', exact: true }).getByText(/First missing at keyword/).waitFor()
  await page.getByRole('region', { name: 'Export release check' }).getByText(/Export is available once the candidate passes/).waitFor()
  assert.equal(await page.getByRole('button', { name: 'Export release check', exact: true }).count(), 0)
  await page.screenshot({ path: join(output, 'candidate-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  await page.screenshot({ path: join(output, 'candidate-mobile.png'), fullPage: true })

  await page.setViewportSize({ width: 1440, height: 1100 })
  await compare('fixed')
  await critical.getByText(/pass: recover-account/).waitFor()
  await page.getByLabel('Release check name').fill('Browser reference release')
  await page.getByRole('button', { name: 'Export release check', exact: true }).click()
  await page.getByText('Before the first CI run', { exact: true }).waitFor()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download files (.zip)' }).click()
  const zip = join(output, 'browser-release.zip')
  await (await download).saveAs(zip)
  const inspected = spawnSync(python, ['-I', '-c', `
import json, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    release = json.loads(archive.read('retrieval/release/release.json'))
    assert release['policy']['sha256'] == sys.argv[2]
    assert release['repeats'] == 3
    assert release['pipeline']['target'] == 'reference_fixed:retrieve'
    assert '.github/workflows/retrieval-release-check.yml' in archive.namelist()
`, zip, summary.policy_sha256], { encoding: 'utf8', timeout: 10_000 })
  assert.equal(inspected.status, 0, inspected.stderr)
  await page.screenshot({ path: join(output, 'fixed-export.png'), fullPage: true })
  assert.deepEqual(errors, [])
  await context.tracing.stop({ path: join(output, 'browser-trace.zip') })
  console.log('Reference browser comparison and export passed. Artifacts: ' + output)
} catch (error) {
  if (page) {
    await page.screenshot({ path: join(output, 'failure.png'), fullPage: true }).catch(() => {})
    await page.context().tracing.stop({ path: join(output, 'browser-trace.zip') }).catch(() => {})
  }
  throw error
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
