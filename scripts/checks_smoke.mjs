// Browser smoke: each new check report opens in the workbench's Checks tab with its summary.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { copyFile, mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const fixtures = join(root, 'frontend/src/lib/fixtures')
const reports = {
  'fusion_replay.json': /Fusion replay/,
  'corpus_scan.json': /Corpus injection scan/,
  'poisoning.json': /Poisoning check/,
  'promptfoo_redteam.json': /promptfoo red team/,
  'selectivity_sweep.json': /Recall by filter pass rate/,
  'capacity.json': /Capacity curve/,
  'label_audit.json': /Label audit/,
  'embedding_drift.json': /Embedding drift/,
  'component_scoreboard.json': /Component scoreboard/,
  'run_header.json': /Run summary: nightly/,
}
const dir = await mkdtemp(join(tmpdir(), 'vectraudit-checks-'))
for (const name of Object.keys(reports)) await copyFile(join(fixtures, name), join(dir, name))
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', dir, '--port', '8810'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8810/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1000 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8810/#/workbench')
  for (const [name, title] of Object.entries(reports)) {
    await page.getByLabel('Open local artifact').selectOption(name)
    await page.getByRole('navigation', { name: 'Workbench' }).getByRole('button', { name: 'Checks' }).waitFor()
    await page.getByText(title).first().waitFor({ timeout: 5000 })
    assert.match(await page.textContent('body'), title, `${name} did not open in the Checks tab`)
  }
  assert.deepEqual(errors, [])
  console.log('Checks browser checks passed for ' + Object.keys(reports).length + ' reports.')
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
