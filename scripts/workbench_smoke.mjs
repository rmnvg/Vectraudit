// Browser integration and visual evidence against the real local workbench server.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const artifacts = await mkdtemp(join(tmpdir(), 'vectraudit-workbench-browser-'))
const cli = join(root, '.venv/bin/vectraudit')
for (const command of [
  ['workbench', 'run', 'examples/workbench/manifest.json', '--out', artifacts],
  ['workbench', 'optimize', 'examples/workbench/manifest.json', '--out', join(artifacts, 'optimization.json')],
  ['workbench', 'benchmark', 'examples/workbench/queries.jsonl', '--out', join(artifacts, 'benchmark.json')],
]) {
  const result = spawnSync(cli, command, { cwd: root, encoding: 'utf8' })
  assert.equal(result.status, 0, result.stderr + result.stdout)
}
const server = spawn(cli, ['workbench', 'serve', artifacts, '--port', '8807'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8807/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8807')
  await page.getByRole('heading', { name: 'Your next release. With evidence.' }).waitFor()
  const options = await page.getByLabel('Open local artifact').locator('option').allTextContents()
  await page.getByLabel('Open local artifact').selectOption(options.find(o => o.endsWith('.run.json')))
  await page.getByRole('heading', { name: 'Stage evidence' }).waitFor()
  await page.getByLabel('Trace document ID').fill('invoice-guide')
  await page.screenshot({ path: join(artifacts, 'pipeline-desktop.png'), fullPage: true })
  await page.getByLabel('Open local artifact').selectOption('optimization.json')
  await page.getByRole('heading', { name: 'Measured candidates' }).waitFor()
  assert.match(await page.textContent('body'), /Held-out validation/)
  await page.screenshot({ path: join(artifacts, 'experiments-desktop.png'), fullPage: true })
  await page.getByLabel('Open local artifact').selectOption('benchmark.json')
  await page.getByRole('heading', { name: 'Benchmark health' }).waitFor()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: join(artifacts, 'benchmark-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  assert.deepEqual(errors, [])
  console.log('Workbench browser checks passed. Screenshots: ' + artifacts)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
