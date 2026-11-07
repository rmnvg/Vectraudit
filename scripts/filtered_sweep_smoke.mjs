// Browser smoke: the Tuner's pass-rate card against a real `vectraudit serve` with metadata.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const dir = await mkdtemp(join(tmpdir(), 'vectraudit-filtered-'))
const made = spawnSync(join(root, '.venv/bin/python'), ['examples/make_synthetic.py', '--out', dir, '--n', '2000', '--d', '32', '--nlist', '16'], { cwd: root, encoding: 'utf8' })
assert.equal(made.status, 0, made.stderr + made.stdout)
const server = spawn(join(root, '.venv/bin/vectraudit'), ['serve', join(dir, 'ivf_flat.index'), '--vectors', join(dir, 'vectors.npy'), '--meta', join(dir, 'chunks.jsonl'), '--port', '8811', '--no-browser'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'], env: { ...process.env, VECTRAUDIT_CACHE_DIR: join(dir, 'cache') } })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 150; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8811/api/health')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8811/#/tuner')
  await page.getByLabel('Pass rates (fraction that passes)').fill('0.5, 0.02')
  await page.getByRole('button', { name: 'Run filtered sweep' }).click()
  const table = page.locator('table', { hasText: 'Pass rate' })
  await table.locator('tbody tr').nth(2).waitFor({ timeout: 20000 })
  assert.equal(await table.locator('tbody tr').count(), 3)
  assert.match(await page.textContent('body'), /Recall falls under filters at nprobe/)
  await page.getByLabel('Pass rates (fraction that passes)').fill('50%')
  await page.getByRole('button', { name: 'Run filtered sweep' }).click()
  await page.getByText(/Pass rates are fractions that pass/).waitFor()
  assert.deepEqual(errors, [])
  console.log('Filtered-sweep browser checks passed.')
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
