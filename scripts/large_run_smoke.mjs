// Browser smoke: a run too large to open is listed, and choosing it shows its validated summary.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const dir = await mkdtemp(join(tmpdir(), 'vectraudit-large-'))
const make = `
import sys
from pathlib import Path
sys.path.insert(0, '.')
from tests.test_large_runs import write_big
print(write_big(Path(sys.argv[1]), 70 * 2**20))
`
const name = 'nightly.run.json'
const made = spawnSync(join(root, '.venv/bin/python'), ['-c', make, join(dir, name)], { cwd: root, encoding: 'utf8' })
assert.equal(made.status, 0, made.stderr + made.stdout)
const questions = made.stdout.trim()
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', dir, '--port', '8814'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8814/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1000 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8814/#/workbench')
  const picker = page.getByLabel('Open local artifact')
  assert.match(await picker.textContent(), /nightly\.run\.json \(summary only: 70 MiB\)/)
  await picker.selectOption(name)
  await page.getByText(`${questions} questions`).first().waitFor({ timeout: 30000 })
  const body = await page.textContent('body')
  assert.match(body, /every one validated without loading the run/)
  assert.match(body, /vectraudit runs table/)
  assert.deepEqual(errors, [])
  console.log(`Large run browser check passed (${questions} questions).`)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
