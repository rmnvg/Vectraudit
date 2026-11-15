// Browser smoke: a pipeline run's view shows where its questions first lost their evidence.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const dir = await mkdtemp(join(tmpdir(), 'vectraudit-components-'))
const make = `
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from tests.test_components import broken_run
Path(sys.argv[1]).write_text(json.dumps(broken_run({"multi_hop": ["retrieve", "rerank"]})))
`
const made = spawnSync(join(root, '.venv/bin/python'), ['-c', make, join(dir, 'broken.run.json')], { cwd: root, encoding: 'utf8' })
assert.equal(made.status, 0, made.stderr + made.stdout)
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', dir, '--port', '8813'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8813/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1200 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8813/#/workbench')
  await page.getByLabel('Open local artifact').selectOption('broken.run.json')
  const panel = page.getByLabel('Component scoreboard')
  await panel.waitFor({ timeout: 5000 })
  const text = await panel.textContent()
  assert.match(text, /rerank71\.4%35\.9 to 91\.86/, 'the broken reranker holds most of the loss')
  assert.match(text, /Not observed \(never assigned\): 2/)
  assert.match(text, /multi_hop \(meant to exercise retrieve, rerank\): 1 of 2 observed questions/)
  assert.deepEqual(errors, [])
  console.log('Component scoreboard browser check passed.')
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
