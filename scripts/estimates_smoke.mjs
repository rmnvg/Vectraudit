// Browser smoke: the workbench shows judge-corrected estimates (or why there are none) for a run.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const dir = await mkdtemp(join(tmpdir(), 'vectraudit-estimates-'))
const make = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, '.')
import tests.test_answer_estimates as T
from vectraudit.evaluation.answers import attach_answers
T.FAMILIES = 100
ds, run, truth, model = T.world(seed=21)
sample = T.sample_for(ds, run, Path(tempfile.mkdtemp()), size=40)
from tests.test_review_sample import _valid_run
valid = _valid_run(ds)
out = attach_answers(valid, T.rows(truth, model, sample), sample=sample)
Path(sys.argv[1]).write_text(json.dumps(out))
`
const made = spawnSync(join(root, '.venv/bin/python'), ['-c', make, join(dir, 'estimated.run.json')], { cwd: root, encoding: 'utf8' })
assert.equal(made.status, 0, made.stderr + made.stdout)
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', dir, '--port', '8812'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8812/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1200 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8812/#/workbench')
  await page.getByLabel('Open local artifact').selectOption('estimated.run.json')
  const panel = page.getByLabel('Judge-corrected estimates')
  await panel.waitFor({ timeout: 10000 })
  const text = await panel.innerText()
  assert.match(text, /ragas\.correctness/)
  assert.match(text, /Judge only \(biased\)/)
  assert.match(text, /People only/)
  assert.match(text, /Corrected \(95%\)/)
  assert.match(text, /40 of 100 families reviewed by people/)
  assert.match(text, /no release policy reads it/)
  assert.deepEqual(errors, [])
  console.log('Estimates browser checks passed.')
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
