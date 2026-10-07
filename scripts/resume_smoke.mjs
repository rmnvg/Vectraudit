// Real browser recovery across a killed workbench server, with a versioned Python adapter.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'

const root = resolve(import.meta.dirname, '..')
const workspace = await mkdtemp(join(tmpdir(), 'vectraudit-resume-browser-'))
const base = 'http://127.0.0.1:8812'
await writeFile(join(workspace, 'adapter.py'), `
import time
from pathlib import Path
from vectraudit.pipelines.models import PipelineResult
root = Path(__file__).parent
def retrieve(request):
    with (root / 'calls').open('a') as f:
        f.write(request.query_id + '\\n')
    time.sleep(0.3)
    return PipelineResult(hits=[])
`)
await writeFile(join(workspace, 'manifest.json'), JSON.stringify({
  dataset: 'unused.jsonl', corpus_version: 'corpus-v1',
  candidates: [{ name: 'Recovery test', pipeline_version: 'deployment-v1', adapter: { kind: 'python', target: 'adapter:retrieve' } }],
}))
let server, browser
let logs = ''
async function start() {
  server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', workspace, '--manifest', join(workspace, 'manifest.json'), '--port', '8812'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
  server.stderr.on('data', chunk => { logs += chunk })
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(logs)
    try { if ((await fetch(`${base}/api/workbench`)).ok) return } catch {}
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  throw new Error('Workbench did not start: ' + logs)
}
try {
  await start()
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
  const errors = []
  page.on('pageerror', error => errors.push(String(error)))
  const session = await (await fetch(`${base}/api/guided/session`)).json()
  const headers = { 'Content-Type': 'application/json', 'X-Workbench-Token': session.token }
  const draft = await (await fetch(`${base}/api/guided/drafts`, { method: 'POST', headers, body: JSON.stringify({
    name: 'Restart benchmark', dataset: Array.from({ length: 12 }, (_, i) => JSON.stringify({ query_id: String(i), text: 'question', no_answer: true })).join('\n'),
  }) })).json()
  await fetch(`${base}/api/guided/jobs`, { method: 'POST', headers, body: JSON.stringify({
    draft_id: draft.id, revision: draft.revision, pipeline: 'configured-0', label: 'Restart evaluation',
  }) })
  let saved
  for (let i = 0; i < 100; i++) {
    saved = (await (await fetch(`${base}/api/guided/evaluations`, { headers })).json())[0]
    if (saved.finished >= 2) break
    await new Promise(resolve => setTimeout(resolve, 50))
  }
  assert.ok(saved.finished >= 2 && saved.finished < 12)
  const stopped = once(server, 'exit')
  server.kill('SIGKILL')
  await stopped
  const before = (await readFile(join(workspace, 'calls'), 'utf8')).trim().split('\n')
  await start()
  await page.goto(base)
  await page.getByRole('navigation', { name: 'Evaluation steps' }).getByRole('button', { name: /Evaluate & inspect/ }).click()
  await page.getByRole('heading', { name: 'Saved progress' }).waitFor()
  await page.getByRole('button', { name: 'Resume evaluation', exact: true }).click()
  await page.getByRole('button', { name: 'Continue saved evaluation' }).click()
  await page.getByRole('status').getByText(/^Evaluation complete/).waitFor({ timeout: 20_000 })
  await page.getByRole('button', { name: 'Inspect failures' }).waitFor()
  const after = (await readFile(join(workspace, 'calls'), 'utf8')).trim().split('\n')
  // Only the in-flight last call may repeat. Earlier saved queries appear once.
  for (const id of before.slice(0, -1)) assert.equal(after.filter(q => q === id).length, 1)
  assert.equal(new Set(after).size, 12)
  await page.setViewportSize({ width: 390, height: 844 })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  assert.deepEqual(errors, [])
  console.log('Browser restart/resume passed: ' + workspace)
} finally {
  await browser?.close()
  server?.kill('SIGTERM')
}
