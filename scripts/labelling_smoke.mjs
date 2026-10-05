// Synthetic local browser regression using the existing Playwright dev dependency.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const temp = await mkdtemp(join(tmpdir(), 'vectraudit-label-browser-'))
const task = join(temp, 'task.json')
const port = Number(process.env.VECTRAUDIT_LABEL_TEST_PORT || 8806)
const base = 'http://127.0.0.1:' + port
const python = process.env.VECTRAUDIT_SMOKE_PYTHON || join(root, '.venv', 'bin', 'python')
const screenshots = process.env.VECTRAUDIT_SCREENSHOT_DIR
const fixture = [
  'import sys,json',
  'from pathlib import Path',
  'from vectraudit.evaluation.dataset import parse_dataset',
  'from vectraudit.experiments.corpus import Document,make_corpus',
  'from vectraudit.labelling.pooling import pool',
  'from vectraudit.labelling.task import save_task',
  'queries="\\n".join(json.dumps({"query_id":"q"+str(i),"text":"How does retrieval evaluation work? "+str(i)}) for i in range(5))',
  'save_task(pool(parse_dataset(queries),make_corpus([Document("doc","A 😀 answer. Human relevance labels tell us which source documents answer a question.")]),k=1),Path(sys.argv[1]))',
].join('\n')
const prepared = spawnSync(python, ['-I', '-c', fixture, task], { cwd: root, encoding: 'utf8' })
assert.equal(prepared.status, 0, prepared.stderr)
const serve = [
  'import sys,uvicorn',
  'from pathlib import Path',
  'from vectraudit.labelling.web import create_app',
  'uvicorn.run(create_app(Path(sys.argv[1])),host="127.0.0.1",port=int(sys.argv[2]),access_log=False)',
].join('\n')
const server = spawn(python, ['-I', '-c', serve, task, String(port)], {
  cwd: root, stdio: ['ignore', 'ignore', 'pipe'],
})
let serverError = ''
server.stderr.on('data', chunk => { serverError += chunk })
let browser
try {
  let ready = false
  for (let attempt = 0; attempt < 100; attempt++) {
    if (server.exitCode !== null) throw new Error(serverError)
    try { ready = (await fetch(base)).ok } catch { /* starting */ }
    if (ready) break
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  assert.ok(ready, serverError)
  browser = await chromium.launch({
    headless: true,
    ...(process.env.VECTRAUDIT_BROWSER_PATH ? { executablePath: process.env.VECTRAUDIT_BROWSER_PATH } : {}),
  })
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => assert.equal(new URL(request.url()).origin, base))
  await page.goto(base + '/?annotator=Alice&q=3')
  // A hidden q=3 input must not intercept the grade-3 shortcut.
  await page.locator('article').focus()
  await page.keyboard.press('3')
  await page.getByText('Saved grade 3').waitFor()
  await page.goto(base + '/?annotator=Alice&q=0')
  await page.evaluate(() => {
    const passage = document.querySelector('.passage')
    const range = document.createRange()
    range.setStart(passage.firstChild, 2)
    range.setEnd(passage.firstChild, 4) // emoji: two UTF-16 units, one Python character
    const selection = window.getSelection()
    selection.removeAllRanges()
    selection.addRange(range)
    document.dispatchEvent(new MouseEvent('mouseup'))
  })
  assert.equal(await page.locator('input[name=start]').inputValue(), '2')
  assert.equal(await page.locator('input[name=end]').inputValue(), '3')
  await page.getByRole('button', { name: 'Grade 2: Relevant', exact: true }).click()
  await page.getByText('Saved grade 2').waitFor()
  assert.equal(await page.locator('input[name=start]').inputValue(), '2')
  assert.equal(await page.locator('input[name=end]').inputValue(), '3')
  const stale = await browser.newPage()
  await stale.goto(base + '/?annotator=Alice&q=0')
  await page.getByRole('button', { name: 'Grade 3: Perfect', exact: true }).click()
  await page.getByText('Saved grade 3').waitFor()
  await stale.getByRole('button', { name: 'Grade 1: Partly', exact: true }).click()
  await stale.getByText('Your judgment changed in another tab.').waitFor()
  await page.goto(base + '/?annotator=Bob&q=0')
  await page.getByRole('button', { name: 'Grade 1: Partly', exact: true }).click()
  await page.getByText('Saved grade 1').waitFor()
  await page.goto(base + '/review?reviewer=Reviewer')
  await page.getByText('Alice: grade 3', { exact: false }).waitFor()
  if (screenshots) {
    await mkdir(screenshots, { recursive: true })
    await page.screenshot({ path: join(screenshots, 'labelling-review.png'), fullPage: true })
  }
  await page.getByRole('button', { name: 'Grade 2: Relevant', exact: true }).click()
  await page.getByText('No unresolved disagreements.', { exact: false }).waitFor()
  await page.goto(base + '/status?annotator=Reviewer')
  await page.getByText('2/5 questions with positive labels.', { exact: false }).waitFor()
  const exported = await page.request.get(base + '/dataset.jsonl')
  assert.equal(exported.status(), 200)
  const rows = (await exported.text()).trim().split('\n').map(line => JSON.parse(line))
  assert.equal(rows[1].relevant.doc, 2)
  if (screenshots) {
    await page.screenshot({ path: join(screenshots, 'labelling-status.png'), fullPage: true })
  }
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(base + '/?annotator=Alice&q=0')
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390)
  if (screenshots) {
    await page.screenshot({ path: join(screenshots, 'labelling-mobile.png'), fullPage: true })
  }
  const saved = JSON.parse(await readFile(task, 'utf8'))
  assert.equal(saved.resolutions[0].grade, 2)
  assert.deepEqual(errors, [])
  console.log('Local labelling browser regression passed.')
} finally {
  if (browser) await browser.close()
  server.kill('SIGTERM')
  await new Promise(resolve => {
    if (server.exitCode !== null) resolve()
    else server.once('exit', resolve)
  })
  await rm(temp, { recursive: true, force: true })
}
