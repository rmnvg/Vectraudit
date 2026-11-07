// Browser smoke for the corpus atlas against the real local workbench server (needs --enable-gpu
// in headless Chrome: the map is WebGL 2).
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const frontend = join(root, 'frontend')
const data = join(frontend, 'e2e/.data-atlas')
const made = spawnSync(join(root, '.venv/bin/python'), ['e2e/atlas_data.py'], { cwd: frontend, encoding: 'utf8' })
assert.equal(made.status, 0, made.stderr + made.stdout)
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', data, '--port', '8809'], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let serverErrors = ''
server.stderr.on('data', chunk => { serverErrors += chunk })
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(serverErrors)
    try { if ((await fetch('http://127.0.0.1:8809/api/workbench')).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true, args: ['--enable-gpu', '--ignore-gpu-blocklist'] })
  const page = await browser.newPage({ viewport: { width: 1300, height: 1400 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto('http://127.0.0.1:8809/#/workbench')
  await page.getByLabel('Open local artifact').selectOption('atlas-e2e.json')
  await page.locator('[data-testid=corpus-atlas] canvas').first().waitFor({ timeout: 15000 })
  const body = await page.getByTestId('corpus-atlas').innerText()
  assert.match(body, /of documents form a group no benchmark question is close to/)
  assert.match(body, /\(hypothesis\)/)
  assert.match(body, /Held-out questions are not drawn/)
  assert.doesNotMatch(body, /heldout-1/)
  assert.match(body, /distorts distance/)
  assert.match(await page.getByLabel('Legend').innerText(), /Question: failed \(13\)/)
  await page.screenshot({ path: join(data, 'atlas-desktop.png') })
  assert.deepEqual(errors, [])
  console.log('Atlas browser checks passed. Screenshot: ' + join(data, 'atlas-desktop.png'))
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
