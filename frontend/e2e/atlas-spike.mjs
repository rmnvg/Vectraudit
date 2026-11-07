// Spike 31.0: offline + GPU check for embedding-atlas. Run: node e2e/atlas-spike.mjs
import { chromium } from '@playwright/test'

const base = 'http://127.0.0.1:8791'
const browser = await chromium.launch({ args: ['--enable-gpu', '--use-gl=angle', '--use-angle=metal', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } })
const blocked = []
const local = []
await page.route('**/*', (route) => {
  const url = route.request().url()
  if (url.startsWith(base) || url.startsWith('data:') || url.startsWith('blob:')) { local.push(url); return route.continue() }
  blocked.push(url)
  return route.abort()
})
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
const t0 = Date.now()
await page.goto(`${base}/#/atlas-spike`)
await page.waitForSelector('[data-testid=atlas] canvas', { timeout: 20000 })
await page.waitForTimeout(2500)
const ms = Date.now() - t0
const gl = await page.evaluate(() => {
  const c = document.createElement('canvas'); const g = c.getContext('webgl2')
  return g ? g.getParameter(g.VERSION) + ' / ' + (g.getExtension('WEBGL_debug_renderer_info') ? g.getParameter(g.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL) : '?') : 'no webgl2'
})
await page.screenshot({ path: '/tmp/atlas-spike.png' })
const box = await page.locator('[data-testid=atlas] canvas').first().boundingBox()
// Hover then click on a cluster (screen coordinates from the screenshot above).
for (const [x, y] of [[620, 240], [500, 360], [845, 440]]) { await page.mouse.move(x, y); await page.waitForTimeout(300); await page.mouse.click(x, y) }
await page.waitForTimeout(800)
const selected = await page.getByTestId('atlas-count').innerText()
console.log(JSON.stringify({ loadMs: ms, gl, blockedExternalRequests: blocked, localRequests: local.length, localWorkers: local.filter((u) => /worker|wasm/.test(u)).map((u) => u.split('/').pop()), selected, errors: errors.slice(0, 5) }, null, 2))
await browser.close()
