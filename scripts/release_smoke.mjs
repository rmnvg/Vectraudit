// Real browser journey: release policy -> comparison -> export -> CLI release check, and a
// stricter policy version changing the comparison and case verification consistently.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
const root = resolve(import.meta.dirname, '..')
const cli = join(root, '.venv/bin/vectraudit')
const workspace = await mkdtemp(join(tmpdir(), 'vectraudit-release-'))
const repo = await mkdtemp(join(tmpdir(), 'vectraudit-release-repo-'))
const base = 'http://127.0.0.1:8819'
// Not connected to a hosted project, whatever the shell has configured.
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !k.startsWith('VECTRAUDIT_CLOUD_')))
const server = spawn(cli, ['workbench', 'serve', workspace, '--port', '8819'], { cwd: root, env, stdio: ['ignore', 'ignore', 'pipe'] })
let browser
let stderr = ''
server.stderr.on('data', b => { stderr += b })
async function choose(select, prefix) {
  await select.selectOption(await select.locator('option', { hasText: prefix }).first().getAttribute('value'))
}
function run(args, expected = 0) {
  const result = spawnSync(cli, args, { cwd: repo, encoding: 'utf-8' })
  assert.equal(result.status, expected, `${args.join(' ')}\n${result.stdout}\n${result.stderr}`)
  return result.stdout
}
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(stderr)
    try { if ((await fetch(`${base}/api/workbench`)).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto(base)
  await page.getByRole('button', { name: 'Use starter example' }).click()
  await page.getByRole('button', { name: 'Save benchmark', exact: true }).click()
  await page.getByText(/Saved locally/).waitFor()
  await page.getByRole('button', { name: 'Continue to evaluation' }).click()
  async function evaluate(label, depth, reranker) {
    await page.getByLabel('Run label').fill(label)
    await page.getByLabel('Retrieval depth').fill(String(depth))
    await page.getByLabel('Reranker').selectOption(reranker)
    await page.getByRole('button', { name: 'Run evaluation', exact: true }).click()
    await page.getByRole('status', { name: `Evaluation ${label}` }).getByText(/Evaluation complete/).waitFor()
  }
  await evaluate('Baseline', 50, 'none')
  await evaluate('Candidate', 2, 'phrase')
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()

  // Without a saved policy, the comparison is explicitly exploratory and cannot be exported.
  const policy = page.getByLabel('Release policy', { exact: true })
  assert.equal(await policy.inputValue(), '')
  await choose(page.getByLabel('Baseline run'), 'Baseline ·')
  await choose(page.getByLabel('Candidate run'), 'Candidate ·')
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText('(exploratory, not a release decision)', { exact: false }).waitFor()
  await page.getByText(/cannot become a release contract/).waitFor()

  // Agree a lenient policy (imported JSON), then compare and export under it.
  await page.getByRole('button', { name: 'Create release policy' }).click()
  await page.getByLabel('Policy name').fill('Pilot release policy')
  await page.getByText('Import or export policy JSON').click()
  await page.getByLabel('Policy JSON').fill(JSON.stringify({ metrics: { ndcg: { max_drop: 1 }, recall: { max_drop: 1 } }, min_queries: 1 }))
  await page.getByRole('button', { name: 'Load into form' }).click()
  assert.equal(await page.getByLabel('Minimum measured questions').inputValue(), '1')
  await page.getByRole('button', { name: 'Add critical assertion' }).click()
  await page.getByLabel('Critical 1 queryId', { exact: true }).fill('password')
  await page.getByLabel('Critical 1 docId', { exact: true }).fill('account-guide')
  await page.getByLabel('Critical 1 maxRank', { exact: true }).fill('1')
  await page.getByRole('region', { name: 'Release policy settings' }).screenshot({ path: join(workspace, 'release-policy.png') })
  await page.getByRole('button', { name: 'Save policy version' }).click()
  await page.getByText(/Comparisons, case verifications and exports use/).waitFor()
  // A/A calibration from the policy settings: measure on recordings, nothing is saved.
  await page.getByText('Calibrate this policy against unchanged runs (A/A)').click()
  const recordings = page.getByRole('group', { name: /Recordings of the unchanged setup/ })
  for (const box of await recordings.getByRole('checkbox').all()) await box.check()
  await page.getByRole('button', { name: /^Measure false alarms \(2 runs\)/ }).click()
  await page.getByRole('region', { name: 'Calibration result' }).getByText(/A\/A comparisons/).first().waitFor()
  // Baseline and candidate are different configurations: the panel must say it is not A/A.
  await page.getByRole('region', { name: 'Calibration result' }).getByText(/not one unchanged setup/).waitFor()
  await page.getByRole('region', { name: 'Release policy settings' }).screenshot({ path: join(workspace, 'calibration.png') })
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText(/under Pilot release policy v1/).waitFor()
  await page.getByText('Release comparison: pass').waitFor()
  await page.getByRole('region', { name: 'Critical question results' }).getByText(/pass: password/).waitFor()
  await page.getByRole('button', { name: 'Track finding' }).click()
  await page.getByText(/· open · Unassigned/).first().waitFor()

  await page.getByLabel('Release check name').fill('Support search')
  await page.getByRole('checkbox', { name: /Include the measured corpus/ }).check()
  await page.getByRole('button', { name: 'Export release check' }).click()
  await page.getByText('Before the first CI run').waitFor()
  await page.getByRole('region', { name: 'Export release check' }).screenshot({ path: join(workspace, 'release-export.png') })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download files (.zip)' }).click()
  assert.match((await download).suggestedFilename(), /^release-check-[a-f0-9]{8}-v1\.zip$/)
  const command = await page.getByText(/vectraudit workbench export [a-f0-9]{32}/).textContent()
  const id = command.match(/[a-f0-9]{32}/)[0]

  // The exported files run from a clean checkout and reach the browser's verdict.
  run(['workbench', 'export', id, '--workspace', workspace, '--out', repo])
  const release = 'retrieval/release/release.json'
  run(['release', 'validate', release])
  run(['release', 'run', release, '--out', 'runs/baseline.run.json'])
  run(['release', 'run', release, '--approved', release, '--out', 'runs/candidate.run.json'])
  const checked = run(['release', 'check', release, 'runs/baseline.run.json', 'runs/candidate.run.json'])
  assert.match(checked, /PASS/)
  assert.match(checked, /Pilot release policy v1/)

  // A stricter version changes the browser comparison and case verification together.
  await page.getByRole('button', { name: 'Edit as new version' }).click()
  await page.getByLabel('Critical 1 docId', { exact: true }).fill('missing-required-document')
  await page.getByRole('button', { name: 'Save policy version' }).click()
  await page.getByText(/use Pilot release policy v2/).waitFor()
  await page.getByText(/The selected policy changed since this comparison/).waitFor()
  await page.getByRole('button', { name: 'Compare runs' }).click()
  await page.getByText('Release comparison: fail').waitFor()
  await page.getByText(/Critical query password: document missing-required-document/).waitFor()
  await page.getByText(/Export is available once the candidate passes Pilot release policy v2/).waitFor()
  // Each failing rule says who may accept it; this policy opens no exceptions.
  const rules = page.getByRole('region', { name: 'Failing rules' })
  await rules.getByText('critical:password:missing-required-document').waitFor()
  await rules.getByText('Cannot be accepted: this policy has no exceptions block.').waitFor()
  await rules.getByRole('button', { name: 'Check accepted regressions in the hosted project' }).click()
  await rules.getByRole('alert').getByText(/Not connected to a hosted project/).waitFor()
  await rules.screenshot({ path: join(workspace, 'failing-rules.png') })

  await page.getByRole('button', { name: 'Evaluate another configuration' }).click()
  await evaluate('Recheck', 2, 'phrase')
  await page.getByRole('button', { name: 'Compare a fix', exact: true }).click()
  await choose(page.getByLabel('Verification run').first(), 'Recheck ·')
  await page.getByRole('button', { name: /^Verify under Pilot release policy v2/ }).first().click()
  await page.getByText(/release fail under Pilot release policy v2/).first().waitFor()

  await page.screenshot({ path: join(workspace, 'release-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: join(workspace, 'release-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  assert.deepEqual(errors, [])
  console.log('Release browser checks passed. Screenshots: ' + workspace)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
