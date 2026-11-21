// Real adapter -> measured runs -> passage review -> saved labels -> fresh comparison.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'

const root = resolve(import.meta.dirname, '..')
const remoteLookup = process.env.VECTRAUDIT_LOOKUP_SMOKE === '1'
const workspace = await mkdtemp(join(tmpdir(), 'vectraudit-passage-review-'))
const base = 'http://127.0.0.1:8820'
await writeFile(join(workspace, 'review_fixture.py'), `from vectraudit.pipelines.models import Hit, PipelineResult
def retrieve(request):
    ids = ["other"] if request.context.get("tenant") == "other" else (
        ["known", "negative", "common", "base-only", "missing"] if request.params["side"] == "base"
        else ["known", "negative", "common", "candidate-only"])
    return PipelineResult(hits=[Hit(id=doc) for doc in ids])
def lookup(request):
    documents = []
    for doc in request.doc_ids:
        if doc == "missing":
            documents.append({"doc_id": doc, "status": "missing"})
        elif (doc == "other") != (request.context.get("tenant") == "other"):
            documents.append({"doc_id": doc, "status": "forbidden"})
        else:
            documents.append({"doc_id": doc, "status": "found", "text": "Remote passage for " + doc,
                              "version": "content-v1"})
    return {"corpus_version": request.corpus_version, "documents": documents}
`)
await writeFile(join(workspace, 'manifest.json'), JSON.stringify({ dataset: 'unused.jsonl', ...(remoteLookup ? { corpus_version: 'snapshot-v1', document_lookup: { target: 'review_fixture:lookup' } } : {}), candidates: ['base', 'candidate'].map(side => ({ name: side, adapter: { kind: 'python', target: 'review_fixture:retrieve' }, params: { side } })) }))
const server = spawn(join(root, '.venv/bin/vectraudit'), ['workbench', 'serve', workspace, '--port', '8820', '--manifest', join(workspace, 'manifest.json')], { cwd: root, stdio: ['ignore', 'ignore', 'pipe'] })
let browser, token, stderr = ''
server.stderr.on('data', b => { stderr += b })
async function api(path, body) {
  const response = await fetch(`${base}/api/guided/${path}`, { method: body ? 'POST' : 'GET', headers: { 'X-Workbench-Token': token || '', 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) })
  assert.ok(response.ok, await response.clone().text())
  return response.json()
}
async function evaluate(draft, side) {
  const job = await api('jobs', { draft_id: draft.id, revision: draft.revision, pipeline: `configured-${side}`, label: side ? 'Candidate' : 'Baseline', k: 10 })
  for (let i = 0; i < 100; i++) {
    const state = await api(`jobs/${job.id}`)
    if (state.status !== 'running') { assert.equal(state.status, 'done'); return state.run }
    await new Promise(r => setTimeout(r, 100))
  }
  throw new Error('Evaluation did not finish')
}
try {
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(stderr)
    try { if ((await fetch(`${base}/api/workbench`)).ok) break } catch {}
    await new Promise(r => setTimeout(r, 100))
  }
  token = (await api('session')).token
  const rows = [
    { query_id: 'invoice', text: 'invoice', relevant: { known: 2 }, context: { tenant: 'acme' } },
    { query_id: 'other', text: 'invoice', context: { tenant: 'other' } },
    { query_id: 'heldout', text: 'final question', split: 'heldout' },
  ]
  const corpus = remoteLookup ? '' : ['known', 'negative', 'common', 'base-only', 'candidate-only', 'other', 'outside-pool'].map(doc_id => JSON.stringify({ doc_id, text: `Local passage for ${doc_id}` })).join('\n')
  let draft = await api('drafts', { name: 'Review benchmark', dataset: rows.map(r => JSON.stringify(r)).join('\n'), corpus, irrelevant: { invoice: ['negative'] } })
  const baseline = await evaluate(draft, 0)
  const candidate = await evaluate(draft, 1)
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  await page.goto(base)
  const steps = page.getByRole('navigation', { name: 'Evaluation steps' })
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  await page.getByLabel('Resume saved benchmark').selectOption({ label: 'Review benchmark' })
  await page.getByLabel('Review baseline').selectOption(baseline.run_id)
  await page.getByLabel('Review candidate').selectOption(candidate.run_id)
  await page.getByRole('button', { name: 'Build passage review queue', exact: true }).click()
  await page.getByText(/4 passages awaiting judgment/).waitFor()
  // The pair's stop rule is shown, and the queue is in decision order.
  const decision = page.getByRole('status', { name: 'Release check decision' })
  assert.match(await decision.textContent(), /could still change the verdict|can change this verdict|Cannot say when judging can stop|settle that before judging/)
  assert.match(await decision.textContent(), /about a third fewer, not half/)
  await page.getByText(/ordered by how much they could move the release check/).waitFor()
  const grader = page.getByRole('region', { name: 'Passage grading' })
  assert.match(await grader.locator('article').first().textContent(), /base-only/)
  assert.equal(await grader.getByRole('button', { name: 'Relevant: known', exact: true }).count(), 0)
  assert.equal(await grader.getByRole('button', { name: 'Irrelevant: negative', exact: true }).count(), 0)
  assert.equal(await grader.getByText(/Show retrieval evidence/).count(), 0)
  assert.ok(await grader.getByRole('button', { name: 'Relevant: missing', exact: true }).isDisabled())
  if (remoteLookup) {
    assert.ok(await grader.getByRole('button', { name: 'Relevant: base-only', exact: true }).isDisabled())
    const fetchPassages = grader.getByRole('button', { name: 'Fetch visible passages', exact: true })
    assert.ok(await fetchPassages.isDisabled())
    await page.getByRole('checkbox', { name: /Send document IDs/ }).check()
    await fetchPassages.click()
    await grader.getByText('Remote passage for base-only', { exact: true }).waitFor()
    await grader.getByText('Document not found in this corpus version.', { exact: true }).waitFor()
    assert.ok(await grader.getByRole('button', { name: 'Relevant: missing', exact: true }).isDisabled())
    // Changing the source invalidates both content and transfer consent.
    await page.getByLabel('Passage source', { exact: true }).selectOption('baseline')
    assert.ok(await grader.getByRole('button', { name: 'Relevant: base-only', exact: true }).isDisabled())
    await page.getByRole('checkbox', { name: /Send document IDs/ }).check()
    await fetchPassages.click()
    await grader.getByText('Remote passage for base-only', { exact: true }).waitFor()
  }

  await grader.getByRole('button', { name: 'Irrelevant: base-only', exact: true }).click()
  await page.getByText(/3 passages awaiting judgment/).waitFor()
  await grader.getByText('Show retrieval evidence for base-only', { exact: true }).click()
  await grader.getByText('Baseline rank: 4', { exact: true }).waitFor()
  await grader.getByRole('button', { name: 'Unjudged: base-only', exact: true }).click()
  await page.getByText(/4 passages awaiting judgment/).waitFor()
  await grader.getByRole('button', { name: 'Irrelevant: base-only', exact: true }).click()
  await grader.getByRole('button', { name: 'Relevant: candidate-only', exact: true }).click()
  if (!remoteLookup) {
    await page.getByRole('button', { name: 'Browse all passages', exact: true }).click()
    await grader.getByRole('button', { name: 'Relevant: outside-pool', exact: true }).waitFor()
    await page.getByRole('button', { name: 'Review pooled passages', exact: true }).click()
  }
  await page.getByText(/2 passages awaiting judgment/).waitFor()
  await page.screenshot({ path: join(workspace, 'review-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: join(workspace, 'review-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  await page.setViewportSize({ width: 1440, height: 1100 })

  // Review another tenant's question before saving; only its results may appear.
  await page.getByRole('button', { name: /^other Explore/ }).click()
  await page.getByRole('button', { name: 'Build passage review queue', exact: true }).click()
  await page.getByText(/1 passages awaiting judgment/).waitFor()
  assert.equal(await grader.locator('article').count(), 1)
  if (remoteLookup) {
    await page.getByRole('checkbox', { name: /Send document IDs/ }).check()
    await grader.getByRole('button', { name: 'Fetch visible passages', exact: true }).click()
    await grader.getByText('Remote passage for other', { exact: true }).waitFor()
    assert.equal(await grader.getByText('Remote passage for base-only', { exact: true }).count(), 0)
  }
  await grader.getByRole('button', { name: 'Relevant: other', exact: true }).click()
  await page.getByLabel('Question', { exact: true }).fill('changed question')
  assert.equal(await page.getByText('Review retrieved passages', { exact: true }).count(), 0)
  assert.ok(await page.getByRole('button', { name: 'Build passage review queue', exact: true }).isDisabled())
  await page.getByLabel('Question', { exact: true }).fill('invoice')
  await page.getByRole('button', { name: /^heldout Held-out/ }).click()
  assert.equal(await page.getByRole('button', { name: 'Build passage review queue', exact: true }).count(), 0)
  await page.getByRole('button', { name: 'Save benchmark', exact: true }).click()
  await page.getByText(/Saved locally/).waitFor()
  await page.reload()
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  await page.getByLabel('Resume saved benchmark').selectOption({ label: 'Review benchmark' })
  if (remoteLookup) {
    const saved = await api(`drafts/${draft.id}`)
    assert.equal(saved.corpus, '')
    assert.ok(saved.irrelevant.invoice.includes('base-only'))
    assert.equal(saved.dataset.includes('Remote passage'), false)
    assert.equal(JSON.parse(saved.dataset.split('\n').find(line => line.includes('candidate-only'))).relevant['candidate-only'], 1)
  } else {
    assert.equal(await page.getByRole('button', { name: 'Irrelevant: base-only', exact: true }).getAttribute('aria-pressed'), 'true')
    assert.equal(await page.getByRole('button', { name: 'Relevant: candidate-only', exact: true }).getAttribute('aria-pressed'), 'true')
  }
  await page.getByLabel('Review baseline').selectOption(baseline.run_id)
  await page.getByLabel('Review candidate').selectOption(candidate.run_id)
  await page.getByRole('button', { name: 'Build passage review queue', exact: true }).click()
  await page.getByText(/Choose exploratory runs from this exact saved benchmark/).waitFor()

  // The reviewed benchmark feeds a fresh, comparable baseline/candidate decision.
  draft = await api(`drafts/${draft.id}`)
  const freshBase = await evaluate(draft, 0)
  const freshCandidate = await evaluate(draft, 1)
  await page.reload()
  await steps.getByRole('button', { name: /Build benchmark/ }).click()
  await page.getByLabel('Resume saved benchmark').selectOption({ label: 'Review benchmark' })
  await steps.getByRole('button', { name: /Compare a fix/ }).click()
  await page.getByLabel('Baseline run', { exact: true }).selectOption(freshBase.run_id)
  await page.getByLabel('Candidate run', { exact: true }).selectOption(freshCandidate.run_id)
  const comparison = page.waitForResponse(r => r.url().endsWith('/api/guided/compare'))
  await page.getByRole('button', { name: 'Compare runs', exact: true }).click()
  const result = await (await comparison).json()
  assert.ok(['pass', 'fail'].includes(result.check.verdict))
  await page.getByRole('button', { name: 'Review passages for this question', exact: true }).click()
  assert.equal(await page.getByLabel('Query ID', { exact: true }).inputValue(), 'invoice')
  assert.equal(await page.getByLabel('Review baseline').inputValue(), freshBase.run_id)
  assert.equal(await page.getByLabel('Review candidate').inputValue(), freshCandidate.run_id)
  await page.getByRole('button', { name: 'Build passage review queue', exact: true }).click()
  await page.getByText(/2 passages awaiting judgment/).waitFor()
  assert.deepEqual(errors, [])
  console.log('Passage review browser checks passed. Screenshots: ' + workspace)
} finally {
  await browser?.close()
  server.kill('SIGTERM')
}
