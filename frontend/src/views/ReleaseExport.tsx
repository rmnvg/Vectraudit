import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import type { Comparison } from '../lib/guided'
import { downloadGuided, policyLabel } from '../lib/release'
import type { ReleaseRecord, ReleaseSummary } from '../lib/release'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'

/** Turn a passing comparison into a versioned release check the team reviews and commits. */
export default function ReleaseExport({ token, comparison, defaultName }: { token: string; comparison: Comparison; defaultName: string }) {
  const [name, setName] = useState(defaultName)
  const [repeats, setRepeats] = useState(3)
  const [includeCorpus, setIncludeCorpus] = useState(false)
  const [created, setCreated] = useState<ReleaseRecord | null>(null)
  const [history, setHistory] = useState<ReleaseSummary[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const policy = comparison.policy
  const exploratory = !policy || policy.source === 'exploratory'
  const passing = comparison.check.verdict === 'pass'
  const config = comparison.candidate.backend?.config || {}
  const localCorpus = config.adapter === 'hybrid'

  useEffect(() => {
    let cancelled = false
    guidedRequest<ReleaseSummary[]>('releases', token).then(r => { if (!cancelled) setHistory(r) }).catch(() => undefined)
    return () => { cancelled = true }
  }, [token, created])

  async function create() {
    if (!policy?.id) return
    setBusy(true); setError('')
    try {
      setCreated(await guidedRequest<ReleaseRecord>('releases', token, {
        baseline: comparison.baseline.run_id, candidate: comparison.candidate.run_id,
        policy: policy.id, name, repeats, include_corpus: includeCorpus,
      }))
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }

  return <section aria-label="Export release check" className="space-y-3 rounded-xl border border-line p-4">
    <h3 className="font-semibold">Export release check</h3>
    <p className="text-sm text-muted">Package this benchmark split, the candidate pipeline and the policy that passed it, so every later change runs the same check locally and in GitHub Actions.</p>
    {exploratory
      ? <p className="text-sm text-amber-600">This comparison used exploratory defaults. Choose or create a release policy above and compare again; exploratory defaults cannot become a release contract.</p>
      : !passing
        ? <p className="text-sm text-amber-600">Export is available once the candidate passes {policyLabel(policy)}. This comparison is {comparison.check.verdict.replace('_', ' ')}.</p>
        : <fieldset disabled={busy} className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-[1fr_160px]">
            <label className="text-sm">Release check name<input aria-label="Release check name" className={input} value={name} maxLength={128} onChange={e => setName(e.target.value)} /><span className="text-xs text-muted">Exporting again under this name creates the next configuration version.</span></label>
            <label className="text-sm">Repeats per question<input aria-label="Repeats per question" className={input} type="number" min={1} max={100} value={repeats} onChange={e => setRepeats(Number(e.target.value))} /><span className="text-xs text-muted">CI latency uses the median.</span></label>
          </div>
          {localCorpus && <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={includeCorpus} onChange={e => setIncludeCorpus(e.target.checked)} /><span>Include the measured corpus in the export<span className="block text-xs text-muted">Off by default: corpora can be large or confidential. Without it, the export lists the corpus as a prerequisite.</span></span></label>}
          <button className={`${button} bg-teal-600 text-white hover:bg-teal-700`} disabled={!name.trim() || repeats < 1} onClick={() => void create()}>{busy ? 'Preparing…' : 'Export release check'}</button>
        </fieldset>}
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {created && <ExportResult token={token} record={created} />}
    {history.length > 0 && <details className="text-sm"><summary className="cursor-pointer">Earlier release configurations ({history.length})</summary><ul className="mt-2 space-y-1 text-xs">{history.map(r => <li key={r.id}>{r.name} v{r.version} · {r.pipeline.name} · {policyLabel(r.policy)} · {r.created_at.slice(0, 19)} · <code>{r.id}</code></li>)}</ul></details>}
  </section>
}

function ExportResult({ token, record }: { token: string; record: ReleaseRecord }) {
  const [error, setError] = useState('')
  const benchmark = record.release.benchmark
  return <div role="status" className="space-y-3 rounded-lg bg-surface-2 p-4 text-sm">
    <p><b>{record.name} v{record.version}</b> · {benchmark.n_queries} {benchmark.split === 'explore' ? 'exploratory' : 'held-out'} questions · k={record.release.k} · {policyLabel(record.policy)}</p>
    {record.issues.warnings.map(w => <p key={w} className="text-amber-600">{w}</p>)}
    <div><h4 className="font-semibold">Before the first CI run</h4><ul className="mt-1 list-disc space-y-1 pl-5">{record.issues.prerequisites.map(p => <li key={p}>{p}</li>)}<li>Add repository secret <code>VECTRAUDIT_DOWNLOAD_TOKEN</code> so the workflow can install Vectraudit.</li></ul></div>
    <div><h4 className="font-semibold">Files</h4><p className="text-xs text-muted">Review them, then commit. Nothing is pushed. The target branch's copy becomes the approved check for later pull requests.</p>
      {Object.entries(record.files).map(([path, content]) => <details key={path} className="mt-1"><summary className="cursor-pointer font-mono text-xs">{path}</summary><pre className="mt-1 max-h-64 overflow-auto rounded bg-page p-2 text-xs">{content.length > 20000 ? `${content.slice(0, 20000)}\n…` : content}</pre></details>)}</div>
    <div className="flex flex-wrap items-center gap-3"><button className={button} onClick={() => void downloadGuided(`releases/${record.id}/export.zip`, token, `release-check-${record.id.slice(0, 8)}-v${record.version}.zip`).catch(e => setError(String(e)))}>Download files (.zip)</button>
      <span className="text-xs text-muted">or write them into a checkout: <code>vectraudit workbench export {record.id} --workspace WORKBENCH_DIR --out REPO</code></span></div>
    {error && <p role="alert" className="text-red-600">{error}</p>}
  </div>
}
