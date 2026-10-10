import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import { DROP_TESTS, METRICS, METRIC_HELP, STARTER_POLICY, formFromPolicy, policyFromForm, policyLabel, selectedRef } from '../lib/release'
import type { PolicyChoice, PolicyForm, PolicyListing, PolicyRecord, PolicyRef, RawPolicy } from '../lib/release'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const input = 'mt-1 w-full rounded-lg border border-line bg-page px-3 py-2 text-sm'
const small = 'mt-1 w-24 rounded-lg border border-line bg-page px-2 py-1 text-sm'


export default function ReleasePolicy({ token, value, onChange }: { token: string; value: PolicyChoice; onChange: (choice: string | null, ref: PolicyRef | null) => void }) {
  const [listing, setListing] = useState<PolicyListing | null>(null)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState(false)

  async function reload(select?: string) {
    try {
      const next = await guidedRequest<PolicyListing>('policies', token)
      setListing(next)
      if (select) onChange(select, selectedRef(next, select))
      // Prefer the customer's own policy: the startup file, then the newest saved policy.
      else if (value === undefined) {
        const fallback = next.startup ? 'startup' : next.policies.find(p => p.latest)?.id || null
        onChange(fallback, selectedRef(next, fallback))
      }
    } catch (e) { setError(String(e)) }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void reload() }, [token])

  const ref = selectedRef(listing, value)
  const record = value === 'startup' ? listing?.startup : listing?.policies.find(p => p.id === value)
  return <section aria-label="Release policy settings" className="space-y-3 rounded-xl border border-line p-4">
    <div className="flex flex-wrap items-end justify-between gap-3">
      <label className="min-w-0 flex-1 text-sm">Release policy<select aria-label="Release policy" className={input} value={value ?? ''} onChange={e => { const id = e.target.value || null; onChange(id, selectedRef(listing, id)) }}>
        <option value="">Exploratory defaults (not a release decision)</option>
        {listing?.startup && <option value="startup">{listing.startup.name} · startup policy</option>}
        {listing?.policies.map(p => <option key={p.id} value={p.id ?? ''}>{p.name} v{p.version}{p.latest ? '' : ' (superseded)'}</option>)}
      </select></label>
      <button className={button} onClick={() => setEditing(!editing)}>{editing ? 'Close policy editor' : record ? 'Edit as new version' : 'Create release policy'}</button>
    </div>
    {ref
      ? <p className="rounded-lg bg-teal-500/10 p-3 text-sm text-teal-700 dark:text-teal-300">Comparisons, case verifications and exports use <b>{policyLabel(ref)}</b>. The same rules decide <code>vectraudit check --policy</code> and the exported CI check.</p>
      : <p className="rounded-lg bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-300">Exploratory defaults judge whichever metrics both runs have, from a single question upward. Use them to explore; save the thresholds your team agrees on as a release policy before relying on a verdict or exporting a release check.</p>}
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {editing && <PolicyEditor key={record?.id ?? 'new'} token={token} base={record} onSaved={saved => { setEditing(false); void reload(saved.id ?? undefined) }} />}
  </section>
}

function PolicyEditor({ token, base, onSaved }: { token: string; base: PolicyRecord | null | undefined; onSaved: (record: PolicyRecord) => void }) {
  const [name, setName] = useState(base?.source === 'workspace' ? base.name : 'Release policy')
  const [form, setForm] = useState<PolicyForm>(() => formFromPolicy(base?.policy || STARTER_POLICY))
  const [note, setNote] = useState('')
  const [errors, setErrors] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [json, setJson] = useState('')
  const update = (change: Partial<PolicyForm>) => setForm(previous => ({ ...previous, ...change }))
  const built = policyFromForm(form)

  async function save() {
    if (!built.policy) { setErrors(built.errors); return }
    setSaving(true); setErrors([])
    try { onSaved(await guidedRequest<PolicyRecord>('policies', token, { name, policy: built.policy, note })) } catch (e) { setErrors([String(e)]) } finally { setSaving(false) }
  }
  function importJson() {
    try { const raw: unknown = JSON.parse(json); if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error(); setForm(formFromPolicy(raw as RawPolicy)); setErrors([]) } catch { setErrors(['Paste a policy JSON object, for example the contents of retrieval/policy.json.']) }
  }
  function download() {
    if (!built.policy) { setErrors(built.errors); return }
    const url = URL.createObjectURL(new Blob([JSON.stringify(built.policy, null, 2) + '\n'], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'policy.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <fieldset disabled={saving} className="space-y-4 rounded-lg bg-surface-2 p-4">
    <p className="text-sm text-muted">Saving creates a new version. Earlier comparisons and verifications keep the version that judged them.</p>
    <label className="block text-sm">Policy name<input aria-label="Policy name" className={input} value={name} maxLength={128} onChange={e => setName(e.target.value)} /><span className="text-xs text-muted">Saving under an existing name adds a version to that policy.</span></label>

    <div><h4 className="text-sm font-semibold">Quality across all questions</h4><p className="text-xs text-muted">Allowed drop is absolute: 0.02 lets nDCG fall from 0.80 to 0.78. Minimum fails the candidate below a fixed score. Every metric you turn on must be measured in both runs, or the comparison is not comparable.</p>
      <div className="mt-2 space-y-2">{METRICS.map(m => <div key={m} className="flex flex-wrap items-end gap-3 text-sm">
        <label className="flex w-28 items-center gap-2 pb-1.5"><input type="checkbox" aria-label={`Require ${m}`} checked={form.metrics[m].enabled} onChange={e => update({ metrics: { ...form.metrics, [m]: { ...form.metrics[m], enabled: e.target.checked } } })} />{m}</label>
        <label className="flex flex-col text-xs">Allowed drop<input aria-label={`${m} allowed drop`} className={small} inputMode="decimal" disabled={!form.metrics[m].enabled} value={form.metrics[m].maxDrop} onChange={e => update({ metrics: { ...form.metrics, [m]: { ...form.metrics[m], maxDrop: e.target.value } } })} /></label>
        <label className="flex flex-col text-xs">Minimum<input aria-label={`${m} minimum`} className={small} inputMode="decimal" disabled={!form.metrics[m].enabled} value={form.metrics[m].min} onChange={e => update({ metrics: { ...form.metrics, [m]: { ...form.metrics[m], min: e.target.value } } })} /></label>
        <span className="pb-1.5 text-xs text-muted">{METRIC_HELP[m]}</span>
      </div>)}</div></div>

    <div className="space-y-2"><h4 className="text-sm font-semibold">Critical questions</h4>
      <p className="text-xs text-muted">Require a particular document within the top results for a question. These rules judge the candidate’s recorded ranking even when average quality improves. The question must be in the selected benchmark split and maximum rank must not exceed retrieval k. Repeated calls currently use the first recorded ranking.</p>
      {form.criticalQueries.map((rule, index) => <div key={index} className="flex flex-wrap items-end gap-3">
        {(['queryId', 'docId', 'maxRank'] as const).map(field => <label key={field} className="min-w-0 text-xs">{field === 'queryId' ? 'Query ID' : field === 'docId' ? 'Document ID' : 'Maximum rank'}<input aria-label={`Critical ${index + 1} ${field}`} className={field === 'maxRank' ? small : input} value={rule[field]} onChange={e => update({ criticalQueries: form.criticalQueries.map((item, i) => i === index ? { ...item, [field]: e.target.value } : item) })} /></label>)}
        <button className={button} onClick={() => update({ criticalQueries: form.criticalQueries.filter((_, i) => i !== index) })}>Remove assertion {index + 1}</button>
      </div>)}
      <button className={button} onClick={() => update({ criticalQueries: [...form.criticalQueries, { queryId: '', docId: '', maxRank: '3' }] })}>Add critical assertion</button>
    </div>

    <div className="grid gap-4 sm:grid-cols-3">
      <label className="text-sm">Minimum measured questions<input aria-label="Minimum measured questions" className={input} inputMode="numeric" value={form.minQueries} onChange={e => update({ minQueries: e.target.value })} /><span className="text-xs text-muted">Each required metric needs this many labelled questions measured in both runs; fewer is not comparable rather than a pass.</span></label>
      <label className="text-sm">Regressed questions allowed<input aria-label="Regressed questions allowed" className={input} inputMode="numeric" value={form.maxRegressed} onChange={e => update({ maxRegressed: e.target.value })} /><span className="text-xs text-muted">Fail when more individual questions get worse than this, even if the average holds. Empty means no limit.</span></label>
      <label className="text-sm">Metric for regressed questions<select aria-label="Metric for regressed questions" className={input} value={form.primaryMetric} onChange={e => update({ primaryMetric: e.target.value })}><option value="">First measured metric</option>{METRICS.map(m => <option key={m} value={m}>{m}</option>)}</select></label>
    </div>

    <div><h4 className="text-sm font-semibold">Cohorts (question tags)</h4><p className="text-xs text-muted">Protect groups such as a language or customer tier. Tags with fewer questions than the minimum are reported, not judged.</p>
      <div className="mt-2 flex flex-wrap items-end gap-4"><label className="flex flex-col text-xs">Minimum questions per cohort<input aria-label="Minimum questions per cohort" className={small} inputMode="numeric" value={form.cohortMinQueries} onChange={e => update({ cohortMinQueries: e.target.value })} /></label>
        {METRICS.map(m => <div key={m} className="flex flex-col text-xs"><label className="flex items-center gap-2"><input type="checkbox" aria-label={`Cohort ${m}`} checked={form.cohortMetrics[m].enabled} onChange={e => update({ cohortMetrics: { ...form.cohortMetrics, [m]: { ...form.cohortMetrics[m], enabled: e.target.checked } } })} />{m} allowed drop</label><input aria-label={`Cohort ${m} allowed drop`} className={small} inputMode="decimal" disabled={!form.cohortMetrics[m].enabled} value={form.cohortMetrics[m].maxDrop} onChange={e => update({ cohortMetrics: { ...form.cohortMetrics, [m]: { ...form.cohortMetrics[m], maxDrop: e.target.value } } })} /></div>)}</div>
      <label className="mt-2 flex items-start gap-2 text-sm"><input type="checkbox" checked={form.cohortControl} onChange={e => update({ cohortControl: e.target.checked })} /><span>Correct for many cohorts (Bonferroni)<span className="block text-xs text-muted">A cohort fails only when its drop clears an interval widened for the number of cohort rules judged, so many small cohorts do not fail by chance. Real drops in small cohorts are harder to catch.</span></span></label></div>

    <div><h4 className="text-sm font-semibold">Latency</h4><p className="text-xs text-muted">Percentage limits compare the candidate with the baseline; the ceiling is absolute. Increases smaller than the noise allowance are ignored, because shared runners vary by a few milliseconds.</p>
      <div className="mt-2 flex flex-wrap gap-4">
        <label className="flex flex-col text-xs">p95 increase allowed (%)<input aria-label="p95 increase allowed" className={small} inputMode="decimal" value={form.p95Pct} onChange={e => update({ p95Pct: e.target.value })} /></label>
        <label className="flex flex-col text-xs">p50 increase allowed (%)<input aria-label="p50 increase allowed" className={small} inputMode="decimal" value={form.p50Pct} onChange={e => update({ p50Pct: e.target.value })} /></label>
        <label className="flex flex-col text-xs">p95 ceiling (ms)<input aria-label="p95 ceiling" className={small} inputMode="decimal" value={form.p95Ms} onChange={e => update({ p95Ms: e.target.value })} /></label>
        <label className="flex flex-col text-xs">Ignore increases below (ms)<input aria-label="Ignore latency increases below" className={small} inputMode="decimal" value={form.minIncreaseMs} onChange={e => update({ minIncreaseMs: e.target.value })} /></label>
        <label className="flex items-center gap-2 self-end pb-1.5 text-xs"><input type="checkbox" checked={form.sameEnvironment} onChange={e => update({ sameEnvironment: e.target.checked })} />Compare latency only between runs from the same machine type</label>
      </div></div>

    <fieldset className="text-sm"><legend className="font-medium">Judge allowed drops by</legend>
      {DROP_TESTS.map(test => <label key={test.value} className="mt-1 flex items-start gap-2"><input type="radio" name="drop-test" checked={form.dropTest === test.value} onChange={() => update({ dropTest: test.value })} /><span>{test.label}<span className="block text-xs text-muted">{test.help}</span></span></label>)}
    </fieldset>
    <label className="block text-sm">Review note<textarea aria-label="Policy review note" className={input} rows={2} maxLength={2000} value={note} placeholder="Who agreed to these thresholds, and why" onChange={e => setNote(e.target.value)} /></label>
    <details className="text-sm"><summary className="cursor-pointer">Import or export policy JSON</summary><textarea aria-label="Policy JSON" className={`${input} font-mono text-xs`} rows={5} value={json} placeholder='{"metrics": {"ndcg": {"max_drop": 0.02}}, "min_queries": 30}' onChange={e => setJson(e.target.value)} /><div className="mt-2 flex gap-2"><button className={button} onClick={importJson}>Load into form</button><button className={button} onClick={download}>Download policy.json</button></div></details>
    {errors.length > 0 && <ul role="alert" className="list-disc pl-5 text-sm text-red-600">{errors.map(e => <li key={e}>{e}</li>)}</ul>}
    <button className={`${button} bg-teal-600 text-white hover:bg-teal-700`} disabled={!name.trim()} onClick={() => void save()}>{saving ? 'Saving…' : 'Save policy version'}</button>
  </fieldset>
}
