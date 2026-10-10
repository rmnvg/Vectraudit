import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import { alarmSummary, changeLine, choosableRuns } from '../lib/calibration'
import type { CalibrationResult, RecordedRun } from '../lib/calibration'
import type { PolicyRecord } from '../lib/release'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'

/** Measure how often the selected policy fails an unchanged system, and review a draft. */
export default function PolicyCalibration({ token, policyId, policyName, onSaved }: { token: string; policyId: string; policyName: string; onSaved: (record: PolicyRecord) => void }) {
  const [runs, setRuns] = useState<RecordedRun[]>([])
  const [chosen, setChosen] = useState<string[]>([])
  const [result, setResult] = useState<CalibrationResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    guidedRequest<RecordedRun[]>('runs', token).then(r => { if (!cancelled) setRuns(r) }).catch(e => { if (!cancelled) setError(String(e)) })
    return () => { cancelled = true }
  }, [token])

  const options = choosableRuns(runs, chosen)
  const toggle = (id: string, on: boolean) => { setResult(null); setChosen(prev => on ? [...prev, id] : prev.filter(r => r !== id)) }

  async function measure() {
    setBusy(true); setError(''); setResult(null)
    try { setResult(await guidedRequest<CalibrationResult>('calibrate', token, { runs: chosen, policy: policyId })) } catch (e) { setError(String(e instanceof Error ? e.message : e)) } finally { setBusy(false) }
  }

  async function saveDraft() {
    if (!result?.report.draft_policy) return
    setBusy(true); setError('')
    const { report } = result
    const note = `A/A calibration on ${report.runs.length} recordings: ${alarmSummary(report.current)} before, ${alarmSummary(report.draft)} with the draft.`
    try { onSaved(await guidedRequest<PolicyRecord>('policies', token, { name: policyName, policy: report.draft_policy, note })) } catch (e) { setError(String(e instanceof Error ? e.message : e)) } finally { setBusy(false) }
  }

  const report = result?.report
  return <details className="rounded-lg border border-line p-3 text-sm">
    <summary className="cursor-pointer font-semibold">Calibrate this policy against unchanged runs (A/A)</summary>
    <div className="mt-3 space-y-3">
      <p className="text-xs text-muted">Record the same, unchanged setup several times on the runner and endpoint your CI uses (5 or more is better), then choose those recordings. Every failure between them is a false alarm. The draft only raises noise-driven thresholds; it never loosens critical, permission, no-answer or absolute rules, and nothing is saved until you choose to.</p>
      <fieldset disabled={busy} className="max-h-48 space-y-1 overflow-auto">
        <legend className="text-xs text-muted">Recordings of the unchanged setup ({chosen.length} chosen)</legend>
        {options.length === 0 && <p className="text-xs text-muted">No complete runs yet.</p>}
        {options.map(r => <label key={r.run_id} className="flex items-center gap-2"><input type="checkbox" checked={chosen.includes(r.run_id)} onChange={e => toggle(r.run_id, e.target.checked)} /><span>{r.label || 'Run'} · {r.created_at.slice(0, 16).replace('T', ' ')} · <span className="font-mono text-xs">{r.run_id.slice(0, 8)}</span></span></label>)}
      </fieldset>
      <button className={button} disabled={busy || chosen.length < 2 || chosen.length > 10} onClick={() => void measure()}>{busy && !report ? 'Measuring…' : `Measure false alarms (${chosen.length} runs)`}</button>
      {report && <section aria-label="Calibration result" className="space-y-2">
        {report.config_differences.length > 0 && <p className="rounded-lg bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-300">These recordings are not one unchanged setup: they differ in {report.config_differences.slice(0, 5).join(', ')}{report.config_differences.length > 5 ? ' and more' : ''}. Choose recordings of a single configuration for a meaningful A/A measurement.</p>}
        <p><b>Current policy:</b> {alarmSummary(report.current)}.</p>
        {Object.keys(report.current.by_rule).length > 0 && <ul className="list-disc pl-5 text-xs">{Object.entries(report.current.by_rule).map(([rule, n]) => <li key={rule}><span className="font-mono">{rule}</span> failed {n}×</li>)}</ul>}
        {report.not_noise.length > 0 && <p className="rounded-lg bg-red-500/10 p-2 text-xs text-red-700 dark:text-red-300">Not noise: {report.not_noise.join(', ')} failed on an unchanged system. Investigate these; the draft does not loosen them.</p>}
        {report.changes.length > 0 ? <>
          <p><b>Draft:</b> {alarmSummary(report.draft)}. Proposed changes (margin ×{report.margin}):</p>
          <ul className="list-disc pl-5 font-mono text-xs">{report.changes.map(c => <li key={c.path}>{changeLine(c)}</li>)}</ul>
          <button className={button} disabled={busy} onClick={() => void saveDraft()}>Save draft as a new version of {policyName}</button>
        </> : <p>No threshold needs to change: the policy is already wider than the measured noise.</p>}
        <ul className="list-disc pl-5 text-xs text-muted">{report.notes.map(n => <li key={n}>{n}</li>)}</ul>
      </section>}
      {error && <p role="alert" className="text-red-600">{error}</p>}
    </div>
  </details>
}
