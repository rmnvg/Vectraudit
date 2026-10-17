import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import type { JobState } from '../lib/guided'
import type { PolicyListing } from '../lib/release'
import { ShareRecord } from '../components/ShareRecord'
import { OUTCOMES, canCloseFixed, lineageLine, lossLabel, rememberName, savedName, trialCalls, trialOutcome } from '../lib/investigation'
import type { Investigation as Data, InvestigationRecord, Trial, Validation } from '../lib/investigation'

const button = 'rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface-2 disabled:opacity-50'
const primary = 'rounded-lg bg-teal-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-50'
const format = (value: number | null | undefined) => value == null ? '—' : value.toFixed(3)

/** Poll a workspace job until it stops; the result arrives under `run`. */
async function finished<T>(token: string, id: string, onProgress: (message: string) => void): Promise<T> {
  for (;;) {
    const state = await guidedRequest<JobState & { run?: unknown }>(`jobs/${id}`, token)
    if (state.status !== 'running') {
      if (state.status !== 'done') throw new Error(state.error || `The job ${state.status}.`)
      return state.run as T
    }
    onProgress(state.message)
    await new Promise(resolve => setTimeout(resolve, 400))
  }
}

/** One failed question: what was lost, possible causes, and the pipeline's declared fixes. */
export default function Investigation({ token, runId, queryId, baselineId, policyId, policyLabel, onClose }: {
  token: string; runId: string; queryId: string; baselineId?: string; policyId: string | null; policyLabel: string; onClose: () => void
}) {
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState('')
  const [sample, setSample] = useState(20)
  const [remote, setRemote] = useState(false)
  const [heldout, setHeldout] = useState(false)
  const [busy, setBusy] = useState('')
  const [trial, setTrial] = useState<Trial | null>(null)
  const [validation, setValidation] = useState<Validation | null>(null)
  const [name, setName] = useState(savedName)
  const [draftName, setDraftName] = useState('')
  const [record, setRecord] = useState<InvestigationRecord | null>(null)
  const [reason, setReason] = useState('')
  const [missing, setMissing] = useState('')
  const [outcome, setOutcome] = useState('fixed')
  const [policies, setPolicies] = useState<PolicyListing['policies']>([])
  const [protectPolicy, setProtectPolicy] = useState('')
  const [protectContext, setProtectContext] = useState(false)
  const [protectDoc, setProtectDoc] = useState('')
  const [protectRank, setProtectRank] = useState('1')
  const [protectedNote, setProtectedNote] = useState('')
  const verified = record?.status === 'closed' && record.outcome === 'fixed'

  useEffect(() => {
    if (!verified) return
    let live = true
    guidedRequest<PolicyListing>('policies', token).then(listing => {
      if (!live) return
      const latest = listing.policies.filter(p => p.latest)
      setPolicies(latest)
      setProtectPolicy(current => current || latest[0]?.id || '')
    }, () => { if (live) setPolicies([]) })
    return () => { live = false }
  }, [token, verified])

  const protect = async () => {
    if (!record) return
    setError('')
    try {
      const result = await guidedRequest<{ record: InvestigationRecord; policy: { name: string; version: number; id: string }; added: unknown[] }>(
        `investigations/${record.id}/protect`, token,
        { policy: protectPolicy, by: name, context: protectContext, ...(protectDoc.trim() ? { doc_id: protectDoc.trim(), max_rank: Number(protectRank) } : {}) })
      setRecord(result.record)
      setProtectedNote(result.added.length ? `Added ${result.added.length} rule(s) to ${result.policy.name} v${result.policy.version}. Select it for the next comparison or release export.` : `${result.policy.name} v${result.policy.version} already protects this question.`)
      setPolicies(current => [...current.filter(p => p.name !== result.policy.name), result.policy as PolicyListing['policies'][number]])
      setProtectPolicy(result.policy.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    let live = true
    guidedRequest<Data>('investigate', token, { run: runId, query_id: queryId, baseline: baselineId ?? null })
      .then(value => { if (live) setData(value) }, (e: unknown) => { if (live) setError(String(e instanceof Error ? e.message : e)) })
    return () => { live = false }
  }, [token, runId, queryId, baselineId])

  // The record opens when the investigation does, so its times start here.
  useEffect(() => {
    if (!name) return
    let live = true
    guidedRequest<InvestigationRecord>('investigations', token, { run: runId, query_id: queryId, baseline: baselineId ?? null, by: name })
      .then(value => { if (live) setRecord(value) }, (e: unknown) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [token, runId, queryId, baselineId, name])

  const act = async (action: string, body: object) => {
    if (!record) return
    setError('')
    try {
      const updated = await guidedRequest<InvestigationRecord>(`investigations/${record.id}/${action}`, token, { ...body, by: name })
      setRecord(action === 'close' ? await guidedRequest<InvestigationRecord>(`investigations/${record.id}`, token) : updated)
      return true
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      return false
    }
  }

  const run = async (path: string, body: object, kind: 'trial' | 'validation') => {
    setError(''); setBusy('Starting…')
    try {
      const { id } = await guidedRequest<{ id: string }>(path, token, body)
      const result = await finished<Trial | Validation>(token, id, setBusy)
      if (record?.status === 'open') await act('results', { result: result.id })
      if (kind === 'trial') { setTrial(result as Trial); setValidation(null) } else setValidation(result as Validation)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  if (error && !data) return <section aria-label={`Investigation ${queryId}`} className="rounded-xl border border-line p-4"><p role="alert" className="text-sm text-red-600">{error}</p><button className={`${button} mt-3`} onClick={onClose}>Close</button></section>
  if (!data) return <p className="text-sm text-muted">Loading investigation…</p>
  const blocked = !data.trial.available || !!busy || (!!data.pipeline.remote && !remote)
  return <section aria-label={`Investigation ${queryId}`} className="space-y-4 rounded-xl border border-teal-500/40 p-4">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h3 className="text-lg font-semibold">Investigate {queryId}</h3>
        {data.question && <p className="mt-1 whitespace-pre-wrap break-words text-sm">{data.question}</p>}
        <p className="mt-1 text-xs text-muted">{data.pipeline.name} · nDCG {format(data.metrics?.ndcg)}</p></div>
      <button className={button} onClick={onClose}>Close</button>
    </div>

    <div><h4 className="text-sm font-semibold">What was lost</h4>
      {data.losses.length ? <ul className="mt-1 list-disc pl-5 text-sm">{data.losses.map((loss, i) => <li key={i}>{lossLabel(loss)}</li>)}</ul> : <p className="mt-1 text-sm text-muted">No loss is recorded for this question in this run.</p>}
      {data.regression && <p className="mt-1 text-xs text-muted">{data.regression.new_losses.length === 1 ? '1 of these is' : `${data.regression.new_losses.length} of these are`} new since the baseline.</p>}</div>
    {data.lineage && <div aria-label="Source lineage"><h4 className="text-sm font-semibold">Source lineage</h4>
      {data.lineage.length ? <ul className="mt-1 list-disc pl-5 text-sm">{data.lineage.map(row => <li key={`${row.source_id}-${row.start}`}>{lineageLine(row)}</li>)}</ul> : <p className="mt-1 text-sm text-muted">The benchmark labels no passages for this question, so lineage cannot follow it.</p>}</div>}
    {data.hypotheses.length > 0 && <div><h4 className="text-sm font-semibold">Possible causes</h4><p className="text-xs text-muted">{data.hypotheses_note}</p><ul className="mt-1 list-disc pl-5 text-sm">{data.hypotheses.map(h => <li key={h}>{h}</li>)}</ul></div>}
    {data.cautions.length > 0 && <div className="rounded-lg bg-amber-500/10 p-3 text-sm"><b>Cautions</b><ul className="mt-1 list-disc pl-5">{data.cautions.map(c => <li key={c}>{c}</li>)}</ul></div>}
    {data.uncertainty.length > 0 && <div><h4 className="text-sm font-semibold">Uncertain</h4><ul className="mt-1 list-disc pl-5 text-sm text-muted">{data.uncertainty.map(u => <li key={u}>{u}</li>)}</ul></div>}

    <div><h4 className="text-sm font-semibold">Fixes this pipeline supports</h4>
      {!data.fixes.length && <p className="mt-1 text-sm text-muted">{data.unaddressed.length ? `No declared fix addresses: ${data.unaddressed.map(c => c.replaceAll('_', ' ')).join(', ')}. Declare fixes for the candidate in the manifest.` : 'Nothing to fix.'}</p>}
      {data.trial.reason && data.fixes.length > 0 && <p className="mt-1 text-sm text-amber-600">{data.trial.reason}</p>}
      {data.fixes.length > 0 && <label className="mt-2 block text-sm">Other exploratory questions to check<input aria-label="Sample size" type="number" min={0} max={200} value={sample} onChange={e => setSample(Number(e.target.value))} className="ml-2 w-20 rounded-lg border border-line bg-page px-2 py-1 text-sm" /></label>}
      {data.pipeline.remote && <label className="mt-2 flex gap-2 rounded-lg bg-amber-500/10 p-3 text-sm"><input type="checkbox" checked={remote} onChange={e => setRemote(e.target.checked)} /><span>Send these questions to {data.pipeline.remote} for the trial.</span></label>}
      <ul className="mt-2 space-y-2">{data.fixes.map(fix => <li key={fix.name} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line p-3">
        <div className="min-w-0"><b className="text-sm">{fix.name}</b><p className="font-mono text-xs text-muted">{JSON.stringify(fix.params)}</p><p className="text-xs text-muted">Addresses {fix.addresses.map(([c, d]) => `${d ? `${d} ` : ''}${c.replaceAll('_', ' ')}`).join(', ')}</p></div>
        <button className={primary} disabled={blocked} onClick={() => void run('fixes/try', { run: runId, fix: fix.name, query_ids: [queryId], sample, confirm_remote: remote }, 'trial')}>Try this fix · {trialCalls(data.trial, 1, sample)} calls</button>
      </li>)}</ul></div>

    {busy && <p role="status" className="text-sm text-muted">{busy}</p>}
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}

    {trial && <div aria-label="Fix trial" className="space-y-2 rounded-lg bg-surface-2 p-3">
      <p className="text-sm"><b>{trial.fix.name}</b>: {trial.recovered_focus} of {trial.focus.length} failing question(s) fully recovered · {trial.regressed.length} of {trial.questions} questions regressed</p>
      <p className="text-xs text-muted">{trial.status_label}</p>
      <table className="w-full text-left text-xs"><thead><tr><th className="py-1">Question</th><th>nDCG</th><th>What changed</th></tr></thead><tbody>
        {trial.rows.filter(r => r.focus || r.regressed).map(r => <tr key={r.query_id} className={r.regressed ? 'text-red-600' : ''}><td className="py-1 pr-2 font-mono">{r.query_id}{r.focus ? ' (failing)' : ''}</td><td className="pr-2 font-mono">{format(r.ndcg_before)} → {format(r.ndcg_after)}</td><td>{trialOutcome(r)}</td></tr>)}
      </tbody></table>
      {data.trial.heldout_questions > 0 ? <>
        <label className="flex gap-2 rounded-lg bg-amber-500/10 p-3 text-sm"><input type="checkbox" checked={heldout} onChange={e => setHeldout(e.target.checked)} /><span>I chose this fix and am validating it on {data.trial.heldout_questions} held-out question(s). Repeated validation against them contaminates the benchmark.</span></label>
        <button className={primary} disabled={!heldout || !!busy || (!!data.pipeline.remote && !remote)} onClick={() => void run('fixes/validate', { run: runId, fix: trial.fix.name, policy: policyId, confirm_heldout: heldout, confirm_remote: remote }, 'validation')}>{policyId ? `Validate under ${policyLabel}` : 'Measure on held-out questions (no release policy selected)'}</button>
      </> : <p className="text-xs text-muted">The benchmark has no held-out questions, so the fix cannot be validated here. Mark questions as held out that were not used to choose it.</p>}
    </div>}

    {validation && <div aria-label="Fix validation" role="status" className={`rounded-lg p-3 text-sm ${validation.approved ? 'bg-teal-500/10' : 'bg-amber-500/10'}`}>
      <b>{validation.status_label}</b>
      {validation.check && <ul className="mt-1 list-disc pl-5 text-xs">{validation.check.reasons.map(r => <li key={r}>{r}</li>)}</ul>}
      {validation.approved && <p className="mt-2 text-xs">Promote as a manifest candidate: <code className="break-all">{JSON.stringify({ name: validation.promote.name, params: validation.promote.params })}</code></p>}
    </div>}

    <div aria-label="Investigation record" className="space-y-2 rounded-lg border border-line p-3 text-sm">
      <h4 className="font-semibold">Investigation record</h4>
      {!name ? <div className="flex flex-wrap items-end gap-2"><label className="text-sm">Your name, recorded with your decisions<input aria-label="Your name" value={draftName} onChange={e => setDraftName(e.target.value)} className="ml-2 rounded-lg border border-line bg-page px-2 py-1 text-sm" /></label><button className={button} disabled={!draftName.trim()} onClick={() => { rememberName(draftName.trim()); setName(draftName.trim()) }}>Start recording</button><p className="w-full text-xs text-muted">Pilots measure time to an accepted diagnosis, disputed diagnoses and verified fixes from these records. Names are not verified.</p></div>
      : !record ? <p className="text-muted">Opening the record…</p> : <>
        <p className="text-xs text-muted">Opened by {record.opened_by} · recording as {name} · {record.status}</p>
        <p>Diagnosis: <b>{record.diagnosis.status}</b>{record.diagnosis.by ? ` by ${record.diagnosis.by}` : ''}{record.diagnosis.reason ? `: ${record.diagnosis.reason}` : ''}</p>
        {record.status === 'open' && <div className="flex flex-wrap items-center gap-2">
          <button className={button} onClick={() => void act('diagnosis', { status: 'accepted' })}>Accept diagnosis</button>
          <input aria-label="Dispute reason" placeholder="Why the diagnosis is wrong or incomplete" value={reason} onChange={e => setReason(e.target.value)} className="min-w-0 flex-1 rounded-lg border border-line bg-page px-2 py-1 text-sm" />
          <button className={button} disabled={!reason.trim()} onClick={() => void act('diagnosis', { status: 'disputed', reason }).then(ok => { if (ok) setReason('') })}>Dispute</button>
        </div>}
        {record.requests.length > 0 && <ul className="list-disc pl-5 text-xs text-muted">{record.requests.map(r => <li key={r.at + r.what}>Evidence requested by {r.by}: {r.what}</li>)}</ul>}
        {record.status === 'open' && <div className="flex flex-wrap items-center gap-2">
          <input aria-label="Missing evidence" placeholder="Evidence needed: a trace, a corpus version, a judgment…" value={missing} onChange={e => setMissing(e.target.value)} className="min-w-0 flex-1 rounded-lg border border-line bg-page px-2 py-1 text-sm" />
          <button className={button} disabled={!missing.trim()} onClick={() => void act('requests', { what: missing }).then(ok => { if (ok) setMissing('') })}>Request evidence</button>
        </div>}
        <p className="text-xs text-muted">{record.trials.length} trial(s) and {record.validations.length} validation(s) linked; {record.validations.filter(v => v.approved).length} approved.</p>
        {record.status === 'open' ? <div className="flex flex-wrap items-center gap-2">
          <select aria-label="Outcome" value={outcome} onChange={e => setOutcome(e.target.value)} className="rounded-lg border border-line bg-page px-2 py-1 text-sm">{OUTCOMES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
          <button className={button} disabled={outcome === 'fixed' && !canCloseFixed(record)} onClick={() => void act('close', { outcome })}>Close investigation</button>
        </div> : <><p role="status">Closed: <b>{record.outcome?.replaceAll('_', ' ')}</b>{record.measures?.minutes_to_accepted_diagnosis != null ? ` · ${record.measures.minutes_to_accepted_diagnosis} min to an accepted diagnosis` : ''}{record.measures?.minutes_to_verified_fix != null ? ` · ${record.measures.minutes_to_verified_fix} min to a verified fix` : ''}</p>
          {verified && <div aria-label="Protect this question" className="space-y-2 rounded-lg bg-surface-2 p-3">
            <h5 className="font-semibold">Protect this question in later releases</h5>
            <p className="text-xs text-muted">Adds rules to a saved release policy as a new version; earlier versions and checks are unchanged.</p>
            {!policies.length ? <p className="text-xs text-muted">Save a release policy first; the startup policy is read-only.</p> : <>
              <label className="block text-sm">Policy<select aria-label="Policy to protect with" value={protectPolicy} onChange={e => setProtectPolicy(e.target.value)} className="ml-2 rounded-lg border border-line bg-page px-2 py-1 text-sm">{policies.map(p => <option key={p.id} value={p.id ?? ''}>{p.name} v{p.version}</option>)}</select></label>
              <label className="flex gap-2 text-sm"><input type="checkbox" checked={protectContext} onChange={e => setProtectContext(e.target.checked)} /><span>Its required evidence must reach the model on every execution</span></label>
              <div className="flex flex-wrap items-end gap-2 text-sm"><label>Document that must rank<input aria-label="Protected document" value={protectDoc} onChange={e => setProtectDoc(e.target.value)} className="ml-2 w-40 rounded-lg border border-line bg-page px-2 py-1 text-sm" /></label><label>within the top<input aria-label="Protected rank" inputMode="numeric" value={protectRank} onChange={e => setProtectRank(e.target.value)} className="ml-2 w-16 rounded-lg border border-line bg-page px-2 py-1 text-sm" /></label></div>
              <button className={primary} disabled={!protectPolicy || (!protectContext && !protectDoc.trim())} onClick={() => void protect()}>Protect this question</button>
            </>}
            {protectedNote && <p role="status" className="text-sm">{protectedNote}</p>}
            {(record.protections ?? []).length > 0 && <ul className="list-disc pl-5 text-xs text-muted">{(record.protections ?? []).map(p => <li key={p.policy_id + p.version}>{p.name} v{p.version}: {p.rules.map(r => r.kind).join(', ') || 'already protected'} · by {p.by}</li>)}</ul>}
          </div>}</>}
        <ShareRecord token={token} kind="investigation" id={record.id} label="this investigation record" />
      </>}
    </div>
  </section>
}
