import { useState } from 'react'
import { acceptanceLine, coverageLine, hostedVerdictLine } from '../lib/exceptions'
import type { Acceptance, Coverage, HostedExceptions } from '../lib/exceptions'
import { guidedRequest } from '../lib/guided'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'
const tone: Record<Coverage['status'], string> = {
  covered: 'text-teal-600', expired: 'text-red-600', used: 'text-red-600', revoked: 'text-red-600',
  not_open: 'text-amber-600', not_applicable: 'text-muted', none: 'text-muted',
}

/** Each failing rule, who may accept it, and (on request) the hosted project's audited exceptions. Read only. */
export function FailingRules({ token, options, checkFile }: { token: string; options: Acceptance[]; checkFile: string }) {
  const check = checkFile.replace(/\.check\.json$/, '')
  const [loaded, setLoaded] = useState<{ check: string; view: HostedExceptions } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const view = loaded?.check === check ? loaded.view : null
  const coverage = new Map(view?.coverage.map(c => [c.rule, c]))

  async function lookup() {
    setBusy(true)
    setError('')
    try {
      setLoaded({ check, view: await guidedRequest<HostedExceptions>('exceptions/hosted', token, { check }) })
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  if (!options.length) return null
  return <section aria-label="Failing rules" className="mt-3 space-y-2 border-t border-line pt-3 text-sm">
    <h4 className="font-semibold">Failing rules</h4>
    <ul className="space-y-2">{options.map(option => {
      const hosted = coverage.get(option.rule)
      return <li key={option.rule}>
        <code className="text-xs">{option.rule}</code>
        <span className={`block text-xs ${option.role ? 'text-muted' : 'text-amber-600'}`}>{acceptanceLine(option)}</span>
        {hosted && <span className={`block text-xs ${tone[hosted.status]}`}>{coverageLine(hosted)}</span>}
      </li>
    })}</ul>
    <p className="text-xs text-muted">Exceptions are granted and audited in the hosted project, never here. This comparison stays a fail on this machine.</p>
    <button className={button} disabled={busy} onClick={() => void lookup()}>{busy ? 'Checking…' : view ? 'Check again' : 'Check accepted regressions in the hosted project'}</button>
    {view && <>
      <p role="status" className={view.hosted_verdict === 'accepted' ? 'text-teal-600' : view.same_policy ? 'text-red-600' : 'text-amber-600'}>{hostedVerdictLine(view)}</p>
      {!view.hosted_policy_allows && <p className="text-xs text-muted">The hosted policy has no exceptions block, so nothing can be accepted there yet.</p>}
      {view.exceptions.length > 0 && <details className="text-xs"><summary className="cursor-pointer">All exceptions in {view.destination.project ?? 'the hosted project'} ({view.exceptions.length}{view.truncated ? '+' : ''})</summary>
        <ul className="mt-2 space-y-1">{view.exceptions.map(e => <li key={e.id}><code>{e.rule}</code> · {e.status} · accepted by {e.accepted_by} until {e.expires_at.slice(0, 10)}{e.max_checks != null ? ` · ${e.uses} of ${e.max_checks} checks` : ''}<span className="block text-muted">{e.reason}</span></li>)}</ul>
      </details>}
    </>}
    {error && <p role="alert" className="text-red-600">{error}</p>}
  </section>
}
