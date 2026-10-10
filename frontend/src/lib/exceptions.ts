/** Audited exceptions beside a local comparison: who may accept each failure, and what the hosted project holds. */
import type { ShareDestination } from './share'

export type Role = 'engineer' | 'owner'
export interface Acceptance {
  rule: string; kind: string; message: string; role: Role | null; max_days: number | null
  why: 'no_exceptions' | 'never' | 'not_allowed' | null
}
export interface HostedException {
  id: string; rule: string; kind: string; reason: string; owner: string; accepted_by: string
  created_at: string; expires_at: string; max_checks: number | null; uses: number
  status: 'active' | 'expired' | 'used' | 'revoked'; origin_check_id: string
}
export type CoverageStatus = 'covered' | 'expired' | 'used' | 'revoked' | 'not_open' | 'not_applicable' | 'none'
export interface Coverage { rule: string; status: CoverageStatus; exception: HostedException | null }
export interface HostedExceptions {
  same_policy: boolean; hosted_policy_sha256: string; hosted_policy_default: boolean; hosted_policy_allows: boolean
  hosted_verdict: string | null; coverage: Coverage[]; exceptions: HostedException[]; truncated: boolean
  destination: ShareDestination
}

/** Who may accept a failing rule under the comparison's policy, or why nobody can. */
export function acceptanceLine(option: Acceptance): string {
  if (option.role) {
    const who = option.role === 'engineer' ? 'an engineer or owner' : 'an owner'
    return `Can be accepted by ${who} in the hosted project${option.max_days ? `, for up to ${option.max_days} days` : ''}.`
  }
  if (option.why === 'never') return 'Cannot be accepted: this policy never waives permission or no-answer failures.'
  if (option.why === 'not_allowed') return `Cannot be accepted: this policy does not open ${option.kind} rules to exceptions.`
  return 'Cannot be accepted: this policy has no exceptions block.'
}

export function day(iso: string): string {
  return iso.slice(0, 10)
}

/** One line for a failing rule's state in the hosted project. */
export function coverageLine(coverage: Coverage): string {
  const e = coverage.exception
  switch (coverage.status) {
    case 'covered': return `Accepted by ${e?.accepted_by} until ${day(e?.expires_at ?? '')}, follow-up by ${e?.owner}: ${e?.reason}`
    case 'expired': return `Exception expired on ${day(e?.expires_at ?? '')}; this failure blocks again.`
    case 'used': return `Exception used up (${e?.uses} of ${e?.max_checks} checks); this failure blocks again.`
    case 'revoked': return 'Exception revoked; this failure blocks again.'
    case 'not_open': return 'An active exception exists, but the hosted policy does not open this rule, so it does not apply.'
    case 'not_applicable': return 'Exceptions never apply to errors or incomparable results.'
    default: return 'No exception in the hosted project.'
  }
}

/** What the hosted project would record for this comparison, or why that can't be said. */
export function hostedVerdictLine(view: HostedExceptions): string {
  const project = view.destination.project ?? 'The hosted project'
  if (!view.same_policy) {
    const which = view.hosted_policy_default ? 'the built-in default policy' : `policy ${view.hosted_policy_sha256.slice(0, 12)}`
    return `${project} judges under ${which}, not this comparison's policy, so its verdict for this change can't be predicted here.`
  }
  if (view.hosted_verdict === 'accepted') return `Recorded in ${project}, this result would read "fail, accepted by exception", never pass.`
  if (view.hosted_verdict === 'fail') {
    const blocking = view.coverage.filter(c => c.status !== 'covered').length
    return `Recorded in ${project}, this result would still fail: ${blocking} failing rule${blocking === 1 ? ' has' : 's have'} no active exception.`
  }
  return `${project} applies exceptions only to failing results.`
}
