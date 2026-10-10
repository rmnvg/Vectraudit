import { describe, expect, it } from 'vitest'
import { acceptanceLine, coverageLine, hostedVerdictLine } from './exceptions'
import type { Acceptance, HostedException, HostedExceptions } from './exceptions'

const option = (extra: Partial<Acceptance>): Acceptance => ({ rule: 'metric:ndcg.max_drop', kind: 'metric', message: '', role: null, max_days: null, why: null, ...extra })
const exception: HostedException = {
  id: 'e1', rule: 'metric:ndcg.max_drop', kind: 'metric', reason: 'Known trade-off', owner: 'lead@acme.test', accepted_by: 'eng@acme.test',
  created_at: '2026-10-01T00:00:00+00:00', expires_at: '2026-10-20T00:00:00+00:00', max_checks: 3, uses: 3, status: 'active', origin_check_id: 'c',
}
const view = (extra: Partial<HostedExceptions>): HostedExceptions => ({
  same_policy: true, hosted_policy_sha256: 'abcdef0123456789', hosted_policy_default: false, hosted_policy_allows: true,
  hosted_verdict: 'fail', coverage: [], exceptions: [], truncated: false, destination: { configured: true, project: 'acme/search' }, ...extra,
})

describe('exceptions', () => {
  it('says who may accept a failure, or why nobody can', () => {
    expect(acceptanceLine(option({ role: 'engineer', max_days: 30 }))).toBe('Can be accepted by an engineer or owner in the hosted project, for up to 30 days.')
    expect(acceptanceLine(option({ role: 'owner', max_days: 7 }))).toContain('by an owner')
    expect(acceptanceLine(option({ why: 'never', kind: 'safety' }))).toContain('never waives permission')
    expect(acceptanceLine(option({ why: 'not_allowed', kind: 'cohort' }))).toContain('does not open cohort rules')
    expect(acceptanceLine(option({ why: 'no_exceptions' }))).toContain('no exceptions block')
  })

  it('makes lapsed exceptions visible as blocking again', () => {
    expect(coverageLine({ rule: 'r', status: 'covered', exception })).toBe('Accepted by eng@acme.test until 2026-10-20, follow-up by lead@acme.test: Known trade-off')
    expect(coverageLine({ rule: 'r', status: 'expired', exception })).toBe('Exception expired on 2026-10-20; this failure blocks again.')
    expect(coverageLine({ rule: 'r', status: 'used', exception })).toContain('3 of 3 checks')
    expect(coverageLine({ rule: 'r', status: 'revoked', exception })).toContain('blocks again')
    expect(coverageLine({ rule: 'r', status: 'none', exception: null })).toBe('No exception in the hosted project.')
  })

  it('never reads as a pass, and never guesses under another policy', () => {
    expect(hostedVerdictLine(view({ hosted_verdict: 'accepted' }))).toContain('"fail, accepted by exception", never pass')
    const blocking = view({ coverage: [{ rule: 'a', status: 'covered', exception }, { rule: 'b', status: 'none', exception: null }] })
    expect(hostedVerdictLine(blocking)).toBe('Recorded in acme/search, this result would still fail: 1 failing rule has no active exception.')
    expect(hostedVerdictLine(view({ same_policy: false, hosted_verdict: null }))).toContain('policy abcdef012345')
    expect(hostedVerdictLine(view({ same_policy: false, hosted_policy_default: true, hosted_verdict: null }))).toContain('built-in default policy')
  })
})
