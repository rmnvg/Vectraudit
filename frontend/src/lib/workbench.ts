export type Hit = { id: string; score?: number }
export type Stage = { name: string; kind: string; parents: string[]; hits: Hit[]; latency_ms: number | null; source: string }
export type ContextEvidence = { coverage: string; method: string; reached: string[]; losses: { doc_id: string; kind: 'omitted' | 'truncated'; reason: string }[]; unknown: string[] }
export type Evidence = { stages?: Stage[]; failures?: { doc_id: string; stage: string | null; reason: string }[]; context?: ContextEvidence }
export type Answer = { verdict: string | null; decided_by: string | null; outcome: string; judgments: { evaluator: string; kind: string; verdict: string | null }[]; diagnosis?: { class: string; facts: string[] }; claim_findings?: ClaimFinding[]; claim_evidence?: { unmatched_documents?: string[] }; safety?: { followed_injection: boolean; attacks: string[]; documents: string[] } }
/** What claim-level evidence (RAGChecker) says about an answer: model-judged, never a root cause. */
export type ClaimFinding = { finding: string; count: number; points_to: string; evidence: string; disputed: boolean; documents?: string[]; more_documents?: number; note?: string }
export type Query = { answer?: Answer; query_id: string; ids: string[] | null; latency_ms: number | null; tags: string[]; metrics: Record<string, number> | null; no_answer_correct?: boolean | null; forbidden_ids?: string[]; error?: unknown; evidence?: Evidence | null; repeat_ids?: string[][] | null; repeat_metrics?: (Record<string, number> | null)[]; repeat_ann_recall?: (number | null)[]; repeat_evidence?: Record<string, Evidence> }
export type Run = { metric_contract?: { version: number; repeat_aggregation?: string }; format: 'vectraudit.run'; run_id: string; created_at: string; status: string; label?: string; queries: Query[]; backend?: { config?: Record<string, unknown> } | null; benchmark_source?: { split: string } | null; summary: { n_queries: number; n_ok: number; metrics: Record<string, number> | null; latency_ms: { p95: number } | null; permissions?: { violations: number }; no_answer?: { n: number; accuracy: number | null } } }
/** A run as listed by the guided workspace: per-query records load on demand. */
export type RunListing = Omit<Run, 'queries'> & { queries?: Query[]; queries_omitted?: number }
export type PolicyScreen = { verdict: string; failing_rules: string[]; reasons: string[] }
export type Candidate = { name: string; ndcg: number | null; p95_ms: number | null; cost_usd: number | null; memory_bytes: number | null; feasible: boolean; reasons: string[]; run: Run; policy_screen?: PolicyScreen | null; check?: PolicyScreen | null; hsd_group?: string; p_vs_leader_adjusted?: number | null }
export type Optimization = {
  format: 'vectraudit.optimization'; candidates: Candidate[]; selected: string | null; eligible: boolean; heldout: Candidate | null
  /** Present from format version 2: older artifacts are exploratory by construction. */
  status?: 'exploratory_only' | 'no_feasible_candidate' | 'baseline_remains_best' | 'approved_under_policy' | 'not_approved'
  status_label?: string; approved?: boolean; policy?: { sha256: string; baseline: string } | null
  /** Three or more candidates: which the best cannot be told apart from (randomized Tukey HSD). */
  significance?: { leader: string; alpha: number; indistinguishable: string[]; note: string } | null
}
export type Benchmark = { format: 'vectraudit.benchmark'; n_queries: number; n_labelled: number; missing_categories: string[]; duplicates: string[][]; unannotated: string[]; suggestions: { category: string; instruction: string }[]; judgment_queue: { query_id: string; doc_id: string; reason: string; kind?: string; evidence?: string }[]; required_evidence_proposals?: EvidenceProposal[] }
/** A group of documents that together cover the reference answer's claims: proposed, never applied. */
export type EvidenceProposal = { query_id: string; docs: string[]; claims_covered: number; reference_claims: number; coverage: number; status: string; needs_judgment: string[]; already_relevant: string[] }
export type Monitor = { format: 'vectraudit.monitor'; created_at: string; verdict: string; drift: { corpus: string; configuration: string; results: unknown[] }; alerts: { message: string }[]; summary: Run['summary'] }
export type Artifact = Run | Optimization | Benchmark | Monitor

export function parseArtifact(value: unknown): Artifact {
  if (!value || typeof value !== 'object') throw new Error('Choose a Vectraudit JSON artifact.')
  const data = value as Record<string, unknown>
  const keys: Record<string, string[]> = {
    'vectraudit.run': ['run_id', 'queries', 'summary'],
    'vectraudit.optimization': ['candidates', 'eligible'],
    'vectraudit.benchmark': ['missing_categories', 'duplicates', 'suggestions', 'judgment_queue'],
    'vectraudit.monitor': ['created_at', 'drift', 'alerts', 'summary'],
  }
  const required = keys[String(data.format)]
  if (!required || required.some(k => data[k] == null)) throw new Error('Unsupported or incomplete artifact.')
  // Validate arrays before rendering imported JSON; a malformed file must stay a recoverable error.
  const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
  const strings = (v: unknown) => Array.isArray(v) && v.every(x => typeof x === 'string')
  const numeric = (v: unknown) => v == null || (typeof v === 'number' && Number.isFinite(v))
  const metrics = (v: unknown) => v == null || (object(v) && Object.values(v).every(numeric))
  const summary = (v: unknown) => object(v) && numeric(v.n_queries) && numeric(v.n_ok) && metrics(v.metrics) && metrics(v.latency_ms)
  const evidence = (value: unknown): boolean => {
      if (value == null) return true
      if (!object(value)) return false
      const stages = value.stages
      if (stages != null && (!Array.isArray(stages) || !stages.every(st => {
        if (!object(st) || typeof st.name !== 'string' || typeof st.kind !== 'string' || typeof st.source !== 'string') return false
        return numeric(st.latency_ms) && strings(st.parents) && Array.isArray(st.hits) && st.hits.every(h => object(h) && typeof h.id === 'string')
      }))) return false
      const failures = value.failures
      if (failures != null && !(Array.isArray(failures) && failures.every(f => object(f) && typeof f.doc_id === 'string' && typeof f.reason === 'string' && (f.stage == null || typeof f.stage === 'string')))) return false
      const context = value.context
      return context == null || (object(context) && typeof context.coverage === 'string' && typeof context.method === 'string' && strings(context.reached) && strings(context.unknown) && Array.isArray(context.losses) && context.losses.every(l => object(l) && typeof l.doc_id === 'string' && (l.kind === 'omitted' || l.kind === 'truncated')))
  }
  const run = (v: unknown): boolean => {
    if (!object(v) || typeof v.run_id !== 'string' || !summary(v.summary) || !Array.isArray(v.queries)) return false
    return v.queries.every(q => {
      if (!object(q) || typeof q.query_id !== 'string' || !strings(q.tags)) return false
      if (q.ids != null && !strings(q.ids)) return false
      if (!numeric(q.latency_ms) || !metrics(q.metrics)) return false
      if (q.forbidden_ids != null && !strings(q.forbidden_ids)) return false
      if (q.repeat_ids != null && (!Array.isArray(q.repeat_ids) || !q.repeat_ids.every(strings))) return false
      if (q.repeat_metrics != null && (!Array.isArray(q.repeat_metrics) || !q.repeat_metrics.every(metrics))) return false
      if (q.repeat_ann_recall != null && (!Array.isArray(q.repeat_ann_recall) || !q.repeat_ann_recall.every(numeric))) return false
      if (q.repeat_evidence != null && (!object(q.repeat_evidence) || !Object.values(q.repeat_evidence).every(evidence))) return false
      return evidence(q.evidence)
    })
  }
  let valid = false
  if (data.format === 'vectraudit.run') valid = run(data)
  if (data.format === 'vectraudit.optimization') valid = Array.isArray(data.candidates) && data.candidates.every(c => object(c) && typeof c.name === 'string' && strings(c.reasons) && [c.ndcg, c.p95_ms, c.cost_usd, c.memory_bytes].every(numeric) && run(c.run)) && (data.heldout == null || (object(data.heldout) && run(data.heldout.run)))
  if (data.format === 'vectraudit.benchmark') valid = strings(data.missing_categories) && strings(data.unannotated) && Array.isArray(data.duplicates) && data.duplicates.every(strings) && Array.isArray(data.suggestions) && data.suggestions.every(x => object(x) && typeof x.category === 'string' && typeof x.instruction === 'string') && Array.isArray(data.judgment_queue) && data.judgment_queue.every(x => object(x) && typeof x.query_id === 'string' && typeof x.doc_id === 'string' && typeof x.reason === 'string')
  if (data.format === 'vectraudit.monitor') valid = typeof data.created_at === 'string' && typeof data.verdict === 'string' && object(data.drift) && typeof data.drift.corpus === 'string' && typeof data.drift.configuration === 'string' && Array.isArray(data.drift.results) && Array.isArray(data.alerts) && data.alerts.every(a => object(a) && typeof a.message === 'string') && summary(data.summary)
  if (!valid) throw new Error('Malformed artifact: check the recorded result schema.')
  return data as unknown as Artifact
}

/** A finding's heading, count and where it points, as shown under the answer. */
export function claimFindingText(f: ClaimFinding): string {
  const plural = f.count === 1 ? 'claim' : 'claims'
  const subject = f.finding === 'contradicting_document' ? (f.count === 1 ? 'document' : 'documents') : plural
  return `${f.finding.replaceAll('_', ' ')}: ${f.count} ${subject}`
}

/** One line saying what a proposed required-evidence group covers and what is still unjudged. */
export function proposalText(p: EvidenceProposal): string {
  const covers = `covers ${p.claims_covered} of ${p.reference_claims} reference claims (${Math.round(p.coverage * 100)}%), model-judged`
  const todo = p.needs_judgment.length ? `; judge ${p.needs_judgment.join(', ')} first` : '; ready to accept'
  return `${p.query_id}: ${p.docs.join(', ')} ${covers}${todo}`
}
