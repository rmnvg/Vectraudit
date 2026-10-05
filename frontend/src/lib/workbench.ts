export type Hit = { id: string; score?: number }
export type Stage = { name: string; kind: string; parents: string[]; hits: Hit[]; latency_ms: number; source: string }
export type Query = { query_id: string; ids: string[] | null; latency_ms: number | null; tags: string[]; metrics: Record<string, number> | null; no_answer_correct?: boolean | null; forbidden_ids?: string[]; error?: unknown; evidence?: { stages?: Stage[]; failures?: { doc_id: string; stage: string | null; reason: string }[] } }
export type Run = { format: 'vectraudit.run'; run_id: string; created_at: string; status: string; label?: string; queries: Query[]; backend?: { config?: Record<string, unknown> } | null; benchmark_source?: { split: string } | null; summary: { n_queries: number; n_ok: number; metrics: Record<string, number> | null; latency_ms: { p95: number } | null; permissions?: { violations: number }; no_answer?: { n: number; accuracy: number | null } } }
export type Candidate = { name: string; ndcg: number | null; p95_ms: number | null; cost_usd: number | null; memory_bytes: number | null; feasible: boolean; reasons: string[]; run: Run }
export type Optimization = { format: 'vectraudit.optimization'; candidates: Candidate[]; selected: string | null; eligible: boolean; heldout: Candidate | null }
export type Benchmark = { format: 'vectraudit.benchmark'; n_queries: number; n_labelled: number; missing_categories: string[]; duplicates: string[][]; unannotated: string[]; suggestions: { category: string; instruction: string }[]; judgment_queue: { query_id: string; doc_id: string; reason: string }[] }
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
  const run = (v: unknown): boolean => {
    if (!object(v) || typeof v.run_id !== 'string' || !summary(v.summary) || !Array.isArray(v.queries)) return false
    return v.queries.every(q => {
      if (!object(q) || typeof q.query_id !== 'string' || !strings(q.tags)) return false
      if (q.ids != null && !strings(q.ids)) return false
      if (!numeric(q.latency_ms) || !metrics(q.metrics)) return false
      if (q.forbidden_ids != null && !strings(q.forbidden_ids)) return false
      if (q.evidence == null) return true
      if (!object(q.evidence)) return false
      const stages = q.evidence.stages
      if (stages != null && (!Array.isArray(stages) || !stages.every(st => {
        if (!object(st) || typeof st.name !== 'string' || typeof st.kind !== 'string' || typeof st.source !== 'string') return false
        return numeric(st.latency_ms) && strings(st.parents) && Array.isArray(st.hits) && st.hits.every(h => object(h) && typeof h.id === 'string')
      }))) return false
      const failures = q.evidence.failures
      return failures == null || (Array.isArray(failures) && failures.every(f => object(f) && typeof f.doc_id === 'string' && typeof f.reason === 'string' && (f.stage == null || typeof f.stage === 'string')))
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
