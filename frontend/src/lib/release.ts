/** Release policies and exported release checks; the server's policy parser stays authoritative. */

export type PolicySource = 'exploratory' | 'startup' | 'workspace'
export interface PolicyRef { id: string | null; name: string; version: number | null; sha256: string; source: PolicySource }
export type RawPolicy = Record<string, unknown>
export interface PolicyRecord extends PolicyRef { policy: RawPolicy; effective: RawPolicy; note?: string; latest?: boolean; created_at?: string; supersedes?: string | null }
export interface PolicyListing { exploratory: PolicyRef & { effective: RawPolicy }; startup: PolicyRecord | null; policies: PolicyRecord[] }
export interface ReleaseSummary {
  id: string; name: string; version: number; created_at: string; policy: PolicyRef
  pipeline: { name: string; kind: string }; issues: { prerequisites: string[]; warnings: string[] }
}
export interface ReleaseRecord extends ReleaseSummary {
  release: { benchmark: { sha256: string; split: string; n_queries: number }; k: number; repeats: number }
  files: Record<string, string>
}

export const METRICS = ['ndcg', 'recall', 'mrr', 'precision', 'ann_recall'] as const
export type Metric = typeof METRICS[number]
export const METRIC_HELP: Record<Metric, string> = {
  ndcg: 'nDCG: are the most relevant documents near the top?',
  recall: 'Recall: how many of the relevant documents were returned at all?',
  mrr: 'MRR: how high is the first relevant document?',
  precision: 'Precision: how many returned documents are relevant?',
  ann_recall: 'ANN recall: how closely approximate search matches exact search?',
}

/** Strings so inputs can be empty while editing; empty means "no limit". */
export interface MetricForm { enabled: boolean; maxDrop: string; min: string }
export interface PolicyForm {
  metrics: Record<Metric, MetricForm>
  minQueries: string
  maxRegressed: string
  primaryMetric: string
  cohortMinQueries: string
  cohortMetrics: Record<Metric, { enabled: boolean; maxDrop: string }>
  p95Pct: string; p50Pct: string; p95Ms: string; minIncreaseMs: string
  sameEnvironment: boolean
  requireSignificance: boolean
}

/** The scaffolded CI policy (`vectraudit init`): a reasonable first release contract. */
export const STARTER_POLICY: RawPolicy = {
  version: 1,
  metrics: { ndcg: { max_drop: 0.02 }, recall: { max_drop: 0.02 } },
  latency: { p95_max_increase_pct: 30 },
  min_queries: 30,
  max_regressed_queries: 10,
  cohorts: { min_queries: 15, metrics: { ndcg: { max_drop: 0.05 } } },
  require_significance: false,
}

const text = (value: unknown) => typeof value === 'number' ? String(value) : ''
const object = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

export function formFromPolicy(raw: RawPolicy): PolicyForm {
  const metrics = object(raw.metrics)
  const latency = object(raw.latency)
  const cohorts = object(raw.cohorts)
  const cohortMetrics = object(cohorts.metrics)
  return {
    metrics: Object.fromEntries(METRICS.map(m => {
      const rule = object(metrics[m])
      return [m, { enabled: m in metrics, maxDrop: text(rule.max_drop), min: text(rule.min) }]
    })) as Record<Metric, MetricForm>,
    minQueries: text(raw.min_queries ?? 1),
    maxRegressed: text(raw.max_regressed_queries),
    primaryMetric: typeof raw.primary_metric === 'string' ? raw.primary_metric : '',
    cohortMinQueries: text(cohorts.min_queries ?? 10),
    cohortMetrics: Object.fromEntries(METRICS.map(m => [m, { enabled: m in cohortMetrics, maxDrop: text(object(cohortMetrics[m]).max_drop) }])) as PolicyForm['cohortMetrics'],
    p95Pct: text(latency.p95_max_increase_pct),
    p50Pct: text(latency.p50_max_increase_pct),
    p95Ms: text(latency.p95_max_ms),
    minIncreaseMs: text(latency.min_increase_ms ?? 2),
    sameEnvironment: latency.require_same_environment !== false,
    requireSignificance: raw.require_significance === true,
  }
}

/** The policy JSON for a form, or the problems to fix first. */
export function policyFromForm(form: PolicyForm): { policy?: RawPolicy; errors: string[] } {
  const errors: string[] = []
  const number = (value: string, label: string, max?: number): number | undefined => {
    if (!value.trim()) return undefined
    const n = Number(value)
    if (!Number.isFinite(n) || n < 0) { errors.push(`${label} must be a number of at least 0.`); return undefined }
    if (max !== undefined && n > max) { errors.push(`${label} must be at most ${max}.`); return undefined }
    return n
  }
  const count = (value: string, label: string, minimum: number): number | undefined => {
    if (!value.trim()) return undefined
    const n = Number(value)
    if (!Number.isInteger(n) || n < minimum) { errors.push(`${label} must be a whole number of at least ${minimum}.`); return undefined }
    return n
  }
  const metrics: Record<string, Record<string, number>> = {}
  for (const m of METRICS) {
    const rule = form.metrics[m]
    if (!rule.enabled) continue
    const maxDrop = number(rule.maxDrop, `${m} allowed drop`, 1)
    const min = number(rule.min, `${m} minimum`, 1)
    if (maxDrop === undefined && min === undefined) errors.push(`Set an allowed drop or a minimum for ${m}, or turn it off.`)
    metrics[m] = { ...(maxDrop !== undefined && { max_drop: maxDrop }), ...(min !== undefined && { min }) }
  }
  const cohortMetrics: Record<string, Record<string, number>> = {}
  for (const m of METRICS) {
    const rule = form.cohortMetrics[m]
    if (!rule.enabled) continue
    const maxDrop = number(rule.maxDrop, `${m} cohort allowed drop`, 1)
    if (maxDrop === undefined) errors.push(`Set an allowed cohort drop for ${m}, or turn it off.`)
    else cohortMetrics[m] = { max_drop: maxDrop }
  }
  const maxRegressed = count(form.maxRegressed, 'Regressed questions allowed', 0)
  if (!Object.keys(metrics).length && maxRegressed === undefined && !Object.keys(cohortMetrics).length) {
    errors.push('Add at least one quality rule: a metric, a regressed-question limit or a cohort rule.')
  }
  const latency: Record<string, number | boolean> = {}
  const latencyFields: [keyof PolicyForm, string, string][] = [
    ['p95Pct', 'p95_max_increase_pct', 'p95 increase'], ['p50Pct', 'p50_max_increase_pct', 'p50 increase'],
    ['p95Ms', 'p95_max_ms', 'p95 ceiling'], ['minIncreaseMs', 'min_increase_ms', 'Ignored latency noise'],
  ]
  for (const [field, key, label] of latencyFields) {
    const value = number(form[field] as string, label)
    if (value !== undefined) latency[key] = value
  }
  if (!form.sameEnvironment) latency.require_same_environment = false
  const policy: RawPolicy = {
    version: 1,
    metrics,
    latency,
    min_queries: count(form.minQueries, 'Minimum measured questions', 1) ?? 1,
    ...(maxRegressed !== undefined && { max_regressed_queries: maxRegressed }),
    ...(form.primaryMetric && { primary_metric: form.primaryMetric }),
    cohorts: { min_queries: count(form.cohortMinQueries, 'Minimum questions per cohort', 1) ?? 10, metrics: cohortMetrics },
    require_significance: form.requireSignificance,
  }
  return errors.length ? { errors } : { policy, errors }
}

/** `undefined` until a default is chosen; `null` is the labelled exploratory mode. */
export type PolicyChoice = string | null | undefined

export function selectedRef(listing: PolicyListing | null, choice: PolicyChoice): PolicyRef | null {
  if (!listing || !choice) return null
  if (choice === 'startup') return listing.startup
  return listing.policies.find(p => p.id === choice) || null
}

export function policyLabel(ref: PolicyRef | null | undefined): string {
  if (!ref || ref.source === 'exploratory') return 'Exploratory defaults'
  return `${ref.name} v${ref.version} · ${ref.sha256.slice(0, 12)}`
}

export async function downloadGuided(path: string, token: string, filename: string): Promise<void> {
  const response = await fetch(`api/guided/${path}`, { headers: { 'X-Workbench-Token': token } })
  if (!response.ok) throw new Error('The download failed. Reload the workbench and try again.')
  const url = URL.createObjectURL(await response.blob())
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
