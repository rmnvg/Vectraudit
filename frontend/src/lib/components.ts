/** Where each question first lost its evidence: a port of evaluation/components.py for the run view.
 *  fixtures/components.json holds runs with the scoreboard Python computes for them; the tests
 *  require this file to give exactly the same result. */

export const COMPONENTS = ['retrieve', 'filter', 'fusion', 'rerank', 'truncate', 'custom', 'context', 'answer'] as const
const STAGE_COMPONENT: Record<string, string> = { keyword: 'retrieve', vector: 'retrieve', filter: 'filter', fusion: 'fusion', rerank: 'rerank', truncate: 'truncate', custom: 'custom' }
const OBSERVED = new Set<string>([...COMPONENTS, 'kept'])

type Row = Record<string, unknown>
const object = (v: unknown): v is Row => !!v && typeof v === 'object' && !Array.isArray(v)
const rows = (v: unknown): Row[] => Array.isArray(v) ? v.filter(object) : []
const round = (v: number) => Math.round(v * 10000) / 10000

export interface Rate { families: number; rate: number | null; ci_low: number; ci_high: number }
export interface ComponentRow extends Rate { component: string; questions: number }
export interface Table {
  observed: { questions: number; families: number }
  components: ComponentRow[]
  kept: Rate & { questions: number }
  not_observed: number
  not_applicable: { unlabelled: number; no_answer: number }
  errored: number
}
export interface Scoreboard extends Table {
  declared: string[]
  by_stage_kind: Record<string, number>
  cohorts: (Table & { tag: string; targets: string[]; observable: string[]; lost_at_targets: number })[]
}

/** The Wilson score interval, as evaluation/significance.py computes it (fractional successes allowed). */
export function wilson(successes: number, n: number, z = 1.959963984540054): [number, number] {
  if (n === 0) return [0, 1]
  const p = successes / n
  const denominator = 1 + z * z / n
  const centre = (p + z * z / (2 * n)) / denominator
  const half = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
  return [Math.min(p, Math.max(0, centre - half)), Math.max(p, Math.min(1, centre + half))]
}

function firstDrop(evidence: Row): Map<string, number | null> {
  const order = new Map(rows(evidence.stages).map((s, i) => [String(s.name), i]))
  const out = new Map<string, number | null>()
  for (const failure of rows(evidence.failures)) {
    const doc = String(failure.doc_id)
    if (out.has(doc) && out.get(doc) !== null) continue
    if (failure.stage == null) out.set(doc, failure.evidence === 'recorded' ? -1 : null)
    else out.set(doc, order.get(String(failure.stage)) ?? null)
  }
  return out
}

/** The outcome (a component, kept, not_observed, unlabelled, no_answer or errored) and stage kind. */
export function attribute(entry: Row): [string, string | null] {
  if (entry.error) return ['errored', null]
  if (entry.no_answer_correct != null) return ['no_answer', null]
  const metrics = entry.metrics
  if (!object(metrics)) return ['unlabelled', null]
  const evidence = object(entry.evidence) ? entry.evidence : {}
  const stages = rows(evidence.stages)
  const sufficiency = object(entry.sufficiency) ? entry.sufficiency : null
  const delivered = sufficiency ? sufficiency.delivered === 'sufficient' : !!metrics.hit_rate
  const missing = sufficiency ? (Array.isArray(sufficiency.missing_from_delivered) ? sufficiency.missing_from_delivered.map(String) : []) : rows(evidence.failures).map(f => String(f.doc_id))
  if (!delivered) {
    const drops = firstDrop(evidence)
    const positions = [...new Set(missing)].map(d => drops.has(d) ? drops.get(d)! : null)
    if (!stages.length || !positions.length || positions.some(p => p === null)) return ['not_observed', null]
    const known = positions as number[]
    const position = sufficiency ? Math.min(...known) : Math.max(...known)
    if (position < 0) return ['retrieve', null]
    const kind = String(stages[position].kind)
    return [STAGE_COMPONENT[kind] ?? 'custom', kind]
  }
  const answer = object(entry.answer) ? entry.answer : {}
  const diagnosis = object(answer.diagnosis) ? answer.diagnosis.class : undefined
  if (sufficiency) {
    if (sufficiency.context === 'insufficient') return ['context', null]
    if (diagnosis === 'answered_wrong_despite_evidence') return ['answer', null]
    if (diagnosis === 'context_unknown') return ['not_observed', null]
    return ['kept', null]
  }
  const context = evidence.context
  if (object(context) && !(Array.isArray(context.reached) && context.reached.length) && rows(context.losses).length) return ['context', null]
  return ['kept', null]
}

function rate(shares: Map<string, number>, families: number): Rate {
  let successes = 0
  for (const v of shares.values()) successes += v
  const [low, high] = wilson(Math.min(successes, families), families)
  return { families: round(successes), rate: families ? round(successes / families) : null, ci_low: round(low), ci_high: round(high) }
}

function table(entries: [string, string][], names: string[]): Table {
  const perFamily = new Map<string, string[]>()
  for (const [family, outcome] of entries) if (OBSERVED.has(outcome)) perFamily.set(family, [...(perFamily.get(family) ?? []), outcome])
  const shares = new Map<string, Map<string, number>>()
  for (const [family, outcomes] of perFamily) for (const outcome of outcomes) {
    const m = shares.get(outcome) ?? new Map<string, number>()
    m.set(family, (m.get(family) ?? 0) + 1 / outcomes.length)
    shares.set(outcome, m)
  }
  const counts = new Map<string, number>()
  for (const [, outcome] of entries) counts.set(outcome, (counts.get(outcome) ?? 0) + 1)
  const n = perFamily.size
  const share = (name: string) => shares.get(name) ?? new Map<string, number>()
  return {
    observed: { questions: entries.filter(([, o]) => OBSERVED.has(o)).length, families: n },
    components: names.map(name => ({ component: name, questions: counts.get(name) ?? 0, ...rate(share(name), n) })),
    kept: { questions: counts.get('kept') ?? 0, ...rate(share('kept'), n) },
    not_observed: counts.get('not_observed') ?? 0,
    not_applicable: { unlabelled: counts.get('unlabelled') ?? 0, no_answer: counts.get('no_answer') ?? 0 },
    errored: counts.get('errored') ?? 0,
  }
}

/** The components a run can observe: its stage kinds, then context and answer. */
export function declared(run: Row): string[] {
  const found = new Set<string>()
  for (const entry of rows(run.queries)) {
    if (!object(entry.evidence)) continue
    for (const stage of rows(entry.evidence.stages)) found.add(STAGE_COMPONENT[String(stage.kind)] ?? 'custom')
    if (object(entry.evidence.context)) found.add('context')
    const answer = object(entry.answer) ? entry.answer : {}
    if (object(answer.diagnosis) && answer.diagnosis.class) found.add('answer')
  }
  if (found.size) found.add('retrieve')
  return COMPONENTS.filter(c => found.has(c))
}

/** The scoreboard of a pipeline run, or null when it is not one. */
export function componentScoreboard(run: unknown): Scoreboard | null {
  if (!object(run) || run.format !== 'vectraudit.run') return null
  const names = declared(run)
  const all: [string, string][] = []
  const tagged = new Map<string, [string, string][]>()
  const byKind: Record<string, number> = {}
  for (const entry of rows(run.queries)) {
    const [outcome, kind] = attribute(entry)
    if (kind !== null) byKind[kind] = (byKind[kind] ?? 0) + 1
    const row: [string, string] = [String(entry.family_id || entry.query_id), outcome]
    all.push(row)
    for (const tag of Array.isArray(entry.tags) ? entry.tags.map(String) : []) tagged.set(tag, [...(tagged.get(tag) ?? []), row])
  }
  const dataset = object(run.dataset) ? run.dataset : {}
  const meta = object(dataset.meta) ? dataset.meta : {}
  const componentTags = object(meta.component_tags) ? meta.component_tags : {}
  const cohorts = Object.keys(componentTags).sort().map(tag => {
    const targets = Array.isArray(componentTags[tag]) ? (componentTags[tag] as unknown[]).map(String) : []
    const t = table(tagged.get(tag) ?? [], names)
    return { tag, targets, observable: targets.filter(x => names.includes(x)), lost_at_targets: t.components.filter(c => targets.includes(c.component)).reduce((s, c) => s + c.questions, 0), ...t }
  })
  const sortedKinds = Object.fromEntries(Object.entries(byKind).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
  return { declared: names, ...table(all, names), by_stage_kind: sortedKinds, cohorts }
}
