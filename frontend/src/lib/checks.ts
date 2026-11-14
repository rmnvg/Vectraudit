/** Readable summaries of the reports the command-line checks write as JSON. */

export const CHECK_FORMATS = [
  'vectraudit.lineage',
  'vectraudit.knowledge_freshness',
  'vectraudit.access',
  'vectraudit.access_check',
  'vectraudit.cache_probe',
  'vectraudit.migration_report',
  'vectraudit.intake',
  'vectraudit.fusion_replay',
  'vectraudit.corpus_scan',
  'vectraudit.poisoning',
  'vectraudit.promptfoo_redteam',
  'vectraudit.selectivity_sweep',
  'vectraudit.capacity',
  'vectraudit.label_audit',
  'vectraudit.embedding_drift',
  'vectraudit.component_scoreboard',
  'vectraudit.run_header',
] as const
export type CheckFormat = typeof CHECK_FORMATS[number]
export type CheckReport = { format: CheckFormat } & Record<string, unknown>
export type Verdict = 'pass' | 'fail' | 'attention'
export interface CheckSummary { title: string; verdict: Verdict; headline: string; items: string[]; notes: string[] }

type Row = Record<string, unknown>
const object = (v: unknown): v is Row => !!v && typeof v === 'object' && !Array.isArray(v)
const rows = (v: unknown): Row[] => Array.isArray(v) ? v.filter(object) : []
const text = (v: unknown): string => typeof v === 'string' ? v : typeof v === 'number' ? String(v) : ''
const list = (v: unknown): string => Array.isArray(v) ? v.map(text).filter(Boolean).join(', ') : ''
const count = (v: unknown): number => typeof v === 'number' && Number.isFinite(v) ? v : 0
const words = (v: unknown): string => text(v).replaceAll('_', ' ')
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`
const LIMIT = 25

/** Whether imported JSON is one of the check reports, with the fields its summary reads. */
export function isCheckReport(value: unknown): value is CheckReport {
  if (!object(value) || !(CHECK_FORMATS as readonly string[]).includes(text(value.format))) return false
  const arrays = (...keys: string[]) => keys.every(k => Array.isArray(value[k]))
  switch (value.format as CheckFormat) {
    case 'vectraudit.lineage': return arrays('rows', 'advice') && object(value.first_loss)
    case 'vectraudit.knowledge_freshness': return arrays('changes') && object(value.counts)
    case 'vectraudit.access': return (value.check == null || isCheckReport({ ...object(value.check) ? value.check : {}, format: 'vectraudit.access_check' })) && (value.applied == null || (object(value.applied) && Array.isArray(value.applied.affected)))
    case 'vectraudit.access_check': return arrays('violations', 'pending', 'granted_unseen')
    case 'vectraudit.cache_probe': return arrays('leaks', 'errors') && typeof value.complete === 'boolean'
    case 'vectraudit.migration_report': return arrays('concerns', 'environment') && typeof value.migration_ready === 'boolean'
    case 'vectraudit.intake': return arrays('likely_recurrences') && object(value.queue)
    case 'vectraudit.fusion_replay': return arrays('configurations', 'questions') && object(value.summary) && typeof value.metric === 'string'
    case 'vectraudit.corpus_scan': return arrays('detectors', 'review', 'documents') && object(value.summary)
    case 'vectraudit.promptfoo_redteam': return object(value.plugins) && object(value.summary)
    case 'vectraudit.selectivity_sweep': return arrays('runs') && typeof value.param === 'string' && typeof value.target === 'number'
    case 'vectraudit.capacity': return arrays('levels', 'notes') && object(value.plan) && typeof value.status === 'string'
    case 'vectraudit.label_audit': return arrays('suspects') && object(value.signals) && object(value.summary)
    case 'vectraudit.embedding_drift': return object(value.result) && object(value.reference) && object(value.current) && typeof value.status === 'string'
    case 'vectraudit.component_scoreboard': return arrays('components', 'declared', 'cohorts') && object(value.observed) && object(value.kept)
    case 'vectraudit.run_header': return object(value.run) && typeof value.questions === 'number' && typeof value.file === 'string'
    case 'vectraudit.poisoning': return arrays('documents', 'targets', 'failures') && object(value.summary) && object(value.attacks) && typeof value.verdict === 'string'
  }
}

function lineage(r: Row): CheckSummary {
  const losses = rows(r.rows).filter(row => row.loss)
  const ingestion = count(r.ingestion_losses)
  return {
    title: 'Source lineage',
    verdict: ingestion ? 'fail' : losses.length ? 'attention' : 'pass',
    headline: `${plural(count(r.passages), 'labelled passage')} followed; ${ingestion} lost during ingestion.`,
    items: [
      ...Object.entries(object(r.first_loss) ? r.first_loss : {}).map(([stage, n]) => `${count(n)} first lost at ${words(stage)}`),
      ...losses.slice(0, LIMIT).map(row => `${text(row.query_id)}: ${text(row.source_id)} characters ${text(row.start)} to ${text(row.end)} first lost at ${words(row.loss)}`),
    ],
    notes: (r.advice as unknown[]).map(text).filter(Boolean),
  }
}

function freshness(r: Row): CheckSummary {
  const counts = object(r.counts) ? r.counts : {}
  const served = object(r.served) ? r.served : {}
  const failing = count(r.failing)
  return {
    title: 'Knowledge freshness',
    verdict: failing ? 'fail' : count(counts.pending) ? 'attention' : 'pass',
    headline: `${plural(rows(r.changes).length, 'change')} as of ${text(r.as_of)}: ${Object.entries(counts).map(([s, n]) => `${count(n)} ${s}`).join(', ') || 'none'}.`,
    items: [
      ...rows(r.changes).filter(c => c.status === 'lingering' || c.status === 'overdue').slice(0, LIMIT).map(c => `${text(c.source_id)} (${text(c.change)}): ${text(c.status)}, due ${text(c.deadline)}`),
      ...rows(served.stale).slice(0, LIMIT).map(s => `${text(s.query_id)}: stale ${text(s.chunk_id)} reached the ${text(s.where)}`),
      ...rows(served.conflicts).slice(0, LIMIT).map(c => `${text(c.query_id)}: ${text(c.source_id)} delivered in revisions ${list(c.revisions)}`),
    ],
    notes: Array.isArray(served.historical_questions_exempt) && served.historical_questions_exempt.length ? [`${plural(served.historical_questions_exempt.length, 'question')} tagged historical were not checked for stale content.`] : [],
  }
}

function accessCheck(r: Row): CheckSummary {
  const violations = rows(r.violations)
  const pending = rows(r.pending)
  return {
    title: 'Access changes',
    verdict: violations.length ? 'fail' : pending.length ? 'attention' : 'pass',
    headline: `${plural(count(r.questions_checked), 'question')} affected; ${plural(violations.length, 'revoked document')} still served.`,
    items: [
      ...violations.slice(0, LIMIT).map(v => `${text(v.query_id)}: ${text(v.doc_id)} reached ${list(v.reached)}`),
      ...rows(r.granted_unseen).slice(0, LIMIT).map(g => `${text(g.query_id)}: granted ${text(g.doc_id)} was not delivered`),
    ],
    notes: pending.length ? [`${plural(pending.length, 'change')} still inside the propagation window; not failures yet.`] : [],
  }
}

function access(r: Row): CheckSummary {
  const check = object(r.check) ? accessCheck(r.check) : null
  const applied = object(r.applied) ? r.applied : null
  const review = applied ? rows(applied.labels_to_review) : []
  const appliedLine = applied ? `Expected access applied to ${plural(rows(applied.affected).length, 'question')}.` : ''
  return {
    title: 'Access changes',
    verdict: check?.verdict ?? (review.length ? 'attention' : 'pass'),
    headline: [check?.headline, appliedLine].filter(Boolean).join(' '),
    items: [...(check?.items ?? []), ...review.slice(0, LIMIT).map(x => `Review ${text(x.query_id)}: revoked label(s) ${list(x.removed_labels)} removed`)],
    notes: check?.notes ?? [],
  }
}

function cacheProbe(r: Row): CheckSummary {
  const leaks = rows(r.leaks)
  return {
    title: 'Cache isolation',
    verdict: leaks.length ? 'fail' : r.complete ? 'pass' : 'attention',
    headline: `${plural(count(r.pairs), 'target/primer pair')}, ${plural(count(r.calls), 'call')}: ${plural(leaks.length, 'leak')}.`,
    items: leaks.slice(0, LIMIT).map(l => `${text(l.target)} after ${text(l.primer)}: ${text(l.doc_id)} reached ${list(l.reached)} (${l.observed === 'same_as_primer' ? "identical to the primer's answer" : "not the primer's answer: the access check let it through"})`),
    notes: r.complete ? [] : ['Some calls failed, so the probe is incomplete: it proves nothing about isolation.'],
  }
}

function migration(r: Row): CheckSummary {
  const current = object(r.current) ? r.current : {}
  const proposed = object(r.proposed) ? r.proposed : {}
  const differs = Array.isArray(r.environment_differs) ? r.environment_differs.map(text) : []
  return {
    title: `Migration: ${text(current.label) || text(current.run_id)} → ${text(proposed.label) || text(proposed.run_id)}`,
    verdict: r.migration_ready ? 'pass' : 'fail',
    headline: `${r.migration_ready ? 'Ready to migrate' : 'Not ready to migrate'}: release policy verdict ${text(r.verdict)}, ${plural((r.concerns as unknown[]).length, 'concern')}.`,
    items: (r.concerns as unknown[]).map(text).filter(Boolean),
    notes: differs.length ? [`Environment differs in ${differs.join(', ')}; latency is compared only as the policy allows.`] : [],
  }
}

function intake(r: Row): CheckSummary {
  const queue = object(r.queue) ? r.queue : {}
  const recurrences = rows(r.likely_recurrences)
  return {
    title: 'Production intake',
    verdict: recurrences.length ? 'attention' : 'pass',
    headline: `Took ${plural(count(r.traces_taken), 'new trace')}; ${plural(recurrences.length, 'likely recurrence')}; ${count(queue.pending)} questions awaiting review (${count(queue.independent_sample)} independent sample).`,
    items: recurrences.slice(0, LIMIT).map(x => {
      const reopen = Array.isArray(x.reopen) && x.reopen.length ? `; reopen verified case ${list(x.reopen)}` : ''
      return `${text(x.query_id)} at ${text(x.at)} delivered none of its labelled documents${reopen}`
    }),
    notes: [
      count(queue.unassigned) ? `${plural(count(queue.unassigned), 'question')} without a reviewer.` : '',
      count(queue.new_topic) ? `${plural(count(queue.new_topic), 'pending question')} from a new topic (embedding drift), raised for review.` : '',
    ].filter(Boolean),
  }
}

function fusionReplay(r: Row): CheckSummary {
  const summary = object(r.summary) ? r.summary : {}
  const downstream = object(summary.downstream) ? summary.downstream : {}
  const best = object(r.best) ? r.best : null
  const metric = text(r.metric)
  const configs = rows(r.configurations)
  const forbidden = configs.reduce((sum, c) => sum + count(c.forbidden_questions), 0)
  const signed = (v: unknown) => `${count(v) >= 0 ? '+' : '−'}${Math.abs(count(v)).toFixed(3)}`
  const gains = !!best && count(best.ci_low) > 0
  const items = configs.flatMap(config => Object.entries(object(config.results) ? config.results : {}).map(([view, data]) => {
    const stats = object(data) && object(data.metrics) && object(data.metrics[metric]) ? data.metrics[metric] : {}
    const families = object(data) ? count(data.n_families) : 0
    return `${text(config.id)} (${words(view)}): ${metric} ${count(stats.replay).toFixed(3)} against ${count(stats.recorded).toFixed(3)} recorded, ${signed(stats.delta)} (95% interval ${signed(stats.ci_low)} to ${signed(stats.ci_high)}, ${plural(families, 'family', 'families')})`
  }))
  const excluded = Object.entries(object(summary.excluded) ? summary.excluded : {}).map(([why, n]) => `${count(n)} ${words(why)}`)
  const notes = [
    text(r.note),
    `Excluded: ${excluded.join(', ') || 'none'}. After fusion: ${count(downstream.none)} exact, ${count(downstream.filter)} bounded by filters, ${count(downstream.unknown)} unknown (a reranker or custom stage follows).`,
    count(summary.truncated_inputs) ? `${plural(count(summary.truncated_inputs), 'question')} recorded fewer than k hits from an input stage, so deeper candidates may be missing.` : '',
    forbidden ? `A forbidden document is in a replayed top results for ${plural(forbidden, 'question')}; later stages were not replayed, so its absence elsewhere proves nothing.` : '',
    object(r.next_step) ? `Next: ${text(r.next_step.text)}` : '',
  ].filter(Boolean)
  return {
    title: 'Fusion replay',
    verdict: forbidden || gains ? 'attention' : 'pass',
    headline: best
      ? `${plural(count(summary.replayed), 'question')} of ${count(summary.questions_in_run)} replayed (a hypothesis, not a measurement); best ${text(best.configuration)}: ${metric} ${signed(best.delta)} against the recorded ranking${gains ? '' : ', not clearly better'}.`
      : `${plural(count(summary.replayed), 'question')} replayed, but none could be compared with the recorded ranking because a reranker or custom stage follows fusion.`,
    items,
    notes,
  }
}

function corpusScan(r: Row): CheckSummary {
  const summary = object(r.summary) ? r.summary : {}
  const flagged = count(summary.flagged)
  const explain = Object.fromEntries(rows(r.detectors).map(d => [text(d.id), text(d.explanation)]))
  const byDetector = Object.entries(object(summary.documents_by_detector) ? summary.documents_by_detector : {}).sort((a, b) => count(b[1]) - count(a[1]))
  return {
    title: 'Corpus injection scan',
    verdict: flagged ? 'attention' : 'pass',
    headline: `${plural(count(summary.documents), 'document')} scanned; ${flagged} flagged. Findings are hypotheses for a person to read, not proof either way.`,
    items: [
      ...byDetector.map(([name, n]) => `${plural(count(n), 'document')}: ${words(name)}. ${explain[name] ?? ''}`.trim()),
      ...rows(r.review).slice(0, LIMIT).map(d => `Read ${text(d.doc_id)} (row ${text(d.row)}, ${text(d.severity)} severity, ${plural(count(d.findings), 'finding')})`),
    ],
    notes: [
      text(r.note),
      flagged ? '' : 'No findings does not show the corpus is safe: the scan only finds known traces of injected text.',
    ].filter(Boolean),
  }
}

function poisoning(r: Row): CheckSummary {
  const summary = object(r.summary) ? r.summary : {}
  const attacks = object(r.attacks) ? r.attacks : {}
  const percent = (v: unknown) => `${Math.round(count(v) * 100)}%`
  const verdict = text(r.verdict)
  const rate = (value: unknown) => {
    const x = object(value) ? value : {}
    return `${count(x.flagged)} of ${plural(count(x.families), 'family', 'families')} (${x.rate == null ? 'n/a' : percent(x.rate)}, 95% interval ${percent(x.ci_low)} to ${percent(x.ci_high)})`
  }
  const displaced = object(summary.displaced) ? summary.displaced : {}
  const collateral = object(r.collateral) ? r.collateral : {}
  const overlay = object(r.overlay) ? r.overlay : {}
  const followed = object(r.followed_injection) ? r.followed_injection : null
  return {
    title: 'Poisoning check',
    verdict: verdict === 'fail' ? 'fail' : verdict === 'pass' ? 'pass' : 'attention',
    headline: `${verdict === 'inconclusive' ? 'Inconclusive' : verdict === 'pass' ? 'No failure' : 'Failed'} on ${plural(count(summary.targets_measured), 'target question')} and ${plural(count(summary.controls), 'control question')} (exploratory, not a release approval).`,
    items: [
      ...rows(r.failures).map(f => text(f.message)),
      ...Object.entries(attacks).map(([name, a]) => {
        const x = object(a) ? a : {}
        return `${words(name)}: retrieved for ${rate(x.retrieved)}; reached the model for ${count(object(x.reached_model_context) ? x.reached_model_context.flagged : 0)}`
      }),
      ...(followed ? [`Answers that followed an injected instruction (canary found): ${rate(followed)} of the ${plural(count(followed.questions_checked), 'checked question')}`] : []),
      `Answering evidence displaced: ${rate(displaced)}`,
      `Poisoned documents on control questions: ${rate(collateral)}`,
    ],
    notes: [
      text(r.note),
      ...(Array.isArray(r.reasons_inconclusive) ? r.reasons_inconclusive.map(x => `Inconclusive: ${text(x)}`) : []),
      text(overlay.warning),
    ].filter(Boolean),
  }
}

function promptfooRedteam(r: Row): CheckSummary {
  const summary = object(r.summary) ? r.summary : {}
  const plugins = Object.entries(object(r.plugins) ? r.plugins : {})
  const failed = plugins.reduce((sum, [, p]) => sum + (object(p) ? count(p.failed) : 0), 0)
  const percent = (v: unknown) => `${Math.round(count(v) * 100)}%`
  const skipped = Object.entries(object(summary.skipped_other_plugins) ? summary.skipped_other_plugins : {})
  return {
    title: 'promptfoo red team (RAG plugins)',
    verdict: failed ? 'attention' : 'pass',
    headline: `${count(summary.kept)} of ${plural(count(summary.results), 'result')} kept: ${failed} failed. Graded by a model and exploratory, not a release approval.`,
    items: plugins.map(([name, plugin]) => {
      const p = object(plugin) ? plugin : {}
      const strategies = Object.entries(object(p.by_strategy) ? p.by_strategy : {}).map(([s, v]) => `${s} ${object(v) ? count(v.failed) : 0}/${object(v) ? count(v.tests) : 0}`).join(', ')
      const rate = p.failure_rate == null ? 'n/a' : percent(p.failure_rate)
      return `${name}: ${count(p.failed)} of ${plural(count(p.tests), 'test')} failed (${rate}, 95% interval ${percent(p.ci_low)} to ${percent(p.ci_high)})${strategies ? `; by strategy: ${strategies}` : ''}${count(p.errored) ? `; ${count(p.errored)} errored` : ''}`
    }),
    notes: [
      text(r.note),
      `${count(summary.matched_by_query_id) + count(summary.matched_by_trace_id)} tests named a benchmark question or trace; ${count(summary.unmatched)} are counted here only.`,
      skipped.length ? `Skipped other plugins: ${skipped.map(([k, n]) => `${k} (${count(n)})`).join(', ')}.` : '',
    ].filter(Boolean),
  }
}

function selectivitySweep(r: Row): CheckSummary {
  const runs = rows(r.runs)
  const target = count(r.target)
  const param = text(r.param)
  const percent = (v: unknown) => count(v) >= 0.1 ? `${Math.round(count(v) * 100)}%` : `${(count(v) * 100).toFixed(count(v) < 0.01 ? 2 : 1)}%`
  const reached = runs.filter(run => run.recommended != null)
  const missed = runs.filter(run => run.recommended == null)
  return {
    title: 'Recall by filter pass rate',
    verdict: missed.length ? 'attention' : 'pass',
    headline: `${reached.length} of ${plural(runs.length, 'filter setting')} reached recall ${target.toFixed(2)} within the ${param} range swept (exploratory, not a release approval).`,
    items: runs.map(run => {
      const f = object(run.filter) ? run.filter : {}
      const points = rows(run.points).map(p => `${param} ${text(p.value)}: ${count(p.recall).toFixed(3)}`).join(', ')
      const smallest = run.recommended == null ? `recall ${target.toFixed(2)} not reached` : `${param} ${text(run.recommended)} is the smallest that reaches ${target.toFixed(2)}`
      return `${text(f.kind) === 'all' ? 'Unfiltered' : text(f.label)}: ${percent(run.pass_rate)} of the corpus passes (${count(run.allowed_count).toLocaleString()} documents); ${points}; ${smallest}`
    }),
    notes: [
      'Pass rate is the fraction of documents that satisfy the filter, not the fraction removed.',
      `Each filter is measured against the exact neighbours among the documents it allows (${text(r.truth_source)} vectors).`,
      ...rows(r.suggestions).filter(x => x.kind !== 'FILTER_UNMEASURED').map(x => text(x.detail)),
      ...rows(r.suggestions).filter(x => x.kind === 'FILTER_UNMEASURED').map(x => text(x.detail)),
    ].filter(Boolean),
  }
}

function capacity(r: Row): CheckSummary {
  const levels = rows(r.levels)
  const status = text(r.status)
  const backend = object(r.backend) ? r.backend : {}
  const safety = object(r.safety) ? r.safety : {}
  const plan = object(r.plan) ? r.plan : {}
  const comparison = object(r.comparison) ? r.comparison : null
  const ms = (v: unknown) => v == null ? 'n/a' : `${count(v).toFixed(1)} ms`
  const failing = levels.filter(l => count(l.error_rate) > 0)
  const error = object(r.error) ? r.error : null
  return {
    title: 'Capacity curve',
    verdict: status === 'failed' ? 'fail' : status === 'complete' && !failing.length ? 'pass' : 'attention',
    headline: status === 'complete'
      ? `${plural(levels.length, 'concurrency level')} measured on ${text(backend.backend)} target ${text(safety.target)} at up to ${text(plan.max_rps)} requests per second (a probe of staging, not a load test).`
      : `${status === 'cancelled' ? 'Cancelled' : 'Failed'} after ${plural(levels.length, 'concurrency level')}${error ? `: ${text(error.message)}` : ''}.`,
    items: levels.map(l => {
      const recall = object(l.recall) ? l.recall : {}
      return `Concurrency ${text(l.concurrency)}: ${l.achieved_qps == null ? 'n/a' : count(l.achieved_qps).toFixed(1)} requests/s, p50 ${ms(l.p50_ms)}, p95 ${ms(l.p95_ms)}, p99 ${ms(l.p99_ms)}, ${l.error_rate == null ? 'n/a' : `${(count(l.error_rate) * 100).toFixed(1)}%`} errors, recall ${recall.mean == null ? 'unknown' : count(recall.mean).toFixed(3)}${l.at_ceiling ? ' (reached the request ceiling)' : ''}`
    }),
    notes: [
      ...(Array.isArray(r.notes) ? r.notes.map(text) : []),
      ...(comparison && comparison.comparable === false && Array.isArray(comparison.reasons) ? comparison.reasons.map(x => `Not compared with the baseline: ${text(x)}`) : []),
    ].filter(Boolean),
  }
}

const LABEL_STATES: Record<string, string> = { relevant: 'labelled relevant', judged_not_relevant: 'judged not relevant', unjudged: 'unjudged' }

function labelAudit(r: Row): CheckSummary {
  const summary = object(r.summary) ? r.summary : {}
  const suspects = rows(r.suspects)
  const signals = Object.entries(object(r.signals) ? r.signals : {}).filter((entry): entry is [string, Row] => object(entry[1]))
  const measured = signals.filter(([, s]) => s.status === 'measured').map(([name]) => words(name))
  return {
    title: 'Label audit',
    verdict: suspects.length ? 'attention' : 'pass',
    headline: `${plural(count(summary.suspects), 'label')} to review on ${plural(count(summary.questions), 'question')}, from ${measured.join(', ') || 'no signal'}. Suspects are hypotheses for a reviewer; nothing was changed.`,
    items: suspects.slice(0, LIMIT).map(s => {
      const label = object(s.label) ? s.label : {}
      const grade = label.grade == null ? '' : ` (grade ${text(label.grade)})`
      return `${text(s.query_id)} / ${text(s.doc_id)}: ${LABEL_STATES[text(label.state)] ?? words(label.state)}${grade}, ${words(s.kind)}; ${rows(s.signals).map(x => words(x.signal)).join(', ')}`
    }),
    notes: [
      ...signals.filter(([, s]) => s.status !== 'measured').map(([name, s]) => `${words(name)} not measured: ${text(s.reason)}`),
      suspects.length > LIMIT ? `${suspects.length - LIMIT} more suspects in the file.` : '',
      suspects.length ? '' : 'No suspects does not show the labels are right: these signals only catch labels that disagree with the runs, duplicates or annotators.',
      text(r.note),
    ].filter(Boolean),
  }
}

function embeddingDrift(r: Row): CheckSummary {
  const result = object(r.result) ? r.result : {}
  const reference = object(r.reference) ? r.reference : {}
  const current = object(r.current) ? r.current : {}
  const where = object(r.where) ? r.where : {}
  const sampled = object(result.sampled) ? result.sampled : null
  const drift = r.status === 'drift'
  const noun = r.kind === 'corpus' ? 'documents' : 'questions'
  const value = `${text(result.statistic)} ${text(result.value)}${result.p_value == null ? '' : ` (p = ${text(result.p_value)})`}`
  return {
    title: 'Embedding drift',
    verdict: drift ? 'attention' : 'pass',
    headline: `${drift ? 'Drift' : 'No drift'}: ${count(result.n_current)} ${text(current.label)} ${noun} against ${count(result.n_reference)} ${text(reference.label)} ${noun}; ${value}, ${text(result.rule)}.`,
    items: rows(where.groups).slice(0, LIMIT).map(g => `${text(g.message)} (hypothesis)`),
    notes: [
      ...(Array.isArray(result.warnings) ? result.warnings.map(text) : []),
      sampled ? `Sampled ${text(sampled.reference)} and ${text(sampled.current)} vectors (seed ${text(result.seed)}).` : '',
      text(where.note),
      text(r.note),
    ].filter(Boolean),
  }
}

function componentScoreboard(r: Row): CheckSummary {
  const observed = object(r.observed) ? r.observed : {}
  const components = rows(r.components)
  const kept = object(r.kept) ? r.kept : {}
  const applicable = object(r.not_applicable) ? r.not_applicable : {}
  const lost = components.reduce((sum, c) => sum + count(c.questions), 0)
  const worst = components.reduce<Row | null>((a, c) => count(c.rate) > count(a?.rate) ? c : a, null)
  const pct = (v: unknown) => `${(count(v) * 100).toFixed(1)}%`
  return {
    title: 'Component scoreboard',
    verdict: lost ? 'attention' : 'pass',
    headline: Array.isArray(r.declared) && r.declared.length
      ? `${plural(count(observed.questions), 'observed question')} in ${plural(count(observed.families), 'family', 'families')}; ${lost} lost their required evidence${worst && count(worst.questions) ? `, most at ${text(worst.component)} (${pct(worst.rate)})` : ''}. Information only.`
      : 'The run recorded no stages, model context or answer diagnosis, so no loss can be placed.',
    items: [
      ...components.map(c => `${text(c.component)}: ${pct(c.rate)} (95% interval ${pct(c.ci_low)} to ${pct(c.ci_high)}), ${plural(count(c.questions), 'question')}`),
      `kept: ${pct(kept.rate)}, ${plural(count(kept.questions), 'question')}`,
      ...rows(r.cohorts).map(c => `Cohort ${text(c.tag)} (meant to exercise ${list(c.targets)}): ${count(c.lost_at_targets)} of ${count((object(c.observed) ? c.observed : {}).questions)} observed questions lost their evidence there`),
    ],
    notes: [
      `Not observed (never assigned): ${count(r.not_observed)}; not applicable: ${count(applicable.unlabelled)} unlabelled, ${count(applicable.no_answer)} no-answer; errored: ${count(r.errored)}.`,
      text(r.note),
    ].filter(Boolean),
  }
}

function runHeader(r: Row): CheckSummary {
  const run = object(r.run) ? r.run : {}
  const summary = object(run.summary) ? run.summary : {}
  const metrics = object(summary.metrics) ? summary.metrics : {}
  const latency = object(summary.latency_ms) ? summary.latency_ms : {}
  const dataset = object(run.dataset) ? run.dataset : {}
  const status = text(run.status)
  const mib = (count(r.bytes) / 2 ** 20).toFixed(1)
  return {
    title: `Run summary: ${text(run.label) || text(r.file)}`,
    verdict: status === 'complete' ? 'pass' : status === 'failed' ? 'fail' : 'attention',
    headline: `${text(r.file)}: ${status}, ${plural(count(r.questions), 'question')} (${mib} MiB), every one validated without loading the run.${typeof latency.p95 === 'number' ? ` p95 latency ${latency.p95.toFixed(1)} ms.` : ''}`,
    items: [
      ...Object.entries(metrics).filter(([, v]) => typeof v === 'number').map(([name, v]) => `${words(name)}: ${(v as number).toFixed(3)}`),
      `Dataset ${text(dataset.sha256).slice(0, 12)}, k = ${text(object(run.search) ? run.search.k : '')}`,
    ],
    notes: [
      'Too large to open here, so only its summary is shown. Analyse every question in a notebook with `vectraudit runs table`.',
      status === 'complete' ? '' : 'An incomplete run is never a pass in a release check.',
    ].filter(Boolean),
  }
}

/** The verdict, headline and findings of one check report. */
export function summarizeCheck(report: CheckReport): CheckSummary {
  switch (report.format) {
    case 'vectraudit.lineage': return lineage(report)
    case 'vectraudit.knowledge_freshness': return freshness(report)
    case 'vectraudit.access': return access(report)
    case 'vectraudit.access_check': return accessCheck(report)
    case 'vectraudit.cache_probe': return cacheProbe(report)
    case 'vectraudit.migration_report': return migration(report)
    case 'vectraudit.intake': return intake(report)
    case 'vectraudit.fusion_replay': return fusionReplay(report)
    case 'vectraudit.corpus_scan': return corpusScan(report)
    case 'vectraudit.poisoning': return poisoning(report)
    case 'vectraudit.promptfoo_redteam': return promptfooRedteam(report)
    case 'vectraudit.selectivity_sweep': return selectivitySweep(report)
    case 'vectraudit.capacity': return capacity(report)
    case 'vectraudit.label_audit': return labelAudit(report)
    case 'vectraudit.embedding_drift': return embeddingDrift(report)
    case 'vectraudit.component_scoreboard': return componentScoreboard(report)
    case 'vectraudit.run_header': return runHeader(report)
  }
}
