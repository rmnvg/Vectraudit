/** Judge-corrected correct rates from a run's attached answers (core/evaluation/answer_estimates.py). */

export interface RateInterval { estimate: number; ci_low: number; ci_high: number; n: number; method: string }
export interface EvaluatorEstimate {
  status: 'ok' | 'absent'
  reason: string | null
  judge_only: { estimate: number | null; families: number; note?: string }
  human_only?: RateInterval
  corrected?: RateInterval & { n_unlabelled: number; lambda: number; warnings: string[] }
  agreement_on_sample?: number
  sample_families?: number
  frame_families?: number
}
export interface Estimates {
  metric: string
  definition: string
  sample: { sha256: string; size: number; frame: number } | null
  by_evaluator: Record<string, EvaluatorEstimate>
}

const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const percent = (v: number) => `${Math.round(v * 100)}%`
const range = (r: RateInterval) => `${percent(r.ci_low)} to ${percent(r.ci_high)}`

/** The estimates block of an attached-answers summary, or null when absent or malformed. */
export function readEstimates(answers: unknown): Estimates | null {
  if (!object(answers) || !object(answers.estimates)) return null
  const e = answers.estimates
  if (!object(e.by_evaluator) || typeof e.metric !== 'string') return null
  return e as unknown as Estimates
}

export interface EstimateRow { evaluator: string; ok: boolean; judgeOnly: string; humanOnly: string | null; corrected: string | null; detail: string }

/** One row per model evaluator: the three numbers side by side, or the reason there is no corrected one. */
export function estimateRows(estimates: Estimates): EstimateRow[] {
  return Object.entries(estimates.by_evaluator).map(([evaluator, row]) => {
    const judge = row.judge_only.estimate == null ? 'n/a' : percent(row.judge_only.estimate)
    if (row.status !== 'ok' || !row.human_only || !row.corrected) {
      return { evaluator, ok: false, judgeOnly: judge, humanOnly: null, corrected: null, detail: row.reason ?? 'No corrected estimate.' }
    }
    return {
      evaluator,
      ok: true,
      judgeOnly: judge,
      humanOnly: `${percent(row.human_only.estimate)} (${range(row.human_only)})`,
      corrected: `${percent(row.corrected.estimate)} (${range(row.corrected)})`,
      detail: `${row.sample_families} of ${row.frame_families} families reviewed by people; the judge agreed with them on ${percent(row.agreement_on_sample ?? 0)}.${row.corrected.warnings.length ? ` ${row.corrected.warnings.join(' ')}` : ''}`,
    }
  })
}
