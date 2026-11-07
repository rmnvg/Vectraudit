import { useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { api } from '../api/client'
import { useSelectivityAdvice } from '../api/hooks'
import { useJob } from '../api/jobs'
import type { Info, SelectivityRequest, SelectivityResult, SweepParam } from '../api/types'
import { Field } from '../components/form'
import { JobStatus } from '../components/JobStatus'
import { Banner, Card, Spinner } from '../components/ui'
import { parseFilterLines, parsePassRates } from '../lib/filterText'
import { prefersLogAxis } from '../lib/tuner'

const AXIS = { fill: 'var(--muted)', fontSize: 11 }
const INPUT =
  'tabular rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1'

const percent = (rate: number) => (rate >= 0.1 ? `${(rate * 100).toFixed(0)}%` : `${(rate * 100).toFixed(rate < 0.01 ? 2 : 1)}%`)

/** The unfiltered run is series-1; the tighter the filter, the closer to series-2. */
const lineColour = (index: number, total: number) =>
  `color-mix(in oklab, var(--series-2) ${total <= 1 ? 0 : Math.round((index / (total - 1)) * 100)}%, var(--series-1))`

/**
 * The same sweep at several pass rates. A pass rate is the share of the corpus that PASSES the
 * filter, and each is measured against the exact neighbours among the documents it allows.
 */
export function TunerSelectivity({
  info,
  param,
  values,
  k,
  nQueries,
  repeats,
  seed,
  target,
}: {
  info: Info
  param: SweepParam
  /** Parsed from the sweep form; null when it holds something that is not a value list. */
  values: number[] | null
  k: number
  nQueries: number
  repeats: number
  seed: number
  target: number
}) {
  const hasMetadata = (info.inputs.metadata_rows ?? 0) > 0
  const [passText, setPassText] = useState('0.5, 0.1, 0.01')
  const [filterText, setFilterText] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const { job, poll, start, cancel, missing } = useJob({
    kind: 'selectivity',
    jobId,
    start: api.startSelectivity,
    fetch: api.selectivity,
    cancel: api.cancelSelectivity,
    onStarted: (j) => setJobId(j.job_id),
  })

  const run = () => {
    if (!values) return setFormError('Fix the sweep values above first.')
    const passRates = passText.trim() ? parsePassRates(passText) : []
    if (passRates === null) {
      return setFormError('Pass rates are fractions that pass, above 0 up to 1 (at most 8), e.g. 0.5, 0.1, 0.01.')
    }
    const filters = filterText.trim() ? parseFilterLines(filterText) : []
    if (filters === null) {
      return setFormError('Write one filter per line (at most 4) as key=value; join conditions with ";".')
    }
    if (passRates.length === 0 && filters.length === 0) return setFormError('Give a pass rate or a filter.')
    setFormError(null)
    const req: SelectivityRequest = {
      param,
      values,
      k,
      n_queries: nQueries,
      repeats,
      seed,
      pass_rates: passRates,
      filters,
    }
    start.mutate(req)
  }

  const result = job?.status === 'done' ? job.result : null
  const running = job?.status === 'running'

  return (
    <Card
      title="Recall by filter pass rate"
      subtitle="Repeat the sweep with a filter applied inside the search, measured against the exact neighbours among the documents that pass"
    >
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault()
          run()
        }}
      >
        <Field label="Pass rates (fraction that passes)" className="min-w-64 flex-1">
          <input value={passText} onChange={(e) => setPassText(e.target.value)} className={INPUT} />
        </Field>
        {hasMetadata && (
          <Field label='Metadata filters, one per line (e.g. lang=de;year={"gte":2020})' className="min-w-72 flex-1">
            <textarea
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              rows={2}
              className={INPUT}
              spellCheck={false}
            />
          </Field>
        )}
        <button
          type="submit"
          disabled={start.isPending || running}
          className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {running ? 'Sweeping…' : 'Run filtered sweep'}
        </button>
        {running && jobId && (
          <button
            type="button"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate(jobId)}
            className="rounded-lg border border-line px-4 py-2 text-sm text-ink"
          >
            Cancel
          </button>
        )}
      </form>
      {formError && <p className="mt-2 text-sm text-critical">{formError}</p>}
      <p className="mt-2 text-xs text-muted">
        Pass rate is the share of documents that pass, not the share removed: 0.01 keeps 1%. The
        id filter keeps the highest ids, so it isolates selectivity from how a real filter
        correlates with the vectors.
        {!hasMetadata && ' Start with --meta to filter on metadata fields.'}
      </p>
      <div className="mt-3">
        <JobStatus
          noun="Filtered sweep"
          job={job}
          missing={missing}
          errors={[start.error, cancel.error, missing ? null : poll.error]}
          onRunAgain={run}
          running={start.isPending}
        />
      </div>
      {result && jobId && <SelectivityResults result={result} jobId={jobId} target={target} />}
    </Card>
  )
}

function SelectivityResults({ result, jobId, target }: { result: SelectivityResult; jobId: string; target: number }) {
  const advice = useSelectivityAdvice(jobId, target)
  const values = result.runs[0]?.points.map((p) => p.value) ?? []
  const logX = prefersLogAxis(values)
  const rows = values.map((v) => ({
    value: v,
    ...Object.fromEntries(result.runs.map((r) => [r.label, r.points.find((p) => p.value === v)?.recall ?? null])),
  }))
  return (
    <div className="mt-4 flex flex-col gap-4">
      {result.truth_source === 'reconstructed' && (
        <Banner tone="warning">Ground truth computed on reconstructed vectors; PQ/SQ error is not measured.</Banner>
      )}
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis
              dataKey="value"
              type="number"
              scale={logX ? 'log' : 'linear'}
              domain={['dataMin', 'dataMax']}
              ticks={values}
              tick={AXIS}
              label={{ value: result.param, position: 'insideBottom', offset: -2, ...AXIS }}
            />
            <YAxis domain={[0, 1]} tick={AXIS} tickFormatter={(v: number) => v.toFixed(1)} />
            <Tooltip
              formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))}
              labelFormatter={(v) => `${result.param} ${v}`}
              contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', fontSize: 12 }}
            />
            <ReferenceLine y={target} stroke="var(--muted)" strokeDasharray="4 4" />
            <Legend />
            {result.runs.map((r, i) => (
              <Line
                key={r.label}
                dataKey={r.label}
                name={`${r.kind === 'all' ? 'unfiltered' : percent(r.pass_rate)} · ${r.label}`}
                stroke={lineColour(i, result.runs.length)}
                strokeWidth={2}
                dot={{ r: 3 }}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="overflow-x-auto">
        <table className="tabular w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-medium">Filter</th>
              <th className="py-1 pr-3 text-right font-medium">Pass rate</th>
              <th className="py-1 pr-3 text-right font-medium">Documents</th>
              {values.map((v) => (
                <th key={v} className="py-1 pr-3 text-right font-medium">
                  {result.param} {v}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.runs.map((r) => (
              <tr key={r.label} className="border-t border-line">
                <td className="py-1 pr-3 text-ink">{r.kind === 'all' ? 'unfiltered' : r.label}</td>
                <td className="py-1 pr-3 text-right text-ink-2">{percent(r.pass_rate)}</td>
                <td className="py-1 pr-3 text-right text-ink-2">{r.allowed_count.toLocaleString()}</td>
                {values.map((v) => {
                  const p = r.points.find((q) => q.value === v)
                  return (
                    <td key={v} className={`py-1 pr-3 text-right ${p && p.recall >= target ? 'text-ink' : 'text-muted'}`}>
                      {p ? p.recall.toFixed(3) : ''}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {advice.isPending ? (
        <Spinner label="Reading the measurements" />
      ) : advice.isError ? (
        <p className="text-sm text-critical">{advice.error.message}</p>
      ) : advice.data && advice.data.suggestions.length > 0 ? (
        <ol className="flex flex-col gap-3">
          {advice.data.suggestions.map((s) => (
            <li key={s.kind + s.title}>
              <h3 className="text-sm font-medium text-ink">{s.title}</h3>
              <p className="mt-1 text-sm text-ink-2">{s.detail}</p>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-sm text-ink-2">
          Every pass rate reaches recall {target.toFixed(2)} somewhere in the sweep range.
        </p>
      )}
    </div>
  )
}
