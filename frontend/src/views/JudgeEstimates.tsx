import { estimateRows, readEstimates } from '../lib/estimates'

const panel = 'rounded-2xl border border-line bg-surface p-5'

/** The judge-only, people-only and corrected correct rates for each model evaluator of a run. */
export default function JudgeEstimates({ answers }: { answers: unknown }) {
  const estimates = readEstimates(answers)
  if (!estimates) return null
  const rows = estimateRows(estimates)
  if (!rows.length) return null
  return (
    <section className={panel} aria-label="Judge-corrected estimates">
      <h2 className="font-semibold">Correct answers, judge-corrected</h2>
      <p className="mt-1 text-xs text-muted">{estimates.definition} Exploratory: it describes the rate, never a single question, and no release policy reads it.</p>
      <div className="mt-3 overflow-x-auto">
        <table className="tabular w-full text-left text-sm">
          <thead className="text-xs text-muted"><tr><th className="py-1 pr-3 font-medium">Judge</th><th className="py-1 pr-3 font-medium">Judge only (biased)</th><th className="py-1 pr-3 font-medium">People only</th><th className="py-1 pr-3 font-medium">Corrected (95%)</th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.evaluator} className="border-t border-line align-top">
                <td className="py-2 pr-3 font-mono text-xs">{r.evaluator}</td>
                <td className="py-2 pr-3">{r.judgeOnly}</td>
                <td className="py-2 pr-3">{r.humanOnly ?? '—'}</td>
                <td className="py-2 pr-3 font-semibold">{r.corrected ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.map(r => <p key={r.evaluator} className="mt-2 text-xs text-muted"><b>{r.evaluator}</b>: {r.ok ? r.detail : `no corrected estimate. ${r.detail}`}</p>)}
    </section>
  )
}
