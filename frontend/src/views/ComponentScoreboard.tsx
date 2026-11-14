import { useMemo } from 'react'
import { componentScoreboard } from '../lib/components'
import type { ComponentRow, Rate } from '../lib/components'

const panel = 'rounded-2xl border border-line bg-surface p-5'
const percent = (r: Rate) => r.rate == null ? '—' : `${(r.rate * 100).toFixed(1)}%`
const range = (r: Rate) => r.rate == null ? '' : `${(r.ci_low * 100).toFixed(1)} to ${(r.ci_high * 100).toFixed(1)}`

/** Where each question of a pipeline run first lost its required evidence, per component. */
export default function ComponentScoreboard({ run }: { run: unknown }) {
  const board = useMemo(() => componentScoreboard(run), [run])
  if (!board || !board.declared.length) return null
  const worst = board.components.reduce<ComponentRow | null>((a, c) => (c.rate ?? 0) > (a?.rate ?? 0) ? c : a, null)
  return (
    <section className={panel} aria-label="Component scoreboard">
      <h2 className="font-semibold">Where questions first lost their evidence</h2>
      <p className="mt-1 text-xs text-muted">
        {board.observed.questions} observed questions in {board.observed.families} families, family-weighted with 95% intervals. A document no recorded stage returned counts at retrieve, though the corpus, the index or query rewriting may be the cause. Information only.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="tabular w-full text-left text-sm">
          <thead className="text-xs text-muted"><tr><th className="py-1 pr-3 font-medium">Component</th><th className="py-1 pr-3 font-medium">Lost first here</th><th className="py-1 pr-3 font-medium">95% interval</th><th className="py-1 pr-3 font-medium">Questions</th></tr></thead>
          <tbody>
            {board.components.map(c => (
              <tr key={c.component} className={`border-t border-line ${c === worst && c.questions ? 'font-semibold' : ''}`}>
                <td className="py-2 pr-3 font-mono text-xs">{c.component}</td>
                <td className="py-2 pr-3">{percent(c)}</td>
                <td className="py-2 pr-3 text-muted">{range(c)}</td>
                <td className="py-2 pr-3">{c.questions}</td>
              </tr>
            ))}
            <tr className="border-t border-line text-muted"><td className="py-2 pr-3 font-mono text-xs">kept</td><td className="py-2 pr-3">{percent(board.kept)}</td><td className="py-2 pr-3">{range(board.kept)}</td><td className="py-2 pr-3">{board.kept.questions}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-muted">Not observed (never assigned): {board.not_observed} · not applicable: {board.not_applicable.unlabelled} unlabelled, {board.not_applicable.no_answer} no-answer · errored: {board.errored}</p>
      {board.cohorts.map(c => <p key={c.tag} className="mt-1 text-xs text-muted"><b>{c.tag}</b> (meant to exercise {c.targets.join(', ')}): {c.lost_at_targets} of {c.observed.questions} observed questions lost their evidence there.</p>)}
    </section>
  )
}
