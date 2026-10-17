import { summarizeCheck } from '../lib/checks'
import type { CheckReport, Verdict } from '../lib/checks'

const BADGE: Record<Verdict, string> = {
  pass: 'bg-emerald-500/10 text-emerald-600',
  fail: 'bg-red-500/10 text-red-600',
  attention: 'bg-amber-500/10 text-amber-700',
}
const LABEL: Record<Verdict, string> = { pass: 'pass', fail: 'fail', attention: 'needs attention' }

/** One card per imported check report, newest import first. */
export default function CheckReports({ reports, names }: { reports: CheckReport[]; names: string[] }) {
  return <div className="space-y-4">{reports.map((report, i) => ({ report, name: names[i] })).reverse().map(({ report, name }, i) => {
    const s = summarizeCheck(report)
    return <section key={`${name}-${i}`} aria-label={s.title} className="rounded-2xl border border-line bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h2 className="font-semibold">{s.title}</h2>{name && <p className="text-xs text-muted">{name}</p>}</div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${BADGE[s.verdict]}`}>{LABEL[s.verdict]}</span>
      </div>
      <p className="mt-3 text-sm">{s.headline}</p>
      {s.items.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-sm">{s.items.map((item, j) => <li key={j}>{item}</li>)}</ul>}
      {s.notes.map((note, j) => <p key={j} className="mt-2 text-xs text-muted">{note}</p>)}
    </section>
  })}</div>
}
