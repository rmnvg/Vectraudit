import { useEffect, useMemo, useState } from 'react'
import { EmbeddingView } from 'embedding-atlas/react'
import type { DataPoint } from 'embedding-atlas'
import { ATLAS_COLORS, useTheme } from '../lib/theme'
import { atlasBinName, buildGrid, decodeAtlas, nearestPoint, pointsWithin, poolJsonl, sha256Hex } from '../lib/atlas'
import type { AtlasPoints, AtlasRegion, AtlasSidecar, Box, XY } from '../lib/atlas'

const panel = 'rounded-2xl border border-line bg-surface p-5'
const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-teal-500'
const MAX_LISTED = 50

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/x-ndjson' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

/**
 * The corpus atlas: every point is a document or a question, coloured by what happened to it in
 * the run. The layout is a 2D projection and distorts distance, so the regions and findings come
 * from the full-dimensional analysis the sidecar carries.
 */
export default function CorpusAtlas({ side, loadedRuns, onOpenQuestion, onOpenGuided }: {
  side: AtlasSidecar
  /** IDs of the run artifacts currently loaded in the workbench. */
  loadedRuns: string[]
  onOpenQuestion: (runId: string, queryId: string) => void
  onOpenGuided: (docId: string) => void
}) {
  const { mode } = useTheme()
  const [points, setPoints] = useState<AtlasPoints | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<number[]>([])
  const [range, setRange] = useState<Box | XY[] | null>(null)
  const colors = ATLAS_COLORS[mode]
  const legend = useMemo(() => new Map(side.legend.map(item => [item.id, item])), [side])
  const byCategory = (side.counts.by_category ?? {}) as Record<string, number>

  useEffect(() => {
    let cancelled = false
    fetch(`api/workbench/atlas/${encodeURIComponent(atlasBinName(side))}`)
      .then(r => { if (!r.ok) throw new Error('The workbench could not read the atlas points.'); return r.arrayBuffer() })
      .then(async buffer => {
        if (await sha256Hex(buffer) !== side.bin.sha256) throw new Error('The atlas points do not match their description: rebuild the atlas.')
        if (!cancelled) setPoints(decodeAtlas(buffer, side.bin.n))
      })
      .catch(e => { if (!cancelled) setError(String(e.message ?? e)) })
    return () => { cancelled = true }
  }, [side])

  const grid = useMemo(() => (points ? buildGrid(points, side.extent) : null), [points, side])
  const describe = (i: number) => {
    const id = side.points.ids[i]
    const title = side.points.titles?.[i]
    const category = legend.get(points?.category[i] ?? 0)?.label ?? ''
    return `${title && title !== id ? `${title} · ` : ''}${id} · ${category}`
  }
  const labels = useMemo(() => side.regions.map(r => ({
    x: r.centre[0], y: r.centre[1], priority: r.uncovered ? 10 : 1, level: 0,
    content: (r.uncovered ? '? ' : '') + (r.terms.slice(0, 3).join(' · ') || `region ${r.cluster}`),
  })), [side])
  const runId = side.runs.selected?.run_id ?? null
  const runLoaded = runId !== null && loadedRuns.includes(runId)
  const groups = side.findings.map(finding => ({ finding, members: side.regions.filter(r => finding.clusters.includes(r.cluster)) }))
  const regions = [...side.regions].sort((a: AtlasRegion, b: AtlasRegion) => Number(b.uncovered) - Number(a.uncovered) || b.documents - a.documents)

  return (
    <div className="space-y-4" data-testid="corpus-atlas">
      <div className={`${panel} text-sm text-muted`}>
        <b className="text-ink">{side.name}</b> · {side.points.n.toLocaleString()} points
        {side.runs.selected && <> · outcomes from run <b>{side.runs.selected.label ?? side.runs.selected.run_id}</b></>}
        {side.runs.baseline && <> · regressions against <b>{side.runs.baseline.label ?? side.runs.baseline.run_id}</b></>}
        <p className="mt-2">{side.projection.note} The groups and findings below come from the full-dimensional vectors. {side.heldout.shown ? 'Held-out questions are drawn uncoloured.' : 'Held-out questions are not drawn.'} This file holds document titles and stays on this machine.</p>
      </div>
      {side.findings.length > 0 && (
        <div className={panel} aria-label="Coverage findings">
          <h2 className="font-semibold">Coverage hypotheses</h2>
          {side.findings.map((f, i) => <p key={i} className="mt-2 text-sm">{f.message} <span className="text-muted">(hypothesis)</span></p>)}
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <div className={`${panel} p-2`}>
          <div className="flex flex-wrap gap-x-4 gap-y-1 px-3 pb-2 pt-1 text-xs text-muted" aria-label="Legend">
            {side.legend.filter(item => (byCategory[item.key] ?? 0) > 0).map(item => (
              <span key={item.id} className="inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: colors[item.id] }} />{item.label} ({byCategory[item.key]})</span>
            ))}
          </div>
          <div style={{ height: 560 }}>
            {error && <p className="p-4 text-sm text-red-600" role="alert">{error}</p>}
            {!error && !points && <p className="p-4 text-sm text-muted">Reading the points…</p>}
            {points && grid && (
              <EmbeddingView
                data={{ x: points.x, y: points.y, category: points.category }}
                categoryColors={colors}
                labels={labels}
                querySelection={async (x: number, y: number, unit: number): Promise<DataPoint | null> => {
                  const i = nearestPoint(points, grid, x, y, unit * 10)
                  return i < 0 ? null : { x: points.x[i], y: points.y[i], category: points.category[i], text: describe(i), identifier: i }
                }}
                selection={selected.map(i => ({ x: points.x[i], y: points.y[i], category: points.category[i], text: describe(i), identifier: i }))}
                onSelection={(s: DataPoint[] | null) => { setRange(null); setSelected((s ?? []).map(p => Number(p.identifier))) }}
                rangeSelection={range as never}
                onRangeSelection={(area: Box | XY[] | null) => { setRange(area); setSelected(area ? pointsWithin(points, area) : []) }}
              />
            )}
          </div>
        </div>
        <div className="space-y-4">
          <div className={panel} aria-label="Selection">
            <h2 className="font-semibold">Selection</h2>
            {!selected.length && <p className="mt-2 text-sm text-muted">Click a point, or draw a box or lasso, to list what is there.</p>}
            {selected.slice(0, MAX_LISTED).map(i => {
              const item = legend.get(points?.category[i] ?? 0)
              const isQuestion = item?.kind === 'question'
              return (
                <div key={i} className="mt-3 border-t border-line pt-3 text-sm">
                  <div className="font-mono break-all">{side.points.ids[i]}</div>
                  {side.points.titles?.[i] && <p className="text-muted">{side.points.titles[i]}</p>}
                  <p className="text-xs text-muted">{item?.label}</p>
                  {isQuestion
                    ? <button className={`${button} mt-2`} disabled={!runLoaded} title={runLoaded ? '' : 'Load the run this atlas was built from to investigate'} onClick={() => runId && onOpenQuestion(runId, side.points.ids[i])}>Investigate in Pipeline runs</button>
                    : <button className={`${button} mt-2`} onClick={() => onOpenGuided(side.points.ids[i])}>Grade in the Guided workflow</button>}
                </div>
              )
            })}
            {selected.length > MAX_LISTED && <p className="mt-3 text-xs text-muted">{selected.length - MAX_LISTED} more not listed.</p>}
          </div>
          <div className={panel} aria-label="Regions">
            <h2 className="font-semibold">Regions</h2>
            <p className="mt-1 text-xs text-muted">Groups of documents in the full-dimensional space.</p>
            {groups.map(({ finding, members }, i) => (
              <div key={`g${i}`} className="mt-3 border-t border-line pt-3 text-sm" data-testid="region-uncovered">
                <div className="flex justify-between gap-2"><b>{finding.terms.slice(0, 3).join(' · ') || `Group of ${members.length} regions`}</b><span className="text-xs font-semibold text-amber-600">no question close</span></div>
                <p className="text-xs text-muted">{finding.documents.toLocaleString()} documents ({Math.round(finding.share * 100)}%) in {members.length} neighbouring regions · 0 questions near</p>
                <button className={`${button} mt-2`} onClick={() => download(`uncovered-group-${i + 1}-to-judge.jsonl`, members.map(poolJsonl).join(''))}>Download {members.reduce((n, r) => n + (r.pool?.length ?? 0), 0)} documents to judge</button>
                <p className="mt-1 text-xs text-muted">The most representative documents, for a person to read and judge. Never labels.</p>
              </div>
            ))}
            {regions.filter(r => !r.uncovered).map(r => (
              <div key={r.cluster} className="mt-3 border-t border-line pt-3 text-sm" data-testid="region">
                <b>{r.terms.slice(0, 3).join(' · ') || `Region ${r.cluster}`}</b>
                <p className="text-xs text-muted">{r.documents.toLocaleString()} documents ({Math.round(r.share * 100)}%) · {r.questions} questions · {r.relevant_documents} labelled relevant</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
