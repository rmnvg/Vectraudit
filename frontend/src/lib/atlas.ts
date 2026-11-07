/** Reading a corpus atlas: the sidecar JSON and the binary points (see docs/corpus-atlas.md). */

export interface AtlasLegendItem { id: number; key: string; kind: 'document' | 'question'; label: string }
export interface AtlasRegion {
  cluster: number
  documents: number
  share: number
  questions: number
  relevant_documents: number
  terms: string[]
  centre: [number, number]
  uncovered: boolean
  distance_ratio?: number
  pool?: { id: string; title: string | null }[]
}
export interface AtlasFinding { kind: string; evidence: string; clusters: number[]; documents: number; share: number; terms: string[]; message: string }
export interface AtlasSidecar {
  format: 'vectraudit.atlas'
  format_version: 1
  name: string
  local_only: boolean
  projection: { method: string; explained_variance_ratio: number[]; note: string }
  counts: Record<string, number | Record<string, number>>
  legend: AtlasLegendItem[]
  sampling: { seed: number; max_points: number; over_cap: boolean; rule: string }
  heldout: { shown: boolean; note: string }
  runs: { selected: { run_id: string | null; label: string | null } | null; baseline: { run_id: string | null; label: string | null } | null }
  extent: { x: [number, number]; y: [number, number] }
  points: { n: number; ids: string[]; titles: (string | null)[] | null; titles_omitted: boolean }
  bin: { n: number; sha256: string; bytes: number; file?: string }
  regions: AtlasRegion[]
  findings: AtlasFinding[]
}

export interface AtlasPoints {
  x: Float32Array<ArrayBuffer>
  y: Float32Array<ArrayBuffer>
  category: Uint8Array<ArrayBuffer>
  row: Uint32Array
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

/** Whether imported JSON is an atlas sidecar with the fields the view reads. */
export function isAtlasSidecar(value: unknown): value is AtlasSidecar {
  if (!isObject(value) || value.format !== 'vectraudit.atlas' || value.format_version !== 1) return false
  const points = value.points
  const bin = value.bin
  return (
    isObject(points) && Array.isArray(points.ids) && typeof points.n === 'number' && points.ids.length === points.n &&
    isObject(bin) && typeof bin.n === 'number' && bin.n === points.n &&
    Array.isArray(value.legend) && Array.isArray(value.regions) && Array.isArray(value.findings) &&
    isObject(value.projection) && isObject(value.heldout) && isObject(value.extent) && isObject(value.runs)
  )
}

/** The binary points: x float32[n], y float32[n], category uint8[n] padded to 4, row uint32[n]. */
export function decodeAtlas(buffer: ArrayBuffer, n: number): AtlasPoints {
  const padded = n + ((4 - (n % 4)) % 4)
  const expected = 8 * n + padded + 4 * n
  if (buffer.byteLength !== expected) throw new Error(`The atlas points are ${buffer.byteLength} bytes, expected ${expected}.`)
  return {
    x: new Float32Array(buffer, 0, n),
    y: new Float32Array(buffer, 4 * n, n),
    category: new Uint8Array(buffer, 8 * n, n),
    row: new Uint32Array(buffer, 8 * n + padded, n),
  }
}

/** The name the workbench serves the points under: ``atlas-<name>.bin``. */
export function atlasBinName(side: AtlasSidecar): string {
  const file = side.bin.file ?? `atlas-${side.name}.bin`
  return file.replace(/^atlas-/, '').replace(/\.bin$/, '')
}

/** SHA-256 hex of a buffer (the sidecar records the points' digest). */
export async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer)
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('')
}

/** A uniform grid over the points, so a hover finds its nearest point without scanning all of them. */
export interface PointGrid { cell: number; cols: number; rows: number; minX: number; minY: number; cells: Map<number, number[]> }

export function buildGrid(points: AtlasPoints, extent: AtlasSidecar['extent'], target = 256): PointGrid {
  const [minX, maxX] = extent.x
  const [minY, maxY] = extent.y
  const cell = Math.max(maxX - minX, maxY - minY, 1e-9) / target
  const cols = Math.floor((maxX - minX) / cell) + 1
  const rows = Math.floor((maxY - minY) / cell) + 1
  const cells = new Map<number, number[]>()
  for (let i = 0; i < points.x.length; i++) {
    const key = Math.floor((points.y[i] - minY) / cell) * cols + Math.floor((points.x[i] - minX) / cell)
    const list = cells.get(key)
    if (list) list.push(i)
    else cells.set(key, [i])
  }
  return { cell, cols, rows, minX, minY, cells }
}

/** The index of the point nearest (px, py) within ``radius``, preferring notable (non-default) points. */
export function nearestPoint(points: AtlasPoints, grid: PointGrid, px: number, py: number, radius: number): number {
  const span = Math.ceil(radius / grid.cell)
  const cx = Math.floor((px - grid.minX) / grid.cell)
  const cy = Math.floor((py - grid.minY) / grid.cell)
  let best = -1
  let bestScore = radius * radius
  for (let gy = cy - span; gy <= cy + span; gy++) {
    for (let gx = cx - span; gx <= cx + span; gx++) {
      if (gx < 0 || gy < 0 || gx >= grid.cols || gy >= grid.rows) continue
      for (const i of grid.cells.get(gy * grid.cols + gx) ?? []) {
        const d = (points.x[i] - px) ** 2 + (points.y[i] - py) ** 2
        // A notable point wins a near tie against a plain document underneath it.
        const score = points.category[i] === 0 ? d : d * 0.5
        if (score < bestScore) { bestScore = score; best = i }
      }
    }
  }
  return best
}

/** Documents of an uncovered region as JSONL for a person to judge. These are never labels. */
export function poolJsonl(region: AtlasRegion): string {
  return (region.pool ?? [])
    .map(d => JSON.stringify({ doc_id: d.id, region: region.cluster, reason: 'uncovered region (hypothesis): judge, do not label automatically' }))
    .join('\n') + '\n'
}

export interface Box { xMin: number; yMin: number; xMax: number; yMax: number }
export interface XY { x: number; y: number }

/**
 * Indices of the points inside a rectangle or a lasso polygon. Notable (non-default) points come
 * first, so a capped list shows the findings before plain documents.
 */
export function pointsWithin(points: AtlasPoints, area: Box | XY[]): number[] {
  const inside: number[] = []
  const polygon = Array.isArray(area) ? area : null
  const box = polygon
    ? { xMin: Math.min(...polygon.map(p => p.x)), xMax: Math.max(...polygon.map(p => p.x)), yMin: Math.min(...polygon.map(p => p.y)), yMax: Math.max(...polygon.map(p => p.y)) }
    : (area as Box)
  for (let i = 0; i < points.x.length; i++) {
    const x = points.x[i]
    const y = points.y[i]
    if (x < box.xMin || x > box.xMax || y < box.yMin || y > box.yMax) continue
    if (polygon && !inPolygon(x, y, polygon)) continue
    inside.push(i)
  }
  return inside.sort((a, b) => Number(points.category[b] !== 0) - Number(points.category[a] !== 0) || a - b)
}

function inPolygon(x: number, y: number, poly: XY[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i]
    const b = poly[j]
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside
  }
  return inside
}
