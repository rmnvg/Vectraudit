import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'
import sidecar from './fixtures/atlas.json'
import { atlasBinName, buildGrid, decodeAtlas, isAtlasSidecar, nearestPoint, pointsWithin, poolJsonl, sha256Hex } from './atlas'
import type { AtlasPoints } from './atlas'

// atlas.json / atlas.bin are written by core/atlas.py; tests/test_atlas.py keeps them in step.
const bin = () => {
  const bytes = readFileSync(new URL('./fixtures/atlas.bin', import.meta.url))
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
}

it('recognises the sidecar the Python builder writes, and nothing looser', () => {
  expect(isAtlasSidecar(sidecar)).toBe(true)
  expect(isAtlasSidecar({ ...sidecar, format: 'vectraudit.run' })).toBe(false)
  expect(isAtlasSidecar({ ...sidecar, format_version: 2 })).toBe(false)
  expect(isAtlasSidecar({ ...sidecar, points: { ...sidecar.points, n: sidecar.points.n + 1 } })).toBe(false)
  expect(isAtlasSidecar({ ...sidecar, regions: undefined })).toBe(false)
  expect(isAtlasSidecar(null)).toBe(false)
})

it('decodes the binary points the Python builder writes', async () => {
  const n = sidecar.bin.n
  const points = decodeAtlas(bin(), n)
  expect(points.x).toHaveLength(n)
  expect(points.category).toHaveLength(n)
  expect(await sha256Hex(bin())).toBe(sidecar.bin.sha256)
  const ids = sidecar.points.ids
  // Categories land on the right points: the question that failed, and the one that passed.
  expect(points.category[ids.indexOf('q0')]).toBe(6) // passed
  expect(points.category[ids.indexOf('q1')]).toBe(7) // failed
  expect(ids).not.toContain('h') // held out: not drawn
  expect(points.category[ids.indexOf('d1-0')]).toBe(2) // relevant, never retrieved
  const [minX, maxX] = sidecar.extent.x
  expect([...points.x].every(v => v >= minX - 1e-3 && v <= maxX + 1e-3)).toBe(true)
  expect(Math.max(...points.row)).toBeLessThan(1000)
})

it('refuses a payload of the wrong length', () => {
  expect(() => decodeAtlas(bin().slice(0, 40), sidecar.bin.n)).toThrow(/expected/)
})

it('decodes any length, including those that need padding', () => {
  for (const n of [1, 2, 3, 4, 5, 7]) {
    const padded = n + ((4 - (n % 4)) % 4)
    const buffer = new ArrayBuffer(8 * n + padded + 4 * n)
    const view = new DataView(buffer)
    for (let i = 0; i < n; i++) {
      view.setFloat32(4 * i, i + 0.5, true)
      view.setFloat32(4 * n + 4 * i, -i, true)
      view.setUint8(8 * n + i, i % 11)
      view.setUint32(8 * n + padded + 4 * i, 1000 + i, true)
    }
    const p = decodeAtlas(buffer, n)
    expect([...p.x]).toEqual(Array.from({ length: n }, (_, i) => i + 0.5))
    expect([...p.y]).toEqual(Array.from({ length: n }, (_, i) => -i))
    expect([...p.category]).toEqual(Array.from({ length: n }, (_, i) => i % 11))
    expect([...p.row]).toEqual(Array.from({ length: n }, (_, i) => 1000 + i))
  }
})

it('names the binary from the sidecar', () => {
  expect(atlasBinName(sidecar as never)).toBe('fixture')
  expect(atlasBinName({ ...sidecar, bin: { ...sidecar.bin, file: undefined } } as never)).toBe('fixture')
})

const small = (): AtlasPoints => ({
  x: new Float32Array([0, 1, 1.05, 5]),
  y: new Float32Array([0, 1, 1.0, 5]),
  category: new Uint8Array([0, 0, 7, 0]),
  row: new Uint32Array([0, 1, 2, 3]),
})

it('finds the nearest point under the pointer, preferring a notable one over a plain document', () => {
  const p = small()
  const grid = buildGrid(p, { x: [0, 5], y: [0, 5] })
  expect(nearestPoint(p, grid, 0.02, 0.01, 0.5)).toBe(0)
  expect(nearestPoint(p, grid, 1.025, 1.0, 0.5)).toBe(2) // equidistant: the failed question beats the document
  expect(nearestPoint(p, grid, 3, 3, 0.5)).toBe(-1)
})

it('lists the points inside a rectangle or a lasso, notable points first', () => {
  const p = small()
  expect(pointsWithin(p, { xMin: 0.5, xMax: 2, yMin: 0.5, yMax: 2 })).toEqual([2, 1])
  const triangle = [{ x: 0.95, y: 0.9 }, { x: 1.2, y: 0.9 }, { x: 1.2, y: 1.2 }]
  expect(pointsWithin(p, triangle)).toEqual([2])
  expect(pointsWithin(p, { xMin: 9, xMax: 10, yMin: 9, yMax: 10 })).toEqual([])
})

it('exports an uncovered region as documents to judge, never as labels', () => {
  const lines = poolJsonl({ cluster: 3, documents: 9, share: 0.1, questions: 0, relevant_documents: 0, terms: ['refund'], centre: [0, 0], uncovered: true, pool: [{ id: 'd1', title: null }, { id: 'd2', title: 'T' }] }).trim().split('\n').map(l => JSON.parse(l))
  expect(lines.map(l => l.doc_id)).toEqual(['d1', 'd2'])
  expect(lines[0].region).toBe(3)
  expect(JSON.stringify(lines)).not.toContain('relevant')
  expect(lines[0].reason).toContain('do not label')
})
