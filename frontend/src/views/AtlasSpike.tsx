// Throwaway spike (31.0): can embedding-atlas render offline, lazily, with permissive licences?
import { useMemo, useState } from 'react'
import { EmbeddingView } from 'embedding-atlas/react'

const N = 20_000
const COLORS = ['#9aa0a6', '#2a78d6', '#eb6834', '#1baf7a', '#c0392b', '#8e44ad']

function build() {
  const x = new Float32Array(N)
  const y = new Float32Array(N)
  const category = new Uint8Array(N)
  let seed = 7
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
  for (let i = 0; i < N; i++) {
    const c = i % 12
    x[i] = Math.cos(c) * 8 + (rand() + rand() + rand() - 1.5) * 2
    y[i] = Math.sin(c) * 8 + (rand() + rand() + rand() - 1.5) * 2
    category[i] = rand() < 0.02 ? 1 + (i % 5) : 0
  }
  return { x, y, category }
}

export default function AtlasSpike() {
  const data = useMemo(build, [])
  const [selection, setSelection] = useState<unknown[] | null>(null)
  ;(window as unknown as Record<string, unknown>).__atlasSelection = selection
  return (
    <div style={{ height: 'calc(100vh - 120px)' }} data-testid="atlas">
      <EmbeddingView
        data={data}
        categoryColors={COLORS}
        selection={selection as never}
        onSelection={(s) => setSelection(s)}
        querySelection={async (px, py, unit) => {
          let best = -1
          let bestD = unit * 8
          for (let i = 0; i < N; i++) {
            const d = Math.hypot(data.x[i] - px, data.y[i] - py)
            if (d < bestD) (bestD = d), (best = i)
          }
          return best < 0 ? null : { x: data.x[best], y: data.y[best], category: data.category[best], text: `point ${best}`, identifier: best }
        }}
      />
      <p data-testid="atlas-count">{selection?.length ?? 0} selected</p>
    </div>
  )
}
