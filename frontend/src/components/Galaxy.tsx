import { useEffect, useRef } from 'react'
import { boundsOf, fitProjection, type Bounds } from '../lib/projection'

export type MapPoint = [id: string, x: number, y: number, plays: number, title?: string, artist?: string, playable?: number]
type XY = [number, number]
export type Marks = { path: XY[]; current?: XY; cands: XY[] }
function accent(el: HTMLElement) {
  return getComputedStyle(el).getPropertyValue('--accent').trim() || '#c8ff4d'
}

/** The whole library as a star field (UMAP of the embeddings). Optional marks overlay the current journey. */
export function Galaxy({ points, width, height, marks, dim, tint, view, pad = 6 }: {
  points: MapPoint[]; width: number; height: number; marks?: Marks; dim?: boolean
  view?: Bounds // zoom to this region instead of the whole library
  pad?: number
  tint?: string // changes when --accent changes, to trigger a redraw
}) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = ref.current
    if (!c || !points.length || width < 10 || height < 10) return
    const dpr = window.devicePixelRatio || 1
    c.width = width * dpr
    c.height = height * dpr
    const g = c.getContext('2d')!
    g.scale(dpr, dpr)
    g.clearRect(0, 0, width, height)

    const { px, py } = fitProjection(view ?? boundsOf(points.map((p) => [p[1], p[2]] as const)), width, height, pad)

    const acc = accent(c)
    for (const [, x, y, plays] of points) {
      const b = Math.min(1, 0.18 + Math.log1p(plays) / 7)
      g.fillStyle = `rgba(235,235,240,${(dim ? 0.22 : 0.42) * b})`
      const r = dim ? 0.7 : 0.6 + Math.log1p(plays) * 0.18
      const X = px(x), Y = py(y)
      if (X < -2 || Y < -2 || X > width + 2 || Y > height + 2) continue
      g.fillRect(X - r / 2, Y - r / 2, r, r)
    }
    if (!marks) return

    g.fillStyle = acc
    g.globalAlpha = 0.55
    for (const [x, y] of marks.cands) g.fillRect(px(x) - 1, py(y) - 1, 2, 2)
    g.globalAlpha = 1
    if (marks.path.length > 1) {
      g.strokeStyle = acc
      g.lineWidth = 1.2
      g.beginPath()
      marks.path.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))))
      g.stroke()
    }
    if (marks.current) {
      const [x, y] = marks.current
      g.beginPath(); g.arc(px(x), py(y), 7, 0, Math.PI * 2)
      g.fillStyle = acc; g.globalAlpha = 0.2; g.fill(); g.globalAlpha = 1
      g.beginPath(); g.arc(px(x), py(y), 2.6, 0, Math.PI * 2); g.fill()
    }
  }, [points, width, height, marks, dim, tint, view, pad])

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />
}
