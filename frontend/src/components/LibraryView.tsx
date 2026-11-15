import { useEffect, useMemo, useRef, useState } from 'react'
import { boundsOf, fitProjection, type Bounds } from '../lib/projection'
import type { MapPoint, Marks } from './Galaxy'

const PAD = 28
const HIT_PX = 10

/**
 * The whole library as an explorable star map (UMAP of the embeddings).
 * Active: scroll to zoom, drag to pan, hover for the song, click to play, double-click to reset.
 * Inactive (shown as a thumbnail): static, whole library, marks drawn larger so they survive the downscale.
 */
export function LibraryView({ points, marks, width, height, active, pointScale, tint, onPlay, nowLabel }: {
  points: MapPoint[]
  marks: Marks | null
  width: number
  height: number
  active: boolean
  pointScale: number
  tint?: string
  onPlay: (p: MapPoint) => void
  nowLabel?: string
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const full = useMemo(() => boundsOf(points.map((p) => [p[1], p[2]] as const)), [points])
  const [zoomed, setZoomed] = useState<Bounds | null>(null)
  const view = active && zoomed ? zoomed : full
  const proj = useMemo(() => fitProjection(view, width, height, PAD), [view, width, height])
  const zoom = (full.x1 - full.x0) / (view.x1 - view.x0)
  const [hover, setHover] = useState<number | null>(null)
  const drag = useRef<{ x: number; y: number; view: Bounds; moved: boolean } | null>(null)

  /* ---- draw ---- */
  useEffect(() => {
    const c = canvas.current
    if (!c || !points.length || width < 10) return
    const dpr = window.devicePixelRatio || 1
    c.width = width * dpr
    c.height = height * dpr
    const g = c.getContext('2d')!
    g.scale(dpr, dpr)
    g.clearRect(0, 0, width, height)
    const acc = getComputedStyle(c).getPropertyValue('--accent').trim() || '#c8ff4d'
    const ps = pointScale
    const k = Math.min(3, Math.sqrt(zoom))
    const { px, py } = proj

    for (const [, x, y, plays] of points) {
      const X = px(x), Y = py(y)
      if (X < -4 || Y < -4 || X > width + 4 || Y > height + 4) continue
      const b = Math.min(1, 0.2 + Math.log1p(plays) / 7)
      const r = (0.7 + Math.log1p(plays) * 0.22) * k * ps
      g.fillStyle = `rgba(235,235,240,${0.5 * b})`
      g.fillRect(X - r / 2, Y - r / 2, r, r)
    }
    if (marks) {
      g.fillStyle = acc
      g.globalAlpha = 0.85
      for (const [x, y] of marks.cands) {
        g.beginPath(); g.arc(px(x), py(y), 2.6 * ps, 0, Math.PI * 2); g.fill()
      }
      g.globalAlpha = 1
      if (marks.path.length > 1) {
        g.strokeStyle = acc
        g.lineWidth = 1.6 * ps
        g.beginPath()
        marks.path.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))))
        g.stroke()
        for (const [x, y] of marks.path) { g.beginPath(); g.arc(px(x), py(y), 2.4 * ps, 0, Math.PI * 2); g.fill() }
      }
      if (marks.current) {
        const X = px(marks.current[0]), Y = py(marks.current[1])
        g.globalAlpha = 0.18; g.beginPath(); g.arc(X, Y, 16 * ps, 0, Math.PI * 2); g.fill()
        g.globalAlpha = 1
        g.strokeStyle = acc; g.lineWidth = 1.2 * ps
        g.beginPath(); g.arc(X, Y, 8 * ps, 0, Math.PI * 2); g.stroke()
        g.beginPath(); g.arc(X, Y, 3.6 * ps, 0, Math.PI * 2); g.fill()
      }
    }
    if (hover != null && points[hover]) {
      const p = points[hover]
      g.strokeStyle = '#fff'; g.lineWidth = 1.5
      g.beginPath(); g.arc(px(p[1]), py(p[2]), 6, 0, Math.PI * 2); g.stroke()
    }
  }, [points, proj, zoom, marks, hover, pointScale, tint, width, height])

  /* ---- wheel zoom around the cursor (non-passive so the page doesn't scroll) ---- */
  useEffect(() => {
    const c = canvas.current
    if (!c || !active) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = c.getBoundingClientRect()
      const ax = proj.ix(e.clientX - r.left), ay = proj.iy(e.clientY - r.top)
      const f = Math.exp(e.deltaY * 0.0015)
      const nb = { x0: ax + (view.x0 - ax) * f, x1: ax + (view.x1 - ax) * f, y0: ay + (view.y0 - ay) * f, y1: ay + (view.y1 - ay) * f }
      const span = nb.x1 - nb.x0
      if (span >= full.x1 - full.x0) setZoomed(null) // never zoom out past the whole library
      else if (span > 0.15) setZoomed(nb)
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, [active, proj, view, full])

  const nearest = (X: number, Y: number) => {
    let best = -1, bd = HIT_PX * HIT_PX
    points.forEach((p, i) => {
      const dx = proj.px(p[1]) - X, dy = proj.py(p[2]) - Y
      const d = dx * dx + dy * dy
      if (d < bd) { bd = d; best = i }
    })
    return best >= 0 ? best : null
  }

  const local = (e: React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect()
    return { X: e.clientX - r.left, Y: e.clientY - r.top }
  }

  const zoomBy = (f: number) => {
    const cx = (view.x0 + view.x1) / 2, cy = (view.y0 + view.y1) / 2
    const nb = { x0: cx + (view.x0 - cx) * f, x1: cx + (view.x1 - cx) * f, y0: cy + (view.y0 - cy) * f, y1: cy + (view.y1 - cy) * f }
    setZoomed(nb.x1 - nb.x0 >= full.x1 - full.x0 ? null : nb)
  }

  const hp = hover != null ? points[hover] : null
  const now = marks?.current

  return (
    <div className={`library ${active ? 'active' : ''}`}>
      <canvas
        ref={canvas}
        style={{ width, height, display: 'block', cursor: hover != null ? 'pointer' : undefined }}
        onMouseDown={(e) => {
          if (!active) return
          const { X, Y } = local(e)
          drag.current = { x: X, y: Y, view, moved: false }
        }}
        onMouseMove={(e) => {
          if (!active) return
          const { X, Y } = local(e)
          const d = drag.current
          if (d) {
            if (!d.moved && Math.hypot(X - d.x, Y - d.y) < 4) return
            d.moved = true
            const s = proj.scale
            const dx = (X - d.x) / s, dy = (Y - d.y) / s
            setZoomed({ x0: d.view.x0 - dx, x1: d.view.x1 - dx, y0: d.view.y0 + dy, y1: d.view.y1 + dy })
            setHover(null)
            return
          }
          setHover(nearest(X, Y))
        }}
        onMouseUp={() => {
          const d = drag.current
          drag.current = null
          if (d && !d.moved && hover != null) onPlay(points[hover])
        }}
        onMouseLeave={() => { drag.current = null; setHover(null) }}
        onDoubleClick={() => setZoomed(null)}
      />

      {active && now && nowLabel && (
        <div className="lib-now" style={{ left: proj.px(now[0]), top: proj.py(now[1]) }}>
          <span>now</span> {nowLabel}
        </div>
      )}

      {active && hp && (
        <div className="tip lib-tip" style={{ left: proj.px(hp[1]), top: proj.py(hp[2]) }}>
          <div className="tip-title">{hp[4]}</div>
          <div className="tip-artist">{hp[5]}</div>
          <div className="tip-grid"><span>Plays</span><b>{hp[3]}</b></div>
          <div className="tip-cta">{hp[6] ? 'click to play' : 'not on Spotify'}</div>
        </div>
      )}

      {active && (
        <>
          <div className="lib-controls">
            <button onClick={() => zoomBy(0.6)} aria-label="Zoom in">+</button>
            <button onClick={() => zoomBy(1 / 0.6)} aria-label="Zoom out">−</button>
            <button onClick={() => setZoomed(null)} aria-label="Fit library" className="wide">fit</button>
          </div>
          <div className="lib-hint">
            {points.length.toLocaleString()} tracks · scroll to zoom · drag to pan · click a dot to play
          </div>
        </>
      )}
    </div>
  )
}
