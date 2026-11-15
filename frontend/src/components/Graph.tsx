import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import type { Neighborhood, Route, Track } from '../lib/types'
import { Galaxy, type MapPoint } from './Galaxy'
import { LibraryView } from './LibraryView'
import { boundsOf, fitProjection, type Bounds } from '../lib/projection'
import { useCover } from '../lib/useCover'

type Role = 'center' | 'candidate' | 'trail' | 'route' | 'route-current' | 'route-past'
type Target = {
  x: number; y: number; role: Role; side: 'l' | 'r' | 't' | 'b'
  age?: number // trail: steps back in time
  rank?: number // candidate: similarity rank (0 = nearest)
  reveal?: number // route map: ms into the replay when this node appears
  endpoint?: boolean
}
type EdgeT = {
  a: string; b: string; why: string[]; sim: number; role: 'candidate' | 'trail' | 'route' | 'routemap'; rank: number
  reveal?: number
}
type Layout = { targets: Map<string, Target>; edges: EdgeT[]; nodes: Track[] }
type P = { x: number; y: number; o: number }
type XY = { x: number; y: number }

const TWEEN_MS = 750
const COMET_MS = 650 // travelling to the next song before the graph re-centres
const RANK_STAGGER_MS = 70 // candidates appear in similarity order
const TRACE_HOP_MS = 420 // path replay: time per hop
const HOLD_MS = 900 // path replay: pause on the found path before unfolding
const MAP_PAD = 90
const CENTER_R = 24
const PREV_MIN_GAP = 170 // px between the centre and the previous song, so their titles don't overlap
const TRAIL_R = 15 // history nodes are album covers, so they need room to read

/** A history node's album cover, cropped round (falls back to the plain dot while loading / if none). */
function TrailArt({ id, r }: { id: string; r: number }) {
  const url = useCover(id)
  if (!url) return null
  return (
    <image key={url} href={url} x={-r} y={-r} width={r * 2} height={r * 2}
      clipPath="url(#art-clip-unit)" className="art" preserveAspectRatio="xMidYMid slice" />
  )
}
const INSET_MARGIN = 14

/** compact edge-label wording: "played back to back ×14" → "b2b ×14", "more energy (+0.12)" → "energy +0.12" */
const shortWhy = (w: string) =>
  w.replace(/^played back to back /, 'b2b ').replace(/^(more|less) energy \((.+)\)$/, 'energy $2').replace(/^harmonic neighbour$/, 'harmonic')
const ease = (t: number) => 1 - Math.pow(1 - t, 3)
const spaceName = (m?: string) => (m === 'columns' ? 'feature' : 'GNN')

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 800, h: 600 })
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}

const centerOf = (l: Layout | null) =>
  l ? [...l.targets].find(([, t]) => t.role === 'center' || t.role === 'route-current')?.[0] : undefined

/* ---------------- layouts ---------------- */

function layoutNeighborhood(h: Neighborhood, w: number, ht: number): Layout {
  const targets = new Map<string, Target>()
  const edges: EdgeT[] = []
  const trail = h.trail.slice(-5)
  const hasTrail = trail.length > 0
  const cx = hasTrail ? Math.max(w * 0.4, 260) : w * 0.5
  const cy = ht * 0.5
  const center = h.nodes[0]
  targets.set(center.id, { x: cx, y: cy, role: 'center', side: 'b' })

  // candidates: most similar straight ahead (right), alternating outward
  const cands = h.nodes.slice(1)
  const sims = cands.map((c) => c.similarity ?? 0)
  const hi = Math.max(...sims), lo = Math.min(...sims)
  const span = hasTrail ? (230 * Math.PI) / 180 : (330 * Math.PI) / 180
  const n = cands.length
  const step = n > 1 ? span / (n - 1) : 0
  const rx = Math.min(hasTrail ? w - cx - 150 : w * 0.5 - 150, ht * 0.62)
  const ry = ht * 0.38
  const order = [...h.edges].sort((a, b) => b.similarity - a.similarity)
  cands.forEach((c, i) => {
    const k = Math.ceil(i / 2) * (i % 2 ? 1 : -1) // 0, +1, -1, +2, -2 …
    // with a trail: fan to the right (left side belongs to history); without: full ring
    const a = hasTrail ? k * step : (i / n) * Math.PI * 2 - Math.PI / 2 + 0.3
    const t = hi === lo ? 0.5 : (hi - (c.similarity ?? lo)) / (hi - lo)
    const r = 0.62 + 0.38 * t + (i % 2 ? 0.035 : -0.035)
    const x = cx + Math.cos(a) * rx * r
    const y = cy + Math.sin(a) * ry * r
    // labels point away from the centre, unless that would run off the panel edge
    const side = Math.abs(x - cx) < 40 ? (y < cy ? 't' : 'b') : (x > cx ? x < w - 190 : x < 190) ? 'r' : 'l'
    targets.set(c.id, { x, y, role: 'candidate', side, rank: order.findIndex((e) => e.to === c.id) })
  })
  h.edges.forEach((e) =>
    edges.push({ a: e.from, b: e.to, why: e.why, sim: e.similarity, role: 'candidate', rank: order.indexOf(e) }),
  )

  // trail: where we came from, drifting off to the left
  // The previous song keeps its title, so it gets a guaranteed gap from the centre and sits a bit
  // above it with its label on top (the centre's label is underneath): the two titles never collide.
  // Older songs (covers only) share whatever room is left further out.
  const prevGap = Math.max(PREV_MIN_GAP, Math.min(200, (cx - 60) * 0.45))
  const older = trail.length - 1
  const oStep = older ? Math.max(0, Math.min(120, (cx - prevGap - 30) / older)) : 0 // never off-panel
  trail.forEach((t, i) => {
    const dist = trail.length - i
    const x = dist === 1 ? cx - prevGap : cx - prevGap - (dist - 1) * oStep
    const y = dist === 1 ? cy - 34 : cy + Math.sin(i * 1.7) * 36
    targets.set(t.id, { x, y, role: 'trail', age: dist, side: dist === 1 ? 't' : i % 2 ? 't' : 'b' })
  })
  const chain = [...trail.map((t) => t.id), center.id]
  h.trail_edges
    .filter((e) => chain.includes(e.from) && chain.includes(e.to))
    .forEach((e) => edges.push({ a: e.from, b: e.to, why: e.why, sim: e.similarity, role: 'trail', rank: 0 }))
  return { targets, edges, nodes: [...h.nodes, ...trail] }
}

/** Readable resting layout for a planned route: a gentle wave across the panel. */
function layoutRoute(r: Route, currentId: string | null, w: number, ht: number): Layout {
  const targets = new Map<string, Target>()
  const n = r.tracks.length
  const pad = Math.min(140, w * 0.12)
  const cur = r.tracks.findIndex((t) => t.id === currentId)
  r.tracks.forEach((t, i) => {
    const f = n > 1 ? i / (n - 1) : 0.5
    const x = pad + f * (w - pad * 2)
    // zig-zag around a gentle wave: neighbours' labels sit on opposite sides, so they never collide
    const up = i % 2 === 0
    const y = ht * 0.5 + Math.sin(f * Math.PI * 1.2 - 0.3) * ht * 0.08 + (up ? -1 : 1) * Math.min(110, ht * 0.15)
    const role: Role = i === cur ? 'route-current' : cur >= 0 && i < cur ? 'route-past' : 'route'
    targets.set(t.id, { x, y, role, side: up ? 't' : 'b' })
  })
  const edges: EdgeT[] = r.hops.map((h) => ({ a: h.from, b: h.to, why: h.why, sim: h.similarity, role: 'route', rank: 0 }))
  return { targets, edges, nodes: r.tracks }
}

function routeTiming(r: Route) {
  const n = r.explored?.length ?? 0
  const search = n ? Math.min(2600, Math.max(1000, n * 14)) : 0
  const trace = Math.max(1, r.tracks.length - 1) * TRACE_HOP_MS
  return { search, trace, total: search + trace + HOLD_MS, step: n ? search / n : 0 }
}

/** Projection of a route (and its search tree) onto the library map, zoomed to fit the panel. */
function routeMap(r: Route, w: number, h: number) {
  if (r.tracks.some((t) => t.x == null || t.y == null)) return null
  const pts: [number, number][] = [...r.tracks, ...(r.explored ?? [])]
    .filter((p) => p.x != null && p.y != null)
    .map((p) => [p.x!, p.y!])
  const b0 = boundsOf(pts)
  // never zoom in closer than a small window, so the local galaxy stays visible
  const minSpan = 2.5
  const cx = (b0.x0 + b0.x1) / 2, cy = (b0.y0 + b0.y1) / 2
  const hx = Math.max(minSpan, b0.x1 - b0.x0) / 2, hy = Math.max(minSpan, b0.y1 - b0.y0) / 2
  const bounds: Bounds = { x0: cx - hx, x1: cx + hx, y0: cy - hy, y1: cy + hy }
  const { px, py } = fitProjection(bounds, w, h, MAP_PAD)
  const pos = new Map<string, XY>()
  for (const p of [...(r.explored ?? []), ...r.tracks]) if (p.x != null && p.y != null) pos.set(p.id, { x: px(p.x), y: py(p.y) })
  return { bounds, pos }
}

function layoutRouteMap(r: Route, pos: Map<string, XY>, search: number): Layout {
  const targets = new Map<string, Target>()
  const n = r.tracks.length
  const mid = [...pos.values()].reduce((s, p) => s + p.y, 0) / Math.max(pos.size, 1)
  r.tracks.forEach((t, i) => {
    const p = pos.get(t.id)!
    const endpoint = i === 0 || i === n - 1
    targets.set(t.id, { ...p, role: 'route', side: p.y < mid ? 't' : 'b', endpoint, reveal: endpoint ? 0 : search + i * TRACE_HOP_MS })
  })
  const edges: EdgeT[] = r.hops.map((h, i) => ({
    a: h.from, b: h.to, why: h.why, sim: h.similarity, role: 'routemap', rank: 0, reveal: search + i * TRACE_HOP_MS,
  }))
  return { targets, edges, nodes: r.tracks }
}

/* ---------------- component ---------------- */

export function Graph({
  hood, route, currentId, highlights, onPlay, mapPoints, loading, emptyNote, session = [],
}: {
  /** the whole listening session (oldest first); the graph shows its tail, the library all of it */
  session?: Track[]
  hood: Neighborhood | null
  route: Route | null
  currentId: string | null
  highlights: Set<string>
  onPlay: (t: Track) => void
  mapPoints: MapPoint[]
  loading: boolean
  emptyNote?: string | null
}) {
  const [wrap, { w, h }] = useSize<HTMLDivElement>()
  const [hover, setHover] = useState<string | null>(null)
  const [hoverEdgeKey, setHoverEdgeKey] = useState<string | null>(null)
  const centerArt = useCover(currentId)
  const hoverArt = useCover(hover)

  /* ---- route replay clock: search tree → path trace → unfold ---- */
  const map = useMemo(() => (route ? routeMap(route, w, h) : null), [route, w, h])
  const timing = useMemo(() => (route ? routeTiming(route) : null), [route])
  const [clock, setClock] = useState<{ ts: number; ms: number }>({ ts: -1, ms: 0 })
  const routeTs = route?.ts
  const hasMap = !!map
  useEffect(() => {
    if (routeTs == null || !hasMap || !timing) return
    const t0 = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const ms = now - t0
      setClock({ ts: routeTs, ms })
      if (ms <= timing.total) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeTs, hasMap])
  const elapsed = route && clock.ts === route.ts ? clock.ms : 0
  const phase: 'search' | 'trace' | 'done' | null = !route
    ? null
    : !map || !timing
      ? 'done'
      : elapsed < timing.search ? 'search' : elapsed < timing.total ? 'trace' : 'done'
  const onMap = phase === 'search' || phase === 'trace'

  const layout = useMemo<Layout | null>(() => {
    if (route && map && timing && phase !== 'done') return layoutRouteMap(route, map.pos, timing.search)
    if (route) return layoutRoute(route, currentId, w, h)
    if (hood) return layoutNeighborhood(hood, w, h)
    return null
    // phase (not the clock) is the dependency: layouts change 3 times per replay, not every frame
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hood, route, currentId, w, h, phase === 'done', map])

  // node data persists so exiting nodes can still render while fading out
  const data = useRef(new Map<string, Track>())
  layout?.nodes.forEach((t) => data.current.set(t.id, { ...data.current.get(t.id), ...t }))

  /* ---- tween displayed positions toward layout targets (after a comet hop, if we moved songs) ---- */
  const [frame, setFrame] = useState<Map<string, P>>(new Map())
  const [comet, setComet] = useState<{ a: XY; b: XY; k: number } | null>(null)
  const [reveal, setReveal] = useState({ key: '', delay: 0 })
  const shown = useRef(frame)
  shown.current = frame
  const prevLayout = useRef<Layout | null>(null)
  const revealKey = !route && hood ? `${hood.center}:${hood.mode}` : ''

  useEffect(() => {
    const targets = layout?.targets ?? new Map<string, Target>()
    const from = new Map(shown.current)
    const oldC = centerOf(prevLayout.current), newC = centerOf(layout)
    prevLayout.current = layout
    // moving to a song that's already on screen: travel there first
    const hop = oldC && newC && oldC !== newC && from.has(oldC) && from.has(newC) ? { a: from.get(oldC)!, b: from.get(newC)! } : null
    const delay = hop ? COMET_MS : 0

    const spawn = (newC && targets.get(newC)) || { x: w / 2, y: h / 2 }
    const ids = new Set([...from.keys(), ...targets.keys()])
    const start = new Map<string, P>(), end = new Map<string, P>()
    ids.forEach((id) => {
      const t = targets.get(id)
      const f = from.get(id)
      const inPlace = t && (t.role === 'trail' || t.reveal != null)
      const anchor = f ?? (inPlace ? { x: t!.x, y: t!.y } : spawn)
      start.set(id, f ?? { x: anchor.x, y: anchor.y, o: 0 })
      end.set(id, t ? { x: t.x, y: t.y, o: 1 } : { ...(f as P), o: 0 })
    })
    const t0 = performance.now()
    let raf = 0, first = true
    const tick = (now: number) => {
      if (first) {
        first = false
        setReveal({ key: revealKey, delay: delay + 120 })
      }
      const dt = now - t0
      if (dt < delay && hop) {
        setComet({ ...hop, k: ease(dt / delay) })
        raf = requestAnimationFrame(tick)
        return
      }
      setComet(null)
      const k = ease(Math.min(1, (dt - delay) / TWEEN_MS))
      const m = new Map<string, P>()
      ids.forEach((id) => {
        const a = start.get(id)!, b = end.get(id)!
        if (k === 1 && b.o === 0) return // drop finished exits
        m.set(id, { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, o: a.o + (b.o - a.o) * k })
      })
      setFrame(m)
      if (k < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout])

  const hovered = hover ? data.current.get(hover) : null
  const hp = hover ? frame.get(hover) : null
  const hoverEdge = layout?.edges.find((e) => e.b === hover || (e.role === 'trail' && e.a === hover))

  // minimap marks: global UMAP coords of what's on screen
  const marks = useMemo(() => {
    const pts = (ts: Track[]) => ts.filter((t) => t.x != null).map((t) => [t.x!, t.y!] as [number, number])
    if (route) return { path: pts(route.tracks), current: pts(route.tracks.filter((t) => t.id === currentId))[0], cands: [] }
    if (!hood) return null
    // full session path: everything before the on-screen trail, then the trail, then now
    const shown = new Set([...hood.trail.map((t) => t.id), hood.nodes[0].id])
    const earlier = session.filter((t) => !shown.has(t.id))
    return { path: pts([...earlier, ...hood.trail, hood.nodes[0]]), current: pts([hood.nodes[0]])[0],
      cands: pts(hood.nodes.slice(1)) }
  }, [hood, route, currentId, session])
  // songs of this session that don't fit in the on-screen trail (they're drawn in the library view)
  const earlierCount = useMemo(() => {
    if (!hood) return 0
    const shown = new Set([...hood.trail.map((t) => t.id), hood.nodes[0].id])
    return new Set(session.filter((t) => !shown.has(t.id)).map((t) => t.id)).size
  }, [hood, session])

  /* ---- path replay overlays ---- */
  const explored = route?.explored?.filter((e) => map?.pos.has(e.id)) ?? []
  const shownExplored = phase === 'search' && timing ? Math.min(explored.length, Math.floor(elapsed / timing.step) + 1) : explored.length
  let tracer: XY | null = null
  if (phase === 'trace' && map && timing && route) {
    const pts = route.tracks.map((t) => map.pos.get(t.id)!)
    const f = Math.min(1, (elapsed - timing.search) / timing.trace) * (pts.length - 1)
    const i = Math.min(pts.length - 2, Math.floor(f)), k = f - i
    if (pts.length > 1 && f < pts.length - 1) tracer = { x: pts[i].x + (pts[i + 1].x - pts[i].x) * k, y: pts[i].y + (pts[i + 1].y - pts[i].y) * k }
  }
  // on the map, only the endpoints and the node the tracer just reached are labelled (clusters overlap)
  const latestReached = onMap && layout
    ? [...layout.targets].filter(([, t]) => !t.endpoint && t.reveal != null && t.reveal <= elapsed).sort((a, b) => b[1].reveal! - a[1].reveal!)[0]?.[0]
    : undefined
  const centerPos = (() => {
    const id = centerOf(layout)
    return id ? frame.get(id) : undefined
  })()

  const status = (() => {
    if (!route || !timing) return null
    if (phase === 'search') return <>searching {spaceName(route.mode)} graph · <b>{shownExplored}</b> tracks explored</>
    if (phase === 'trace')
      return route.tool === 'arc'
        ? <>energy arc · <b>{route.tracks.length}</b> tracks</>
        : route.tool === 'set'
          ? <>set · <b>{route.tracks.length}</b> tracks · style drifts start → end, energy {route.curve ?? 'rise'}</>
        : route.tool === 'queue'
          ? <>queue · <b>{route.tracks.length}</b> tracks</>
        : route.method === 'interpolation'
          ? <>no graph route · <b>interpolating</b> through embedding space</>
          : <>path found · <b>{route.tracks.length - 1}</b> hops · {explored.length}{explored.length >= 160 ? '+' : ''} explored</>
    return null
  })()

  /* ---- focus: graph (library as thumbnail) <-> library (graph as thumbnail) ---- */
  // remembers which route was on screen when the library opened: a new path/arc pulls focus back to the graph
  const [lib, setLib] = useState<{ routeTs: number | null } | null>(null)
  const libFocus = !!lib && lib.routeTs === (route?.ts ?? null) && !onMap && !!layout
  useEffect(() => {
    if (!libFocus) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setLib(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [libFocus])
  const S = Math.min(200 / Math.max(w, 1), 150 / Math.max(h, 1))
  const insetBox = { x: w - w * S - INSET_MARGIN, y: h - h * S - INSET_MARGIN, w: w * S, h: h * S }
  const layerStyle = (inset: boolean): CSSProperties =>
    inset ? ({ transform: `translate(${insetBox.x}px, ${insetBox.y}px) scale(${S})`, ['--s' as string]: S } as CSSProperties) : { transform: 'none' }
  const nowTitle = currentId ? data.current.get(currentId)?.title : undefined

  return (
    <div className="graph" ref={wrap}>
      {!layout && (
        <div className="graph-empty">
          <Galaxy points={mapPoints} width={Math.min(w * 0.8, 720)} height={Math.min(h * 0.7, 520)} />
          <div className="graph-empty-copy">
            <div className="kicker">{mapPoints.length ? `${mapPoints.length.toLocaleString()} tracks · your listening graph` : 'loading library…'}</div>
            <div className="big">{emptyNote ?? 'Start talking. Pick a vibe.'}</div>
          </div>
        </div>
      )}

      {layout && (
      <div
        className={`layer layer-graph ${libFocus ? 'inset' : ''}`}
        style={layerStyle(libFocus)}
        onClick={libFocus ? () => setLib(null) : undefined}
      >
      {map && (
        <div className={`route-map ${onMap ? '' : 'out'}`}>
          <Galaxy points={mapPoints} width={w} height={h} view={map.bounds} pad={MAP_PAD} tint={route?.mode} />
        </div>
      )}

      {layout && (
        <svg width={w} height={h} className="graph-svg">
          <defs>
            <radialGradient id="halo">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.35" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </radialGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <clipPath id="art-clip"><circle r={CENTER_R} /></clipPath>
            {/* any-size round crop for trail covers */}
            <clipPath id="art-clip-unit" clipPathUnits="objectBoundingBox"><circle cx={0.5} cy={0.5} r={0.5} /></clipPath>
          </defs>

          {/* nearest-neighbour scan: a radar ring sweeps out as candidates are ranked */}
          {revealKey && reveal.key === revealKey && centerPos && (
            <circle
              key={`radar:${reveal.key}`}
              className="radar"
              cx={centerPos.x}
              cy={centerPos.y}
              r={Math.max(w, h) * 0.6}
              style={{ animationDelay: `${reveal.delay}ms` }}
            />
          )}

          {/* path search tree, replayed in the order Dijkstra settled the nodes */}
          {map && explored.length > 0 && (
            <g className={`search ${onMap ? '' : 'out'}`}>
              {explored.slice(0, shownExplored).map((e) => {
                const p = map.pos.get(e.id)!
                const q = e.parent ? map.pos.get(e.parent) : undefined
                return (
                  <g key={e.id}>
                    {q && <line className="sx-line" x1={q.x} y1={q.y} x2={p.x} y2={p.y} />}
                    <circle className="sx-dot" cx={p.x} cy={p.y} r={2.4} />
                  </g>
                )
              })}
            </g>
          )}

          {/* edges */}
          <g>
            {layout.edges.map((e) => {
              const a = frame.get(e.a), b = frame.get(e.b)
              if (!a || !b) return null
              if (e.reveal != null && elapsed < e.reveal) return null
              const key = `${e.a}>${e.b}`
              const isHot = hoverEdgeKey === key || hover === e.b || hover === e.a || (e.role === 'candidate' && highlights.has(e.b))
              const strong = e.role !== 'candidate'
              const op = Math.min(a.o, b.o)
              const draws = e.role === 'candidate' || e.role === 'routemap'
              const drawStyle: CSSProperties | undefined =
                e.role === 'candidate' ? { animationDelay: `${reveal.delay + e.rank * RANK_STAGGER_MS}ms` }
                  : e.role === 'routemap' ? { animationDuration: `${TRACE_HOP_MS}ms` } : undefined
              return (
                <g key={key} opacity={op} className={`edge ${e.role} ${isHot ? 'hot' : ''}`}>
                  <line
                    key={draws ? `${reveal.key}` : undefined}
                    className={draws ? 'draw' : undefined}
                    pathLength={draws ? 1 : undefined}
                    style={drawStyle}
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    strokeOpacity={strong ? 1 : 0.25 + e.sim * 0.6}
                  />
                  {/* fat invisible line: easy hover target for the "why" popup */}
                  {e.role !== 'routemap' && (
                    <line
                      className="edge-hit"
                      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      onMouseEnter={() => setHoverEdgeKey(key)}
                      onMouseLeave={() => setHoverEdgeKey((k) => (k === key ? null : k))}
                    />
                  )}
                </g>
              )
            })}
          </g>

          {/* comet: travelling to the chosen next song */}
          {comet && (
            <g className="comet">
              <line x1={comet.a.x} y1={comet.a.y} x2={comet.a.x + (comet.b.x - comet.a.x) * comet.k} y2={comet.a.y + (comet.b.y - comet.a.y) * comet.k} />
              <circle cx={comet.a.x + (comet.b.x - comet.a.x) * comet.k} cy={comet.a.y + (comet.b.y - comet.a.y) * comet.k} r={6} filter="url(#glow)" />
            </g>
          )}
          {tracer && (
            <g className="comet">
              <circle cx={tracer.x} cy={tracer.y} r={6} filter="url(#glow)" />
            </g>
          )}

          {/* nodes */}
          <g>
            {[...frame.entries()].map(([id, p]) => {
              const t = data.current.get(id)
              const tg = layout.targets.get(id)
              if (!t) return null
              if (tg?.reveal != null && elapsed < tg.reveal) return null
              const role = tg?.role ?? 'candidate'
              const isCenter = role === 'center' || role === 'route-current'
              const hl = highlights.has(id)
              const age = tg?.age ?? 0
              const isTrail = role === 'trail'
              const isCand = role === 'candidate'
              // covers instead of names so labels don't pile up: history keeps only the previous song's title,
              // candidates only the top 3 (everything else on hover)
              const showLabel = hover === id || (isTrail ? age <= 1 : isCand ? (tg?.rank ?? 0) < 3 || hl : true)
              const r = isCenter ? CENTER_R : isTrail ? TRAIL_R : tg?.endpoint && onMap ? 9
                : isCand ? Math.max(10, Math.min(15, 6 + 1.4 * Math.log1p(t.plays ?? 0)))
                : Math.max(5, Math.min(13, 4 + 1.6 * Math.log1p(t.plays ?? 0)))
              const side = tg?.side ?? 'r'
              const lx = side === 'r' ? r + 9 : side === 'l' ? -(r + 9) : 0
              const ly = side === 't' ? -(r + 22) : side === 'b' ? r + 16 : -2
              const anchor = side === 'r' ? 'start' : side === 'l' ? 'end' : 'middle'
              const op = p.o * (role === 'route-past' ? 0.55 : 1)
              const title = t.title.length > 28 ? t.title.slice(0, 27) + '…' : t.title
              // candidates pop in by similarity rank; route-map nodes pop when the tracer reaches them
              const pop = role === 'candidate' ? { key: reveal.key, delay: reveal.delay + (tg?.rank ?? 0) * RANK_STAGGER_MS }
                : tg?.reveal != null && onMap ? { key: `r:${route?.ts}`, delay: 0 } : null
              return (
                <g
                  key={id}
                  transform={`translate(${p.x},${p.y})`}
                  opacity={op}
                  className={`node ${role} ${tg?.endpoint && onMap ? 'endpoint' : ''} ${onMap && !tg?.endpoint && id !== latestReached ? 'quiet' : ''} ${hl ? 'hl' : ''} ${t.playable === false ? 'unplayable' : ''} ${hover === id ? 'hover' : ''}`}
                  onMouseEnter={() => setHover(id)}
                  onMouseLeave={() => setHover((x) => (x === id ? null : x))}
                  onClick={() => !isCenter && onPlay(t)}
                >
                  <g key={pop?.key} className={pop ? 'pop' : undefined} style={pop ? { animationDelay: `${pop.delay}ms` } : undefined}>
                    {isCenter && <circle r={72} fill="url(#halo)" className="halo" />}
                    {isCenter && <circle r={r + 7} className="pulse" />}
                    <circle r={r + 10} className="hit" />
                    <circle r={r} className="dot" filter={hl || isCenter || (tg?.endpoint && onMap) ? 'url(#glow)' : undefined} />
                    {isCenter && id === currentId && centerArt && (
                      <image key={centerArt} href={centerArt} x={-r} y={-r} width={r * 2} height={r * 2} clipPath="url(#art-clip)" className="art" preserveAspectRatio="xMidYMid slice" />
                    )}
                    {(isTrail || isCand) && <TrailArt id={id} r={r} />}
                    {(isTrail || isCand) && (hl || hover === id) && <circle r={r + 2.5} className="art-ring" />}
                    {role === 'candidate' && t.similarity != null && (
                      <text className="score" y={-r - 7} textAnchor="middle">{t.similarity.toFixed(3)}</text>
                    )}
                    {showLabel && (
                      <g className="node-label" transform={`translate(${lx},${ly})`} textAnchor={anchor}>
                        <text className={`t ${isCenter ? 'big' : ''}`}>{title}</text>
                        <text className="a" dy={isCenter ? 16 : 14}>{t.artist}</text>
                      </g>
                    )}
                  </g>
                </g>
              )
            })}
          </g>
        </svg>
      )}

      {/* narration chip: what the model is doing right now */}
      {status && <div className="status-chip live"><span className="spin" />{status}</div>}
      {!route && hood && reveal.key === revealKey && revealKey && (
        <div key={reveal.key} className="status-chip fleeting" style={{ animationDelay: `${reveal.delay}ms` }}>
          <span className="spin" />
          nearest neighbours in <b>{spaceName(hood.mode)}</b> space · top {hood.nodes.length - 1} of {mapPoints.length ? mapPoints.length.toLocaleString() : '…'}
        </div>
      )}

      {(() => {
        // edge popup: why this hop, floating above the edge's midpoint
        const e = !hover && hoverEdgeKey ? layout?.edges.find((x) => `${x.a}>${x.b}` === hoverEdgeKey) : undefined
        const a = e && frame.get(e.a), b = e && frame.get(e.b)
        if (!e || !a || !b) return null
        const from = data.current.get(e.a), to = data.current.get(e.b)
        return (
          <div className="edge-tip" style={{ left: (a.x + b.x) / 2, top: (a.y + b.y) / 2 }}>
            <div className="edge-tip-route">{from?.title} <span>→</span> {to?.title}</div>
            {e.why.length > 0 && (
              <div className="edge-tip-why">{e.why.map((w) => <span key={w}>{shortWhy(w)}</span>)}</div>
            )}
            <div className="edge-tip-sim">similarity {e.sim.toFixed(3)}</div>
          </div>
        )
      })()}

      {hovered && hp && (
        <div className="tip" style={{ left: hp.x, top: hp.y }}>
          <div className="tip-head">
            {hoverArt ? <img src={hoverArt} alt="" className="tip-art" /> : <div className="tip-art blank" />}
            <div>
              <div className="tip-title">{hovered.title}</div>
              <div className="tip-artist">{hovered.artist}</div>
            </div>
          </div>
          <div className="tip-grid">
            <span>BPM</span><b>{hovered.bpm ? Math.round(hovered.bpm) : '—'}</b>
            <span>Key</span><b>{hovered.camelot ?? '—'}</b>
            <span>Energy</span><b>{hovered.energy != null ? hovered.energy.toFixed(2) : '—'}</b>
            <span>Plays</span><b>{hovered.plays ?? '—'}</b>
            {hovered.similarity != null && (<><span>Sim</span><b>{hovered.similarity.toFixed(3)}</b></>)}
          </div>
          {hoverEdge && hoverEdge.why.length > 0 && <div className="tip-why">{hoverEdge.why.join(' · ')}</div>}
          {hovered.id !== currentId && (
            <div className="tip-cta">{hovered.playable === false ? 'not on Spotify' : 'click to play'}</div>
          )}
        </div>
      )}

      </div>
      )}

      {layout && !onMap && (
        <div
          className={`layer layer-lib ${libFocus ? '' : 'inset'}`}
          style={layerStyle(!libFocus)}
          onClick={libFocus ? undefined : () => setLib({ routeTs: route?.ts ?? null })}
        >
          <LibraryView
            points={mapPoints}
            marks={marks}
            width={w}
            height={h}
            active={libFocus}
            pointScale={libFocus ? 1 : 0.55 / S}
            tint={hood?.mode}
            nowLabel={nowTitle}
            onPlay={(p) => p[6] && onPlay({ id: p[0], title: p[4] ?? '', artist: p[5] ?? '' })}
          />
        </div>
      )}
      {layout && !onMap && !route && !libFocus && earlierCount > 0 && (
        <button className="session-more" onClick={() => setLib({ routeTs: null })}
          title="The whole session is drawn as a path in the library view">
          … +{earlierCount} earlier this session · library <span>↗</span>
        </button>
      )}
      {layout && !onMap && (
        <button
          className="inset-label"
          style={{ left: insetBox.x + 10, top: insetBox.y + 8 }}
          onClick={() => (libFocus ? setLib(null) : setLib({ routeTs: route?.ts ?? null }))}
        >
          {libFocus ? 'graph' : 'library'} <span>↗</span>
        </button>
      )}

      {loading && <div className="graph-loading" />}
    </div>
  )
}
