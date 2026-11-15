import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ConversationProvider,
  useConversationControls,
  useConversationStatus,
  type MessagePayload,
} from '@elevenlabs/react'
import { Architecture } from './components/Architecture'
import { Graph } from './components/Graph'
import type { MapPoint } from './components/Galaxy'
import { Transcript } from './components/Transcript'
import { Mic, ModeToggle, NowPlaying, Transport, type Playback } from './components/Dock'
import { summarize, type Line, type Mode, type Neighborhood, type Route, type Track } from './lib/types'
import './App.css'

const NEIGHBOURS = 10

export default function App() {
  const [tools, setTools] = useState<{ name: string; endpoint: string }[] | null>(null)
  const [lines, setLines] = useState<Line[]>([])
  const push = useCallback((l: Line) => setLines((ls) => [...ls, l]), [])

  useEffect(() => {
    fetch('/agent/tools').then((r) => r.json()).then(setTools).catch(() => setTools([]))
  }, [])

  // Every ElevenLabs client tool is a thin POST to our FastAPI tools service.
  const clientTools = useMemo(() => {
    const map: Record<string, (params: Record<string, unknown>) => Promise<string>> = {}
    for (const t of tools ?? []) {
      map[t.name] = async (params) => {
        const r = await fetch(t.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params ?? {}),
        })
        const result = await r.json().catch(() => ({ detail: r.statusText }))
        push({ kind: 'tool', name: t.name, args: params ?? {}, summary: summarize(result), ok: r.ok, ts: Date.now() })
        return JSON.stringify(result)
      }
    }
    return map
  }, [tools, push])

  if (!tools) return <div className="boot"><span className="boot-dot" /></div>

  return (
    <ConversationProvider
      clientTools={clientTools}
      onMessage={(m: MessagePayload) =>
        // strip expressive voice tags like "[calm]" from the transcript (the voice still uses them)
        push({ kind: m.role === 'user' ? 'user' : 'agent', text: m.message.replace(/\[[a-z ]+\]\s*/gi, ''), ts: Date.now() })
      }
      onError={(e: unknown) => push({ kind: 'agent', text: `⚠ ${String(e)}`, ts: Date.now() })}
    >
      <Session lines={lines} push={push} />
    </ConversationProvider>
  )
}

/** ids of tracks a tool result points at — glow them in the graph */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mentioned(result: any): string[] {
  if (!result) return []
  const list: Track[] = result.results ?? result.tracks ?? result.top_tracks ?? result.followed_by ?? []
  return [...list.map((t) => t.id), result.probe?.id].filter(Boolean)
}

function Session({ lines, push }: { lines: Line[]; push: (l: Line) => void }) {
  const { startSession, sendContextualUpdate } = useConversationControls()
  const { status } = useConversationStatus()
  const connected = status === 'connected'
  const [error, setError] = useState<string | null>(null)

  const [nowPlaying, setNowPlaying] = useState<Track | null>(null)
  const [queue, setQueue] = useState<Track[]>([])
  const [history, setHistory] = useState<string[]>([]) // played track ids, oldest first
  const [session, setSession] = useState<Track[]>([]) // the whole listening session, from the backend
  const [probe, setProbe] = useState<Track | null>(null)
  const [mode, setMode] = useState<Mode>('gnn')
  const [playback, setPlayback] = useState<Playback>(() => ({ isPlaying: true, progress: 0, duration: 0, at: Date.now() }))
  const [route, setRoute] = useState<(Route & { fresh: boolean }) | null>(null)
  const [highlights, setHighlights] = useState<Set<string>>(new Set())
  const [hood, setHood] = useState<Neighborhood | null>(null)
  const [loading, setLoading] = useState(false)
  const [mapPoints, setMapPoints] = useState<MapPoint[]>([])
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [graphNews, setGraphNews] = useState<GraphNews | null>(null)
  const [showArch, setShowArch] = useState(false)

  // The session lives in the scrobbles (KG + Last.fm), not in this tab: seed it on load so a reload
  // keeps the whole trace. Songs played here since then are merged in after it.
  const adoptSession = (s: { active?: boolean; tracks?: (Track & { in_graph?: boolean })[] }) => {
    if (!s?.active || !s.tracks) return
    const tracks = s.tracks.filter((t) => t.in_graph !== false)
    setSession(tracks)
    setHistory((h) => {
      const seed: string[] = []
      for (const t of tracks) if (seed[seed.length - 1] !== t.id) seed.push(t.id)
      return [...seed.filter((id) => !h.includes(id)), ...h]
    })
  }

  useEffect(() => {
    fetch('/graph/map').then((r) => r.json()).then(setMapPoints).catch(() => {})
    fetch('/api/health').then((r) => r.json()).then((h) => h.mode && setMode(h.mode)).catch(() => {})
    fetch('/live/session').then((r) => r.json()).then(adoptSession).catch(() => {})
    fetch('/api/metrics').then((r) => r.json()).then(setMetrics).catch(() => {})
  }, [])

  // Backend event bus: playback, tool results (from any client) and mode changes.
  const lastTs = useRef(0)
  useEffect(() => {
    // The stream can die silently (backend restart behind the Vite proxy) and EventSource won't notice:
    // the server pings every 8 s, and if nothing arrives for 25 s we reopen it. On reconnect the server
    // replays the latest now_playing/playback/up_next, so the UI catches up on whatever it missed.
    let es: EventSource
    let lastSeen = Date.now()
    const seen = new Set<string>()
    const connect = () => {
      es?.close()
      lastSeen = Date.now()
      es = new EventSource('/events')
      es.onmessage = onMessage
    }
    const watchdog = setInterval(() => {
      if (Date.now() - lastSeen > 25_000 || es.readyState === EventSource.CLOSED) connect()
    }, 5_000)
    const onMessage = (e: MessageEvent) => {
      lastSeen = Date.now()
      const ev = JSON.parse(e.data)
      if (ev.type === 'ping') return
      // replayed events are skipped by identity, not by time: events published from worker threads
      // can arrive slightly out of order, and a time cut-off would drop them
      const key = `${ev.type}:${ev.ts}`
      if (seen.has(key)) return
      seen.add(key)
      if (seen.size > 2000) seen.clear()
      if (ev.ts > lastTs.current) lastTs.current = ev.ts
      if (ev.type === 'now_playing') {
        setNowPlaying(ev.track)
        setPlayback((p) => ({ ...p, isPlaying: true, progress: 0, at: Date.now() }))
        setQueue(ev.queue ?? [])
        if (ev.track.in_library !== false)
          setHistory((h) => (h[h.length - 1] === ev.track.id ? h : [...h.filter((id) => id !== ev.track.id), ev.track.id]))
        setRoute((r) => (r ? { ...r, fresh: false } : r))
      } else if (ev.type === 'playback') {
        setPlayback({ isPlaying: ev.is_playing, progress: ev.progress_ms ?? 0, duration: ev.duration_ms ?? 0, at: Date.now() })
      } else if (ev.type === 'mode') {
        setMode(ev.mode)
      } else if (ev.type === 'graph_update' && ev.session) {
        adoptSession({ active: true, tracks: ev.session.tracks }) // Last.fm caught up with new plays
        if (Date.now() / 1000 - ev.ts < 60 && ev.new_edges?.length) setGraphNews({ ...ev, key: ev.ts })
      } else if (ev.type === 'tool') {
        const r = ev.result
        const recent = Date.now() / 1000 - ev.ts < 120 // don't resurrect old routes from the replayed history
        if (['path', 'arc', 'set', 'queue'].includes(ev.tool) && r?.tracks?.length && recent) {
          const next = { tool: ev.tool, tracks: r.tracks, hops: r.hops ?? [], method: r.method, curve: r.curve, mode: r.mode,
            explored: r.explored, ts: ev.ts, fresh: true }
          const ids = (t: Track[]) => t.map((x) => x.id).join(',')
          // playing a set/path already on screen also emits a queue: keep the drawn route, don't replay it
          setRoute((cur) => (ev.tool === 'queue' && cur && ids(cur.tracks) === ids(r.tracks) ? cur : next))
        } else if (['similar', 'probe', 'mix', 'reset'].includes(ev.tool)) {
          setRoute(null)
        }
        if (ev.tool === 'probe' && r?.probe) setProbe(r.probe)
        if (ev.tool === 'reset') setProbe(null)
        const ids = mentioned(r)
        if (ids.length) setHighlights(new Set(ids))
      }
    }
    connect()
    return () => {
      clearInterval(watchdog)
      es.close()
    }
  }, [])

  // keep the agent aware of what's actually playing
  useEffect(() => {
    if (connected && nowPlaying) {
      sendContextualUpdate(
        `now playing: "${nowPlaying.title}" by ${nowPlaying.artist} (track id ${nowPlaying.id}, ` +
          `${nowPlaying.bpm ? Math.round(nowPlaying.bpm) + ' BPM, ' : ''}energy ${nowPlaying.energy ?? '?'})`,
      )
    }
  }, [connected, nowPlaying, sendContextualUpdate])

  // a song outside the listening history has no node to centre on
  const outside = nowPlaying?.in_library === false
  const center = outside ? null : (nowPlaying?.id ?? probe?.id ?? null)
  const trail = history.filter((id) => id !== center).slice(-5).join(',')
  const activeRoute = route && (route.fresh || route.tracks.some((t) => t.id === center)) ? route : null

  useEffect(() => {
    if (!center) return
    const ac = new AbortController()
    let retry: ReturnType<typeof setTimeout> | undefined
    setLoading(true)
    // retry until it lands: a failed fetch (e.g. backend restarting) must not leave the old song's graph up
    const load = (attempt: number) =>
      fetch(`/graph/neighborhood?track=${encodeURIComponent(center)}&k=${NEIGHBOURS}&trail=${encodeURIComponent(trail)}`, { signal: ac.signal })
        .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
        .then((h: Neighborhood) => {
          setHood({ ...h, trail: h.trail ?? [], trail_edges: h.trail_edges ?? [] })
          setLoading(false)
        })
        .catch(() => {
          if (!ac.signal.aborted && attempt < 6) retry = setTimeout(() => load(attempt + 1), 800 * (attempt + 1))
          else if (!ac.signal.aborted) setLoading(false)
        })
    load(0)
    return () => {
      ac.abort()
      clearTimeout(retry)
    }
  }, [center, mode, trail])

  const start = async () => {
    setError(null)
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true })
      const s = await fetch('/agent/session').then((r) => r.json())
      if (!s.conversationToken) throw new Error(s.detail ?? 'no token')
      startSession({ conversationToken: s.conversationToken })
    } catch (e) {
      setError(String(e).replace(/^Error: /, ''))
    }
  }

  const changeMode = async (m: Mode) => {
    if (m === mode) return
    setMode(m)
    const r = await fetch('/tools/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: m }) })
    push({ kind: 'tool', name: 'mode', args: { mode: m }, summary: r.ok ? `switched to ${m} embeddings` : 'failed', ok: r.ok, ts: Date.now() })
    if (connected) sendContextualUpdate(`The user switched the similarity space to "${m}" embeddings from the UI.`)
  }

  // transport: optimistic locally, Spotify confirms via the next playback event
  const position = () => Math.min(playback.duration, playback.progress + (playback.isPlaying ? Date.now() - playback.at : 0))
  const control = async (action: 'play' | 'pause' | 'seek', body?: object) => {
    const r = await fetch(`/playback/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })
    if (!r.ok) {
      const d = await r.json().catch(() => ({}))
      push({ kind: 'tool', name: action, args: {}, summary: `error: ${d.detail ?? r.statusText}`, ok: false, ts: Date.now() })
    }
  }
  const togglePlay = () => {
    const next = !playback.isPlaying
    setPlayback((p) => ({ ...p, isPlaying: next, progress: position(), at: Date.now() }))
    control(next ? 'play' : 'pause')
  }
  const seek = (ms: number) => {
    setPlayback((p) => ({ ...p, progress: ms, at: Date.now() }))
    control('seek', { position_ms: Math.round(ms) })
  }

  const play = async (t: Track) => {
    const r = await fetch('/tools/play', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tracks: [t.id] }) })
    const result = await r.json().catch(() => ({ detail: r.statusText }))
    push({ kind: 'tool', name: 'play', args: { tracks: [t.title] }, summary: summarize(result), ok: r.ok, ts: Date.now() })
  }

  const graphTitle = activeRoute
    ? `${activeRoute.tool === 'path' ? 'path' : activeRoute.tool === 'set' ? 'set' : activeRoute.tool === 'queue' ? 'queue' : `${activeRoute.curve ?? ''} arc`} · ${activeRoute.tracks.length} tracks`
    : hood
      ? `neighbourhood · ${hood.nodes.length - 1} candidates`
      : 'library'

  return (
    <div className="app" data-mode={mode}>
      <aside className="panel left">
        <header className="panel-head">
          <div className="brand"><span className="brand-mark" />Listening Graph</div>
          <span className="panel-tag">transcript</span>
        </header>
        <Transcript lines={lines} connected={connected} />
      </aside>

      <main className="panel right">
        <header className="panel-head">
          <span className="panel-tag">graph · <b>{graphTitle}</b></span>
          <div className="head-right">
            <button className="arch-btn" onClick={() => setShowArch(true)} title="How it works">
              <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden><path d="M2 3h4v3H2zM10 3h4v3h-4zM6 11h4v3H6zM4 6v2.5h8V6M8 8.5V11" fill="none" stroke="currentColor" strokeWidth="1.2" /></svg>
              architecture
            </button>
            {metrics && <MetricBadge metrics={metrics} mode={mode} />}
            <ModeToggle mode={mode} onChange={changeMode} />
          </div>
        </header>
        {graphNews && <GraphNewsToast key={graphNews.key} news={graphNews} onDone={() => setGraphNews(null)} />}
        <Graph
          hood={center ? hood : null}
          emptyNote={outside ? `“${nowPlaying!.title}” isn’t in your listening history yet` : null}
          route={activeRoute}
          currentId={center}
          highlights={highlights}
          onPlay={play}
          mapPoints={mapPoints}
          loading={loading}
          session={session}
        />
      </main>

      {showArch && <Architecture onClose={() => setShowArch(false)} />}

      <footer className="dock">
        <NowPlaying track={nowPlaying} queue={queue} playing={playback.isPlaying} />
        <Transport playback={playback} disabled={!nowPlaying} onToggle={togglePlay} onSeek={seek} />
        <Mic onStart={start} error={error} />
      </footer>
    </div>
  )
}

type Metrics = { metric: string; gnn: number; columns: number; same_artist: number }
type GraphNews = {
  key: number
  scrobbles?: number
  new_edges: { from: string; to: string }[]
  effects: { from_title: string; to_title: string; rank_before: number; rank_after: number }[]
  status?: { last_reembed?: { seconds: number; nodes_moved: number } | null }
}

/** Held-out next-song recall@10 for the active space — the number that says why the GNN is there. */
function MetricBadge({ metrics, mode }: { metrics: Metrics; mode: Mode }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`
  const gnn = mode === 'gnn'
  return (
    <div key={mode} className={`metric ${gnn ? 'good' : 'weak'}`}
      title={`Share of held-out next plays (after the train/test split) found in the top 10. GNN ${pct(metrics.gnn)} · same artist ${pct(metrics.same_artist)} · audio features ${pct(metrics.columns)}`}>
      <span className="metric-label">next-song hit rate</span>
      <b>{pct(gnn ? metrics.gnn : metrics.columns)}</b>
      <span className="metric-vs">{gnn
        ? `${(metrics.gnn / metrics.columns).toFixed(1)}× features`
        : `GNN ${pct(metrics.gnn)}`}</span>
    </div>
  )
}

/** A scrobble just landed: new transition edges, re-embedded, and how the pair moved. */
function GraphNewsToast({ news, onDone }: { news: GraphNews; onDone: () => void }) {
  const done = useRef(onDone)
  done.current = onDone
  useEffect(() => {
    const t = setTimeout(() => done.current(), 9000)
    return () => clearTimeout(t)
  }, [])
  const e = [...news.effects].sort((a, b) => (b.rank_before - b.rank_after) - (a.rank_before - a.rank_after))[0]
  const n = news.new_edges.length
  return (
    <div className="graph-news" onClick={onDone}>
      <div className="gn-head"><span className="gn-dot" />graph updated · <b>+{n} edge{n === 1 ? '' : 's'}</b> · re-embedded{news.status?.last_reembed ? ` ${news.status.last_reembed.nodes_moved.toLocaleString()} songs in ${news.status.last_reembed.seconds}s` : ''}</div>
      {e && (
        <div className="gn-body">
          {e.from_title} <span>→</span> {e.to_title}
          {e.rank_before !== e.rank_after && (
            <em> neighbour rank #{e.rank_before.toLocaleString()} → #{e.rank_after.toLocaleString()}</em>
          )}
        </div>
      )}
    </div>
  )
}
