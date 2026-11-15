export type Track = {
  id: string
  title: string
  artist: string
  bpm?: number | null
  energy?: number | null
  valence?: number | null
  camelot?: string | null
  plays?: number | null
  playable?: boolean
  similarity?: number | null
  x?: number | null
  y?: number | null
  in_library?: boolean
  cover?: string | null // from Spotify's playback state, when known
}

export type Hop = { from: string; to: string; why: string[]; similarity: number }

export type Mode = 'gnn' | 'columns'

export type Line =
  | { kind: 'user' | 'agent'; text: string; ts: number }
  | { kind: 'tool'; name: string; args: Record<string, unknown>; summary: string; ok: boolean; ts: number }

export type Neighborhood = {
  center: string
  mode: Mode
  nodes: Track[] // nodes[0] is the center
  edges: Hop[]
  trail: Track[] // played before center, oldest first
  trail_edges: Hop[]
}

/** A planned sequence from a path/arc tool call. */
export type Explored = { id: string; x: number | null; y: number | null; parent: string | null }
export type Route = {
  tool: 'path' | 'arc' | 'set' | 'queue'; tracks: Track[]; hops: Hop[]; method?: string; curve?: string
  mode?: Mode
  ts: number // event time: identifies this route (replays its animation once)
  explored?: Explored[] // search-tree settle order (path only)
}

export const label = (t: Track) => `${t.title} — ${t.artist}`

/** One-line human summary of a tool result, for the grey transcript rows. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function summarize(result: any): string {
  if (!result || typeof result !== 'object') return String(result ?? '')
  if (result.detail) return `error: ${typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail)}`
  const list: Track[] | undefined = result.results ?? result.tracks ?? result.top_tracks ?? result.followed_by
  if (result.probe) return `probe → ${label(result.probe)}`
  if (result.now_playing) return `${result.ok ? '▶' : '✕'} ${label(result.now_playing)}${result.error ? ` (${result.error})` : ''}`
  if (list?.length) return list.slice(0, 3).map(label).join(' · ') + (list.length > 3 ? ` +${list.length - 3}` : '')
  if (result.sentiment) return `${result.sentiment}${result.ready_to_build ? ' · ready to build' : ''}`
  if (result.mode) return `mode: ${result.mode}`
  return JSON.stringify(result).slice(0, 120)
}

export const fmtArgs = (args: Record<string, unknown>) =>
  Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? `"${v}"` : JSON.stringify(v)}`)
    .join(', ')

/** Deterministic hue from an id, for generated cover tiles. */
export function hue(id: string): number {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  return h % 360
}
