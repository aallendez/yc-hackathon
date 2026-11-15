import { useEffect, useRef, useState } from 'react'
import { useConversationControls, useConversationMode, useConversationStatus } from '@elevenlabs/react'
import { hue, type Mode, type Track } from '../lib/types'
import { useCover } from '../lib/useCover'

function Energy({ value }: { value?: number | null }) {
  const n = value == null ? 0 : Math.max(1, Math.round(value * 8))
  return (
    <span className="energy" title={value != null ? `energy ${value.toFixed(2)}` : 'no energy data'}>
      {Array.from({ length: 8 }, (_, i) => <i key={i} className={i < n ? 'on' : ''} />)}
    </span>
  )
}

function Equalizer({ bpm }: { bpm?: number | null }) {
  const beat = bpm ? 60 / bpm : 0.5
  return (
    <span className="eq" style={{ ['--beat' as string]: `${beat}s` }}>
      <i /><i /><i /><i />
    </span>
  )
}

export type Playback = { isPlaying: boolean; progress: number; duration: number; at: number }

const fmt = (ms: number) => {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/** Spotify-style scrubber: interpolates between polls, click to seek. */
function Timeline({ playback, onSeek }: { playback: Playback; onSeek: (ms: number) => void }) {
  const [now, setNow] = useState(() => Date.now())
  const [ghost, setGhost] = useState<number | null>(null)
  useEffect(() => {
    if (!playback.isPlaying) return
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [playback.isPlaying])
  const { duration } = playback
  const pos = Math.min(duration, playback.progress + (playback.isPlaying ? Math.max(0, now - playback.at) : 0))
  const pct = duration ? (pos / duration) * 100 : 0
  const frac = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
  }
  return (
    <div className={`tl ${duration ? '' : 'off'}`}>
      <span className="tl-t">{fmt(pos)}</span>
      <div
        className="tl-bar"
        onClick={(e) => duration && onSeek(frac(e) * duration)}
        onMouseMove={(e) => setGhost(frac(e))}
        onMouseLeave={() => setGhost(null)}
        title={ghost != null && duration ? fmt(ghost * duration) : undefined}
      >
        <div className="tl-track">
          {ghost != null && <div className="tl-ghost" style={{ width: `${ghost * 100}%` }} />}
          <div className="tl-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="tl-knob" style={{ left: `${pct}%` }} />
      </div>
      <span className="tl-t">{fmt(duration)}</span>
    </div>
  )
}

export function NowPlaying({ track, queue, playing }: { track: Track | null; queue: Track[]; playing: boolean }) {
  const fetched = useCover(track?.cover ? null : track?.id)
  const art = track?.cover ?? fetched
  if (!track) {
    return (
      <div className="np idle">
        <div className="cover blank" />
        <div className="np-meta">
          <div className="np-title muted">Nothing playing</div>
          <div className="np-artist">Spotify is waiting for the DJ</div>
        </div>
      </div>
    )
  }
  const h = hue(track.id)
  return (
    <div className={`np ${playing ? '' : 'paused'}`} key={track.id}>
      <div className="cover" style={{ background: `linear-gradient(135deg, hsl(${h} 70% 55%), hsl(${(h + 60) % 360} 60% 22%))` }}>
        {art && <img key={art} src={art} alt="" />}
        <Equalizer bpm={track.bpm} />
      </div>
      {art && <div className="np-glow" style={{ backgroundImage: `url(${art})` }} />}
      <div className="np-meta">
        <div className="np-title">{track.title}</div>
        <div className="np-artist">
          {track.artist}
          {track.in_library === false && <span className="np-new">new to you</span>}
        </div>
        <div className="np-stats">
          <span className="stat"><b>{track.bpm ? Math.round(track.bpm) : '—'}</b> BPM</span>
          <span className="stat"><b>{track.camelot ?? '—'}</b> key</span>
          <Energy value={track.energy} />
          {queue.length > 0 && (
            <span className="np-next" title={queue.map((q) => `${q.title} — ${q.artist}`).join('\n')}>
              next · {queue[0].title}{queue.length > 1 && ` +${queue.length - 1}`}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

/** Centre of the dock, Spotify-style: play/pause above the scrubber. */
export function Transport({ playback, disabled, onToggle, onSeek }: {
  playback: Playback; disabled: boolean; onToggle: () => void; onSeek: (ms: number) => void
}) {
  const playing = playback.isPlaying && !disabled
  return (
    <div className={`transport ${disabled ? 'off' : ''} ${playing ? 'playing' : 'paused'}`}>
      <button className="pp" onClick={onToggle} disabled={disabled} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14"><rect x="3" y="2" width="3" height="10" rx="1" fill="currentColor" /><rect x="8" y="2" width="3" height="10" rx="1" fill="currentColor" /></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14"><path d="M4 2.2v9.6a.6.6 0 0 0 .9.5l7.6-4.8a.6.6 0 0 0 0-1L4.9 1.7a.6.6 0 0 0-.9.5z" fill="currentColor" /></svg>
          )}
      </button>
      <Timeline playback={disabled ? { ...playback, progress: 0, duration: 0 } : playback} onSeek={onSeek} />
    </div>
  )
}

export function Mic({ onStart, error }: { onStart: () => void; error: string | null }) {
  const { endSession, getInputVolume, getOutputVolume } = useConversationControls()
  const { status } = useConversationStatus()
  const { isSpeaking } = useConversationMode()
  const orb = useRef<HTMLButtonElement>(null)
  const connected = status === 'connected'

  // drive the orb from live audio levels
  useEffect(() => {
    const el = orb.current
    if (!connected || !el) return
    let raf = 0
    const loop = () => {
      let v = 0
      try { v = isSpeaking ? getOutputVolume() : getInputVolume() } catch { /* not ready yet */ }
      el.style.setProperty('--lvl', String(Math.min(1, v * 2.2)))
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(raf); el.style.setProperty('--lvl', '0') }
  }, [connected, isSpeaking, getInputVolume, getOutputVolume])

  const state = connected ? (isSpeaking ? 'speaking' : 'listening') : status === 'connecting' ? 'connecting' : 'idle'
  const caption = { speaking: 'DJ speaking', listening: 'Listening', connecting: 'Connecting…', idle: 'Tap to talk' }[state]

  return (
    <div className={`mic ${state}`}>
      <button
        ref={orb}
        className="orb"
        onClick={connected ? endSession : onStart}
        disabled={state === 'connecting'}
        aria-label={connected ? 'End conversation' : 'Start conversation'}
      >
        <span className="orb-ring" />
        <span className="orb-core">
          {connected ? (
            <svg width="14" height="14" viewBox="0 0 14 14"><rect x="3" y="3" width="8" height="8" rx="1.5" fill="currentColor" /></svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
            </svg>
          )}
        </span>
      </button>
      <div className="mic-copy">
        <div className="mic-state">{caption}</div>
        <div className={`mic-sub ${error ? 'err' : ''}`}>{error ?? (connected ? 'tap to end' : 'voice DJ · ElevenLabs')}</div>
      </div>
    </div>
  )
}

export function ModeToggle({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="mode">
      <span className="mode-label">embedding</span>
      <div className="seg" data-mode={mode}>
        <span className="seg-thumb" />
        <button className={mode === 'gnn' ? 'on' : ''} onClick={() => onChange('gnn')}
          title="Graph neural network: each song learned from its sound AND what you play around it">GNN</button>
        <button className={mode === 'columns' ? 'on' : ''} onClick={() => onChange('columns')}
          title="No graph: songs compared only by their raw features (BPM, key, energy, genre tags)">features only</button>
      </div>
    </div>
  )
}
