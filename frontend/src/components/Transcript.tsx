import { useEffect, useRef } from 'react'
import { fmtArgs, type Line } from '../lib/types'

const SUGGESTIONS = [
  'Play something like Radiohead, but less sad',
  'Take me from John Mayer to Daft Punk in 6 songs',
  'What was I into in summer 2023?',
  'Build me a playlist that peaks in the middle',
]

export function Transcript({ lines, connected }: { lines: Line[]; connected: boolean }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) // don't return the Promise newer Chrome gives back
  }, [lines])

  if (!lines.length) {
    return (
      <div className="transcript empty">
        <div className="hint">{connected ? 'Listening… try:' : 'Hit the mic and try:'}</div>
        {SUGGESTIONS.map((s) => (
          <div key={s} className="suggestion">“{s}”</div>
        ))}
      </div>
    )
  }

  return (
    <div className="transcript">
      {lines.map((l, i) =>
        l.kind === 'tool' ? (
          <div key={i} className={`line tool ${l.ok ? '' : 'failed'}`} title={`${l.name}(${fmtArgs(l.args)})\n→ ${l.summary}`}>
            <span className="tool-name">{l.name}</span>
            <span className="tool-args">({fmtArgs(l.args)})</span>
            <span className="tool-arrow">→</span>
            <span className="tool-sum">{l.summary}</span>
          </div>
        ) : (
          <div key={i} className={`line msg ${l.kind}`}>
            <div className="who">
              <span className={`who-pic ${l.kind === 'user' ? 'me' : 'dj'}`}>
                <img src={l.kind === 'user' ? '/me.jpg' : '/dj.png'} alt=""
                  onError={(e) => (e.currentTarget.parentElement!.style.display = 'none')} />
              </span>
              {l.kind === 'user' ? 'you' : 'dj'}
            </div>
            <div className="text">{l.text}</div>
          </div>
        ),
      )}
      <div ref={end} />
    </div>
  )
}
