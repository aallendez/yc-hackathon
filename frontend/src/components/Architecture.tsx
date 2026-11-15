import { useEffect } from 'react'

/** Full-screen architecture diagram for the demo's "how it works" moment. Esc / click outside closes. */
export function Architecture({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="arch-backdrop" onClick={onClose}>
      <div className="arch" onClick={(e) => e.stopPropagation()}>
        <header className="arch-head">
          <div>
            <div className="arch-kicker">how it works</div>
            <h2>A DJ that lives inside your listening history</h2>
          </div>
          <button className="arch-close" onClick={onClose} aria-label="Close">esc ✕</button>
        </header>

        <div className="arch-flow">
          {/* 1. data */}
          <section className="arch-col">
            <div className="arch-step">1 · data</div>
            <Box title="Last.fm scrobbles" sub="5 years · every play, timestamped" />
            <Box title="Enrichment" sub="Deezer → ISRC → ReccoBeats: BPM, key, energy · Last.fm tags & similar artists" />
            <Box title="Knowledge graph" sub="scrobble · track · album · artist · tag" accent
              chips={['FOLLOWED_BY', 'LISTENED_TO', 'SIMILAR_TO', 'TAGGED', 'BY / ON']} />
          </section>

          <Arrow />

          {/* 2. model */}
          <section className="arch-col">
            <div className="arch-step">2 · model</div>
            <Box title="GraphSAGE" sub="2 layers · 46 features per song (audio, key, tags, artist) · trained to predict your next song" accent />
            <div className="arch-metric">
              <div><b>8.2%</b><span>GNN</span></div>
              <div><b>6.1%</b><span>same artist</span></div>
              <div><b>3.8%</b><span>audio features</span></div>
              <em>next-song hit rate (top 10), held-out plays after the train/test split</em>
            </div>
            <Box title="Song embeddings" sub="9,164 songs in one space: neighbours = songs that fit together for you" />
          </section>

          <Arrow />

          {/* 3. tools */}
          <section className="arch-col">
            <div className="arch-step">3 · tools (FastAPI)</div>
            <Box title="Graph tools" sub="similar · next_track · path · build_set · arc · mix · taste · query"
              chips={['every pick + why', 'from real plays']} />
            <Box title="Playback" sub="play · add_to_queue · catalog (new albums) · always one 'up next' queued" />
            <Box title="Live UI" sub="React graph · every tool call streamed over SSE" />
          </section>

          <Arrow />

          {/* 4. voice */}
          <section className="arch-col">
            <div className="arch-step">4 · voice</div>
            <Box title="ElevenLabs agent" sub="Scribe speech-to-text · Claude Haiku for tool calls · Turbo voice · English + Spanish" accent />
            <Box title="Spotify" sub="plays it, verifies it actually started" />
          </section>
        </div>

        <GnnDiagram />

        {/* live loop */}
        <div className="arch-loop">
          <span className="arch-loop-dot" />
          <b>live loop</b>
          <span>you listen → Last.fm every 30 s → new edges in the graph → GraphSAGE re-embeds in <b>0.07 s</b>, no retraining → next picks change</span>
        </div>
      </div>
    </div>
  )
}

function Box({ title, sub, chips, accent }: { title: string; sub: string; chips?: string[]; accent?: boolean }) {
  return (
    <div className={`arch-box ${accent ? 'accent' : ''}`}>
      <div className="arch-box-title">{title}</div>
      <div className="arch-box-sub">{sub}</div>
      {chips && <div className="arch-chips">{chips.map((c) => <span key={c}>{c}</span>)}</div>}
    </div>
  )
}

function Arrow() {
  return (
    <div className="arch-arrow" aria-hidden>
      <svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 12h15m-5-6 6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </div>
  )
}

/** GraphSAGE in one picture: a song pulls in its neighbours (2 hops), then gets a 64-d embedding. */
function GnnDiagram() {
  // target song in the middle, 1-hop ring, 2-hop outer ring
  const cx = 170, cy = 125
  const hop1 = [0, 1, 2, 3, 4].map((k) => {
    const a = (k / 5) * Math.PI * 2 - Math.PI / 2
    return { x: cx + Math.cos(a) * 70, y: cy + Math.sin(a) * 62, a }
  })
  const kinds = ['transition', 'same', 'link', 'transition', 'same'] as const
  const hop2 = hop1.flatMap((p, k) => [-0.38, 0.38].map((d) => ({
    x: cx + Math.cos(p.a + d) * 135, y: cy + Math.sin(p.a + d) * 112, from: k,
  })))
  return (
    <div className="gnn">
      <div className="arch-step">inside the model · GraphSAGE</div>
      <div className="gnn-body">
        <svg viewBox="0 0 340 250" className="gnn-graph" role="img" aria-label="A song aggregating its 1-hop and 2-hop neighbours">
          {hop2.map((q, k) => (
            <line key={`e2${k}`} x1={hop1[q.from].x} y1={hop1[q.from].y} x2={q.x} y2={q.y} className="gnn-e hop2" />
          ))}
          {hop1.map((p, k) => (
            <line key={`e1${k}`} x1={cx} y1={cy} x2={p.x} y2={p.y} className={`gnn-e ${kinds[k]}`} />
          ))}
          {hop2.map((q, k) => <circle key={`n2${k}`} cx={q.x} cy={q.y} r={6} className="gnn-n hop2" />)}
          {hop1.map((p, k) => <circle key={`n1${k}`} cx={p.x} cy={p.y} r={9} className="gnn-n hop1" />)}
          <circle cx={cx} cy={cy} r={26} className="gnn-halo" />
          <circle cx={cx} cy={cy} r={14} className="gnn-n target" />
          <text x={cx} y={cy + 40} textAnchor="middle" className="gnn-label">the song</text>
          <text x={18} y={20} className="gnn-label dim">2 hops = what's around its neighbours</text>
        </svg>

        <div className="gnn-legend">
          <div><i className="sw transition" />played back to back <span>(PMI over 5 years of sessions)</span></div>
          <div><i className="sw same" />same artist</div>
          <div><i className="sw link" />related artists <span>(Last.fm similar)</span></div>
        </div>

        <div className="gnn-pipe">
          <Stage n="46" label="features per song" sub="audio · key (sin/cos) · 32-d tags · listeners" />
          <PipeArrow />
          <Stage n="128" label="projection" sub="linear" />
          <PipeArrow />
          <Stage n="hop 1" label="SAGEConv · mean" sub="+ LayerNorm" accent />
          <PipeArrow />
          <Stage n="hop 2" label="SAGEConv · mean" sub="+ LayerNorm" accent />
          <PipeArrow />
          <Stage n="64" label="embedding" sub="+ skip from features" accent />
          <div className="gnn-train">
            <b>trained on</b> predicting held-out edges — <i>which song you play next</i> · cosine × learned
            temperature · inductive: new songs & edges embed with one forward pass, no retraining
          </div>
        </div>
      </div>
    </div>
  )
}

function Stage({ n, label, sub, accent }: { n: string; label: string; sub: string; accent?: boolean }) {
  return (
    <div className={`gnn-stage ${accent ? 'accent' : ''}`}>
      <b>{n}</b>
      <span>{label}</span>
      <em>{sub}</em>
    </div>
  )
}

function PipeArrow() {
  return <span className="gnn-arrow" aria-hidden>→</span>
}
