import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import playback, tools
from .events import bus
from .live import LiveGraph
from .store import Store, clean

load_dotenv(Path(__file__).resolve().parents[2] / ".env")



@asynccontextmanager
async def lifespan(_: FastAPI):
    watcher = asyncio.create_task(playback.watch())  # follow song changes made in Spotify itself
    scrobbles = asyncio.create_task(live_graph.watch())  # Last.fm plays -> new edges -> re-embed
    yield
    watcher.cancel()
    scrobbles.cancel()


app = FastAPI(title="yc-hackathon API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tools.store = Store()
app.include_router(tools.router)
from . import catalog  # noqa: E402

catalog.store = tools.store
catalog.load_links()  # Spotify links found from discographies in earlier runs
app.include_router(catalog.router)
app.include_router(playback.router)
live_graph = LiveGraph(tools.store)


@app.get("/api/health")
def health() -> dict:
    s = tools.store
    return {"status": "ok", "tracks": len(s.ids), "playable": int(s.playable.sum()),
            "with_audio": int(s.df["bpm"].notna().sum()), "mode": s.mode, "ui_streams": len(bus.queues)}


@app.get("/api/metrics")
def metrics() -> dict:
    """Held-out next-track recall@10 (test plays after the train/test split): GNN vs baselines."""
    import json

    m = json.loads((Path(__file__).resolve().parents[2] / "data" / "gnn" / "metrics.json").read_text())
    sage = m.get(f"sage_L{m.get('best_layers', 2)}", {})
    return {"metric": "recall@10", "gnn": sage.get("test"), "columns": m["features_knn"]["test"],
            "same_artist": m["same_artist"]["test"], "markov": m["markov"]["test"],
            "popularity": m["popularity"]["test"]}


@app.get("/events")
async def events() -> StreamingResponse:
    return StreamingResponse(bus.stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/tracks/search")
def search(q: str, k: int = 8) -> list[dict]:
    """Library search for the UI/agent (no Spotify call)."""
    s = tools.store
    try:
        best = s.resolve(q)
    except KeyError:
        return []
    nq = q.lower()
    hits = s.df[s.df["title"].str.lower().str.contains(nq, regex=False)
                | s.df["artist"].str.lower().str.contains(nq, regex=False)]
    order = [best] + [i for i in hits.sort_values("plays", ascending=False).index if i != best]
    return [s.track(i) for i in order[:k]]


_covers: dict[str, str | None] = {}


@app.get("/tracks/cover")
def cover(id: str) -> dict:
    """Album art via Spotify's public oEmbed (no auth, no API quota). Cached in memory."""
    import requests

    s = tools.store
    if id.startswith("spotify:track:"):  # tracks outside the listening history
        uri = id
    else:
        try:
            uri = s.df.loc[s.resolve(id), "spotify_uri"]
        except KeyError as e:
            raise HTTPException(404, str(e))
    if not isinstance(uri, str) or not uri:
        return {"url": None}
    if uri not in _covers:
        try:
            r = requests.get("https://open.spotify.com/oembed", params={"url": uri}, timeout=6)
            _covers[uri] = r.json().get("thumbnail_url") if r.ok else None
        except requests.RequestException:
            return {"url": None}  # transient: don't cache
    return {"url": _covers[uri]}


@app.get("/graph/neighborhood")
def neighborhood(track: str, k: int = 12, trail: str = "") -> dict:
    """Subgraph for the UI: a track, its k nearest neighbours (current mode), 2D coords and why-labels.

    `trail` = comma-separated ids played before `track` (oldest first); returned as a hop chain into it.
    """
    s = tools.store
    try:
        i = s.resolve(track)
    except KeyError as e:
        raise HTTPException(404, str(e))
    prev = [s.idx[t] for t in trail.split(",") if t in s.idx and s.idx[t] != i]

    def node(j: int, **extra) -> dict:
        return s.track(j, x=clean(s.df.loc[j, "x"]), y=clean(s.df.loc[j, "y"]), **extra)

    nb, sims = s.knn()
    cand = [(int(j), float(sim)) for j, sim in zip(nb[i], sims[i]) if int(j) not in prev][:k]
    nodes = [node(i)] + [node(j, similarity=round(sim, 3)) for j, sim in cand]
    edges = [tools.hop(i, j) for j, _ in cand]
    chain = prev + [i]
    return {"center": s.ids[i], "mode": s.mode, "nodes": nodes, "edges": edges,
            "trail": [node(j) for j in prev], "trail_edges": [tools.hop(a, b) for a, b in zip(chain, chain[1:])]}


@app.get("/graph/map")
def graph_map() -> list[list]:
    """All tracks as [id, x, y, plays, title, artist, playable] for the library map."""
    s = tools.store
    return [[r.id, round(r.x, 3), round(r.y, 3), int(r.plays), r.title, r.artist, int(p)]
            for r, p in zip(s.df.itertuples(), s.playable)]


AGENT_STATE = Path(__file__).resolve().parents[2] / "data" / "agent" / "elevenlabs.json"


@app.get("/agent/session")
def agent_session() -> dict:
    """Short-lived WebRTC conversation token for the private ElevenLabs agent (key stays server-side)."""
    import json

    import requests

    if not AGENT_STATE.exists():
        raise HTTPException(503, "no agent yet — run scripts/create_agent.py")
    agent_id = json.loads(AGENT_STATE.read_text())["agent_id"]
    r = requests.get("https://api.elevenlabs.io/v1/convai/conversation/token",
                     params={"agent_id": agent_id},
                     headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}, timeout=20)
    if not r.ok:
        raise HTTPException(502, f"elevenlabs token: {r.status_code} {r.text[:200]}")
    return {"agentId": agent_id, "conversationToken": r.json()["token"]}


@app.get("/agent/tools")
def agent_tools() -> list[dict]:
    """Client tool name -> backend endpoint, so the frontend can register every tool generically."""
    from .agent_spec import TOOLS

    return [{"name": t["name"], "endpoint": f"/tools/{t['endpoint']}"} for t in TOOLS]


@app.post("/admin/reload")
def reload() -> dict:
    """Re-read embeddings / KG after retraining or enrichment (live edges are re-applied)."""
    tools.store.load()
    catalog.load_links()
    if live_graph.new_edges:
        live_graph.reembed()
    return health()


# ---------- live graph (Last.fm scrobbles -> GraphSAGE re-embed) ----------

@app.get("/live/status")
def live_status() -> dict:
    return live_graph.status()


@app.get("/live/session")
def live_session() -> dict:
    """The current listening session as a path through the graph (tracks with 2D coords + hop labels).

    Rebuilt from the scrobbles themselves (KG + Last.fm since), with the KG's own session rule
    (< 30 min between plays), so a UI reload gets the whole session back. active=false when the
    last play is long gone, so the UI doesn't resurrect a session that's over."""
    import pandas as pd

    plays = live_graph.session()
    active = bool(plays) and pd.Timestamp.now(tz="UTC") - plays[-1]["played_at"] < pd.Timedelta(minutes=40)
    return {**live_graph.describe(plays), "active": active,
            "started_at": plays[0]["played_at"].isoformat() if plays else None,
            "status": live_graph.status()}


class FakeScrobbleIn(BaseModel):
    tracks: list[str] = Field(..., min_length=1, description="titles ('title - artist') or track ids, in play order")
    minutes_apart: float = 3.5


@app.post("/admin/fake_scrobble")
def fake_scrobble(body: FakeScrobbleIn) -> dict:
    """Inject plays through the same path as Last.fm (testing / demo), as if just listened to."""
    import pandas as pd

    s = tools.store
    try:
        idxs = [s.resolve(t) for t in body.tracks]
    except KeyError as e:
        raise HTTPException(404, str(e))
    now = pd.Timestamp.now(tz="UTC")
    n = len(idxs)
    scrobbles = [{"artist": s.df.loc[i, "artist"], "title": s.df.loc[i, "title"],
                  "played_at": now - pd.Timedelta(minutes=body.minutes_apart * (n - 1 - k))}
                 for k, i in enumerate(idxs)]
    change = live_graph.ingest(scrobbles, "fake")
    live_graph.publish(change)
    return {"new_edges": [tools.hop(a, b) for a, b in change["new_edges"]], "effects": change["effects"],
            "status": live_graph.status()}
