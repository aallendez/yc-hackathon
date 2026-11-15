"""Generic tools over the listening graph. Each is POST /tools/<name> with a JSON body, so the
voice agent can call them 1:1 as functions. Every call is also published on the event bus.

  similar      nearest tracks to a track or artist
  path         a hop-by-hop bridge between two tracks (vetoes / BPM / key constraints)
  taste        what you listened to in a period
  mix          embedding arithmetic: positive - negative (+ "with the energy/BPM of X")
  query        parameterised KG questions (no free-form Cypher/SQL)
  probe/react/arc   discovery by reactions -> playlist with an energy arc
  play/pause/save_playlist/mode
"""

import heapq
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import spotify_client
from .events import bus
from .music import explain, is_num, key_relation
from .stories import story
from .store import Store, clean, norm, parse_period

router = APIRouter(prefix="/tools", tags=["tools"])
store: Store  # set by main.py on startup


# ---------- shared filters ----------

class Filters(BaseModel):
    bpm_min: float | None = None
    bpm_max: float | None = None
    energy_min: float | None = Field(None, ge=0, le=1)
    energy_max: float | None = Field(None, ge=0, le=1)
    max_plays: int | None = Field(None, description="only tracks played at most this many times (deep cuts)")
    min_plays: int | None = None
    exclude_artists: list[str] = []
    exclude: list[str] = Field([], description="track ids/titles to skip (vetoes)")
    exclude_tags: list[str] = Field([], description="skip tracks whose artist tags contain any of these, "
                                                    "e.g. ['soundtrack', 'score']")
    playable_only: bool = True
    use_discovery_exclusions: bool = Field(True, description="also apply styles ruled out during discovery")


def mask(f: Filters) -> np.ndarray:
    df = store.df
    m = np.ones(len(df), dtype=bool)
    if f.playable_only:
        m &= store.playable
    for col, lo, hi in (("bpm", f.bpm_min, f.bpm_max), ("energy", f.energy_min, f.energy_max)):
        if lo is not None:
            m &= (df[col] >= lo).to_numpy()
        if hi is not None:
            m &= (df[col] <= hi).to_numpy()
    if f.max_plays is not None:
        m &= (df["plays"] <= f.max_plays).to_numpy()
    if f.min_plays is not None:
        m &= (df["plays"] >= f.min_plays).to_numpy()
    for a in f.exclude_artists:
        m[store.resolve_artist(a)] = False
    for t in f.exclude:
        try:
            m[store.resolve(t)] = False
        except KeyError:
            pass
    for tag in set(f.exclude_tags) | (disco.exclude_tags if f.use_discovery_exclusions else set()):
        m &= ~store.tag_text.str.contains(tag.lower().strip(), regex=False).to_numpy()
    return m


def hop(a: int, b: int) -> dict:
    ta, tb = store.track(a), store.track(b)
    return {"from": ta["id"], "to": tb["id"],
            "why": explain(ta, tb, store.back_to_back(a, b), store.shared_tag(a, b)),
            "similarity": round(float(store.V[a] @ store.V[b]), 3)}


def placed(i: int, **extra) -> dict:
    """track + its 2D library-map position, for UI animations."""
    df = store.df
    return store.track(i, x=clean(df.loc[i, "x"]), y=clean(df.loc[i, "y"]), **extra)


def nearest(v: np.ndarray, m: np.ndarray, k: int, exclude: list[int] = ()) -> list[int]:
    s = store.V @ v
    s[~m] = -np.inf
    s[list(exclude)] = -np.inf
    top = np.argpartition(-s, min(k, len(s) - 1))[:k]
    return [int(i) for i in top[np.argsort(-s[top])] if np.isfinite(s[i])]


def safe(fn):
    """Unknown track/artist -> 404 with a message the agent can read back."""
    try:
        return fn()
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))


# ---------- similar ----------

class SimilarIn(Filters):
    entity: str = Field(..., description="track title ('title - artist' ok), track id, or artist name")
    k: int = Field(8, le=50)
    exclude_same_artist: bool | None = Field(None, description="default: true for artists, false for tracks")


@router.post("/similar")
def similar(body: SimilarIn) -> dict:
    v, members, label = safe(lambda: store.entity_vector(body.entity))
    m = mask(body)
    is_artist = label.startswith("artist:")
    is_genre = label.startswith("genre:")
    if is_genre:
        # the genre's own songs, most typical of how you listen to it first, max 2 per artist
        inside = np.zeros(len(m), dtype=bool)
        inside[members] = True
        m &= inside
        m[_recently_played()] = False
        ranked = nearest(v, m, body.k * 6, exclude=[])
        per_artist: dict[str, int] = {}
        hits = []
        for i in ranked:
            a = store.df.loc[i, "artist"]
            if per_artist.get(a, 0) < 2:
                hits.append(i)
                per_artist[a] = per_artist.get(a, 0) + 1
            if len(hits) == body.k:
                break
    else:
        if body.exclude_same_artist if body.exclude_same_artist is not None else is_artist:
            m[store.df.index[store.df["artist"] == store.df.loc[members[0], "artist"]]] = False
        hits = nearest(v, m, body.k, exclude=members)
    anchor = members[0] if not (is_artist or is_genre) else None
    if is_genre:
        genre = label.split(":", 1)[1].strip()
        n_genre = len(members)
        cur = _now_idx()
        # why THIS song out of the whole genre: it's the most central to how the user listens to it
        results = [store.track(i, similarity=round(float(store.V[i] @ v), 3), why=[
            f"#{r} of your {n_genre} {genre} songs: the closest to the centre of the {genre} you actually play "
            f"(each song weighted by your plays), measured in the GNN space",
            f"you've played it {int(store.df.loc[i, 'plays'])} times"],
            story=story(store, i, cur if cur != i else None)) for r, i in enumerate(hits, 1)]
        out = {"query": label, "mode": store.mode, "results": results,
               "how": f"Out of {n_genre} {genre} songs in your history, ranked by how central each is to your "
                      f"own {genre} listening in the GNN embedding (not by global popularity)."}
    else:
        out = {"query": label, "mode": store.mode,
               "results": [store.track(i, similarity=round(float(store.V[i] @ v), 3),
                                       why=hop(anchor, i)["why"] if anchor is not None else None) for i in hits]}
    bus.publish("tool", tool="similar", args=body.model_dump(exclude_defaults=True), result=out)
    return out


# ---------- path ----------

class PathIn(Filters):
    origin: str
    destination: str
    max_steps: int = Field(8, ge=2, le=20)
    max_bpm_jump: float | None = Field(None, description="penalise hops with a bigger BPM change")
    harmonic: bool = Field(False, description="prefer hops between compatible keys (Camelot)")


@router.post("/path")
def path(body: PathIn) -> dict:
    a, b = safe(lambda: (store.resolve(body.origin), store.resolve(body.destination)))
    m = mask(body)
    m[[a, b]] = True
    nb, sims = store.knn()
    df = store.df
    bpm, cam = df["bpm"].to_numpy(), df["camelot"].to_numpy()

    def cost(u: int, v: int, s: float) -> float:
        c = (1 - s) + 0.03
        if body.max_bpm_jump and is_num(bpm[u]) and is_num(bpm[v]) and abs(bpm[u] - bpm[v]) > body.max_bpm_jump:
            c += 0.5
        if body.harmonic and cam[u] and cam[v] and not key_relation(cam[u], cam[v]):
            c += 0.2
        return c

    dist, prev, hops, heap = {a: 0.0}, {}, {a: 0}, [(0.0, a)]
    explored: list[int] = []  # settle order, so the UI can replay the search
    while heap:
        d, u = heapq.heappop(heap)
        if u == b:
            break
        if d > dist[u] or hops[u] >= body.max_steps:
            continue
        explored.append(u)
        for v, s in zip(nb[u], sims[u]):
            v = int(v)
            if not m[v]:
                continue
            nd = d + cost(u, v, float(s))
            if nd < dist.get(v, np.inf):
                dist[v], prev[v], hops[v] = nd, u, hops[u] + 1
                heapq.heappush(heap, (nd, v))

    if b in prev:
        method, seq = "graph", [b]
        while seq[-1] != a:
            seq.append(prev[seq[-1]])
        seq.reverse()
    else:  # not reachable within max_steps: walk the straight line between the two embeddings
        method, seq, used = "interpolation", [a], {a, b}
        for t in np.linspace(0, 1, body.max_steps + 1)[1:-1]:
            target = (1 - t) * store.V[a] + t * store.V[b]
            nxt = nearest(target / np.linalg.norm(target), m, 1, exclude=list(used))
            if nxt:
                seq.append(nxt[0])
                used.add(nxt[0])
        seq.append(b)

    out = {"mode": store.mode, "method": method, "tracks": [placed(i) for i in seq],
           "hops": [hop(u, v) for u, v in zip(seq, seq[1:])],
           # search tree for the UI (not needed by the agent): first 160 settled nodes + their parent
           "explored": [{"id": store.ids[u], "x": clean(store.df.loc[u, "x"]), "y": clean(store.df.loc[u, "y"]),
                         "parent": store.ids[prev[u]] if u in prev else None} for u in explored[:160]]}
    bus.publish("tool", tool="path", args=body.model_dump(exclude_defaults=True), result=out)
    return {k: v for k, v in out.items() if k != "explored"}  # the agent doesn't need the search tree


# ---------- taste ----------

class TasteIn(BaseModel):
    period: str = Field(..., description="'2023', '2023-06', '2023-01-01..2023-03-31' or 'last 90 days'")
    k: int = Field(8, le=30)


@router.post("/taste")
def taste(body: TasteIn) -> dict:
    try:
        start, end = parse_period(body.period)
    except ValueError:
        raise HTTPException(422, f"can't parse period {body.period!r}")
    s = store.scrobbles
    s = s[(s["played_at"] >= start) & (s["played_at"] < end)]
    if s.empty:
        return {"period": body.period, "plays": 0}
    counts = s["track"].value_counts()
    artist_counts = store.df.loc[s["track"], "artist"].value_counts()
    tag_counts: dict[str, int] = {}
    for i, n in counts.head(300).items():
        for t in store.tags_of(i)[:3]:
            tag_counts[t] = tag_counts.get(t, 0) + int(n)
    df = store.df.loc[counts.index]
    out = {"period": body.period, "plays": int(len(s)), "unique_tracks": int(len(counts)),
           "top_artists": [{"artist": a, "plays": int(n)} for a, n in artist_counts.head(body.k).items()],
           "top_tracks": [store.track(i, plays_in_period=int(n)) for i, n in counts.head(body.k).items()],
           "top_tags": sorted(tag_counts, key=tag_counts.get, reverse=True)[:6],
           "avg_energy": round(float(np.nanmean(df["energy"].repeat(counts.values))), 3)
           if df["energy"].notna().any() else None,
           "avg_bpm": round(float(np.nanmean(df["bpm"].repeat(counts.values))), 1)
           if df["bpm"].notna().any() else None}
    bus.publish("tool", tool="taste", args=body.model_dump(), result=out)
    return out


# ---------- mix ----------

class MixIn(Filters):
    positive: list[str] = Field(..., min_length=1, description="tracks/artists to add")
    negative: list[str] = Field([], description="tracks/artists to subtract")
    match_features_of: str | None = Field(None, description="track whose features the result should match")
    match_features: list[Literal["energy", "bpm", "valence", "danceability"]] = ["energy"]
    k: int = Field(6, le=30)


@router.post("/mix")
def mix(body: MixIn) -> dict:
    def ent(q):
        return safe(lambda: store.entity_vector(q))
    pos, neg = [ent(q) for q in body.positive], [ent(q) for q in body.negative]
    v = sum(p[0] for p in pos) - sum((n[0] for n in neg), np.zeros_like(pos[0][0]))
    v = v / np.linalg.norm(v)
    m = mask(body)
    target = None
    if body.match_features_of:
        t = safe(lambda: store.resolve(body.match_features_of))
        target = store.track(t)
        tol = {"energy": 0.08, "valence": 0.1, "danceability": 0.1, "bpm": 6}
        for f in body.match_features:
            val = store.df.loc[t, f]
            if is_num(val):
                m &= ((store.df[f] - val).abs() <= tol[f]).to_numpy()
    used = [i for p in pos + neg for i in p[1]]
    hits = nearest(v, m, body.k, exclude=used)
    out = {"mode": store.mode, "positive": [p[2] for p in pos], "negative": [n[2] for n in neg],
           "matched_to": target, "results": [store.track(i, similarity=round(float(store.V[i] @ v), 3))
                                             for i in hits]}
    bus.publish("tool", tool="mix", args=body.model_dump(exclude_defaults=True), result=out)
    return out


# ---------- query (parameterised KG questions) ----------

class QueryIn(BaseModel):
    kind: Literal["track_info", "artist_info", "top_artists", "top_tracks", "played_after"]
    track: str | None = None
    artist: str | None = None
    period: str | None = None
    k: int = Field(8, le=30)


@router.post("/query")
def query(body: QueryIn) -> dict:
    df, s = store.df, store.scrobbles
    if body.kind == "track_info":
        i = safe(lambda: store.resolve(body.track or ""))
        played = s.loc[s["track"] == i, "played_at"]
        out = store.track(i, first_played=str(played.min().date()), last_played=str(played.max().date()),
                          tags=store.tags_of(i)[:5], story=story(store, i, _now_idx()))
    elif body.kind == "artist_info":
        members = store.resolve_artist(body.artist or "")
        if not members:
            raise HTTPException(404, f"no artist {body.artist!r} in your library")
        aid = df.loc[members[0], "artist_id"]
        played = s.loc[s["track"].isin(members), "played_at"]
        ranked = df.loc[members].sort_values("plays", ascending=False)
        top = ranked.head(body.k)
        sim = [store.artists.loc[a, "name"] for a in store.similar_artists.get(aid, []) if a in store.artists.index]
        out = {"artist": df.loc[members[0], "artist"], "plays": int(df.loc[members, "plays"].sum()),
               "heard_as": body.artist,
               "exact_match": df.loc[members[0], "norm_artist"] == norm(body.artist or ""),
               "playable_top_tracks": [store.track(i) for i in ranked.index if store.playable[i]][:body.k],
               "tracks_in_library": len(members), "first_played": str(played.min().date()),
               "last_played": str(played.max().date()), "tags": store.artist_tags.get(aid, [])[:6],
               "similar_artists": sim[:6], "top_tracks": [store.track(i) for i in top.index]}
    elif body.kind in ("top_artists", "top_tracks"):
        start, end = parse_period(body.period)
        if start is not None:
            s = s[(s["played_at"] >= start) & (s["played_at"] < end)]
        if body.kind == "top_tracks":
            out = {"period": body.period or "all time", "tracks": [
                store.track(i, plays_in_period=int(n)) for i, n in s["track"].value_counts().head(body.k).items()]}
        else:
            vc = df.loc[s["track"], "artist"].value_counts().head(body.k)
            out = {"period": body.period or "all time",
                   "artists": [{"artist": a, "plays": int(n)} for a, n in vc.items()]}
    else:  # played_after
        i = safe(lambda: store.resolve(body.track or ""))
        nxt = sorted(((n, b) for (a, b), n in store.transitions.items() if a == i), reverse=True)[:body.k]
        out = {"track": store.track(i), "followed_by": [store.track(b, times=n) for n, b in nxt]}
    bus.publish("tool", tool="query", args=body.model_dump(exclude_none=True), result=out)
    return out


# ---------- next: the GNN's prediction after the current song ----------

def _recently_played(n: int = 25) -> list[int]:
    """Library tracks that played recently (newest last), from now_playing events."""
    out: list[int] = []
    for e in bus.history:
        if e.get("type") == "now_playing":
            i = store.idx.get((e.get("track") or {}).get("id"))
            if i is not None:
                out.append(i)
    return out[-n:]


class NextIn(Filters):
    current: str | None = Field(None, description="track to continue from; default = what's playing now")
    exclude_same_artist: bool = False
    play: bool = Field(False, description="also start playing the pick (saves a round trip)")


@router.post("/next")
async def next_track(body: NextIn) -> dict:
    """What naturally comes after the current song: GNN neighbours of the song that's playing,
    boosted by what you've actually played right after it, nudged by the last few songs of the
    session, never repeating something that just played. A plain "next" plays the same song the
    UI shows as up next."""
    from . import up_next

    plain = body.model_dump(exclude_defaults=True).keys() <= {"play"}
    out = up_next.current_prediction() if plain else None
    if out is None:
        out = predict_next(body)
    if body.play:
        out = {**out, "played": await play(PlayIn(tracks=[out["next"]["id"]]))}
    bus.publish("tool", tool="next", args=body.model_dump(exclude_defaults=True),
                result={k: v for k, v in out.items() if k != "played"})
    return out


def predict_next(body: NextIn) -> dict:
    """Score every playable track as the continuation of the current one (no side effects)."""
    if body.current:
        cur = safe(lambda: store.resolve(body.current))
    else:
        playing = (bus.last("now_playing") or {}).get("track") or {}
        cur = store.idx.get(playing.get("id"))
        if cur is None:
            recent = _recently_played()
            if not recent:
                raise HTTPException(409, "nothing from the library is playing — pick a song first "
                                         "(similar / probe_next / play)")
            cur = recent[-1]
    V, df = store.V, store.df
    recent = _recently_played()
    m = mask(body)
    m[recent + [cur]] = False
    if body.exclude_same_artist:
        m &= (df["artist"] != df.loc[cur, "artist"]).to_numpy()

    score = V @ V[cur]
    context = [i for i in recent[-4:] if i != cur]
    if context:  # the session's direction, not just the last song
        ctx = V[context].mean(0)
        score = 0.75 * score + 0.25 * (V @ (ctx / np.linalg.norm(ctx)))
    after = np.zeros(len(score))
    for (a, b), n in store.transitions.items():
        if a == cur:
            after[b] = n
    score = score + 0.08 * np.log1p(after)  # "you usually play this after that"
    if disco.pref is not None:
        score = score + 0.15 * (V @ disco.pref)
    if disco.energy_target is not None:
        score = score - 1.5 * np.nan_to_num((df["energy"] - disco.energy_target).abs().to_numpy(), nan=0.3)
    score[~m] = -np.inf
    top = [int(i) for i in np.argsort(-score)[:5] if np.isfinite(score[i])]
    if not top:
        raise HTTPException(409, "no playable track left after these filters")
    pick = top[0]
    return {"after": store.track(cur), "next": placed(pick, played_after_current=int(after[pick])),
            "why": hop(cur, pick)["why"], "story": story(store, pick, cur),
            "alternatives": [placed(i) for i in top[1:]],
            "basis": "GNN neighbours of the current song + what you usually play after it + this session"}


# ---------- discovery by reactions ----------

class Discovery:
    WEIGHTS = {"hot": 1.0, "warm": 0.4, "cold": -0.8}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pref: np.ndarray | None = None
        self.history: list[dict] = []  # {idx, reaction}
        self.seen: set[int] = set()
        self.current: int | None = None
        self.bpm_target: float | None = None
        self.energy_target: float | None = None
        self.exclude_tags: set[str] = set()  # styles the user ruled out ("no soundtracks"), sticky


disco = Discovery()


class ProbeIn(Filters):
    seed: str | None = Field(None, description="optional starting track/artist")


@router.post("/probe")
def probe_next(body: ProbeIn) -> dict:
    disco.exclude_tags |= {t.lower().strip() for t in body.exclude_tags}
    m = mask(body)
    m[list(disco.seen)] = False
    df = store.df
    rng = np.random.default_rng()  # fresh randomness: don't open every session with the same song
    if body.seed and disco.pref is None:
        v, members, _ = safe(lambda: store.entity_vector(body.seed))
        disco.pref = v
        m[members] = False
    if disco.pref is None:
        # cold start: something you know well, far from what we've already tried
        pool = [i for i in df.sort_values("plays", ascending=False).index[:400] if m[i]]
        if disco.seen:
            seen = store.V[list(disco.seen)]
            pool = sorted(pool, key=lambda i: float((seen @ store.V[i]).max()))[:20]
        choice = int(rng.choice(pool[:20] if disco.seen else pool[:150]))
        reason = ("a different style from the songs tried so far" if disco.seen
                  else "one of your regulars, to start")
    else:
        s = store.V @ disco.pref
        if disco.energy_target is not None:
            s = s - 2.0 * np.nan_to_num((df["energy"] - disco.energy_target).abs().to_numpy(), nan=0.3)
        if disco.bpm_target is not None:
            s = s - np.nan_to_num((df["bpm"] - disco.bpm_target).abs().to_numpy() / 25, nan=0.5)
        recent_artists = {df.loc[h["idx"], "artist"] for h in disco.history[-2:]}
        m &= ~df["artist"].isin(recent_artists).to_numpy()
        s[~m] = -np.inf
        top = np.argsort(-s)[:max(1, 6 - len(disco.history))]  # explore early, exploit later
        choice = int(rng.choice(top))
        liked = [h["idx"] for h in disco.history if h["reaction"] == "hot"]
        reason = ("closest to the songs you liked so far" if liked
                  else "moving away from the songs you didn't like")
    disco.current = choice
    disco.seen.add(choice)
    prev = disco.history[-1]["idx"] if disco.history else None
    played = store.scrobbles.loc[store.scrobbles["track"] == choice, "played_at"]
    out = {"probe": store.track(choice), "reason": reason,
           # exact listening facts, so the agent never has to guess how familiar a song is
           "your_plays": int(store.df.loc[choice, "plays"]),
           "last_played": str(played.max().date()) if len(played) else None,
           "reactions_so_far": len(disco.history), "excluding_styles": sorted(disco.exclude_tags),
           "hop_from_previous": hop(prev, choice) if prev is not None else None,
           "story": story(store, choice, prev if prev is not None else _now_idx())}
    bus.publish("tool", tool="probe", result=out)
    return out


class ReactIn(BaseModel):
    sentiment: Literal["hot", "warm", "cold"] = Field(..., description="hot=love it, warm=close, cold=no")
    tempo: Literal["slower", "faster", "same"] | None = None
    energy: Literal["more", "less", "same"] | None = None
    note: str | None = None


@router.post("/react")
def react(body: ReactIn) -> dict:
    if disco.current is None:
        raise HTTPException(409, "nothing to react to — call probe first")
    i = disco.current
    w = Discovery.WEIGHTS[body.sentiment]
    v = store.V[i]
    raw = (disco.pref if disco.pref is not None else np.zeros_like(v)) + w * v
    disco.pref = raw / (np.linalg.norm(raw) or 1.0)
    cur = store.df.loc[i]
    if body.tempo in ("slower", "faster") and is_num(cur["bpm"]):
        disco.bpm_target = float(cur["bpm"]) * (0.88 if body.tempo == "slower" else 1.12)
    if body.energy in ("more", "less") and is_num(cur["energy"]):
        disco.energy_target = float(np.clip(cur["energy"] + (0.15 if body.energy == "more" else -0.15), 0, 1))
    disco.history.append({"idx": i, "reaction": body.sentiment, "note": body.note})
    hot = sum(h["reaction"] == "hot" for h in disco.history)
    ready = (hot >= 2 and body.sentiment == "hot") or len(disco.history) >= 6
    wants_change = body.sentiment == "cold" or body.tempo in ("slower", "faster") \
        or body.energy in ("more", "less")
    out = {"reacted_to": store.track(i), "sentiment": body.sentiment, "reactions": len(disco.history),
           "bpm_target": disco.bpm_target and round(disco.bpm_target), "energy_target": disco.energy_target,
           "ready_to_build": ready,
           "next_step": ("change the song now: next_track(play=true) — it applies this tempo/energy too"
                         if wants_change and body.sentiment != "cold" else
                         "change the song now: probe_next, then play" if wants_change else
                         "keep this song playing; don't change it until the user asks for another")}
    bus.publish("tool", tool="react", args=body.model_dump(exclude_none=True), result=out)
    from . import up_next
    up_next.refresh(force=True)  # taste moved: the up-next pick may change too
    return out


@router.post("/reset")
def reset_discovery() -> dict:
    disco.reset()
    bus.publish("tool", tool="reset")
    return {"ok": True}


class SetIn(Filters):
    start: list[str] = Field(..., min_length=1, description="where the set begins: songs, artists or genres")
    end: list[str] = Field(..., min_length=1, description="where it ends: songs, artists or genres")
    n: int = Field(15, ge=4, le=40)
    energy: Literal["rise", "flat", "fall"] = "rise"


def _blend(entities: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    """Several songs / artists / genres -> one direction in the embedding (+ their tracks)."""
    vecs, members, labels, missing = [], [], [], []
    for e in entities:
        try:
            v, mem, label = store.entity_vector(e)
        except KeyError:
            missing.append(e)
            continue
        vecs.append(v)
        members += mem
        labels.append(label)
    if not vecs:
        raise HTTPException(404, f"none of {entities} is in your library")
    v = np.mean(vecs, 0)
    return v / np.linalg.norm(v), members, labels + [f"not found: {m}" for m in missing]


@router.post("/set")
def build_set(body: SetIn) -> dict:
    """A DJ set that travels from one sound to another: every song is a small step from the last one,
    the style drifts from `start` to `end` along the way, and the energy follows the requested shape."""
    df, V = store.df, store.V
    va, ma, la = _blend(body.start)
    vb, mb, lb = _blend(body.end)
    energy = df["energy"].to_numpy()

    def mean_energy(members: list[int]) -> float:
        e = energy[members]
        w = df.loc[members, "plays"].to_numpy(dtype=float) + 1
        ok = ~np.isnan(e)
        return float(np.average(e[ok], weights=w[ok])) if ok.any() else 0.6

    # songs the user named (not artists/genres) are played as-is: start songs open, end songs close
    def named_songs(entities: list[str], labels: list[str]) -> list[int]:
        out = []
        for e, l in zip(entities, labels):
            if not (l.startswith("artist:") or l.startswith("genre:") or l.startswith("not found")):
                i = store.resolve(e)
                if store.playable[i]:
                    out.append(i)
        return out

    open_with, close_with = named_songs(body.start, la), named_songs(body.end, lb)
    unplayable = [f"{df.loc[i, 'title']} — {df.loc[i, 'artist']}" for e, l in zip(body.start + body.end, la + lb)
                  if not (l.startswith(("artist:", "genre:", "not found"))) and not store.playable[i := store.resolve(e)]]
    ea, eb = mean_energy(ma), mean_energy(mb)
    if body.energy == "rise":
        ea, eb = min(ea, eb), max(ea, eb, min(ea, eb) + 0.15)
    elif body.energy == "fall":
        ea, eb = max(ea, eb), min(ea, eb)
    else:
        ea = eb = (ea + eb) / 2
    in_a, in_b = np.zeros(len(df)), np.zeros(len(df))
    in_a[ma], in_b[mb] = 1, 1
    familiarity = 0.03 * np.log1p(df["plays"].to_numpy())

    m = mask(body)
    m &= store.playable
    seq: list[int] = list(open_with)
    per_artist: dict[str, int] = {}
    for i in open_with + close_with:
        per_artist[df.loc[i, "artist"]] = per_artist.get(df.loc[i, "artist"], 0) + 1
    m[close_with] = False  # saved for the end
    free = body.n - len(open_with) - len(close_with)
    for k in range(len(open_with), len(open_with) + max(0, free)):
        t = k / (body.n - 1)
        target = (1 - t) * va + t * vb
        target /= np.linalg.norm(target)
        e_t = ea + (eb - ea) * t
        score = V @ target - 1.2 * np.nan_to_num(np.abs(energy - e_t), nan=0.25) + familiarity
        score += 0.12 * (1 - t) * in_a + 0.12 * t * in_b  # the named artists/songs anchor both ends
        if seq:
            sim_prev = V @ V[seq[-1]]
            # smooth: each song close to the one before, and big jumps are expensive
            score += 0.35 * sim_prev - 1.5 * np.clip(0.72 - sim_prev, 0, None)
        ok = m.copy()
        ok[seq] = False
        if seq:
            ok &= (df["artist"] != df.loc[seq[-1], "artist"]).to_numpy()
        full = [a for a, c in per_artist.items() if c >= 2]
        if full:
            ok &= ~df["artist"].isin(full).to_numpy()
        score[~ok] = -np.inf
        if not np.isfinite(score.max()):
            break
        pick = int(np.argmax(score))
        seq.append(pick)
        a = df.loc[pick, "artist"]
        per_artist[a] = per_artist.get(a, 0) + 1
    seq += [i for i in close_with if i not in seq]

    out = {"mode": store.mode, "method": "bridge", "curve": body.energy, "start": la, "end": lb,
           "named_but_unplayable": unplayable,
           "energy_targets": [round(ea + (eb - ea) * k / max(1, body.n - 1), 2) for k in range(len(seq))],
           "tracks": [placed(i) for i in seq], "hops": [hop(u, v) for u, v in zip(seq, seq[1:])]}
    bus.publish("tool", tool="set", args=body.model_dump(exclude_defaults=True), result=out)
    return out


class ArcIn(Filters):
    n: int = Field(8, ge=3, le=30)
    curve: Literal["rise", "fall", "peak", "valley", "flat"] = "peak"
    seed: str | None = Field(None, description="build around this track/artist instead of the discovery preference")


@router.post("/arc")
def build_arc(body: ArcIn) -> dict:
    if body.seed:
        pref = safe(lambda: store.entity_vector(body.seed))[0]
    elif disco.pref is not None:
        pref = disco.pref
    else:
        raise HTTPException(409, "no preference yet — react to some probes or pass a seed")
    df = store.df
    m = mask(body) & df["energy"].notna().to_numpy()
    sim = store.V @ pref
    sim[~m] = -np.inf
    cands = [int(i) for i in np.argsort(-sim)[:300] if np.isfinite(sim[i])]
    if len(cands) < body.n:
        raise HTTPException(409, "not enough tracks with audio features match these filters")
    e = df.loc[cands, "energy"]
    lo, hi = float(e.quantile(0.15)), float(e.quantile(0.85))
    t = np.linspace(0, 1, body.n)
    shape = {"rise": t, "fall": 1 - t, "peak": 1 - np.abs(2 * t - 1), "valley": np.abs(2 * t - 1),
             "flat": np.full_like(t, 0.5)}[body.curve]
    targets = lo + (hi - lo) * shape
    seq: list[int] = []
    artist_count: dict[str, int] = {}
    for target in targets:
        best, best_score = None, -np.inf
        for c in cands:
            a = df.loc[c, "artist"]
            if c in seq or artist_count.get(a, 0) >= 2 or (seq and df.loc[seq[-1], "artist"] == a):
                continue
            score = sim[c] - 3 * abs(df.loc[c, "energy"] - target)
            if seq:
                score += 0.5 * float(store.V[seq[-1]] @ store.V[c])
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        seq.append(best)
        artist_count[df.loc[best, "artist"]] = artist_count.get(df.loc[best, "artist"], 0) + 1
    out = {"mode": store.mode, "curve": body.curve,
           "energy_targets": [round(float(x), 2) for x in targets],
           "tracks": [placed(i) for i in seq], "hops": [hop(u, v) for u, v in zip(seq, seq[1:])]}
    bus.publish("tool", tool="arc", args=body.model_dump(exclude_defaults=True), result=out)
    return out


# ---------- playback / output ----------

class PlayIn(BaseModel):
    tracks: list[str] = Field(..., min_length=1,
                              description="track ids or 'title - artist'; plays the first playable, queues the rest")


def _now_idx() -> int | None:
    """Library index of the song playing now (from the last now_playing event), if any."""
    return store.idx.get(((bus.last("now_playing") or {}).get("track") or {}).get("id"))


async def _resolve_for_play(q: str) -> int | str:
    """A track id / uri / 'title - artist' -> library index, or Spotify uri for a song outside the KG.

    A named song must really be that song: the fuzzy library match is only accepted when the title
    matches; otherwise we look for it on Spotify (artist's releases), and if it isn't reachable we say so
    instead of playing something else."""
    from .catalog import _key, find_song, newest_release

    if q.startswith("spotify:track:") or q in store.idx:
        return q if q.startswith("spotify:track:") else store.idx[q]
    i = safe(lambda: store.resolve(q))
    if " - " not in q:
        return i
    title, artist = (x.strip() for x in q.rsplit(" - ", 1))
    got = _key(store.df.loc[i, "title"])
    if got == _key(title) or got.startswith(_key(title)) or _key(title).startswith(got):
        return i
    found = await run_in_threadpool(find_song, title, artist)
    if found:
        return found["spotify_uri"]
    newest = await run_in_threadpool(newest_release, artist)
    hint = (f" Their newest release is '{newest['name']}' ({newest['release_date']})." if newest else "")
    raise HTTPException(404, f"'{title}' by {artist} isn't in the listening history and can't be reached on "
                             f"Spotify from here right now (brand-new releases may not be linkable yet).{hint} "
                             "Tell the user plainly; do NOT play a different song instead.")


@router.post("/play")
async def play(body: PlayIn) -> dict:
    from .catalog import OUTSIDE

    # each item is a library index, or a Spotify uri for a song outside the KG (from catalog)
    items: list[int | str] = [await _resolve_for_play(q) for q in body.tracks]
    lib = [i for i in items if isinstance(i, int)]
    # no Spotify link in the KG? look the song up in its artist's discography (no search API needed)
    from .catalog import adopt_link, find_link
    for i in dict.fromkeys(lib):
        if not store.playable[i]:
            uri = await run_in_threadpool(find_link, i)
            if uri:
                adopt_link(i, uri)
    skipped = [f"{store.df.loc[i, 'title']} — {store.df.loc[i, 'artist']}" for i in lib if not store.playable[i]]
    ok = [i for i in items if not isinstance(i, int) or store.playable[i]]
    if not ok:
        raise HTTPException(409, f"can't play {', '.join(skipped)}: it's in the listening history but has no "
                                 "Spotify link (not found in the artist's discography either). Tell the user and offer a playable "
                                 "track instead.")

    def info(i: int | str) -> dict:
        if isinstance(i, int):
            return store.track(i)
        return OUTSIDE.get(i) or {"id": i, "title": "?", "artist": "?", "spotify_uri": i, "playable": True,
                                  "in_library": False}

    uris = [store.df.loc[i, "spotify_uri"] if isinstance(i, int) else i for i in ok]
    first = info(ok[0])
    res = await run_in_threadpool(spotify_client.play, uris, first["title"] if first["title"] != "?" else None)
    # now_playing is the track that actually started: the agent must name THIS one, not what it asked for
    prev = _now_idx()
    out = {**res, "now_playing": first, "skipped_unplayable": skipped,
           # specific reasons to say when revealing the song (relative to what played before)
           "story": story(store, ok[0], prev if prev != ok[0] else None) if isinstance(ok[0], int) else
           ["new for you — not in your listening history yet"]}
    if res.get("ok"):
        bus.publish("now_playing", track={**first, "in_library": isinstance(ok[0], int)},
                    queue=[info(i) for i in ok[1:]])
        ok_lib = [i for i in ok if isinstance(i, int)]
        if len(ok_lib) > 2 and len(ok_lib) == len(ok):  # a queue: draw it in the graph like a path
            bus.publish("tool", tool="queue", result={
                "mode": store.mode, "tracks": [placed(i) for i in ok_lib],
                "hops": [hop(u, v) for u, v in zip(ok_lib, ok_lib[1:])]})
        from . import up_next
        up_next.refresh(force=True)
    bus.publish("tool", tool="play", args=body.model_dump(), result=out)
    return out


class QueueIn(BaseModel):
    tracks: list[str] = Field(..., min_length=1, description="track ids, uris or 'title - artist', in order")


@router.post("/queue")
async def queue(body: QueueIn) -> dict:
    """Add songs AFTER the current one, without interrupting it (Spotify's queue)."""
    from .catalog import OUTSIDE

    added, failed = [], []
    for q in body.tracks:
        try:
            i = await _resolve_for_play(q)
        except HTTPException as e:
            failed.append({"track": q, "why": e.detail})
            continue
        if isinstance(i, int) and not store.playable[i]:
            from .catalog import adopt_link, find_link
            uri = await run_in_threadpool(find_link, i)
            if not uri:
                failed.append({"track": q, "why": "no Spotify link"})
                continue
            adopt_link(i, uri)
        uri = store.df.loc[i, "spotify_uri"] if isinstance(i, int) else i
        try:
            await run_in_threadpool(lambda u=uri: spotify_client.client().add_to_queue(u))
        except Exception as e:  # noqa: BLE001
            failed.append({"track": q, "why": f"spotify: {e}"[:160]})
            continue
        added.append(placed(i) if isinstance(i, int) else
                     {**OUTSIDE.get(i, {"id": i, "title": q}), "spotify_uri": uri, "in_library": False})
    from . import up_next
    up_next.user_queue.extend(added)
    up_next.refresh(force=True)  # up-next now shows the head of the on-demand queue
    out = {"queued": added, "failed": failed,
           "queue_now": [f"{t.get('title')} — {t.get('artist', '')}" for t in up_next.user_queue]}
    bus.publish("tool", tool="queue_add", args=body.model_dump(), result=out)
    return out


@router.post("/pause")
async def pause() -> dict:
    res = await run_in_threadpool(spotify_client.pause)
    bus.publish("tool", tool="pause", result=res)
    return res


class PlaylistIn(BaseModel):
    name: str
    tracks: list[str] = Field(..., min_length=1)


@router.post("/save_playlist")
async def save_playlist(body: PlaylistIn) -> dict:
    idxs = [safe(lambda q=q: store.resolve(q)) for q in body.tracks]
    tracks = [store.track(i) for i in idxs]
    uris = [t["spotify_uri"] for t in tracks if t["spotify_uri"]]
    res = await run_in_threadpool(spotify_client.save_playlist, body.name, uris, tracks)
    bus.publish("tool", tool="save_playlist", args={"name": body.name}, result=res)
    return {**res, "tracks": len(tracks)}


class ModeIn(BaseModel):
    mode: Literal["gnn", "columns", "columnas", "features"]


@router.post("/mode")
def set_mode(body: ModeIn) -> dict:
    mode = store.set_mode(body.mode)
    bus.publish("mode", mode=mode)
    return {"mode": mode}
