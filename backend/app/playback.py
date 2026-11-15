"""Watch Spotify so the UI follows song changes made outside the DJ (skips, autoplay, the Spotify app).

Polls current playback while at least one UI is connected to /events and publishes:
  now_playing  {track, queue: [], source: "spotify"}   when the track changes
  playback     {is_playing, progress_ms, duration_ms, uri}   every poll (not kept in replay history)
Also serves the UI's transport controls: POST /playback/{play,pause,seek}.
Tracks not in the listening history get in_library=False and their Spotify uri as id.
"""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import spotify_client, tools, up_next
from .events import bus

log = logging.getLogger(__name__)
POLL_S, IDLE_S, ERROR_S = 3.0, 2.0, 30.0
SETTLE_S = 6.0  # after the DJ plays a song, how long Spotify may still report the previous one

_uri_index: tuple[int, dict[str, int]] | None = None
_poke = asyncio.Event()  # set to poll right away (after a transport control)
router = APIRouter(prefix="/playback", tags=["playback"])


def _by_uri() -> dict[str, int]:
    """spotify uri -> store index, rebuilt if the store reloaded."""
    global _uri_index
    df = tools.store.df
    if _uri_index is None or _uri_index[0] != id(df):
        _uri_index = (id(df), {u: i for i, u in df["spotify_uri"].items() if isinstance(u, str) and u})
    return _uri_index[1]


def _by_name(title: str | None, artist: str | None) -> int | None:
    """Same song, different Spotify id (single vs album release): exact normalised title + artist."""
    from .music import norm

    if not title or not artist:
        return None
    df = tools.store.df
    first_artist = norm(artist.split(",")[0])
    hit = df.index[(df["norm_title"] == norm(title)) & (df["norm_artist"] == first_artist)]
    return int(hit[0]) if len(hit) else None


def _last_published_uri() -> str | None:
    e = bus.last("now_playing")
    return e["track"].get("spotify_uri") if e else None


def _snapshot() -> dict | None:
    cur = spotify_client.client().current_playback()
    item = (cur or {}).get("item")
    if not item or item.get("type") != "track":
        return None
    sid = (item.get("linked_from") or {}).get("id") or item["id"]  # original id if Spotify relinked
    images = (item.get("album") or {}).get("images") or []
    return {"uri": f"spotify:track:{sid}", "is_playing": bool(cur.get("is_playing")),
            "progress_ms": cur.get("progress_ms") or 0, "duration_ms": item.get("duration_ms") or 0,
            "title": item.get("name"), "artist": ", ".join(a["name"] for a in item.get("artists", [])),
            "cover": images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)}


async def _sleep(s: float) -> None:
    """Sleep, but wake early when poked."""
    try:
        await asyncio.wait_for(_poke.wait(), timeout=s)
    except asyncio.TimeoutError:
        pass
    _poke.clear()


async def watch() -> None:
    playing: bool | None = None
    while True:
        if not bus.queues or not spotify_client.has_token():  # nobody watching / never logged in
            await _sleep(IDLE_S)
            continue
        try:
            snap = await run_in_threadpool(_snapshot)
        except Exception as e:  # network, auth, rate limit: back off, never crash the loop
            log.warning("spotify poll failed: %s", e)
            await _sleep(ERROR_S)
            continue
        if snap:
            last = bus.last("now_playing")
            # the DJ just started a song: Spotify keeps reporting the old one for a moment — don't
            # push that stale song back to the UI
            settling = bool(last) and last.get("source") != "spotify" and time.time() - last["ts"] < SETTLE_S \
                and snap["uri"] != _last_published_uri()
            if snap["uri"] != _last_published_uri() and not settling:
                i = _by_uri().get(snap["uri"])
                if i is None:
                    i = _by_name(snap["title"], snap["artist"])  # another release of a library song
                track = (tools.store.track(i, in_library=True, cover=snap["cover"], spotify_uri=snap["uri"]) if i is not None else
                         {"id": snap["uri"], "title": snap["title"], "artist": snap["artist"],
                          "spotify_uri": snap["uri"], "playable": True, "in_library": False,
                          "cover": snap["cover"]})
                bus.publish("now_playing", track=track, queue=[], source="spotify")
                up_next.refresh(force=True)  # song changed outside the DJ: predict what follows it
            playing = snap["is_playing"]
            bus.publish("playback", keep=False, is_playing=playing, progress_ms=snap["progress_ms"],
                        duration_ms=snap["duration_ms"], uri=snap["uri"])
            await up_next.on_progress(snap["uri"], snap["progress_ms"], snap["duration_ms"], playing)
        elif playing:
            playing = False
            bus.publish("playback", keep=False, is_playing=False, progress_ms=0, duration_ms=0, uri=None)
        await _sleep(POLL_S)


# ---------- transport controls for the UI ----------

async def _control(fn, *args) -> dict:
    res = await run_in_threadpool(fn, *args)
    if not res.get("ok"):
        raise HTTPException(502, res.get("error", "spotify error"))
    await asyncio.sleep(0.25)  # give Spotify a beat to apply it, then report the real state
    _poke.set()
    return res


@router.post("/play")
async def play() -> dict:
    return await _control(spotify_client.resume)


@router.post("/pause")
async def pause() -> dict:
    return await _control(spotify_client.pause)


class SeekIn(BaseModel):
    position_ms: int


@router.post("/seek")
async def seek(body: SeekIn) -> dict:
    return await _control(spotify_client.seek, body.position_ms)
