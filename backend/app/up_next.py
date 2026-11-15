"""There is always one "up next": the GNN's prediction after the song that's playing.

- Recomputed whenever the song changes (agent play, Spotify skip, autoplay) or the session's taste
  changes (a reaction), and published as an `up_next` event so the UI can draw it in the graph.
- Late binding: it's only added to Spotify's queue ~QUEUE_AT_MS before the current song ends.
  Spotify's API can't remove queued items, so queueing early would leave stale picks behind when
  the prediction changes; until that moment the user (or agent) can still change it.
- A plain "next" from the agent plays exactly this pick (tools.next_track uses current_prediction).
"""

import logging

from fastapi.concurrency import run_in_threadpool

from . import spotify_client, tools
from .events import bus

log = logging.getLogger(__name__)
QUEUE_AT_MS = 15_000

_state: dict = {"for": None, "prediction": None, "queued": False}
# songs the user/agent queued on demand (already in Spotify's queue), oldest first: they come before the
# GNN's own pick, so up-next shows the head of this list and the auto-queue waits until it's empty
user_queue: list[dict] = []


def _playing_id() -> str | None:
    e = bus.last("now_playing")
    return ((e or {}).get("track") or {}).get("id")


def current_prediction() -> dict | None:
    """The published pick, if it's still for the song that's playing."""
    if _state["prediction"] and _state["for"] == _playing_id():
        return _state["prediction"]
    return None


def refresh(force: bool = False) -> None:
    """Recompute for the current song (no-op if already computed, unless force) and publish."""
    cur = _playing_id()
    playing_uri = (((bus.last("now_playing") or {}).get("track")) or {}).get("spotify_uri")
    # drop queued songs that have started (or that we skipped past)
    while user_queue and (user_queue[0].get("id") == cur or user_queue[0].get("spotify_uri") == playing_uri):
        user_queue.pop(0)
    if user_queue:
        head = user_queue[0]
        pred = {"next": head, "after": {"id": cur}, "why": ["you queued it"], "alternatives": [],
                "story": [], "basis": "on-demand queue", "queue": user_queue[1:]}
        _state.update({"for": cur, "prediction": pred, "queued": True})
        bus.publish("up_next", track=head, after=cur, why=pred["why"], alternatives=[], queued=True,
                    queue=user_queue[1:])
        return
    if cur is None or (not force and _state["for"] == cur and _state["prediction"]):
        return
    if _state["for"] == cur and _state["queued"]:
        return  # already handed to Spotify: changing it now would leave a stale queue entry
    try:
        pred = tools.predict_next(tools.NextIn())
    except Exception as e:  # nothing from the library playing, etc.
        _state.update({"for": cur, "prediction": None, "queued": False})
        bus.publish("up_next", track=None, reason=str(getattr(e, "detail", e)))
        return
    _state.update({"for": cur, "prediction": pred, "queued": False})
    bus.publish("up_next", track=pred["next"], after=pred["after"]["id"], why=pred["why"],
                alternatives=pred["alternatives"], queued=False)


async def on_progress(uri: str | None, progress_ms: int, duration_ms: int, is_playing: bool) -> None:
    """Called on every playback poll: hand the pick to Spotify's queue near the end of the song."""
    if user_queue:
        return  # Spotify plays the on-demand queue next; the GNN pick waits until it's empty
    pred = current_prediction()
    if (not pred or _state["queued"] or not is_playing or not duration_ms
            or duration_ms - progress_ms > QUEUE_AT_MS):
        return
    track_uri = pred["next"].get("spotify_uri")
    if not track_uri:
        return
    try:
        await run_in_threadpool(lambda: spotify_client.client().add_to_queue(track_uri))
        _state["queued"] = True
        bus.publish("up_next", track=pred["next"], after=pred["after"]["id"], why=pred["why"],
                    alternatives=pred["alternatives"], queued=True)
    except Exception as e:
        log.warning("queueing up-next failed: %s", e)
