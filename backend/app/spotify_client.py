"""Spotify playback only (play / pause / devices / playlist). No search: dev-mode search quota is tiny.

Shares the token cache with scripts/play_song.py. Adding the playlist scopes means the first
call after this change opens a browser once to re-consent.
"""

import json
import os
import time
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
SCOPES = ("user-read-playback-state user-modify-playback-state "
          "playlist-modify-private playlist-modify-public")

_sp: spotipy.Spotify | None = None
_device: tuple[str, float] | None = None  # (device_id, fetched_at) — avoid polling /devices


def client() -> spotipy.Spotify:
    global _sp
    if _sp is None:
        auth = SpotifyPKCE(
            client_id=os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY_CLIENT_KEY"),
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            scope=SCOPES,
            cache_handler=spotipy.CacheFileHandler(cache_path=str(ROOT / ".spotify_cache")),
        )
        _sp = spotipy.Spotify(auth_manager=auth, retries=0, status_retries=0, requests_timeout=10)
    return _sp


def has_token() -> bool:
    """True once the user has logged in; background polling must never trigger the browser OAuth flow."""
    return (ROOT / ".spotify_cache").exists()


def device_id(refresh: bool = False) -> str | None:
    global _device
    if _device and not refresh and time.time() - _device[1] < 300:
        return _device[0]
    devices = client().devices()["devices"]
    if not devices:
        return None
    d = next((d for d in devices if d["is_active"]), devices[0])
    _device = (d["id"], time.time())
    return d["id"]


def _playing_id() -> tuple[str | None, str | None]:
    """(track id actually playing — original id if Spotify relinked it, track name)."""
    cur = client().current_playback()
    item = (cur or {}).get("item") or {}
    return (item.get("linked_from") or {}).get("id") or item.get("id"), item.get("name")


def _wait_for(want: str, checks: int = 4, title: str | None = None) -> tuple[bool, str | None]:
    """True once Spotify reports the wanted track — by id, or by exact title (Spotify sometimes plays
    the same song from another release under a different id, e.g. single vs album)."""
    name = None
    for _ in range(checks):  # Spotify takes a moment to report the new item
        time.sleep(0.5)
        got, name = _playing_id()
        if got == want or (title and name and name.strip().lower() == title.strip().lower()):
            return True, name
    return False, name


def play(uris: list[str], title: str | None = None) -> dict:
    """Start playback and VERIFY it. The desktop app sometimes answers 202 and just restarts the
    current song; pausing first reliably unsticks it, and queue+skip is the last resort."""
    want = uris[0].rsplit(":", 1)[-1]
    sp = client()
    try:
        dev = device_id()
        if not dev:
            return {"ok": False, "error": "no Spotify device open — open Spotify on any device"}
        # pause first: a plain start while something plays is often ignored by the desktop app
        try:
            sp.pause_playback(device_id=dev)
        except spotipy.SpotifyException as e:
            if e.http_status == 404:
                dev = device_id(refresh=True)  # stale device
            # 403 = already paused, fine
        time.sleep(0.3)
        method = "pause+start"
        sp.start_playback(device_id=dev, uris=uris, offset={"position": 0}, position_ms=0)
        ok, name = _wait_for(want, title=title)
        if not ok:
            method = "queue+skip"
            sp.add_to_queue(uris[0], device_id=dev)
            time.sleep(0.3)
            sp.next_track(device_id=dev)
            ok, name = _wait_for(want, title=title)
        if ok:
            return {"ok": True, "verified": True, "queued": len(uris), "method": method}
        return {"ok": False, "verified": False, "error": f"Spotify accepted the command but is still playing {name!r}"}
    except spotipy.SpotifyException as e:
        return {"ok": False, "error": f"spotify {e.http_status}: {e.msg[:200]}"}


def pause() -> dict:
    try:
        client().pause_playback(device_id=device_id())
        return {"ok": True}
    except spotipy.SpotifyException as e:
        return {"ok": False, "error": f"spotify {e.http_status}: {e.msg[:200]}"}


def resume() -> dict:
    try:
        client().start_playback(device_id=device_id())
        return {"ok": True}
    except spotipy.SpotifyException as e:
        return {"ok": False, "error": f"spotify {e.http_status}: {e.msg[:200]}"}


def seek(position_ms: int) -> dict:
    try:
        client().seek_track(max(0, int(position_ms)), device_id=device_id())
        return {"ok": True}
    except spotipy.SpotifyException as e:
        return {"ok": False, "error": f"spotify {e.http_status}: {e.msg[:200]}"}


def save_playlist(name: str, uris: list[str], tracks: list[dict]) -> dict:
    """Try Spotify; on failure (dev-mode 403s are common) export to data/exports/ instead."""
    try:
        sp = client()
        pl = sp.user_playlist_create(sp.current_user()["id"], name, public=False,
                                     description="Made by voice with the listening-graph DJ")
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(pl["id"], uris[i:i + 100])
        return {"ok": True, "url": pl["external_urls"]["spotify"]}
    except spotipy.SpotifyException as e:
        out = ROOT / "data" / "exports"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{int(time.time())}_{''.join(c for c in name if c.isalnum() or c in ' -_')[:40]}.json"
        path.write_text(json.dumps({"name": name, "tracks": tracks}, indent=2, ensure_ascii=False))
        return {"ok": False, "error": f"spotify {e.http_status}", "exported_to": str(path)}
