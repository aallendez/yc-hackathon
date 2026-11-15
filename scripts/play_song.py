# /// script
# requires-python = ">=3.10"
# dependencies = ["spotipy>=2.24", "python-dotenv>=1.0", "requests>=2.31"]
# ///
"""Quick smoke test: play "I'm the Problem" by Morgan Wallen on your Spotify account.

Requirements:
  - Spotify Premium (the playback API refuses to play on free accounts)
  - In your Spotify app dashboard (developer.spotify.com/dashboard), add this
    Redirect URI exactly: http://127.0.0.1:8888/callback
  - SPOTIFY_CLIENT_ID (or SPOTIFY_CLIENT_KEY) in the repo-root .env

Run:
  uv run scripts/play_song.py
  # or: pip install spotipy python-dotenv && python scripts/play_song.py

The first run opens a browser to log in; the token is cached in .spotify_cache.
Uses the PKCE flow, so no client secret is needed.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TITLE, ARTIST = "I'm the Problem", "Morgan Wallen"
TRACK_QUERY = f'track:"{TITLE}" artist:"{ARTIST}"'
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SCOPES = "user-read-playback-state user-modify-playback-state"


def get_client() -> spotipy.Spotify:
    client_id = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY_CLIENT_KEY")
    if not client_id:
        sys.exit("Missing SPOTIFY_CLIENT_ID (or SPOTIFY_CLIENT_KEY) in .env")
    auth = SpotifyPKCE(
        client_id=client_id,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_handler=spotipy.CacheFileHandler(cache_path=str(ROOT / ".spotify_cache")),
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth)


def find_track(sp: spotipy.Spotify) -> dict:
    """Look the track up via ReccoBeats (free, title search, returns Spotify links) so we
    don't spend Spotify's tight dev-mode search quota; fall back to Spotify search."""
    try:
        r = requests.get("https://api.reccobeats.com/v1/track/search",
                         params={"searchText": TITLE, "size": 25}, timeout=20)
        for it in r.json().get("content", []):
            if it["trackTitle"].lower() == TITLE.lower() and any(
                    a["name"].lower() == ARTIST.lower() for a in it["artists"]):
                tid = it["href"].rsplit("/", 1)[-1]
                return {"name": it["trackTitle"], "uri": f"spotify:track:{tid}",
                        "artists": [{"name": a["name"]} for a in it["artists"]]}
    except (requests.RequestException, ValueError, KeyError):
        pass
    results = sp.search(q=TRACK_QUERY, type="track", limit=5)["tracks"]["items"]
    if not results:
        sys.exit("Couldn't find the track.")
    return results[0]


def pick_device(sp: spotipy.Spotify) -> str:
    """Return an active device id, launching the Spotify desktop app if none exist."""
    for attempt in range(10):
        devices = sp.devices()["devices"]
        if devices:
            active = next((d for d in devices if d["is_active"]), devices[0])
            print(f"Using device: {active['name']} ({active['type']})")
            return active["id"]
        if attempt == 0:
            print("No Spotify devices found — trying to open the Spotify app...")
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Spotify"], check=False)
        time.sleep(2)
    sys.exit("No Spotify device available. Open Spotify on any device and retry.")


def main() -> None:
    sp = get_client()
    track = find_track(sp)
    artists = ", ".join(a["name"] for a in track["artists"])
    print(f"Found: {track['name']} — {artists}")

    device_id = pick_device(sp)
    try:
        sp.start_playback(device_id=device_id, uris=[track["uri"]])
    except spotipy.SpotifyException as e:
        if e.http_status == 403:
            sys.exit(f"Playback refused (403). Spotify Premium is required.\n{e.msg}")
        raise
    print("▶ Playing!")


if __name__ == "__main__":
    main()
