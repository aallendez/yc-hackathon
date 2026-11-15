"""One-time login for Spotify 'recently played' (separate token, so playback's login is untouched).

Opens the browser for consent, then saves .spotify_cache_recent. The backend uses it to get exact Spotify
ids for songs you've played lately that aren't in the KG yet (e.g. new singles).
Run: backend/.venv/bin/python scripts/auth_recently_played.py
"""

import os
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
auth = SpotifyPKCE(
    client_id=os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY_CLIENT_KEY"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
    scope="user-read-recently-played",
    cache_handler=spotipy.CacheFileHandler(cache_path=str(ROOT / ".spotify_cache_recent")),
)
sp = spotipy.Spotify(auth_manager=auth, retries=0, requests_timeout=10)
items = sp.current_user_recently_played(limit=50)["items"]
print(f"ok — {len(items)} recently played songs visible, e.g.:")
for it in items[:5]:
    print(" ", it["track"]["name"], "—", it["track"]["artists"][0]["name"])
