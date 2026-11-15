"""Fill missing Spotify links for well-played library songs from their artist's discography.

No search API (blocked for this account): library song of the artist -> artist id -> releases -> tracks.
Throttled, and stops at the first 429 so the player API stays healthy. Writes
data/enriched/spotify_links_catalog.jsonl, which the backend applies at startup / POST /admin/reload.
Run: backend/.venv/bin/python scripts/backfill_links_from_catalog.py [min_plays]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import spotipy  # noqa: E402

from app import catalog, spotify_client  # noqa: E402
from app.store import Store  # noqa: E402

MIN_PLAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PAUSE_S = 0.35

# throttle every Spotify call made through the shared client
sp = spotify_client.client()
orig = sp._internal_call


def throttled(*a, **k):
    time.sleep(PAUSE_S)
    return orig(*a, **k)


sp._internal_call = throttled
sp.retries = 0  # a 429 should stop us, not be retried for minutes

catalog.store = store = Store()
catalog.load_links()
todo = store.df[(~store.playable) & (store.df["plays"] >= MIN_PLAYS)].sort_values("plays", ascending=False)
print(f"{len(todo)} unlinked songs with >= {MIN_PLAYS} plays", flush=True)
found = 0
for n, (i, row) in enumerate(todo.iterrows(), 1):
    try:
        uri = catalog.find_link(i)
    except spotipy.SpotifyException as e:
        if e.http_status == 429:
            print("rate limited — stopping", flush=True)
            break
        uri = None
    except Exception as e:  # noqa: BLE001
        print("skip", row["title"], type(e).__name__, e, flush=True)
        uri = None
    if uri:
        catalog.adopt_link(i, uri)
        found += 1
    if n % 20 == 0:
        print(f"{n}/{len(todo)} checked, {found} linked", flush=True)
print(f"done: {found} linked of {len(todo)}", flush=True)
