# /// script
# requires-python = ">=3.10"
# dependencies = ["spotipy>=2.24", "python-dotenv>=1.0", "pandas>=2.0", "pyarrow>=15"]
# ///
"""Find Spotify links for tracks that enrichment couldn't (no ReccoBeats record / no Deezer match),
using a SECOND Spotify app so the main app's (rate-limited) search quota is never touched.

Most-played first, exact title+artist match only, ~1 search per 1.5 s, and it stops cleanly on a
long 429 instead of hammering. Resumable: results go to data/enriched/spotify_links_extra.jsonl and
are merged into data/kg/nodes_track.parquet (spotify_id / spotify_uri only — no audio features).

Setup (2 min): developer.spotify.com/dashboard -> Create app -> redirect URI
http://127.0.0.1:8888/callback, Web API -> put its Client ID in .env as SPOTIFY_CLIENT_ID_2.
Run: uv run scripts/fill_spotify_links.py [--max 400] [--min-plays 2]
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

ROOT = Path(__file__).resolve().parent.parent
KG = ROOT / "data" / "kg"
OUT = ROOT / "data" / "enriched" / "spotify_links_extra.jsonl"
load_dotenv(ROOT / ".env")


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]", "", s.casefold().replace("&", "and"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def merge() -> None:
    links = {r["track_id"]: r["spotify_id"] for r in read_jsonl(OUT) if r.get("spotify_id")}
    t = pd.read_parquet(KG / "nodes_track.parquet")
    fill = t["spotify_uri"].isna() & t["id"].isin(links)
    t.loc[fill, "spotify_id"] = t.loc[fill, "id"].map(links)
    t.loc[fill, "spotify_uri"] = "spotify:track:" + t.loc[fill, "spotify_id"]
    t.to_parquet(KG / "nodes_track.parquet", index=False)
    print(f"merged {int(fill.sum())} new links; playable now {int(t['spotify_uri'].notna().sum())}/{len(t)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400, help="max tracks to search this run")
    ap.add_argument("--min-plays", type=int, default=2)
    args = ap.parse_args()
    cid = os.getenv("SPOTIFY_CLIENT_ID_2")
    if not cid:
        sys.exit("Set SPOTIFY_CLIENT_ID_2 in .env (a second Spotify app) — see the docstring.")

    t = pd.read_parquet(KG / "nodes_track.parquet")
    by = pd.read_parquet(KG / "edges_track__BY__artist.parquet").set_index("src")["dst"]
    names = pd.read_parquet(KG / "nodes_artist.parquet").set_index("id")["name"]
    plays = pd.read_parquet(KG / "edges_scrobble__LISTENED_TO__track.parquet")["dst"].value_counts()
    t["artist"], t["plays"] = t["id"].map(by).map(names), t["id"].map(plays).fillna(0)
    done = {r["track_id"] for r in read_jsonl(OUT)}
    todo = t[t["spotify_uri"].isna() & (t["plays"] >= args.min_plays) & t["artist"].notna()
             & ~t["id"].isin(done)].sort_values("plays", ascending=False).head(args.max)
    print(f"searching {len(todo)} unplayable tracks (>= {args.min_plays} plays)")

    sp = spotipy.Spotify(auth_manager=SpotifyPKCE(
        client_id=cid, redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
        scope="user-read-private", cache_handler=spotipy.CacheFileHandler(str(ROOT / ".spotify_cache_2"))),
        retries=0, status_retries=0, requests_timeout=15)
    found = 0
    with OUT.open("a") as f:
        try:
            for n, row in enumerate(todo.itertuples(), 1):
                hit = None
                for q in (f'track:"{row.title}" artist:"{row.artist}"', f"{row.title} {row.artist}"):
                    try:
                        items = sp.search(q=q, type="track", limit=10)["tracks"]["items"]
                    except spotipy.SpotifyException as e:
                        if e.http_status == 429:
                            print(f"rate limited after {n - 1} tracks — stopping (re-run later to resume)")
                            raise KeyboardInterrupt
                        raise
                    hit = next((it for it in items if norm(it["name"]) == norm(row.title)
                                and norm(row.artist) in {norm(a["name"]) for a in it["artists"]}), None)
                    time.sleep(1.5)
                    if hit:
                        break
                found += hit is not None
                f.write(json.dumps({"track_id": row.id, "spotify_id": hit["id"] if hit else None}) + "\n")
                f.flush()
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)}  found {found}", flush=True)
        except KeyboardInterrupt:
            pass
    merge()


if __name__ == "__main__":
    main()
