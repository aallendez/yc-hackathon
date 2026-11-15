# /// script
# requires-python = ">=3.10"
# dependencies = ["python-dotenv>=1.0", "pandas>=2.0", "pyarrow>=15", "requests>=2.31"]
# ///
"""Fill bpm / key / mode / energy (+ valence, danceability, ...) on track nodes.

No Spotify search is used (Spotify rate-limits search for ~24h after ~1k calls in dev mode).

1. Deezer search (free, no key) finds each KG track, most-played first. Only EXACT matches
   are accepted: same normalized title (case/accents/punctuation ignored, so
   "X - 2019 Remaster" == "X (2019 Remaster)") and the KG artist as Deezer's main artist.
   Gives us the ISRC.            -> data/enriched/deezer_matches.jsonl
   Tracks already matched exactly by the earlier Spotify pass (spotify_ids.jsonl) are skipped.
2. ReccoBeats (free, no key) returns Spotify-style audio features + the Spotify track URL
   for ISRCs, 40 per request.     -> data/enriched/reccobeats_isrc_features.jsonl
3. BPM sanity check: tempo > BPM_CHECK_ABOVE is often double-time. Cross-checked with the
   Deezer track's bpm, then GetSongBPM (if GETSONGBPM_API_KEY is set).
   confirmed / halved / disagrees / unverified   -> data/enriched/bpm_checks.jsonl
4. Merges into data/kg/nodes_track.parquet (from the pristine nodes_track.orig.parquet).

Run:  uv run scripts/enrich_audio_features.py [--limit N] [--workers 4]
Safe to Ctrl-C and re-run; every API response is cached in the jsonl files.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
KG = ROOT / "data" / "kg"
OUT = ROOT / "data" / "enriched"
SPOTIFY_IDS = OUT / "spotify_ids.jsonl"              # from the earlier Spotify-search pass
SPOTIFY_FEATS = OUT / "reccobeats_features.jsonl"    # features keyed by spotify_id
DEEZER_FILE = OUT / "deezer_matches.jsonl"
ISRC_FEATS = OUT / "reccobeats_isrc_features.jsonl"  # features keyed by isrc
BPM_FILE = OUT / "bpm_checks.jsonl"
FEATURES = ["tempo", "key", "mode", "energy", "valence", "danceability",
            "acousticness", "instrumentalness", "liveness", "loudness", "speechiness"]
BPM_CHECK_ABOVE = 170
BPM_TOLERANCE = 0.04

load_dotenv(ROOT / ".env")


# ---------- helpers ----------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(s: str | None) -> str:
    """Case-, accent- and punctuation-insensitive: 'Nunca Te Olvidé!' -> 'nuncateolvide'."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]", "", s.casefold().replace("&", "and"))


BZRP = re.compile(r"(bzrp music sessions,? vol\.? ?\d+)\s*/\s*\d+", re.IGNORECASE)


def title_key(s: str | None) -> str:
    """norm() plus known catalog quirks: 'Duki: Bzrp Music Sessions, Vol. 50/66' == '..., Vol. 50'."""
    return norm(BZRP.sub(r"\1", s or ""))


def close(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) <= BPM_TOLERANCE


class RateLimiter:
    """At most `per_sec` calls per second across threads."""

    def __init__(self, per_sec: float):
        self.gap, self.lock, self.next = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            time.sleep(delay)


DEEZER_RATE = RateLimiter(8)  # Deezer allows 50 requests / 5 s per IP


def get_json(url: str, params: dict | None = None, rate: RateLimiter | None = None, tries: int = 6):
    for attempt in range(tries):
        if rate:
            rate.wait()
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 5))
            if wait > 120:
                raise RuntimeError(f"{url}: rate limited for {wait}s, giving up")
            time.sleep(wait + 1)
            continue
        if not r.ok:
            time.sleep(2 * (attempt + 1))
            continue
        data = r.json()
        if isinstance(data, dict) and data.get("error", {}).get("code") == 4:  # Deezer quota
            time.sleep(5)
            continue
        return data
    return None


def load_tracks() -> pd.DataFrame:
    src = KG / "nodes_track.orig.parquet"  # pristine copy, never overwritten
    if not src.exists():
        pd.read_parquet(KG / "nodes_track.parquet").to_parquet(src, index=False)
    tracks = pd.read_parquet(src)
    by = pd.read_parquet(KG / "edges_track__BY__artist.parquet")
    artists = pd.read_parquet(KG / "nodes_artist.parquet")[["id", "name"]]
    plays = pd.read_parquet(KG / "edges_scrobble__LISTENED_TO__track.parquet")["dst"].value_counts()
    artist_name = by.merge(artists, left_on="dst", right_on="id").set_index("src")["name"]
    tracks["artist"] = tracks["id"].map(artist_name)
    tracks["plays"] = tracks["id"].map(plays).fillna(0).astype(int)
    return tracks.sort_values("plays", ascending=False)


def spotify_exact() -> dict[str, dict]:
    return {r["track_id"]: r for r in read_jsonl(SPOTIFY_IDS) if r.get("match") == "exact"}


# ---------- 1. Deezer exact match -> ISRC ----------

def deezer_exact(title: str, artist: str) -> tuple[dict | None, str | None]:
    candidate = None
    for q in (f"{title} {artist}", title):
        data = get_json("https://api.deezer.com/search", {"q": q, "limit": 25}, DEEZER_RATE)
        items = (data or {}).get("data", [])
        for it in items:
            if norm(it["artist"]["name"]) == norm(artist) and title_key(title) in (
                    title_key(it["title"]),
                    title_key(it.get("title_short")) if not it.get("title_version") else None):
                return it, None
        if items and candidate is None:
            candidate = f'{items[0]["artist"]["name"]} - {items[0]["title"]}'
    return None, candidate


def map_deezer(tracks: pd.DataFrame, limit: int | None, workers: int) -> None:
    # Misses whose title hits a quirk rule added later (e.g. Bzrp) get one more try.
    retry = {r["track_id"] for r in read_jsonl(DEEZER_FILE)
             if r.get("match") == "none" and BZRP.search(r.get("candidate") or "")}
    done = ({r["track_id"] for r in read_jsonl(DEEZER_FILE)} - retry) | set(spotify_exact())
    pool = tracks[tracks["artist"].notna()]
    if limit is not None:
        pool = pool.head(limit)
    todo = [r for r in pool.itertuples() if r.id not in done]
    print(f"Deezer: {len(done)} already matched/cached, searching {len(todo)}", flush=True)
    lock, stats = threading.Lock(), {"n": 0, "exact": 0}
    f = DEEZER_FILE.open("a")

    def work(row) -> None:
        hit, cand = deezer_exact(row.title, row.artist)
        rec = {"track_id": row.id, "match": "exact" if hit else "none",
               "deezer_id": hit["id"] if hit else None, "isrc": hit.get("isrc") if hit else None,
               "deezer_title": hit["title"] if hit else None, "candidate": cand}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            stats["n"] += 1
            stats["exact"] += hit is not None
            if stats["n"] % 250 == 0:
                print(f"  {stats['n']}/{len(todo)}  exact so far {stats['exact'] / stats['n']:.0%}", flush=True)

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    finally:
        f.close()


# ---------- 2. ReccoBeats features by ISRC ----------

def fetch_spotify_features() -> None:
    """Features for tracks matched in the earlier Spotify pass whose features were never fetched."""
    ids = {r["spotify_id"] for r in spotify_exact().values()}
    todo = sorted(ids - {r["spotify_id"] for r in read_jsonl(SPOTIFY_FEATS)})
    print(f"ReccoBeats (spotify ids): fetching {len(todo)}", flush=True)
    with SPOTIFY_FEATS.open("a") as f:
        for i in range(0, len(todo), 40):
            batch = todo[i:i + 40]
            data = get_json("https://api.reccobeats.com/v1/audio-features", {"ids": ",".join(batch)})
            found = {it["href"].rsplit("/", 1)[-1]: {k: it.get(k) for k in FEATURES}
                     for it in (data or {}).get("content", [])}
            for sid in batch:
                f.write(json.dumps({"spotify_id": sid, **found.get(sid, {})}) + "\n")
            time.sleep(0.8)


def fetch_isrc_features() -> None:
    isrcs = {r["isrc"] for r in read_jsonl(DEEZER_FILE) if r.get("isrc")}
    have = {r["isrc"] for r in read_jsonl(ISRC_FEATS)}
    todo = sorted(isrcs - have)
    print(f"ReccoBeats: {len(have)} ISRCs cached, fetching {len(todo)}", flush=True)
    with ISRC_FEATS.open("a") as f:
        for i in range(0, len(todo), 40):
            batch = todo[i:i + 40]
            data = get_json("https://api.reccobeats.com/v1/audio-features", {"ids": ",".join(batch)})
            found: dict[str, dict] = {}
            for it in (data or {}).get("content", []):
                if it.get("isrc") in batch and it["isrc"] not in found and it.get("tempo") is not None:
                    found[it["isrc"]] = {"spotify_id": it["href"].rsplit("/", 1)[-1],
                                         **{k: it.get(k) for k in FEATURES}}
            for isrc in batch:  # misses recorded too, so they aren't refetched
                f.write(json.dumps({"isrc": isrc, **found.get(isrc, {})}) + "\n")
            f.flush()
            if (i // 40) % 25 == 0:
                print(f"  {i + len(batch)}/{len(todo)}", flush=True)
            time.sleep(0.8)  # ReccoBeats allows ~70 requests/min


# ---------- 3. BPM cross-check ----------

def deezer_bpm(deezer_id, title: str, artist: str) -> float | None:
    if not deezer_id:
        hit, _ = deezer_exact(title, artist)
        deezer_id = hit["id"] if hit else None
    if not deezer_id:
        return None
    bpm = (get_json(f"https://api.deezer.com/track/{deezer_id}", rate=DEEZER_RATE) or {}).get("bpm")
    return float(bpm) if bpm else None


def getsongbpm_bpm(title: str, artist: str) -> float | None:
    key = os.getenv("GETSONGBPM_API_KEY")
    if not key:
        return None
    data = get_json("https://api.getsong.co/search/",
                    {"api_key": key, "type": "both", "lookup": f"song:{title} artist:{artist}"})
    results = (data or {}).get("search")
    for it in results if isinstance(results, list) else []:
        if norm((it.get("artist") or {}).get("name")) == norm(artist) and it.get("tempo"):
            return float(it["tempo"])
    return None


def check_bpm(merged: pd.DataFrame, workers: int) -> None:
    have = {r["track_id"] for r in read_jsonl(BPM_FILE)}
    todo = merged[(merged["bpm"] > BPM_CHECK_ABOVE) & ~merged["id"].isin(have)]
    print(f"BPM check (> {BPM_CHECK_ABOVE}): {len(have)} cached, checking {len(todo)}", flush=True)
    lock = threading.Lock()
    f = BPM_FILE.open("a")

    def work(row) -> None:
        ref, source = deezer_bpm(row.deezer_id, row.title, row.artist), "deezer"
        if not ref:
            ref, source = getsongbpm_bpm(row.title, row.artist), "getsongbpm"
        if ref and close(ref, row.bpm):
            status, bpm = "confirmed", row.bpm
        elif ref and close(ref, row.bpm / 2):
            status, bpm = "halved", row.bpm / 2
        else:
            status, bpm = ("disagrees" if ref else "unverified"), row.bpm
        with lock:
            f.write(json.dumps({"track_id": row.id, "bpm_raw": row.bpm, "bpm": bpm, "ref_bpm": ref,
                                "ref_source": source if ref else None, "bpm_status": status}) + "\n")
            f.flush()

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo.itertuples()))
    finally:
        f.close()


# ---------- 4. merge ----------

def merged_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per KG track with spotify_id/uri + features from whichever path matched."""
    rows = []
    sp_feats = {r["spotify_id"]: r for r in read_jsonl(SPOTIFY_FEATS)}
    for tid, r in spotify_exact().items():
        feat = sp_feats.get(r["spotify_id"], {})
        rows.append({"id": tid, "match_source": "spotify", "spotify_id": r["spotify_id"],
                     "deezer_id": None, **{k: feat.get(k) for k in FEATURES}})
    isrc_feats = {r["isrc"]: r for r in read_jsonl(ISRC_FEATS)}
    seen = {row["id"] for row in rows}
    dz = {r["track_id"]: r for r in read_jsonl(DEEZER_FILE)}  # last record wins
    for tid, r in dz.items():
        if tid in seen:
            continue
        if r.get("match") != "exact":
            rows.append({"id": tid, "match_source": "none"})
            continue
        feat = isrc_feats.get(r.get("isrc"), {})
        rows.append({"id": tid, "match_source": "deezer", "spotify_id": feat.get("spotify_id"),
                     "deezer_id": r["deezer_id"], **{k: feat.get(k) for k in FEATURES}})
    m = pd.DataFrame(rows, columns=["id", "match_source", "spotify_id", "deezer_id", *FEATURES])
    m = tracks.drop(columns=["bpm", "key", "mode", "energy"], errors="ignore").merge(m, on="id", how="left")
    return m.rename(columns={"tempo": "bpm"})


def write_kg(m: pd.DataFrame) -> None:
    # links found later by scripts/fill_spotify_links.py (second Spotify app) survive a re-run
    extra = {r["track_id"]: r["spotify_id"] for r in read_jsonl(OUT / "spotify_links_extra.jsonl")
             if r.get("spotify_id")}
    missing = m["spotify_id"].isna() & m["id"].isin(extra)
    m.loc[missing, "spotify_id"] = m.loc[missing, "id"].map(extra)
    m["spotify_uri"] = m["spotify_id"].map(lambda s: f"spotify:track:{s}" if isinstance(s, str) else None)
    m["bpm_raw"] = m["bpm"]
    m["bpm_status"] = None
    m.loc[m["bpm"].notna(), "bpm_status"] = "ok"
    checks = read_jsonl(BPM_FILE)
    if checks:
        c = pd.DataFrame(checks).drop_duplicates("track_id", keep="last").set_index("track_id")
        hit = m["id"].isin(c.index) & m["bpm"].notna()
        m.loc[hit, "bpm"] = m.loc[hit, "id"].map(c["bpm"])
        m.loc[hit, "bpm_status"] = m.loc[hit, "id"].map(c["bpm_status"])
    m["key_status"] = "pending"
    m.loc[m["match_source"] == "none", "key_status"] = "no_exact_match"
    m.loc[m["match_source"].isin(["spotify", "deezer"]), "key_status"] = "no_features"
    m.loc[m["bpm"].notna(), "key_status"] = "ok"
    m = m.drop(columns=["artist", "plays"])
    m.to_parquet(KG / "nodes_track.parquet", index=False)
    print("key_status:", m["key_status"].value_counts().to_dict())
    print("bpm_status:", m["bpm_status"].value_counts().to_dict())
    print("match_source:", m["match_source"].value_counts(dropna=False).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process the N most-played tracks")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks()
    try:
        map_deezer(tracks, args.limit, args.workers)
    except KeyboardInterrupt:
        print("\nInterrupted; continuing with what we have.")
    fetch_spotify_features()
    fetch_isrc_features()
    m = merged_features(tracks)
    check_bpm(m, args.workers)
    write_kg(m)


if __name__ == "__main__":
    sys.exit(main())
