"""Spotify catalog beyond the listening history: an artist's releases and an album's tracks.

No Spotify search or artist-albums (both rate-limited for this account): every song of the artist that's
in the KG with a Spotify id -> GET /tracks/{id} -> the release it's on -> /albums/{id}/tracks. Deezer (free)
lists every release with dates, so we also know when there's something newer than what we can play.
"""

import difflib
import json
import re
from pathlib import Path

import spotipy
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import spotify_client
from .events import bus
from .music import norm

router = APIRouter(prefix="/tools", tags=["tools"])
store = None  # set by main (same Store as tools)

OUTSIDE: dict[str, dict] = {}  # spotify uri -> track dict for songs outside the KG (play() uses it)


def _sp():
    """Catalog calls never retry: a 429 here must fail fast, not stall a tool call for hours."""
    global _client
    if _client is None:
        _client = spotipy.Spotify(auth_manager=spotify_client.client().auth_manager,
                                  retries=0, status_retries=0, requests_timeout=8)
    return _client


_client = None
MAX_ALBUMS = 15  # one /tracks call each: keep the request volume low
_track_album: dict[str, dict] = {}  # spotify track uri -> its album (persisted)
ALBUM_CACHE = Path(__file__).resolve().parents[2] / "data" / "enriched" / "spotify_track_albums.json"
if ALBUM_CACHE.exists():
    _track_album.update(json.loads(ALBUM_CACHE.read_text()))


def _album_of(uri: str) -> dict:
    """GET /tracks/{id} -> the release it's on (cached on disk)."""
    if uri not in _track_album:
        a = _sp().track(uri.rsplit(":", 1)[-1])["album"]
        _track_album[uri] = {"id": a["id"], "name": a["name"], "type": a["album_type"],
                             "release_date": a["release_date"], "tracks": a.get("total_tracks")}
        ALBUM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ALBUM_CACHE.write_text(json.dumps(_track_album))
    return _track_album[uri]


def _artist_members(artist: str) -> tuple[str, list[int]]:
    members = store.resolve_artist(artist)
    if not members:
        raise HTTPException(404, f"no artist {artist!r} in your library")
    return store.df.loc[members[0], "artist"], members


def _releases_from_library(members: list[int]) -> list[dict]:
    """The artist's releases that hold at least one song from the listening history: one track lookup per
    album the KG knows (no artist-albums / search endpoint needed)."""
    import pandas as pd

    df = store.df.loc[members]
    linked = df[store.playable[members]].sort_values("plays", ascending=False)
    sc = store.scrobbles[store.scrobbles["track"].isin(linked.index)]
    first = pd.to_datetime(sc["played_at"], utc=True).groupby(sc["track"]).min()
    linked = linked.assign(first=first.reindex(linked.index).to_numpy())
    groups = sorted(linked.groupby(linked["album"].fillna(linked["title"]), sort=False),
                    key=lambda kv: kv[1]["first"].min(), reverse=True)[:MAX_ALBUMS]  # newest discoveries
    out, seen = [], set()
    for _, g in groups:
        a = _album_of(g["spotify_uri"].iloc[0])
        if a["id"] not in seen:
            seen.add(a["id"])
            out.append(a)
    return sorted(out, key=lambda a: a["release_date"], reverse=True)


_blocked_until = 0.0  # Spotify artist-albums rate limit (seconds since epoch)


def _spotify_releases(members: list[int]) -> list[dict] | None:
    """Spotify's own list of the artist's albums+singles — None while that endpoint is rate-limited."""
    global _blocked_until
    import time

    if time.time() < _blocked_until:
        return None
    uri = next((store.df.loc[i, "spotify_uri"] for i in members if store.playable[i]), None)
    if not uri:
        return None
    try:
        t = _sp().track(uri.rsplit(":", 1)[-1])
        name = store.df.loc[members[0], "artist"]
        aid = next((a for a in t["artists"] if norm(a["name"]) == norm(name)), t["artists"][0])["id"]
        items, offset = [], 0
        while True:
            page = _sp().artist_albums(aid, include_groups="album,single", limit=10, offset=offset)
            items += page["items"]
            if not page.get("next") or offset >= 40:
                break
            offset += 10
    except spotipy.SpotifyException as e:
        if e.http_status == 429:
            _blocked_until = time.time() + int((e.headers or {}).get("Retry-After") or 3600)
        return None
    seen, out = set(), []
    for a in sorted(items, key=lambda x: x["release_date"], reverse=True):
        if norm(a["name"]) not in seen:
            seen.add(norm(a["name"]))
            out.append({"id": a["id"], "name": a["name"], "type": a["album_type"],
                        "release_date": a["release_date"], "tracks": a.get("total_tracks")})
    return out


def _spotify_album_via_musicbrainz(deezer_id: int) -> dict | None:
    """Deezer album -> barcode (UPC) -> MusicBrainz release -> its Spotify album link."""
    import requests

    ua = {"User-Agent": "listening-graph-demo/0.1 (hackathon)"}

    def mb(url: str, params: dict) -> dict:
        global _mb_last
        import time

        for _ in range(3):  # MusicBrainz allows ~1 request/second per client
            time.sleep(max(0.0, 1.1 - (time.time() - _mb_last)))
            _mb_last = time.time()
            r = requests.get(url, params=params, headers=ua, timeout=8)
            if r.status_code != 503:
                return r.json()
        raise ValueError("musicbrainz busy")

    try:
        al = requests.get(f"https://api.deezer.com/album/{deezer_id}", timeout=6).json()
        upc = al.get("upc")
        if not upc:
            return None
        rel = mb("https://musicbrainz.org/ws/2/release/", {"query": f"barcode:{upc}", "fmt": "json"}).get("releases", [])
        for r in rel[:2]:
            rels = mb(f"https://musicbrainz.org/ws/2/release/{r['id']}", {"inc": "url-rels", "fmt": "json"})
            for x in rels.get("relations", []):
                m = re.search(r"open\.spotify\.com/album/(\w+)", x.get("url", {}).get("resource", ""))
                if m:
                    return {"id": m.group(1), "name": al["title"], "type": al.get("record_type"),
                            "release_date": al.get("release_date"), "tracks": al.get("nb_tracks")}
    except (requests.RequestException, ValueError):
        return "error"  # transient: the caller must not cache this
    return None


_mb_last = 0.0


def _deezer_releases(artist: str) -> list[dict]:
    """Every release with its date, from Deezer (free, no auth): to know what's actually newest."""
    import requests

    try:
        found = requests.get("https://api.deezer.com/search/artist", params={"q": artist, "limit": 10},
                             timeout=6).json().get("data", [])
        same = [a for a in found if norm(a["name"]) == norm(artist)] or found[:1]
        if not same:
            return []
        aid = max(same, key=lambda a: a.get("nb_fan", 0))["id"]
        data = requests.get(f"https://api.deezer.com/artist/{aid}/albums", params={"limit": 100},
                            timeout=6).json().get("data", [])
    except requests.RequestException:
        return []
    seen, out = set(), []
    for a in sorted(data, key=lambda a: a.get("release_date", ""), reverse=True):
        if norm(a["title"]) not in seen:
            seen.add(norm(a["title"]))
            out.append({"name": a["title"], "type": a.get("record_type"), "release_date": a.get("release_date"),
                        "deezer_id": a["id"]})
    return out


def _album_tracks(album: dict, artist: str) -> list[dict]:
    items = _sp().album_tracks(album["id"], limit=50)["items"]
    lib = store.df[store.df["norm_artist"] == norm(artist)]
    by_title = dict(zip(lib["norm_title"], lib.index))
    out = []
    for t in items:
        uri = f"spotify:track:{t['id']}"
        i = by_title.get(norm(t["name"]))
        row = {"title": t["name"], "artist": ", ".join(a["name"] for a in t["artists"]), "spotify_uri": uri,
               "album": album["name"], "release_date": album["release_date"],
               "in_library": i is not None, "your_plays": int(store.df.loc[i, "plays"]) if i is not None else 0}
        # library songs keep their KG id (graph, stories); new ones are played by uri
        row["id"] = store.ids[i] if i is not None else uri
        if i is not None and not store.playable[i]:
            adopt_link(i, uri)  # a library song that had no Spotify link: now it has one
        if i is None:
            OUTSIDE[uri] = {**row, "playable": True}
        out.append(row)
    return out


class CatalogIn(BaseModel):
    artist: str = Field(..., description="artist name")
    album: str | None = Field(None, description="album/single name, or 'latest' for the newest album; "
                                                "omit to list releases")


_all_cache: dict[str, tuple[list[dict], list[dict]]] = {}


def _all_releases(name: str, members: list[int]) -> tuple[list[dict], list[dict]]:
    """(releases playable on Spotify, newest first; every release Deezer knows). Cached per artist."""
    if name in _all_cache:
        return _all_cache[name]
    rel = _spotify_releases(members)  # the full list, when Spotify allows it
    if rel is None:
        rel = _releases_from_library(members)
    everything = _deezer_releases(name)
    # new releases nobody linked yet: Deezer barcode -> MusicBrainz -> Spotify album (best effort, newest 3)
    known_names = {norm(r["name"]) for r in rel}
    newest_known = rel[0]["release_date"] if rel else ""
    complete = True
    for r in [r for r in everything if norm(r["name"]) not in known_names][:3]:
        if r["release_date"] <= newest_known:
            break
        found = _spotify_album_via_musicbrainz(r["deezer_id"])
        if found == "error":
            complete = False
        elif found:
            rel.append(found)
    rel.sort(key=lambda r: r["release_date"], reverse=True)
    if complete:
        _all_cache[name] = (rel, everything)
    return rel, everything


@router.post("/catalog")
async def catalog(body: CatalogIn) -> dict:
    def work() -> dict:
        name, members = _artist_members(body.artist)
        try:
            rel, everything = _all_releases(name, members)
        except spotipy.SpotifyException as e:
            raise HTTPException(503, f"Spotify catalog unavailable right now ({e.http_status})")
        known = {norm(r["name"]) for r in rel}
        for r in everything:
            r["playable_here"] = norm(r["name"]) in known
        newest = everything[0] if everything else None
        if not body.album:
            return {"artist": name, "latest_album": next((r for r in rel if r["type"] == "album"), None),
                    "releases": rel[:12], "newest_release_anywhere": newest,
                    "note": "releases = albums/singles you can play here (they hold songs from your history); "
                            "newest_release_anywhere may be newer and not playable here yet"}
        if body.album.lower() in ("latest", "last", "new", "newest", "último", "ultimo", "nuevo"):
            pick = next((r for r in rel if r["type"] == "album"), rel[0] if rel else None)
        else:
            names = [norm(r["name"]) for r in rel]
            close = difflib.get_close_matches(norm(body.album), names, n=1, cutoff=0.6)
            pick = rel[names.index(close[0])] if close else None
        if not pick:
            raise HTTPException(404, f"no playable release {body.album!r} by {name} (newest anywhere: "
                                     f"{newest['name'] if newest else '?'})")
        try:
            tracks = _album_tracks(pick, name)
        except spotipy.SpotifyException as e:
            raise HTTPException(503, f"Spotify catalog unavailable right now ({e.http_status})")
        return {"artist": name, "album": pick, "tracks": tracks,
                "newer_release_not_playable_here": newest if newest and not newest["playable_here"]
                and newest["release_date"] > pick["release_date"] else None}

    out = await run_in_threadpool(work)
    bus.publish("tool", tool="catalog", args=body.model_dump(exclude_defaults=True), result=out)
    return out


# ---------- filling missing Spotify links from the artist's discography ----------

LINKS_FILE = Path(__file__).resolve().parents[2] / "data" / "enriched" / "spotify_links_catalog.jsonl"
_discography: dict[str, dict[str, str]] = {}  # artist -> {normalised title: spotify uri}


def _key(title: str) -> str:
    """Title key that ignores '(feat. X)', ' - Remix'-style suffixes and case/accents."""
    return norm(re.split(r"\s[\(\[-]\s?(feat|ft|with|con)\b", title, flags=re.I)[0])


def _scan(artist: str, title: str) -> str | None:
    """Spotify uri of `title` among the artist's reachable releases (own album first if the KG knows it)."""
    want = _key(title)
    songs = _discography.setdefault(norm(artist), {})
    if want in songs:
        return songs[want]
    try:
        name, members = _artist_members(artist)
        rel, _ = _all_releases(name, members)
        album = next((store.df.loc[i, "album"] for i in members if _key(store.df.loc[i, "title"]) == want), None)
        same = [r for r in rel if isinstance(album, str) and norm(r["name"]) == norm(album)]
        named = [r for r in rel if norm(r["name"]) == norm(title)]  # a single named after the song
        for r in (named + same + [r for r in rel if r not in same and r not in named])[:12]:
            if r.get("_scanned"):
                continue
            for t in _sp().album_tracks(r["id"], limit=50)["items"]:
                songs.setdefault(_key(t["name"]), f"spotify:track:{t['id']}")
            r["_scanned"] = True
            if want in songs:
                return songs[want]
    except (spotipy.SpotifyException, HTTPException):
        return None
    return None


def find_link(i: int) -> str | None:
    """Spotify uri for a library song that has none."""
    t, a = store.df.loc[i, "title"], store.df.loc[i, "artist"]
    return _recently_played_uri(t, a) or _scan(a, t)


_recent: tuple[float, list[dict]] = (0.0, [])
RECENT_CACHE = Path(__file__).resolve().parents[2] / ".spotify_cache_recent"


def _recently_played_uri(title: str, artist: str) -> str | None:
    """Exact Spotify id from the user's own recently-played list (last 50 plays) — covers new singles
    they've been listening to, with no search at all. Needs scripts/auth_recently_played.py once."""
    global _recent
    import os
    import time

    if not RECENT_CACHE.exists():
        return None
    if time.time() - _recent[0] > 60:
        try:
            from spotipy.oauth2 import SpotifyPKCE

            auth = SpotifyPKCE(client_id=os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIFY_CLIENT_KEY"),
                               redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
                               scope="user-read-recently-played", open_browser=False,
                               cache_handler=spotipy.CacheFileHandler(cache_path=str(RECENT_CACHE)))
            sp = spotipy.Spotify(auth_manager=auth, retries=0, status_retries=0, requests_timeout=8)
            _recent = (time.time(), [it["track"] for it in sp.current_user_recently_played(limit=50)["items"]])
        except Exception:  # noqa: BLE001 — never block a tool on this
            _recent = (time.time(), _recent[1])
    for t in _recent[1]:
        if _key(t["name"]) == _key(title) and any(norm(a["name"]) == norm(artist) for a in t["artists"]):
            return f"spotify:track:{t['id']}"
    return None


_search_blocked_until = 0.0


def _by_isrc(title: str, artist: str) -> str | None:
    """Deezer (song -> ISRC) + Spotify search 'isrc:...' — exact for any song, once search isn't
    rate-limited for this account. While it is, one failed try sets a breaker for the Retry-After time."""
    global _search_blocked_until
    import time

    import requests

    if time.time() < _search_blocked_until:
        return None
    try:
        hits = requests.get("https://api.deezer.com/search", params={"q": f"{artist} {title}", "limit": 10},
                            timeout=6).json().get("data", [])
        hit = next((h for h in hits if _key(h["title"]) == _key(title)
                    and norm(h["artist"]["name"]) == norm(artist)), None)
        if not hit:
            return None
        isrc = requests.get(f"https://api.deezer.com/track/{hit['id']}", timeout=6).json().get("isrc")
        if not isrc:
            return None
        items = _sp().search(q=f"isrc:{isrc}", type="track", limit=1)["tracks"]["items"]
        return f"spotify:track:{items[0]['id']}" if items else None
    except spotipy.SpotifyException as e:
        if e.http_status == 429:
            _search_blocked_until = time.time() + int((e.headers or {}).get("Retry-After") or 3600)
        return None
    except (requests.RequestException, ValueError, KeyError):
        return None


def find_song(title: str, artist: str) -> dict | None:
    """A song that isn't in the listening history (e.g. a new single): a playable track dict, or None."""
    uri = _recently_played_uri(title, artist) or _scan(artist, title) or _by_isrc(title, artist)
    if not uri:
        return None
    t = {"id": uri, "title": title, "artist": artist, "spotify_uri": uri, "playable": True, "in_library": False}
    OUTSIDE[uri] = t
    return t


def newest_release(artist: str) -> dict | None:
    try:
        name, members = _artist_members(artist)
        return (_all_releases(name, members)[1] or [None])[0]
    except (spotipy.SpotifyException, HTTPException):
        return None


def adopt_link(i: int, uri: str) -> None:
    """Make a library song playable from now on (and after restarts)."""
    store.df.loc[i, "spotify_uri"] = uri
    store.playable[i] = True
    LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LINKS_FILE.open("a") as f:
        f.write(json.dumps({"id": store.ids[i], "spotify_uri": uri}) + "\n")


def load_links() -> int:
    """Re-apply links found in earlier runs (called at startup)."""
    if not LINKS_FILE.exists():
        return 0
    n = 0
    for line in LINKS_FILE.read_text().splitlines():
        r = json.loads(line)
        i = store.idx.get(r["id"])
        if i is not None and not store.playable[i]:
            store.df.loc[i, "spotify_uri"] = r["spotify_uri"]
            store.playable[i] = True
            n += 1
    return n
