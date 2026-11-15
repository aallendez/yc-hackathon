"""In-memory store: track table, embeddings (GNN or feature columns), kNN graph, listening history.

Loaded once at startup; POST /admin/reload re-reads everything after a retrain.
"""

import difflib
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .gnn.config import GNN_DIR, GRAPH_FILE, KG
from .music import camelot, norm

MODES = {"gnn": "gnn", "columns": "columns", "columnas": "columns", "features": "columns"}
KNN_K = 20
TRACK_FIELDS = ["id", "title", "artist", "bpm", "energy", "valence", "danceability", "camelot",
                "plays", "spotify_uri"]


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    return v


def parse_period(period: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """'2023' | '2023-06' | '2023-06-01..2023-08-31' | 'last 90 days' -> [start, end) in UTC."""
    if not period:
        return None, None
    p = period.strip().lower()
    if p.startswith("last ") and p.endswith(" days"):
        end = pd.Timestamp.now(tz="UTC")
        return end - pd.Timedelta(days=int(p.split()[1])), end
    if ".." in p:
        a, b = p.split("..")
        return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
    start = pd.Timestamp(p, tz="UTC")
    if len(p) == 4:
        return start, start + pd.DateOffset(years=1)
    if len(p) == 7:
        return start, start + pd.DateOffset(months=1)
    return start, start + pd.Timedelta(days=1)


class Store:
    def __init__(self) -> None:
        self.mode = "gnn"
        self.load()

    # ---------- loading ----------

    def load(self) -> None:
        g = torch.load(GRAPH_FILE, weights_only=False)
        self.ids: list[str] = g["track_ids"]
        self.idx = {t: i for i, t in enumerate(self.ids)}
        x = g["x"].numpy()
        self.vectors = {
            "columns": x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9),
            "gnn": np.load(GNN_DIR / "embeddings.npy"),
        }

        tracks = pd.read_parquet(KG / "nodes_track.parquet").set_index("id").reindex(self.ids)
        for c in ["bpm", "key", "mode", "energy", "valence", "danceability", "spotify_uri"]:
            if c not in tracks:
                tracks[c] = None
        df = pd.DataFrame({"id": self.ids, "title": g["titles"], "artist": g["artists"],
                           "artist_id": g["artist_ids"]})
        for c in ["bpm", "energy", "valence", "danceability", "key", "mode"]:
            df[c] = pd.to_numeric(tracks[c], errors="coerce").to_numpy()
        df["spotify_uri"] = tracks["spotify_uri"].to_numpy()
        df["camelot"] = [camelot(k, m) for k, m in zip(df["key"], df["mode"])]
        umap = pd.read_parquet(GNN_DIR / "tracks.parquet").set_index("id").reindex(self.ids)
        df["x"], df["y"] = umap["x"].to_numpy(), umap["y"].to_numpy()

        scrobbles = pd.read_parquet(KG / "nodes_scrobble.parquet")
        listened = pd.read_parquet(KG / "edges_scrobble__LISTENED_TO__track.parquet").set_index("src")["dst"]
        scrobbles["track"] = scrobbles["id"].map(listened).map(self.idx)
        self.scrobbles = scrobbles.dropna(subset=["track"]).astype({"track": int})
        df["plays"] = np.bincount(self.scrobbles["track"], minlength=len(df))
        on = pd.read_parquet(KG / "edges_track__ON__album.parquet").drop_duplicates("src").set_index("src")["dst"]
        album_titles = pd.read_parquet(KG / "nodes_album.parquet").set_index("id")["title"]
        df["album"] = pd.Series(self.ids).map(on).map(album_titles).to_numpy()
        df["norm_title"] = df["title"].map(norm)
        df["norm_artist"] = df["artist"].map(norm)
        self.df = df
        self.playable = df["spotify_uri"].notna().to_numpy().copy()  # writable: links get filled in at runtime

        # back-to-back counts (all data, directed) for "why" labels and played_after
        followed = pd.read_parquet(KG / "edges_scrobble__FOLLOWED_BY__scrobble.parquet")
        by_scrobble = self.scrobbles.set_index("id")["track"]
        a, b = followed["src"].map(by_scrobble), followed["dst"].map(by_scrobble)
        ok = a.notna() & b.notna() & (a != b)
        self.transitions = Counter(zip(a[ok].astype(int), b[ok].astype(int)))

        # artist metadata
        artists = pd.read_parquet(KG / "nodes_artist.parquet").set_index("id")
        self.artists = artists
        tags = pd.read_parquet(KG / "edges_artist__TAGGED_AT__tag.parquet")
        tag_names = pd.read_parquet(KG / "nodes_tag.parquet").set_index("id")["name"]
        tags["name"] = tags["dst"].map(tag_names)
        tags = tags.sort_values("weight", ascending=False)
        self.artist_tags = tags.groupby("src")["name"].apply(lambda s: s.head(6).tolist()).to_dict()
        all_tags = tags.groupby("src")["name"].apply(lambda s: " | ".join(s.head(12)).lower())
        self.tag_text = df["artist_id"].map(all_tags).fillna("").reset_index(drop=True)
        sim = pd.read_parquet(KG / "edges_artist__SIMILAR_TO__artist.parquet").sort_values("weight", ascending=False)
        self.similar_artists = sim.groupby("src")["dst"].apply(lambda s: s.head(8).tolist()).to_dict()
        self._knn: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ---------- vectors ----------

    def set_mode(self, mode: str) -> str:
        if mode.lower() not in MODES:
            raise ValueError(f"mode must be one of {sorted(set(MODES))}")
        self.mode = MODES[mode.lower()]
        return self.mode

    @property
    def V(self) -> np.ndarray:
        return self.vectors[self.mode]

    def knn(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mode not in self._knn:
            V = self.V
            nb = np.zeros((len(V), KNN_K), dtype=np.int64)
            sims = np.zeros((len(V), KNN_K), dtype=np.float32)
            for i in range(0, len(V), 2048):
                s = V[i:i + 2048] @ V.T
                s[np.arange(len(s)), np.arange(i, i + len(s))] = -np.inf
                top = np.argpartition(-s, KNN_K, axis=1)[:, :KNN_K]
                ts = np.take_along_axis(s, top, 1)
                order = np.argsort(-ts, axis=1)
                nb[i:i + len(s)] = np.take_along_axis(top, order, 1)
                sims[i:i + len(s)] = np.take_along_axis(ts, order, 1)
            self._knn[self.mode] = (nb, sims)
        return self._knn[self.mode]

    # ---------- lookup ----------

    def resolve(self, q: str | int) -> int:
        """Track id, index, 'title', 'title - artist' or 'title by artist' -> index (most-played wins)."""
        if isinstance(q, int) or (isinstance(q, str) and q.isdigit() and int(q) < len(self.ids)):
            return int(q)
        if q in self.idx:
            return self.idx[q]
        df = self.df
        title, artist = q, ""
        for sep in (" by ", " - ", " — ", " de "):
            if sep in q.lower():
                i = q.lower().rindex(sep)
                title, artist = q[:i], q[i + len(sep):]
                break
        nt, na = norm(title), norm(artist)
        score = (df["norm_title"] == nt) * 4 + df["norm_title"].str.contains(nt, regex=False) * 2
        if na:
            score = score + (df["norm_artist"] == na) * 3 + df["norm_artist"].str.contains(na, regex=False) * 1
        else:  # whole query may be "artist title" or similar
            nq = norm(q)
            score = score + (df["norm_title"] + df["norm_artist"]).str.contains(nq, regex=False) * 1
        if score.max() <= 0:
            raise KeyError(f"no track matches {q!r}")
        best = df[score == score.max()]
        return int(best["plays"].idxmax())

    def resolve_artist(self, q: str, fuzzy: bool = True) -> list[int]:
        """Exact (normalized) artist match; otherwise the closest library artist name, which absorbs
        speech-recognition slips like 'Lucho Rica' -> 'Lucho RK'."""
        na = norm(q)
        hits = self.df.index[self.df["norm_artist"] == na]
        if len(hits) or not fuzzy or len(na) < 4:
            return hits.tolist()
        names = self.df["norm_artist"].drop_duplicates()
        close = difflib.get_close_matches(na, names.tolist(), n=1, cutoff=0.72)
        return self.df.index[self.df["norm_artist"] == close[0]].tolist() if close else []

    def genre_members(self, q: str) -> list[int]:
        """Tracks whose artist carries the Last.fm tag `q` ('country', 'reggaeton', 'indie rock')."""
        tag = q.lower().strip()
        artists = {a for a, tags in self.artist_tags.items() if tag in (t.lower() for t in tags)}
        return self.df.index[self.df["artist_id"].isin(artists)].tolist() if artists else []

    def entity_vector(self, q: str) -> tuple[np.ndarray, list[int], str]:
        """A track, an artist or a genre tag (play-weighted centroid) -> (unit vector, members, label)."""
        if not len(self.df.index[self.df["norm_artist"] == norm(q)]):
            genre = self.genre_members(q)
            if len(genre) >= 5:
                w = self.df.loc[genre, "plays"].to_numpy(dtype=float) + 1
                v = (self.V[genre] * w[:, None]).sum(0)
                return v / np.linalg.norm(v), genre, f"genre: {q.lower().strip()}"
        members = self.resolve_artist(q)
        if members:
            w = self.df.loc[members, "plays"].to_numpy(dtype=float) + 1
            v = (self.V[members] * w[:, None]).sum(0)
            return v / np.linalg.norm(v), members, f"artist: {self.df.loc[members[0], 'artist']}"
        i = self.resolve(q)
        return self.V[i], [i], f"{self.df.loc[i, 'artist']} — {self.df.loc[i, 'title']}"

    def track(self, i: int, **extra) -> dict:
        row = self.df.loc[i]
        return {**{k: clean(row[k]) for k in TRACK_FIELDS}, "playable": bool(self.playable[i]), **extra}

    def tags_of(self, i: int) -> list[str]:
        return self.artist_tags.get(self.df.loc[i, "artist_id"], [])

    def shared_tag(self, a: int, b: int) -> str | None:
        tb = set(self.tags_of(b))
        return next((t for t in self.tags_of(a) if t in tb), None)

    def back_to_back(self, a: int, b: int) -> int:
        return self.transitions.get((a, b), 0) + self.transitions.get((b, a), 0)
