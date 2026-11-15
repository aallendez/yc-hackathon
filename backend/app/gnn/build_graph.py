"""KG parquet -> homogeneous track graph for GraphSAGE (data/gnn/graph.pt).

Nodes: every track. Features: audio (z-scored, key on the circle of fifths as sin/cos,
mode, has_features flag) + the artist's tags (TF-IDF -> SVD) + artist log(listeners).

Edges (undirected), all derived from TRAIN sessions only:
  transition   consecutive plays in a session, kept if PMI > 0 (stops hubs linking to everything)
  same_artist  each track <-> its artist's top-K most-played tracks
  artist_link  each track <-> top track of its artist's top-K SIMILAR_TO / PLAYED_WITH artists

Also stored: test-split transitions (for recall@k) and the all-data transition edges
(train + test, used only at inference time to export final embeddings).

Run: cd backend && .venv/bin/python -m app.gnn.build_graph
"""

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

from .config import (ARTIST_LINK_K, AUDIO_COLS, EDGE_TYPES, GNN_DIR, GRAPH_FILE, KG,
                     SAME_ARTIST_K, TAG_DIMS)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(KG / f"{name}.parquet")


def zscore(col: pd.Series) -> np.ndarray:
    v = col.astype(float).to_numpy()
    mu, sd = np.nanmean(v), np.nanstd(v) or 1.0
    return np.nan_to_num((v - mu) / sd)


def key_features(key: pd.Series, mode: pd.Series) -> np.ndarray:
    """Pitch class + mode -> position on the circle of fifths (Camelot wheel) as sin/cos.
    Relative major/minor share a position (A minor == C major), so harmonic neighbours are close."""
    k, m = key.astype(float).to_numpy(), mode.astype(float).to_numpy()
    valid = ~np.isnan(k) & (k >= 0) & ~np.isnan(m)
    major_root = np.where(m == 1, k, (k + 3) % 12)
    angle = 2 * np.pi * ((np.nan_to_num(major_root) * 7) % 12) / 12
    return np.stack([np.where(valid, np.sin(angle), 0), np.where(valid, np.cos(angle), 0),
                     np.where(valid, m, 0.5)], axis=1)


def transition_pairs(scrobbles: pd.DataFrame, followed: pd.DataFrame, listened: pd.Series,
                     split: str | None) -> pd.DataFrame:
    """Directed track->next-track pairs (with repeats removed: a != b)."""
    if split:
        keep = set(scrobbles.loc[scrobbles["split"] == split, "id"])
        followed = followed[followed["src"].isin(keep)]
    e = pd.DataFrame({"a": followed["src"].map(listened), "b": followed["dst"].map(listened)}).dropna()
    return e[e["a"] != e["b"]]


def pmi_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    u = np.minimum(pairs["a"], pairs["b"]), np.maximum(pairs["a"], pairs["b"])
    und = pd.DataFrame({"a": u[0], "b": u[1]}).value_counts().rename("n").reset_index()
    deg = pd.concat([und.groupby("a")["n"].sum(), und.groupby("b")["n"].sum()]).groupby(level=0).sum()
    total = und["n"].sum()
    und["pmi"] = np.log(und["n"] * total / (und["a"].map(deg) * und["b"].map(deg)))
    return und[und["pmi"] > 0]


def main() -> None:
    GNN_DIR.mkdir(parents=True, exist_ok=True)
    tracks = load("nodes_track")
    artists = load("nodes_artist").set_index("id")
    track_artist = load("edges_track__BY__artist").set_index("src")["dst"]
    scrobbles = load("nodes_scrobble")
    listened = load("edges_scrobble__LISTENED_TO__track").set_index("src")["dst"]
    followed = load("edges_scrobble__FOLLOWED_BY__scrobble")

    ids = tracks["id"].tolist()
    idx = {t: i for i, t in enumerate(ids)}
    tracks["artist_id"] = tracks["id"].map(track_artist)
    tracks["artist"] = tracks["artist_id"].map(artists["name"])
    train_ids = set(scrobbles.loc[scrobbles["split"] == "train", "id"])
    train_plays = listened[listened.index.isin(train_ids)].value_counts()
    tracks["train_plays"] = tracks["id"].map(train_plays).fillna(0).astype(int)

    # ---- node features ----
    for c in AUDIO_COLS + ["key", "mode"]:
        if c not in tracks:
            tracks[c] = np.nan
    has_feat = tracks["bpm"].notna().to_numpy(dtype=float)[:, None]
    audio = np.stack([zscore(tracks[c]) for c in AUDIO_COLS], axis=1)
    keyf = key_features(tracks["key"], tracks["mode"])

    tags = load("edges_artist__TAGGED_AT__tag")
    a_ids = pd.Index(tracks["artist_id"].dropna().unique())
    tags = tags[tags["src"].isin(a_ids)]
    t_ids = pd.Index(tags["dst"].unique())
    m = sparse.csr_matrix((tags["weight"] / 100.0, (a_ids.get_indexer(tags["src"]), t_ids.get_indexer(tags["dst"]))),
                          shape=(len(a_ids), len(t_ids)))
    tag_emb = TruncatedSVD(TAG_DIMS, random_state=0).fit_transform(TfidfTransformer().fit_transform(m))
    tag_emb /= np.linalg.norm(tag_emb, axis=1, keepdims=True).clip(1e-9)
    pos = a_ids.get_indexer(tracks["artist_id"])
    track_tags = np.where((pos >= 0)[:, None], tag_emb[pos.clip(0)], 0.0)
    listeners = np.log1p(tracks["artist_id"].map(artists["listeners"]).astype(float))
    artist_pop = zscore(listeners)[:, None]

    x = np.concatenate([audio, keyf, has_feat, track_tags, artist_pop], axis=1).astype(np.float32)
    feature_names = AUDIO_COLS + ["key_sin", "key_cos", "mode", "has_features"] + \
        [f"tag_{i}" for i in range(TAG_DIMS)] + ["artist_listeners"]

    # ---- edges ----
    edges: dict[str, pd.DataFrame] = {}
    train_pairs = transition_pairs(scrobbles, followed, listened, "train")
    edges["transition"] = pmi_edges(train_pairs)[["a", "b"]]

    by_artist = tracks.sort_values("train_plays", ascending=False).groupby("artist_id")["id"]
    top = by_artist.apply(lambda s: s.head(SAME_ARTIST_K).tolist())
    sa = [(t, h) for a, ts in tracks.groupby("artist_id")["id"] for t in ts for h in top[a] if t != h]
    edges["same_artist"] = pd.DataFrame(sa, columns=["a", "b"])

    sim = load("edges_artist__SIMILAR_TO__artist").rename(columns={"weight": "w"})
    pw = load("edges_artist__PLAYED_WITH__artist").rename(columns={"pmi": "w"})
    links = pd.concat([sim[["src", "dst", "w"]], pw[["src", "dst", "w"]]])
    links = links[links["src"].isin(top.index) & links["dst"].isin(top.index) & (links["src"] != links["dst"])]
    links = links.sort_values("w", ascending=False).drop_duplicates(["src", "dst"])
    links = links.groupby("src").head(ARTIST_LINK_K)
    link_top = links.assign(top=links["dst"].map(lambda a: top[a][0]))
    al = tracks[["id", "artist_id"]].merge(link_top, left_on="artist_id", right_on="src")
    edges["artist_link"] = al.rename(columns={"id": "a", "top": "b"})[["a", "b"]]

    ei, et = [], []
    for t, df in edges.items():
        a, b = df["a"].map(idx).to_numpy(), df["b"].map(idx).to_numpy()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        pair = np.unique(np.stack([lo, hi], 1)[lo != hi], axis=0)
        ei.append(pair)
        et.append(np.full(len(pair), EDGE_TYPES.index(t)))
        print(f"{t:12s} {len(pair):>7,} undirected edges")
    ei, et = np.concatenate(ei), np.concatenate(et)

    def directed(df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(np.stack([df["a"].map(idx), df["b"].map(idx)]), dtype=torch.long)

    test_pairs = transition_pairs(scrobbles, followed, listened, "test").drop_duplicates()
    all_trans = pmi_edges(transition_pairs(scrobbles, followed, listened, None))

    data = {
        "x": torch.tensor(x),
        "feature_names": feature_names,
        "edge_index": torch.tensor(ei.T, dtype=torch.long),  # undirected, each pair once
        "edge_type": torch.tensor(et, dtype=torch.long),
        "train_transitions": directed(train_pairs.drop_duplicates()),
        "test_transitions": directed(test_pairs),
        "all_transition_edges": directed(all_trans),
        "track_ids": ids,
        "titles": tracks["title"].tolist(),
        "artists": tracks["artist"].fillna("?").tolist(),
        "artist_ids": tracks["artist_id"].fillna("").tolist(),
        "train_plays": torch.tensor(tracks["train_plays"].to_numpy()),
        "spotify_uri": tracks.get("spotify_uri", pd.Series([None] * len(tracks))).tolist(),
    }
    torch.save(data, GRAPH_FILE)
    deg = np.bincount(ei.ravel(), minlength=len(ids))
    print(f"{len(ids):,} tracks, {x.shape[1]} features, audio coverage {has_feat.mean():.0%}, "
          f"isolated tracks {(deg == 0).sum():,}, median degree {np.median(deg):.0f}")
    print(f"test transitions (unique directed pairs): {test_pairs.shape[0]:,}")
    print(f"saved {GRAPH_FILE}")


if __name__ == "__main__":
    main()
