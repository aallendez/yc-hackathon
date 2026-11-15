"""Live graph: new Last.fm scrobbles -> new transition edges -> fresh GraphSAGE embeddings.

Every POLL_SECONDS we read scrobbles newer than the last one seen (on startup: newer than the
KG, so the weeks since the export are backfilled too). Each scrobble is resolved to the KG's
stable track id with AI-DJ's own resolver (same normalization + aliases), so a known song maps
to its existing node. Consecutive plays inside one session (< SESSION_GAP) become transition
edges; then the trained model.pt is re-run over the updated graph — no retraining, GraphSAGE is
inductive over edges — and the store's GNN vectors are swapped in place.

Unknown songs (not in the KG) are kept in the session list (in_graph=false) but get no node yet.

POST /admin/fake_scrobble injects plays through the same path, for testing and for the video.
"""

import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from .events import bus
from .gnn.config import GNN_DIR, GRAPH_FILE
from .gnn.model import TrackSAGE

ROOT = Path(__file__).resolve().parents[2]
AI_DJ = Path(os.getenv("AI_DJ_DIR", ROOT.parent / "AI-DJ"))
sys.path.insert(0, str(AI_DJ / "src"))
from avicii.domain import ids  # noqa: E402
from avicii.resolve import normalize  # noqa: E402
from avicii.resolve.entities import ArtistResolver  # noqa: E402

POLL_SECONDS = 30
SESSION_GAP = pd.Timedelta(minutes=30)
LASTFM = "https://ws.audioscrobbler.com/2.0/"


class LiveGraph:
    def __init__(self, store) -> None:
        self.store = store
        self.resolver = ArtistResolver.from_aliases(pd.read_parquet(AI_DJ / "data" / "clean" / "aliases.parquet"))
        g = torch.load(GRAPH_FILE, weights_only=False)
        ckpt = torch.load(GNN_DIR / "model.pt", weights_only=False)
        self.model = TrackSAGE(ckpt["in_dim"], layers=ckpt["layers"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.x = g["x"]
        # same message-passing graph the exported embeddings were computed on
        self.base_edges = torch.cat([g["edge_index"][:, g["edge_type"] != 0], g["all_transition_edges"]], 1)
        self.new_edges: Counter = Counter()  # undirected (min, max) -> count, from live plays
        self.plays: list[dict] = []          # every live scrobble, oldest first
        self.last_ts = int(store.scrobbles["played_at"].max().timestamp())
        self.last_poll: float | None = None
        self.last_error: str | None = None
        self.updates = 0

    # ---------- resolve ----------

    def track_id(self, artist: str, title: str, artist_mbid: str | None = None) -> str:
        return ids.track_id(self.resolver.resolve(artist, artist_mbid), normalize.normalize_track(title))

    # ---------- ingest ----------

    def fetch_lastfm(self) -> list[dict]:
        key, user = os.getenv("LASTFM_API_KEY"), os.getenv("LASTFM_USERNAME")
        if not key or not user:
            raise RuntimeError("LASTFM_API_KEY / LASTFM_USERNAME missing in .env")
        out, page = [], 1
        while page <= 10:
            r = requests.get(LASTFM, params={"method": "user.getrecenttracks", "user": user, "api_key": key,
                                             "format": "json", "limit": 200, "page": page,
                                             "from": self.last_ts + 1}, timeout=20)
            r.raise_for_status()
            rt = r.json()["recenttracks"]
            items = rt["track"] if isinstance(rt["track"], list) else [rt["track"]]
            for it in items:
                if it.get("@attr", {}).get("nowplaying") == "true" or "date" not in it:
                    continue
                out.append({"artist": it["artist"]["#text"], "title": it["name"],
                            "artist_mbid": it["artist"].get("mbid") or None,
                            "played_at": pd.Timestamp(int(it["date"]["uts"]), unit="s", tz="UTC")})
            if page >= int(rt["@attr"]["totalPages"]):
                break
            page += 1
        return sorted(out, key=lambda s: s["played_at"])

    def ingest(self, scrobbles: list[dict], source: str) -> dict:
        """Add scrobbles (oldest first); returns what changed."""
        s = self.store
        added_edges: list[tuple[int, int]] = []
        for sc in scrobbles:
            tid = self.track_id(sc["artist"], sc["title"], sc.get("artist_mbid"))
            idx = s.idx.get(tid)
            play = {"track_id": tid, "idx": idx, "artist": sc["artist"], "title": sc["title"],
                    "played_at": sc["played_at"], "in_graph": idx is not None, "source": source}
            prev = self.plays[-1] if self.plays else None
            self.plays.append(play)
            self.last_ts = max(self.last_ts, int(sc["played_at"].timestamp()))
            if idx is not None:
                s.df.loc[idx, "plays"] += 1
                s.scrobbles = pd.concat([s.scrobbles, pd.DataFrame(
                    [{"id": f"live_{len(self.plays)}", "played_at": sc["played_at"], "track": idx}])],
                    ignore_index=True)
            if (prev and idx is not None and prev["idx"] is not None and prev["idx"] != idx
                    and sc["played_at"] - prev["played_at"] < SESSION_GAP):
                a, b = prev["idx"], idx
                self.new_edges[(min(a, b), max(a, b))] += 1
                s.transitions[(a, b)] += 1  # "played back to back ×N" labels pick it up too
                added_edges.append((a, b))
        effects = []
        if added_edges:
            old = self.store.vectors["gnn"]
            self.reembed()
            new = self.store.vectors["gnn"]
            for a, b in dict.fromkeys(added_edges):  # unique, in order
                effects.append({"from": s.ids[a], "to": s.ids[b],
                                "similarity_before": round(float(old[a] @ old[b]), 3),
                                "similarity_after": round(float(new[a] @ new[b]), 3),
                                "rank_before": self._rank(old, a, b), "rank_after": self._rank(new, a, b)})
        return {"scrobbles": len(scrobbles), "new_edges": added_edges, "effects": effects}

    @staticmethod
    def _rank(V: np.ndarray, a: int, b: int) -> int:
        """Position of b among a's nearest neighbours (1 = closest)."""
        sims = V @ V[a]
        sims[a] = -np.inf
        return int((sims > sims[b]).sum()) + 1

    # ---------- re-embed ----------

    def reembed(self) -> None:
        t0 = time.time()
        extra = torch.tensor(list(self.new_edges), dtype=torch.long).T if self.new_edges else \
            torch.zeros((2, 0), dtype=torch.long)
        ei = torch.cat([self.base_edges, extra], 1)
        ei = torch.cat([ei, ei.flip(0)], 1)
        with torch.no_grad():
            z = self.model(self.x, ei).numpy()
        old = self.store.vectors["gnn"]
        self.store.vectors["gnn"] = z
        self.store._knn.pop("gnn", None)  # neighbour lists rebuilt lazily
        moved = np.linalg.norm(z - old, axis=1)
        self.updates += 1
        self.last_reembed = {"seconds": round(time.time() - t0, 2), "nodes_moved": int((moved > 1e-4).sum()),
                             "max_shift": round(float(moved.max()), 4)}

    # ---------- views ----------

    def session(self) -> list[dict]:
        """The current session: trailing plays with < SESSION_GAP between them."""
        out: list[dict] = []
        for p in reversed(self.plays):
            if out and out[-1]["played_at"] - p["played_at"] >= SESSION_GAP:
                break
            out.append(p)
        return list(reversed(out))

    def describe(self, plays: list[dict]) -> dict:
        from . import tools
        s = self.store
        tracks = [{**(s.track(p["idx"]) if p["in_graph"] else
                      {"id": p["track_id"], "title": p["title"], "artist": p["artist"], "playable": False}),
                   "in_graph": p["in_graph"], "played_at": p["played_at"].isoformat(), "source": p["source"],
                   "x": float(s.df.loc[p["idx"], "x"]) if p["in_graph"] else None,
                   "y": float(s.df.loc[p["idx"], "y"]) if p["in_graph"] else None} for p in plays]
        hops = [tools.hop(a["idx"], b["idx"]) for a, b in zip(plays, plays[1:])
                if a["in_graph"] and b["in_graph"] and a["idx"] != b["idx"]]
        return {"tracks": tracks, "hops": hops}

    def status(self) -> dict:
        return {"live_scrobbles": len(self.plays), "in_graph": sum(p["in_graph"] for p in self.plays),
                "new_edges": len(self.new_edges), "reembeds": self.updates,
                "last_reembed": getattr(self, "last_reembed", None),
                "last_poll_ago_s": round(time.time() - self.last_poll) if self.last_poll else None,
                "last_error": self.last_error,
                "since": pd.Timestamp(self.last_ts, unit="s", tz="UTC").isoformat()}

    def publish(self, change: dict) -> None:
        if not change["scrobbles"]:
            return
        from . import tools
        names = {i: self.store.track(self.store.idx[i]) for e in change.get("effects", []) for i in (e["from"], e["to"])}
        effects = [{**e, "from_title": names[e["from"]]["title"], "to_title": names[e["to"]]["title"]}
                   for e in change.get("effects", [])]
        bus.publish("graph_update", new_edges=[tools.hop(a, b) for a, b in change["new_edges"]],
                    scrobbles=change["scrobbles"], effects=effects, session=self.describe(self.session()),
                    status=self.status())

    # ---------- loop ----------

    async def watch(self) -> None:
        while True:
            try:
                new = await asyncio.to_thread(self.fetch_lastfm)
                change = await asyncio.to_thread(self.ingest, new, "lastfm")
                self.publish(change)
                self.last_error = None
            except Exception as e:  # keep polling; surface the error in /live/status
                self.last_error = f"{type(e).__name__}: {e}"[:200]
            self.last_poll = time.time()
            await asyncio.sleep(POLL_SECONDS)
