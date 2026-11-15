"""Manual checkpoint: GNN neighbours vs feature-column neighbours for a track.

Run: cd backend && .venv/bin/python -m app.gnn.neighbors "slow dancing" [--k 10] [--artist "john mayer"]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .config import GNN_DIR, GRAPH_FILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--artist", default="")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    g = torch.load(GRAPH_FILE, weights_only=False)
    z = torch.tensor(np.load(GNN_DIR / "embeddings.npy"))
    feats = F.normalize(g["x"], dim=-1)
    titles, artists, plays = g["titles"], g["artists"], g["train_plays"]

    matches = [i for i, (t, a) in enumerate(zip(titles, artists))
               if args.title.lower() in t.lower() and args.artist.lower() in a.lower()]
    if not matches:
        raise SystemExit("no track matches")
    q = max(matches, key=lambda i: plays[i])
    print(f"Query: {artists[q]} — {titles[q]}  ({plays[q]} plays)\n")

    def top(mat: torch.Tensor) -> list[int]:
        s = mat @ mat[q]
        s[q] = -1
        return s.topk(args.k).indices.tolist()

    w = 48
    print(f"{'GNN':<{w}} | features only")
    for a, b in zip(top(z), top(feats)):
        fmt = lambda i: f"{artists[i]} — {titles[i]}"[: w - 1]
        print(f"{fmt(a):<{w}} | {fmt(b)}")


if __name__ == "__main__":
    main()
