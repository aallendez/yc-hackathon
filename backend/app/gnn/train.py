"""Train GraphSAGE (self-supervised link prediction on transition edges) and compare
against baselines on next-track recall@k for the TEST period (sessions after the split).

Training: each epoch, transition edges are split 70/30 into message-passing vs
supervision edges (so the model can't just "see" the edge it must predict); 10% of
transitions are held out for validation / early stopping. Negatives: random tracks.

Baselines on the same test pairs:
  popularity    most-played tracks (train)
  markov        tracks that followed this one most often in train, padded by popularity
  same_artist   artist's most-played tracks, padded by popularity
  features_knn  cosine on the raw feature columns (audio + artist tags) — the "no GNN" mode

Outputs (data/gnn/): embeddings.npy, tracks.parquet (id, title, artist, umap x/y),
model.pt, metrics.json

Run: cd backend && .venv/bin/python -m app.gnn.train [--layers 1 2 3] [--epochs 300]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import GNN_DIR, GRAPH_FILE
from .model import TrackSAGE

K = 10


def undirected(ei: torch.Tensor) -> torch.Tensor:
    return torch.cat([ei, ei.flip(0)], dim=1)


def recall_at_k(score_fn, pairs: torch.Tensor, k: int = K, chunk: int = 1024) -> float:
    """pairs: [2, P] (query, target). score_fn(query_idx) -> [len(q), N] scores."""
    hits = 0
    for i in range(0, pairs.shape[1], chunk):
        q, t = pairs[0, i:i + chunk], pairs[1, i:i + chunk]
        s = score_fn(q).clone()
        s[torch.arange(len(q)), q] = -float("inf")  # never recommend the query itself
        top = s.topk(k, dim=1).indices
        hits += (top == t[:, None]).any(1).sum().item()
    return hits / pairs.shape[1]


def baselines(g: dict, n: int) -> dict:
    pop = g["train_plays"].float()
    pop_scores = pop / pop.max()  # in [0, 1], used as tiebreak/padding

    tt = g["train_transitions"]
    markov = torch.sparse_coo_tensor(tt, torch.ones(tt.shape[1]), (n, n)).coalesce().to_dense()
    artist_codes = torch.tensor(pd.factorize(pd.Series(g["artist_ids"]))[0])
    feats = F.normalize(g["x"], dim=-1)

    return {
        "popularity": lambda q: pop_scores.expand(len(q), n),
        "markov": lambda q: markov[q] + pop_scores * 1e-3,
        "same_artist": lambda q: (artist_codes[q][:, None] == artist_codes[None, :]).float() + pop_scores * 1e-3,
        "features_knn": lambda q: feats[q] @ feats.T,
    }


def train_one(g: dict, layers: int, epochs: int, seed: int = 0) -> tuple[TrackSAGE, float]:
    torch.manual_seed(seed)
    x, ei, et = g["x"], g["edge_index"], g["edge_type"]
    n = x.shape[0]
    trans, other = ei[:, et == 0], ei[:, et != 0]
    perm = torch.randperm(trans.shape[1])
    n_val = int(0.1 * len(perm))
    val, tr = trans[:, perm[:n_val]], trans[:, perm[n_val:]]
    full_train_graph = undirected(torch.cat([other, tr], 1))

    model = TrackSAGE(x.shape[1], layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        p = torch.randperm(tr.shape[1])
        cut = int(0.7 * len(p))
        msg, sup = tr[:, p[:cut]], tr[:, p[cut:]]
        z = model(x, undirected(torch.cat([other, msg], 1)))
        src = torch.cat([sup[0], sup[1]])
        dst = torch.cat([sup[1], sup[0]])
        neg = torch.randint(0, n, (3 * len(src),))
        pos_l, neg_l = model.score(z, src, dst), model.score(z, src.repeat(3), neg)
        loss = F.binary_cross_entropy_with_logits(pos_l, torch.ones_like(pos_l)) + \
            F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model(x, full_train_graph)
                r = recall_at_k(lambda q: z[q] @ z.T, undirected(val))
            if r > best:
                best, bad = r, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if ep % 50 == 0:
                print(f"    ep {ep:3d}  loss {loss.item():.3f}  val recall@{K} {r:.3f}", flush=True)
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    g = torch.load(GRAPH_FILE, weights_only=False)
    n = g["x"].shape[0]
    test = g["test_transitions"]
    train_pairs = {(a, b) for a, b in g["train_transitions"].T.tolist()}
    novel = torch.tensor([[a, b] for a, b in test.T.tolist() if (a, b) not in train_pairs]).T
    print(f"{n:,} tracks · test pairs {test.shape[1]:,} (of which never seen in train: {novel.shape[1]:,})")

    metrics = {}
    for name, fn in baselines(g, n).items():
        metrics[name] = {"test": recall_at_k(fn, test), "test_novel": recall_at_k(fn, novel)}
        print(f"  {name:14s} recall@{K} test {metrics[name]['test']:.3f}  novel {metrics[name]['test_novel']:.3f}")

    train_graph = undirected(g["edge_index"])
    best_model, best_val, best_layers = None, -1.0, None
    for L in args.layers:
        t0 = time.time()
        print(f"  training GraphSAGE layers={L}")
        model, val = train_one(g, L, args.epochs)
        model.eval()
        with torch.no_grad():
            z = model(g["x"], train_graph)
        m = {"val": val, "test": recall_at_k(lambda q: z[q] @ z.T, test),
             "test_novel": recall_at_k(lambda q: z[q] @ z.T, novel), "seconds": round(time.time() - t0, 1)}
        metrics[f"sage_L{L}"] = m
        print(f"  sage_L{L:<9d} recall@{K} test {m['test']:.3f}  novel {m['test_novel']:.3f}  "
              f"(val {val:.3f}, {m['seconds']}s)")
        if val > best_val:
            best_model, best_val, best_layers = model, val, L

    # Export: final embeddings use ALL transitions (train + test) for message passing.
    # No retraining needed — GraphSAGE is inductive over edges and features.
    other = g["edge_index"][:, g["edge_type"] != 0]
    inference_graph = undirected(torch.cat([other, g["all_transition_edges"]], 1))
    with torch.no_grad():
        z = best_model(g["x"], inference_graph).numpy()
    np.save(GNN_DIR / "embeddings.npy", z)
    torch.save({"state_dict": best_model.state_dict(), "layers": best_layers,
                "in_dim": g["x"].shape[1]}, GNN_DIR / "model.pt")

    import umap  # slow import, only needed here
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(z)
    pd.DataFrame({"id": g["track_ids"], "title": g["titles"], "artist": g["artists"],
                  "spotify_uri": g["spotify_uri"], "train_plays": g["train_plays"].numpy(),
                  "x": xy[:, 0], "y": xy[:, 1]}).to_parquet(GNN_DIR / "tracks.parquet", index=False)
    metrics["best_layers"] = best_layers
    (GNN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best: layers={best_layers}. Saved embeddings, model, tracks.parquet, metrics.json to {GNN_DIR}")


if __name__ == "__main__":
    main()
