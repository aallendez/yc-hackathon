from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KG = ROOT / "data" / "kg"
GNN_DIR = ROOT / "data" / "gnn"
GRAPH_FILE = GNN_DIR / "graph.pt"

AUDIO_COLS = ["bpm", "energy", "valence", "danceability", "acousticness",
              "instrumentalness", "liveness", "speechiness", "loudness"]
TAG_DIMS = 32            # SVD size of artist tag TF-IDF
SAME_ARTIST_K = 5        # each track links to the artist's top-K tracks
ARTIST_LINK_K = 3        # each track links to the top track of its artist's top-K related artists

EDGE_TYPES = ["transition", "same_artist", "artist_link"]
