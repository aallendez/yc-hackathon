"""The ontology as code: node and edge types, their required columns, and validation.

On disk the graph is one Parquet table per node type (`nodes_<type>.parquet`, keyed by `id`) and one per
edge type (`edges_<src>__<REL>__<dst>.parquet`, columns `src`, `dst` + properties).
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd


class NodeType(StrEnum):
    SCROBBLE = "scrobble"
    TRACK = "track"
    ALBUM = "album"
    ARTIST = "artist"
    TAG = "tag"


@dataclass(frozen=True)
class EdgeType:
    src: NodeType
    rel: str
    dst: NodeType

    @property
    def name(self) -> str:
        return f"{self.src}__{self.rel}__{self.dst}"


FOLLOWED_BY = EdgeType(NodeType.SCROBBLE, "FOLLOWED_BY", NodeType.SCROBBLE)
LISTENED_TO = EdgeType(NodeType.SCROBBLE, "LISTENED_TO", NodeType.TRACK)
TRACK_ON_ALBUM = EdgeType(NodeType.TRACK, "ON", NodeType.ALBUM)
TRACK_BY_ARTIST = EdgeType(NodeType.TRACK, "BY", NodeType.ARTIST)
ALBUM_BY_ARTIST = EdgeType(NodeType.ALBUM, "BY", NodeType.ARTIST)
ARTIST_SIMILAR_TO = EdgeType(NodeType.ARTIST, "SIMILAR_TO", NodeType.ARTIST)
TAGGED_AT = EdgeType(NodeType.ARTIST, "TAGGED_AT", NodeType.TAG)
PLAYED_WITH = EdgeType(NodeType.ARTIST, "PLAYED_WITH", NodeType.ARTIST)
# Not built yet: Track SIMILAR_TO Track (source undecided), Track HARMONIC_MATCH Track (needs audio features).

NODE_COLUMNS: dict[NodeType, set[str]] = {
    NodeType.SCROBBLE: {"id", "played_at", "session_id", "split"},
    NodeType.TRACK: {"id", "title", "duration_s", "bpm", "energy", "key", "mode", "key_status"},
    NodeType.ALBUM: {"id", "title"},
    NodeType.ARTIST: {"id", "name", "mbid", "played", "listeners", "playcount", "image_url"},
    NodeType.TAG: {"id", "name"},
}

EDGE_COLUMNS: dict[EdgeType, set[str]] = {
    FOLLOWED_BY: set(),
    LISTENED_TO: set(),
    TRACK_ON_ALBUM: set(),
    TRACK_BY_ARTIST: set(),
    ALBUM_BY_ARTIST: set(),
    ARTIST_SIMILAR_TO: {"weight"},
    TAGGED_AT: {"weight"},
    PLAYED_WITH: {"count", "pmi"},
}

EDGE_TYPES = list(EDGE_COLUMNS)


class SchemaError(ValueError):
    pass


def node_path(kg_dir: Path, node_type: NodeType) -> Path:
    return kg_dir / f"nodes_{node_type}.parquet"


def edge_path(kg_dir: Path, edge_type: EdgeType) -> Path:
    return kg_dir / f"edges_{edge_type.name}.parquet"


def validate_nodes(df: pd.DataFrame, node_type: NodeType) -> None:
    missing = NODE_COLUMNS[node_type] - set(df.columns)
    if missing:
        raise SchemaError(f"{node_type} nodes missing columns {sorted(missing)}")
    if df["id"].isna().any():
        raise SchemaError(f"{node_type} nodes have null ids")
    dupes = df["id"][df["id"].duplicated()]
    if len(dupes):
        raise SchemaError(f"{node_type} nodes have duplicate ids, e.g. {dupes.iloc[0]!r}")


def validate_edges(df: pd.DataFrame, edge_type: EdgeType,
                   nodes: dict[NodeType, pd.DataFrame] | None = None) -> None:
    missing = ({"src", "dst"} | EDGE_COLUMNS[edge_type]) - set(df.columns)
    if missing:
        raise SchemaError(f"{edge_type.name} edges missing columns {sorted(missing)}")
    if df[["src", "dst"]].isna().any().any():
        raise SchemaError(f"{edge_type.name} edges have null endpoints")
    if nodes is None:
        return
    for end, node_type in (("src", edge_type.src), ("dst", edge_type.dst)):
        dangling = ~df[end].isin(nodes[node_type]["id"])
        if dangling.any():
            raise SchemaError(f"{edge_type.name}: {dangling.sum()} {end} ids not in {node_type} nodes, "
                              f"e.g. {df.loc[dangling, end].iloc[0]!r}")
