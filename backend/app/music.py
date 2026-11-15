"""Music helpers: Camelot keys and human-readable "why" labels for a hop between two tracks."""

import math
import re
import unicodedata


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]", "", s.casefold().replace("&", "and"))


def camelot(key, mode) -> str | None:
    """Spotify pitch class (0=C) + mode (1=major) -> Camelot code, e.g. C major = 8B, A minor = 8A."""
    if key is None or mode is None or (isinstance(key, float) and math.isnan(key)) or key < 0 \
            or (isinstance(mode, float) and math.isnan(mode)):
        return None
    k, m = int(key), int(mode)
    major_root = k if m == 1 else (k + 3) % 12
    return f"{(7 * major_root + 7) % 12 + 1}{'B' if m == 1 else 'A'}"


def key_relation(a: str | None, b: str | None) -> str | None:
    """'same key' / 'relative key' / 'harmonic neighbour' / None (a real key change)."""
    if not a or not b:
        return None
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    if a == b:
        return "same key"
    if na == nb:
        return "relative key"
    if la == lb and (na - nb) % 12 in (1, 11):
        return "harmonic neighbour"
    return None


def is_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def explain(a: dict, b: dict, back_to_back: int = 0, shared_tag: str | None = None) -> list[str]:
    """Short labels for the hop a -> b, most informative first."""
    out = []
    if back_to_back:
        out.append(f"played back to back ×{back_to_back}")
    if is_num(a.get("bpm")) and is_num(b.get("bpm")):
        d = round(b["bpm"] - a["bpm"])
        out.append("same BPM" if d == 0 else f"{d:+d} BPM")
    rel = key_relation(a.get("camelot"), b.get("camelot"))
    if rel:
        out.append(rel)
    elif a.get("camelot") and b.get("camelot"):
        out.append(f"key {a['camelot']}→{b['camelot']}")
    if is_num(a.get("energy")) and is_num(b.get("energy")):
        d = b["energy"] - a["energy"]
        if abs(d) >= 0.08:
            out.append(f"{'more' if d > 0 else 'less'} energy ({d:+.2f})")
    if a.get("artist") == b.get("artist"):
        out.append("same artist")
    elif shared_tag:
        out.append(f"both {shared_tag}")
    return out
