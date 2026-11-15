"""Specific, checkable reasons a song was picked — the things a friend who knows your history would say.

Everything comes straight from the scrobbles / KG (no LLM guessing), most specific first, e.g.
  "you've played it right after Fix You 7 times"
  "both usually come right after 'Yellow' in your sessions"
  "you had it on repeat in March 2023 (41 plays that month)"
  "you haven't played it since June 2022"
"""


import pandas as pd

from .music import key_relation, is_num

SESSION_GAP = pd.Timedelta(minutes=30)


def _sessions(store) -> dict[int, set[int]]:
    """track idx -> listening-session ids it appears in (KG rule: < 30 min between plays)."""
    key = len(store.scrobbles)
    if getattr(store, "_sess_key", None) != key:
        sc = store.scrobbles.assign(played_at=pd.to_datetime(store.scrobbles["played_at"], utc=True))
        sc = sc.sort_values("played_at")
        sid = (sc["played_at"].diff() > SESSION_GAP).cumsum()
        store._sess = sid.groupby(sc["track"]).apply(set).to_dict()
        store._sess_end = sc.groupby(sid)["played_at"].max()
        store._sess_key = key
    return store._sess


def _plays(store, i: int) -> pd.Series:
    # live Last.fm rows get appended to the KG ones, which can leave the column as object dtype
    return pd.to_datetime(store.scrobbles.loc[store.scrobbles["track"] == i, "played_at"], utc=True)


def _month(ts: pd.Timestamp) -> str:
    return ts.strftime("%B %Y")


def story(store, pick: int, cur: int | None = None) -> list[str]:
    """Up to ~4 facts about why `pick` (after `cur`, if given), most specific first. Never raises:
    a missing story must not break the tool that asked for it."""
    try:
        return _story(store, pick, cur)
    except Exception as e:  # noqa: BLE001
        print(f"story failed for {pick}: {type(e).__name__}: {e}")
        return []


def _story(store, pick: int, cur: int | None = None) -> list[str]:
    df = store.df
    facts: list[tuple[float, str]] = []  # (specificity, text)
    title = lambda i: f"'{df.loc[i, 'title']}'"

    if cur is not None and cur != pick:
        after = store.transitions.get((cur, pick), 0)
        followers = {b: n for (a, b), n in store.transitions.items() if a == cur}
        if after and followers and after == max(followers.values()) and after >= 2:
            facts.append((10, f"it's what you play most often right after {title(cur)} ({after} times)"))
        elif after >= 2:
            facts.append((9, f"you've played it right after {title(cur)} {after} times"))

        sess = _sessions(store)
        shared = sess.get(cur, set()) & sess.get(pick, set())
        if len(shared) >= 2:
            last = store._sess_end.loc[max(shared)]
            facts.append((8, f"they've shared {len(shared)} of your listening sessions, last in {_month(last)}"))

        # a bridge song: something you regularly play right before both
        before = {}
        for (a, b), n in store.transitions.items():
            if b in (cur, pick) and a not in (cur, pick):
                before.setdefault(a, {})[b] = n
        both = [(min(d.values()), z) for z, d in before.items() if len(d) == 2]
        if both:
            n, z = max(both)
            facts.append((7 + min(n, 3) / 10, f"both usually come right after {title(z)} in your sessions"))

        if not after and not store.transitions.get((pick, cur)) and len(shared) == 0:
            facts.append((6, "you've never played these two together — the GNN connects them through how "
                             "you listen around them"))

        a_art, b_art = df.loc[cur, "artist_id"], df.loc[pick, "artist_id"]
        if a_art != b_art and (b_art in store.similar_artists.get(a_art, [])
                               or a_art in store.similar_artists.get(b_art, [])):
            facts.append((4, f"Last.fm counts {df.loc[pick, 'artist']} as close to {df.loc[cur, 'artist']}"))

        ta, tb = store.track(cur), store.track(pick)
        rel = key_relation(ta.get("camelot"), tb.get("camelot"))
        if rel and is_num(ta.get("bpm")) and is_num(tb.get("bpm")) and abs(ta["bpm"] - tb["bpm"]) <= 4:
            facts.append((5, f"{rel} and within {max(1, round(abs(ta['bpm'] - tb['bpm'])))} BPM — a clean mix"))

    played = _plays(store, pick)
    n = len(played)
    if n >= 6:
        by_month = played.dt.tz_localize(None).dt.to_period("M").value_counts()
        top_m, top_n = by_month.index[0], int(by_month.iloc[0])
        if top_n >= 5 and top_n / n >= 0.3:
            facts.append((7.5, f"you had it on repeat in {top_m.strftime('%B %Y')} ({top_n} plays that month)"))
    if n:
        last, first = played.max(), played.min()
        now = pd.Timestamp.now(tz=last.tz)
        if n >= 5 and now - last > pd.Timedelta(days=365):
            facts.append((6.5, f"an old favourite you haven't played since {_month(last)}"))
        elif now - first < pd.Timedelta(days=60):
            facts.append((6.5, f"a new one for you — first played {first.strftime('%-d %B')}"))
        if n <= 3:
            facts.append((5.5, f"a deep cut: you've only played it {n} time{'s' if n > 1 else ''}"))

    facts.sort(key=lambda f: -f[0])
    return [t for _, t in facts[:4]]
