# AI DJ: a DJ that lives inside your listening history

A voice DJ (English and Spanish) built on top of **five years of one person's listening**. It uses a knowledge graph
plus a **GraphSAGE** graph neural network. You talk to it, it plays on Spotify, and it explains every pick with
facts from your own history.

> "Play a song I've had on repeat this month." · "Switch to country." · "Why, out of all country, this one?" ·
> "I'm driving 40 minutes: start with Coldplay and ease me into Morgan Wallen." · "Add Quevedo's new album to the queue."

## What it does

- **Plays by voice** and **explains every pick from real data**, never invented:
  - "you always play this right after *Human*";
  - "you had it on repeat in March";
  - "out of your 794 country songs, it's the closest to how *you* listen to country".
- **Journeys, not playlists:** sets that drift from one style to another. Every song is a small step from the
  last, and the route is drawn live on a map of the library.
- **Always one "up next"**, predicted by the model and queued in Spotify just before the current song ends. On-demand
  queueing ("add X to the queue") goes ahead of it.
- **Learns live:** new Last.fm plays add edges to the graph and the model re-embeds in about 70 ms, with no retraining
  (GraphSAGE is inductive).
- **Beyond the history:** new albums and singles by artists you know, even without Spotify search.

## How it works

```
Last.fm scrobbles (5 years) ──► knowledge graph ──► GraphSAGE ──► song embeddings (64-d)
   + Deezer/ISRC/ReccoBeats        plays · songs · albums                 │
     audio features                artists · tags                         ▼
   + Last.fm tags & similar                         FastAPI tools (similar, next, path, set, catalog, play, queue…)
                                                                          │  client tools
                                                   ElevenLabs voice agent ┘ ──► Spotify playback
                                                                          │
                                                   React UI ◄── SSE events (every tool call, hop, playback change)

live loop: Last.fm every 30 s → new transition edges → one forward pass → picks change
```

### The knowledge graph
One Parquet table per node and edge type ([backend/app/kg/schema.py](backend/app/kg/schema.py)).
- **Nodes:** scrobble, track, album, artist, tag.
- **Edges:** `FOLLOWED_BY` (play → next play), `LISTENED_TO`, `ON`, `BY`, `SIMILAR_TO` (artists), `TAGGED_AT`.
- Audio features come from Deezer (exact title + artist match) → ISRC → ReccoBeats, with a sanity check on suspicious
  BPMs.

### The model: GraphSAGE ([backend/app/gnn](backend/app/gnn))
- **Graph:** one node per song. Edges are songs *played back to back* (PMI over sessions), *same artist* and
  *related artists*.
- **Input:** 46 features per song:
  - 9 audio values;
  - key on the circle of fifths (sin/cos) and major/minor;
  - a "has audio data" flag;
  - the artist's tags (TF-IDF → SVD, 32-d);
  - artist popularity.
- **Architecture:** linear projection (128) → 2 × SAGEConv (mean, LayerNorm) → 64-d output, plus a skip connection from
  the input features.
- **Training:** link prediction on *which song you play next*. Each epoch, edges are split 70/30 into message-passing
  and supervision edges, scored with cosine × a learned temperature.

### Evaluation
All plays after a cut-off date are held out. On those **11,682 real transitions (song A → song B)**, each method sees
only A, ranks all 9,164 songs, and scores a hit if B is in its top 10.

| method | recall@10 | on pairs never played together before |
|---|---|---|
| **GraphSAGE embeddings** | **8.2%** | **7.0%** |
| same artist | 6.1% | 4.9% |
| "what came next before" (Markov) | 5.0% | 0.5% |
| raw features, same 46 inputs + cosine ("features only" in the UI) | 3.8% | 3.4% |
| most-played songs | 1.3% | 1.0% |

Same input data and the same cosine similarity: the graph and the training alone make it **2.2× better than raw
features**. The UI has a toggle to switch the whole app between the two spaces.

### Tools the agent can call ([backend/app/tools.py](backend/app/tools.py), [catalog.py](backend/app/catalog.py))
| tool | what it does |
|---|---|
| `similar` | Nearest songs to a song, artist or genre. |
| `next_track` | What comes next: GNN neighbours plus what you actually play after this song. |
| `path` | A hop-by-hop bridge between two songs. |
| `build_set` | A journey from one sound to another, with an energy shape. |
| `build_arc` | A playlist that follows an energy curve. |
| `taste` | What you listened to in a given period. |
| `query` | Exact facts: top tracks, artist info, what followed what. |
| `catalog` | An artist's releases and new albums. |
| `play` | Plays on Spotify and verifies it actually started. |
| `add_to_queue` | Adds songs after the current one, without interrupting it. |
| `probe_next` / `react` | Discovery by reactions. |
| `set_mode` | Switch between the GNN space and raw features. |

Every pick comes with a `story`: specific, checkable reasons computed from the scrobbles
([stories.py](backend/app/stories.py)), so the agent never has to make one up.

### Working around Spotify limits
Spotify search was rate-limited during the build, and dev-mode apps can't batch-fetch tracks or read top tracks.
Every lookup therefore has a route that avoids search:
- **Audio features:** Deezer → ISRC → ReccoBeats.
- **New albums:** library songs → their Spotify albums.
- **Brand-new releases:** Deezer barcode → MusicBrainz → Spotify album.
- **Recent singles:** your own recently played list.
- **Search by ISRC:** only once the rate limit lifts.

## Run it

```bash
cp .env.example .env            # fill in the keys

# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.gnn.build_graph && python -m app.gnn.train     # needs the knowledge graph in data/kg
uvicorn app.main:app --reload --reload-dir app --port 8001

# voice agent (creates/updates the ElevenLabs agent and its tools)
python ../scripts/create_agent.py

# frontend (new terminal)
cd frontend && npm install && npm run dev                    # http://localhost:5173
```

You need:
- Spotify Premium, with `http://127.0.0.1:8888/callback` added as a redirect URI in your Spotify app;
- an ElevenLabs API key;
- optionally, Last.fm keys for live updates.

**Data is not included.** `data/` (listening history, knowledge graph, embeddings) is git-ignored because it's personal.
To run this on your own history, build the graph in the schema above from your Last.fm export, then run the
enrichment script ([scripts/enrich_audio_features.py](scripts/enrich_audio_features.py)) and the GNN steps.

## Repo map
```
backend/app/
  main.py          API, live graph, SSE
  tools.py         agent tools
  catalog.py       releases and missing links without search
  gnn/             graph build, model, training, evaluation
  store.py         embeddings and kNN in memory
  live.py          Last.fm → new edges → re-embed
  up_next.py       always one "up next"
  stories.py       grounded "why" facts
  agent_spec.py    voice agent prompt and tool definitions
frontend/src/      React UI: live graph, library map, transcript, architecture view
scripts/           enrichment, agent creation, Spotify helpers
```

## What's next
- New songs added to the graph as nodes, embedded with one forward pass.
- Session-aware predictions, evaluated as the full live pipeline.
- Onboarding for any Last.fm or Spotify history.
- Shared sessions that blend two people's taste.
