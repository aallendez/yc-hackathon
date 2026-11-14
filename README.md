# yc-hackathon

- `frontend/` — React + Vite (TypeScript)
- `backend/` — FastAPI
- `scripts/` — one-off scripts

## Setup

```bash
cp .env.example .env   # fill in keys

# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend (new terminal)
cd frontend
npm install
npm run dev            # http://localhost:5173, /api proxied to :8000
```

## Spotify playback test

Needs Spotify Premium, and `http://127.0.0.1:8888/callback` added as a Redirect URI in your Spotify app dashboard.

```bash
uv run scripts/play_song.py
```
