"""Create or update the ElevenLabs agent from backend/app/agent_spec.py.

Idempotent: the agent id and tool ids are kept in data/agent/elevenlabs.json. Re-running
recreates the tools, points the agent at them and deletes the old ones.

Run: backend/.venv/bin/python scripts/create_agent.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.agent_spec import FIRST_MESSAGE, FIRST_MESSAGE_ES, LLM, PROMPT, TOOLS, elevenlabs_tool_config  # noqa: E402

load_dotenv(ROOT / ".env")
API = "https://api.elevenlabs.io/v1/convai"
HEADERS = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"}
STATE = ROOT / "data" / "agent" / "elevenlabs.json"
TURN_TIMEOUT = -1  # never nudge on silence: silence = listening to music (default is 7s)


def call(method: str, path: str, **kw) -> dict:
    r = requests.request(method, f"{API}{path}", headers=HEADERS, timeout=60, **kw)
    if not r.ok:
        sys.exit(f"{method} {path} -> {r.status_code}\n{r.text[:2000]}")
    return r.json() if r.text else {}


def top_artist_keywords(n_all: int = 60, n_recent: int = 40) -> list[str]:
    """Boost speech recognition for the artist names this user actually says: all-time favourites
    plus the last year's (recent artists like Lucho RK are what they'll ask for)."""
    kg = ROOT / "data" / "kg"
    by = pd.read_parquet(kg / "edges_track__BY__artist.parquet").set_index("src")["dst"]
    names = pd.read_parquet(kg / "nodes_artist.parquet").set_index("id")["name"]
    listened = pd.read_parquet(kg / "edges_scrobble__LISTENED_TO__track.parquet").set_index("src")["dst"]
    scrobbles = pd.read_parquet(kg / "nodes_scrobble.parquet").set_index("id")
    artist = listened.map(by)
    recent = scrobbles["played_at"] >= scrobbles["played_at"].max() - pd.Timedelta(days=365)
    ranked = list(artist.value_counts().index[:n_all]) + \
        list(artist[artist.index.isin(scrobbles.index[recent])].value_counts().index[:n_recent])
    genres = ["country", "reggaeton", "indie", "rock", "pop", "trap", "techno", "house", "jazz",
              "flamenco", "rumba", "soundtrack", "hip hop", "Pon", "Ponme", "Quítala", "Siguiente"]
    return list(dict.fromkeys([*(names[a] for a in ranked if a in names.index), *genres]))


def main() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    tool_ids = [call("POST", "/tools", json={"tool_config": elevenlabs_tool_config(t)})["id"] for t in TOOLS]
    print(f"created {len(tool_ids)} tools")

    body = {
        "name": "listening-graph-dj",
        "conversation_config": {
            "agent": {"first_message": FIRST_MESSAGE, "language": "en",
                      "prompt": {"prompt": PROMPT, "llm": LLM, "tool_ids": tool_ids, "temperature": 0.3,
                                 # lets the agent switch ASR/TTS language when the user switches
                                 "built_in_tools": {"language_detection": {
                                     "type": "system", "name": "language_detection", "description": "",
                                     "params": {"system_tool_type": "language_detection"}}}}},
            "language_presets": {"es": {"overrides": {"agent": {"first_message": FIRST_MESSAGE_ES}}}},
            "asr": {"keywords": top_artist_keywords()},
            # long turn timeout: silence = listening to music, not an absent user
            "turn": {"turn_eagerness": "normal", "turn_timeout": TURN_TIMEOUT},
        },
        # private agent: browsers need a short-lived token from our backend (GET /agent/session)
        "platform_settings": {"auth": {"enable_auth": True}},
    }
    if state.get("agent_id"):
        call("PATCH", f"/agents/{state['agent_id']}", json=body)
        print(f"updated agent {state['agent_id']}")
        for old in state.get("tool_ids", []):
            requests.delete(f"{API}/tools/{old}", headers=HEADERS, timeout=30)
    else:
        state["agent_id"] = call("POST", "/agents/create", json=body)["agent_id"]
        print(f"created agent {state['agent_id']}")
    state["tool_ids"] = tool_ids
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    print(f"saved {STATE}")


if __name__ == "__main__":
    main()
