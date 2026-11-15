"""Tiny in-process pub/sub for the UI: every tool call, hop and playback change is an event.

GET /events streams them as Server-Sent Events.
"""

import asyncio
import json
import time

# state-like events: a fresh UI always gets the latest of each, however old
PING_S = 8  # heartbeat; the UI reconnects after ~3 missed

STATEFUL = {"now_playing", "playback", "mode", "up_next"}


class EventBus:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None  # the server loop, set by the first stream
        self.queues: set[asyncio.Queue] = set()
        self.history: list[dict] = []  # last events, so a fresh UI can catch up
        self.latest: dict[str, dict] = {}

    def publish(self, type_: str, keep: bool = True, **data) -> None:
        """keep=False: frequent ticks (playback progress) that shouldn't crowd the replay history."""
        event = {"type": type_, "ts": time.time(), **data}
        if keep:
            self.history = (self.history + [event])[-200:]
        if type_ in STATEFUL:
            self.latest[type_] = event
        # sync endpoints run in a thread pool: asyncio.Queue isn't thread-safe, so hand the event to the
        # loop (otherwise the UI may get it seconds late, or never)
        try:
            on_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:
            on_loop = False
        for q in list(self.queues):
            if on_loop or self.loop is None:
                q.put_nowait(event)
            else:
                self.loop.call_soon_threadsafe(q.put_nowait, event)

    def last(self, type_: str) -> dict | None:
        return self.latest.get(type_)

    async def stream(self):
        self.loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self.queues.add(q)
        try:
            catchup = {id(e): e for e in [*self.history[-20:], *self.latest.values()]}
            for e in sorted(catchup.values(), key=lambda e: e["ts"]):
                yield f"data: {json.dumps(e)}\n\n"
            while True:
                try:
                    e = await asyncio.wait_for(q.get(), timeout=PING_S)
                    yield f"data: {json.dumps(e)}\n\n"
                except asyncio.TimeoutError:
                    # a real event (not an SSE comment) so the UI can tell a live stream from a dead one
                    yield f"data: {json.dumps({'type': 'ping', 'ts': time.time()})}\n\n"
        finally:
            self.queues.discard(q)


bus = EventBus()
