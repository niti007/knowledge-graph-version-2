"""In-memory conversation sessions, with a TTL and a hard cap.

**Volatility is the design, and it is documented rather than hidden.** Sessions
live in this process's memory and nowhere else: restarting the API loses every
session, and two API processes behind a load balancer would not share them. That
is an accepted Phase 6 limitation, not an oversight -- the alternative is a
Redis the deployment does not otherwise need, for state whose useful lifetime is
one demo conversation. The README says so, `/health` reports the store size, and
`GET /sessions/{id}` returns 404 for an id from before a restart, which is the
honest answer.

**Concurrency is the part that has to be right.** Phase 5 shipped a bug in
exactly this class -- shared per-request state in a module-level slot, two
overlapping requests, one served the other's citations. A dict of sessions
touched by many concurrent handlers is the same hazard wearing different
clothes, so every mutation here happens under one `asyncio.Lock`, and the
critical sections contain no `await` on anything slow. Reads return deep copies:
handing a caller the live turn list would let a serialising response race a
concurrent append.

The lock is an `asyncio.Lock` rather than a `threading.Lock` because the API is
async end to end (see `Guardrails.arun` -- rails traffic must stay on one loop),
so all access is from a single loop and blocking it would be both unnecessary
and harmful.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_TTL_SECONDS = 60 * 60          # one hour idle
DEFAULT_MAX_SESSIONS = 500
DEFAULT_MAX_TURNS = 40                 # per session; oldest turns drop first


@dataclass
class Session:
    session_id: str
    created_at: float
    last_seen_at: float
    turns: list[dict] = field(default_factory=list)


class SessionStore:
    """TTL + LRU-capped session store. Safe under concurrent handlers."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 max_sessions: int = DEFAULT_MAX_SESSIONS,
                 max_turns: int = DEFAULT_MAX_TURNS,
                 clock=time.time):
        self.ttl = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self.max_turns = int(max_turns)
        self._clock = clock
        # OrderedDict, not dict: eviction needs least-recently-used order and
        # move_to_end gives it for free.
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = asyncio.Lock()
        self.created_count = 0
        self.evicted_count = 0
        self.expired_count = 0

    # ------------------------------------------------------------- helpers
    def _is_expired(self, s: Session, now: float) -> bool:
        return (now - s.last_seen_at) > self.ttl

    def _sweep(self, now: float) -> None:
        """Drop expired sessions. Called under the lock, on every access.

        Sweeping on access rather than from a background task means there is no
        second concurrency domain to reason about, and the cost is bounded by
        `max_sessions`.
        """
        dead = [k for k, s in self._sessions.items() if self._is_expired(s, now)]
        for k in dead:
            del self._sessions[k]
            self.expired_count += 1

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)     # least recently used
            self.evicted_count += 1

    # --------------------------------------------------------------- api
    async def touch(self, session_id: str | None) -> str:
        """Return a live session id, creating one if needed.

        A client-supplied id for a session that has expired or was never seen
        is *adopted* rather than rejected: the id is opaque to the server, and
        refusing it would strand a UI holding an id across a restart with no way
        to recover but to clear its own state.
        """
        now = self._clock()
        async with self._lock:
            self._sweep(now)
            sid = session_id or uuid.uuid4().hex
            existing = self._sessions.get(sid)
            if existing is None:
                self._sessions[sid] = Session(session_id=sid, created_at=now,
                                              last_seen_at=now)
                self.created_count += 1
            else:
                existing.last_seen_at = now
            self._sessions.move_to_end(sid)
            self._evict_if_needed()
            return sid

    async def append(self, session_id: str, *turns: dict) -> None:
        """Append turns to a session, creating it if it has since vanished."""
        now = self._clock()
        async with self._lock:
            self._sweep(now)
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(session_id=session_id, created_at=now, last_seen_at=now)
                self._sessions[session_id] = s
                self.created_count += 1
            for t in turns:
                s.turns.append({**t, "at": t.get("at", now)})
            if len(s.turns) > self.max_turns:
                # Trim from the front: recent context is what a follow-up needs.
                del s.turns[: len(s.turns) - self.max_turns]
            s.last_seen_at = now
            self._sessions.move_to_end(session_id)
            self._evict_if_needed()

    async def get(self, session_id: str) -> Session | None:
        """A snapshot, not the live object -- see the module docstring."""
        now = self._clock()
        async with self._lock:
            self._sweep(now)
            s = self._sessions.get(session_id)
            if s is None:
                return None
            self._sessions.move_to_end(session_id)
            return Session(session_id=s.session_id, created_at=s.created_at,
                           last_seen_at=s.last_seen_at,
                           turns=copy.deepcopy(s.turns))

    async def size(self) -> int:
        now = self._clock()
        async with self._lock:
            self._sweep(now)
            return len(self._sessions)

    async def clear(self) -> None:
        async with self._lock:
            self._sessions.clear()

    def expires_at(self, s: Session) -> float:
        return s.last_seen_at + self.ttl
