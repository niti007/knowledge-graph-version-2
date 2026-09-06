"""FastAPI service: the one HTTP boundary in front of the guarded agent.

Three decisions carry this module.

**1. Everything is `async`, and every rails call is `await guardrails.arun(...)`.**

Not a style preference -- a correctness constraint measured in Phase 5. NeMo
caches loop-bound `asyncio` primitives inside `LLMRails`, so driving one
instance from two different event loops makes its LLM client fail with
`Stale event loop: <asyncio.locks.Event> is bound to a different event loop`
and retry. Mixing a sync `run()` path with an async one produced 7 retries on a
single request; keeping every request on the server's own loop via `arun`
produced 0. `Guardrails.run()` now raises if called inside a running loop
precisely so this cannot regress silently, so there is no sync endpoint here
and no `run_in_threadpool` around the rails.

**2. Warmup happens in the lifespan handler, never on the first request.**

The rails cost ~7-9s to construct (Colang parse, flow-matching embeddings,
spaCy pipeline) and the BGE cross-encoder another ~14s to load. Paid lazily,
the first user request takes half a minute and looks like a broken system
rather than a cold one. Paid at startup, `/health` simply reports
`warmup_complete: false` until it is done -- which is the honest thing for a
health check to say and is why it returns 503 while starting.

**3. Meta questions are answered before the rails see them.**

See `app/api/system_card.py`. This is the routing fix for Phase 5's three
remaining benign false positives, placed here rather than inside the grounding
rail because it is routing, not safety.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import system_card
from app.api.metrics import Metrics
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DependencyHealth,
    HealthResponse,
    MetricsResponse,
    SessionResponse,
)
from app.api.sessions import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_MAX_TURNS,
    DEFAULT_TTL_SECONDS,
    SessionStore,
)
from app.config import Settings, get_settings

log = logging.getLogger(__name__)

Probe = Callable[[], DependencyHealth]


# ------------------------------------------------------------------ probes

def _timed(name: str, fn: Callable[[], str]) -> DependencyHealth:
    """Run a blocking connectivity check and report it, never raise.

    A probe that throws would take the health endpoint down with it, which is
    the one endpoint that must keep answering when a dependency is broken.
    """
    t0 = time.perf_counter()
    try:
        detail = fn()
        ok = True
    except Exception as exc:  # noqa: BLE001 - reporting IS the job here
        detail, ok = f"{type(exc).__name__}: {exc}"[:200], False
    return DependencyHealth(name=name, ok=ok, detail=detail,
                            latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def probe_qdrant(settings: Settings | None = None) -> DependencyHealth:
    def _check() -> str:
        from app.ingestion.vector_index import qdrant_client

        s = settings or get_settings()
        with qdrant_client(s) as client:
            names = [c.name for c in client.get_collections().collections]
            if s.qdrant_collection not in names:
                raise RuntimeError(
                    f"collection '{s.qdrant_collection}' missing; run `make ingest`")
            count = client.count(s.qdrant_collection, exact=True).count
            if count == 0:
                raise RuntimeError(f"collection '{s.qdrant_collection}' is empty")
            return f"{count} points in '{s.qdrant_collection}'"

    return _timed("qdrant", _check)


def probe_neo4j(settings: Settings | None = None) -> DependencyHealth:
    def _check() -> str:
        from app.ingestion.graph_builder import get_driver

        driver = get_driver(settings or get_settings())
        try:
            driver.verify_connectivity()
            with driver.session() as sess:
                n = sess.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            if n == 0:
                raise RuntimeError("graph is empty; run `make ingest`")
            return f"{n} nodes"
        finally:
            driver.close()

    return _timed("neo4j", _check)


def default_probes(settings: Settings | None = None) -> list[Probe]:
    return [lambda: probe_qdrant(settings), lambda: probe_neo4j(settings)]


# ------------------------------------------------------------------- state

@dataclass
class AppState:
    settings: Settings
    store: SessionStore
    metrics: Metrics
    probes: list[Probe]
    guardrails: object | None = None
    started_at: float = field(default_factory=time.time)
    warmup_complete: bool = False
    warmup_seconds: float | None = None
    warmup_error: str | None = None

    async def rails(self):
        """The process-wide Guardrails instance, built on demand if warmup ran late."""
        if self.guardrails is None:
            from app.guardrails.runner import get_guardrails

            self.guardrails = get_guardrails(self.settings)
        return self.guardrails


def _state(request: Request) -> AppState:
    return request.app.state.app_state


# -------------------------------------------------------------------- app

def create_app(*, settings: Settings | None = None,
               guardrails: object | None = None,
               probes: list[Probe] | None = None,
               warm: bool = True,
               store: SessionStore | None = None) -> FastAPI:
    """Build the ASGI app.

    A factory, not a module-level singleton with monkeypatched globals: the test
    suite needs an app whose guardrails are a stub and whose probes are fakes,
    and injecting those at construction is the difference between testing the
    real wiring and testing a patched version of it. `warm=False` skips the
    ~20s startup cost for tests.
    """
    s = settings or get_settings()
    state = AppState(
        settings=s,
        store=store or SessionStore(ttl_seconds=DEFAULT_TTL_SECONDS,
                                    max_sessions=DEFAULT_MAX_SESSIONS,
                                    max_turns=DEFAULT_MAX_TURNS),
        metrics=Metrics(),
        probes=probes if probes is not None else default_probes(s),
        guardrails=guardrails,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = state
        if warm:
            await _warmup(state)
        else:
            state.warmup_complete = True
            state.warmup_seconds = 0.0
        yield
        # The rails own a background event-loop thread for sync callers; close
        # it so a reload does not leak one per restart.
        g = state.guardrails
        if g is not None and hasattr(g, "close"):
            try:
                g.close()
            except Exception:  # noqa: BLE001
                log.warning("guardrails close failed", exc_info=True)

    app = FastAPI(title="ACME Enterprise Knowledge Assistant",
                  version="0.6.0", lifespan=lifespan)
    app.state.app_state = state
    _register_routes(app)
    return app


async def _warmup(state: AppState) -> None:
    """Pay every cold cost before the first request, in one place.

    Three separate loads, all blocking and all CPU/IO bound, so each goes
    through `asyncio.to_thread` and the server can already answer `/health`
    (with `warmup_complete: false`) while they run. A failure is recorded and
    re-reported by `/health` rather than raised: a system that cannot load its
    reranker should come up and say so, not refuse to start and leave an
    operator reading logs to find out why.
    """
    t0 = time.perf_counter()
    try:
        rails = await state.rails()

        def _load() -> None:
            from app.ingestion.embedding_model import get_embedder, get_reranker

            get_embedder(state.settings)          # bi-encoder, ~2s
            get_reranker(state.settings)          # cross-encoder, ~14s cold
            rails.warmup()                        # Colang + flows + spaCy, ~7-9s

        await asyncio.to_thread(_load)
        state.warmup_complete = True
    except Exception as exc:  # noqa: BLE001
        log.exception("warmup failed")
        state.warmup_error = f"{type(exc).__name__}: {exc}"[:300]
        state.warmup_complete = False
    finally:
        state.warmup_seconds = round(time.perf_counter() - t0, 2)
        log.info("warmup finished in %.2fs (ok=%s)",
                 state.warmup_seconds, state.warmup_complete)


# ----------------------------------------------------------------- routes

def _register_routes(app: FastAPI) -> None:

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """Never leak a traceback, and never let an error go uncounted."""
        log.exception("unhandled error on %s", request.url.path)
        try:
            _state(request).metrics.record_error()
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "error": type(exc).__name__})

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, state: AppState = Depends(_state)) -> ChatResponse:
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        session_id = await state.store.touch(req.session_id)
        question = req.question.strip()

        meta = system_card.classify(question)
        if meta is not None:
            # Answered before retrieval and before the rails. See system_card.
            response = ChatResponse(
                answer=system_card.answer(question),
                trace_id=trace_id, session_id=session_id, route="system_card",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        else:
            # arun, NOT run: see the module docstring. This is the only call
            # into the guardrails layer in the whole service.
            rails = await state.rails()
            guarded = await rails.arun(question)
            response = _to_response(guarded, trace_id=trace_id,
                                    session_id=session_id, state=state)
            response.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        state.metrics.record_chat(
            route=response.route, latency_ms=response.latency_ms,
            agent_latency_ms=0.0 if meta else getattr(guarded, "agent_latency_ms", 0.0),
            blocked=response.blocked, blocked_by=response.blocked_by,
            rails_fired=response.rails_fired, tools_used=response.tools_used,
            cached=response.cached,
            status="blocked" if response.blocked else "ok")

        await state.store.append(
            session_id,
            {"role": "user", "content": question, "at": time.time()},
            {"role": "assistant", "content": response.answer, "at": time.time(),
             "trace_id": trace_id, "blocked": response.blocked,
             "blocked_by": response.blocked_by, "route": response.route,
             "citations": len(response.citations),
             "latency_ms": response.latency_ms},
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health(state: AppState = Depends(_state)):
        # Probes are blocking network calls; off the loop they go, in parallel,
        # so one hung dependency does not serialise behind another.
        deps: list[DependencyHealth] = list(
            await asyncio.gather(*(asyncio.to_thread(p) for p in state.probes))
        ) if state.probes else []
        deps_ok = all(d.ok for d in deps)
        ready = state.warmup_complete and deps_ok
        if not state.warmup_complete:
            status = "starting"
        elif not deps_ok:
            status = "degraded"
        else:
            status = "ok"
        body = HealthResponse(
            status=status, ready=ready,
            warmup_complete=state.warmup_complete,
            warmup_seconds=state.warmup_seconds,
            warmup_error=state.warmup_error,
            uptime_seconds=round(time.time() - state.started_at, 2),
            dependencies=deps,
        )
        # 200 only when the system can actually answer a question. A green
        # check over a dead Qdrant is worse than no check at all: it tells a
        # load balancer to keep sending traffic that cannot succeed.
        return JSONResponse(status_code=200 if ready else 503,
                            content=body.model_dump())

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str, state: AppState = Depends(_state)):
        s = await state.store.get(session_id)
        if s is None:
            # Also the answer for an id from before a restart -- sessions are
            # in memory and do not survive one. Documented in sessions.py.
            return JSONResponse(
                status_code=404,
                content={"detail": "session not found or expired",
                         "session_id": session_id,
                         "note": "sessions are in-memory and are lost on API restart"})
        return SessionResponse(
            session_id=s.session_id, created_at=s.created_at,
            last_seen_at=s.last_seen_at,
            expires_at=state.store.expires_at(s),
            turns=s.turns, n_turns=len(s.turns))

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics(state: AppState = Depends(_state)):
        snap = state.metrics.snapshot()
        return MetricsResponse(
            uptime_seconds=round(time.time() - state.started_at, 2),
            sessions_active=await state.store.size(),
            sessions_created=state.store.created_count,
            sessions_evicted=state.store.evicted_count,
            **snap)


def _to_response(guarded, *, trace_id: str, session_id: str,
                 state: AppState) -> ChatResponse:
    """Map the guardrails result onto the wire contract.

    Fields are copied, never recomputed: `blocked_by` and `rails_fired` are the
    rails' own telemetry, which is what makes the Phase 9 scorecard a read of
    real decisions rather than a string-match over refusals.
    """
    d = guarded.to_dict()
    return ChatResponse(
        answer=d.get("answer") or "",
        citations=d.get("citations") or [],
        provenance=d.get("provenance") or [],
        tools_used=d.get("tools_used") or [],
        trace_id=trace_id,
        trace_url=None,          # Phase 7
        session_id=session_id,
        latency_ms=round(d.get("latency_ms") or 0.0, 2),
        cached=False,            # Phase 7 owns the semantic cache
        blocked=bool(d.get("blocked")),
        blocked_by=d.get("blocked_by"),
        blocked_stage=d.get("blocked_stage"),
        route="guardrails",
        rails_fired=d.get("fired") or [],
        rails=d.get("rails") or [],
        grounding=d.get("grounding"),
        pii_masked=bool(d.get("pii_input")),
        agent_ran=bool(d.get("agent_ran")),
        error=d.get("error"),
    )


app = create_app()
