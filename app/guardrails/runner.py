"""The guardrails boundary: NeMo rails wrapped around `run_agent`.

The architecture in one sentence: **the agent is the rails' generation step, not
the other way round.** `passthrough: true` in config.yml tells NeMo not to
answer anything itself, and `Guardrails._generate` installs `run_agent` in that
slot. Input rails therefore run before retrieval sees the question, output rails
run over the finished answer, and the LangGraph loop stays a pure reasoning loop
with no idea it is being policed.

The alternative -- calling rails from inside the agent, or checking the answer
in the API layer -- was rejected for a specific reason: rail decisions would
then be spread across two modules and a middleware, and the Phase 9 scorecard
would have to reconstruct them by matching refusal strings. Here every decision
crosses one boundary and lands in one `RunLedger`, so `GuardrailedResponse.rails`
is machine-readable telemetry rather than an inference.

`run_guarded()` returns a structure, never a bare string. Which rail fired, at
which stage, with what reason and what evidence, is the product of this module.

**Concurrency.** This module is async-first, and that is a correctness
requirement rather than a performance one. The first cut kept the in-flight
`RunLedger` in a module-level single-slot dict, on the stated assumption of "one
process, one in-flight run". That assumption is false for the FastAPI service
this feeds, and validation demonstrated the consequence with two overlapping
requests through one shared instance: one answer was destroyed by a false
`check_grounding` block while the other was served the *first* request's
citations. Serving one user's retrieval provenance under another user's answer
is a disclosure-shaped bug that produces plausible output, which is the worst
way for it to fail.

The fix has three parts, and all three are needed:

1. The ledger lives in a `ContextVar` set *inside* the per-request coroutine.
   Every `asyncio` Task copies the context at creation, so concurrent requests
   on one loop cannot see each other's ledger. The module-level dict is gone.
2. The agent -- blocking, network-bound -- runs via `asyncio.to_thread`, which
   propagates the calling context, so the worker thread reads the right ledger
   and would otherwise stall every other request on the loop.
3. All rail traffic runs on ONE event loop owned by this object. NeMo holds
   loop-bound `asyncio` primitives internally; driving `generate()` from several
   threads produced `Stale event loop: <asyncio.locks.Event> is bound to a
   different event loop` and killed the request. A single long-lived loop keeps
   those primitives valid while still running requests as concurrent tasks.

`LLMRails` construction is also not thread-safe -- two concurrent builds raise
`ValueError: Framework 'default' is already registered.` from a NeMo global
registry -- so construction is serialised under a lock.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from app.guardrails.actions import (
    PiiScanner,
    current_ledger,
    RailDecision,
    RunLedger,
    build_actions,
    reset_ledger,
    set_ledger,
)

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config"

# The exact texts in the .co files. Used only to classify an already-blocked
# response for reporting; the ledger, not this list, decides `blocked`.
REFUSAL_MESSAGES = {
    "jailbreak": "I can't help with that request.",
    "off_topic": "That's outside what I can help with.",
    "ungrounded": "I don't have enough support in ACME's knowledge base",
    "unsafe_output": "I generated an answer I'm not able to share.",
}


@dataclass
class GuardrailedResponse:
    """One guarded turn. Every field is derived, none is guessed."""

    question: str
    answer: str
    masked_question: str | None = None
    citations: list[dict] = field(default_factory=list)
    # Retrieval provenance that is NOT being claimed as support -- populated
    # when check_citations demotes a refusal's citations.
    provenance: list[dict] = field(default_factory=list)
    blocked: bool = False
    blocked_by: str | None = None
    blocked_stage: str | None = None
    rails: list[dict] = field(default_factory=list)
    pii_input: list[dict] = field(default_factory=list)
    pii_output: list[dict] = field(default_factory=list)
    grounding: dict | None = None
    agent: Any | None = None            # AgentResponse, or None if input-blocked
    agent_ran: bool = False
    tools_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    agent_latency_ms: float = 0.0
    error: str | None = None

    @property
    def fired(self) -> list[str]:
        """Names of every rail that fired, in order."""
        return [r["rail"] for r in self.rails if r["triggered"]]

    @property
    def rails_evaluated(self) -> list[str]:
        return [r["rail"] for r in self.rails]

    def rail(self, name: str) -> dict | None:
        for r in self.rails:
            if r["rail"] == name:
                return r
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("agent", None)
        d["fired"] = self.fired
        return d


class Guardrails:
    """Loads the rails config once and guards many questions.

    Construction is expensive (a Colang parse, an embedding model for flow
    matching, a spaCy pipeline), so build one and reuse it. Phase 6 holds a
    single instance for the process.
    """

    def __init__(self, settings: Settings | None = None,
                 config_path: Path | str = CONFIG_PATH,
                 agent_fn: Callable[..., Any] | None = None,
                 scanner: PiiScanner | None = None):
        self.settings = settings or get_settings()
        self.config_path = Path(config_path)
        # Injection seam: tests pass a stub so the suite never calls the agent.
        self._agent_fn = agent_fn
        self._scanner = scanner or PiiScanner()
        self._rails = None
        # Serialises LLMRails construction: NeMo registers its framework in a
        # process-global registry and a concurrent second build raises.
        self._build_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

    # ------------------------------------------------------------ wiring
    def _agent(self, question: str):
        """Blocking. Always reached through asyncio.to_thread -- see _generate."""
        if self._agent_fn is not None:
            return self._agent_fn(question)
        from app.agent.graph import run_agent
        return run_agent(question)

    def _build_config(self):
        from nemoguardrails import RailsConfig

        config = RailsConfig.from_path(str(self.config_path))
        s = self.settings
        for model in config.models:
            if model.type != "main":
                continue
            # config.yml carries no credentials and no host. The rails LLM is
            # the fast tier, resolved through the same tiering module the agent
            # uses, so there is one place that decides which model does what.
            from app.llm.tiering import Task, pick_model
            model.model = pick_model(Task.SELF_CHECK, s)
            params = dict(model.parameters or {})
            params.setdefault("temperature", 0.0)
            params["base_url"] = s.openrouter_base_url
            params["api_key"] = s.openrouter_api_key or "missing"
            # Note: no `streaming` kwarg. NeMo 0.24's default framework
            # rejects it, and it would be redundant -- nothing here calls
            # stream_async, so every rail sees a complete message.
            model.parameters = params
        return config

    @property
    def rails(self):
        if self._rails is not None:
            return self._rails
        with self._build_lock:
            # Double-checked: another thread may have finished while we waited.
            if self._rails is None:
                from nemoguardrails import LLMRails

                rails = LLMRails(self._build_config(), verbose=False)
                for name, fn in build_actions(self._scanner).items():
                    rails.register_action(fn, name)
                rails.passthrough_fn = self._generate
                self._rails = rails
        return self._rails

    # --------------------------------------------------------- event loop
    def _owned_loop(self) -> asyncio.AbstractEventLoop:
        """One long-lived loop for all rail traffic from sync callers.

        Not an optimisation. NeMo caches loop-bound asyncio primitives, so a
        fresh loop per call (what `LLMRails.generate` does via
        `get_or_create_event_loop`) raises `Stale event loop` as soon as a
        second thread calls in. One loop, many tasks.
        """
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _spin():
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(target=_spin, name="guardrails-loop",
                                      daemon=True)
            thread.start()
            ready.wait(timeout=10)
            self._loop, self._loop_thread = loop, thread
        return self._loop

    def close(self) -> None:
        """Stop the owned loop. Phase 6 calls this on shutdown."""
        loop, self._loop = self._loop, None
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
        self._loop_thread = None

    async def _generate(self, context: dict | None = None, events: list | None = None):
        """The rails' generation step. This is where the agent runs.

        `context["user_message"]` is the question *after* the input rails, so
        the agent receives the PII-masked text and the raw value never reaches
        Qdrant, Neo4j or the synthesis model.
        """
        ledger = current_ledger()
        question = (context or {}).get("user_message") or ledger.question
        ledger.masked_question = question
        t0 = time.perf_counter()
        # to_thread, not a direct call: run_agent blocks on Qdrant, Neo4j and
        # two LLM round-trips, and blocking this loop would serialise every
        # other in-flight request behind it. to_thread copies the current
        # context, so the worker still sees this request's ledger.
        response = await asyncio.to_thread(self._agent, question)
        ledger.agent_response = response
        ledger.agent_latency_ms = (time.perf_counter() - t0) * 1000
        answer = getattr(response, "answer", "") or ""
        # Second element becomes NeMo's $passthrough_output. We do not read it
        # back -- the ledger is the channel -- but it keeps the framework's own
        # explain/log view honest about what generation produced.
        return answer, {"tools_used": list(getattr(response, "tools_used", []) or []),
                        "n_citations": len(getattr(response, "citations", []) or [])}

    def warmup(self) -> None:
        """Pay the one-off costs now: Colang parse, flow embeddings, spaCy.

        First call is ~5s of model loading that would otherwise land on the
        first user request and look like a slow system rather than a cold one.
        """
        self._scanner.scan("warmup")
        _ = self.rails

    # -------------------------------------------------------------- run
    async def arun(self, question: str) -> GuardrailedResponse:
        """Guard one question. The real entry point; `run` wraps this.

        The ledger is bound to the ContextVar HERE, inside the coroutine, which
        is what makes concurrency safe: every request is its own asyncio Task,
        every Task gets its own copy of the context, so `current_ledger()` in an
        action resolves to the ledger of the request that action belongs to.
        Binding it in the caller instead -- or in a module-level slot -- is how
        request A ends up serving request B's citations.
        """
        t0 = time.perf_counter()
        ledger = RunLedger(question=question)
        token = set_ledger(ledger)
        error = None
        text = ""
        try:
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": question}])
            text = result.get("content", "") if isinstance(result, dict) else str(result)
        except Exception as exc:  # noqa: BLE001 - a rail failure must not 500
            log.exception("guardrails run failed")
            error = f"{type(exc).__name__}: {exc}"
            text = "I couldn't complete that request safely. Please try again."
        finally:
            reset_ledger(token)

        return self._assemble(question, text, ledger, error,
                              (time.perf_counter() - t0) * 1000)

    def run(self, question: str) -> GuardrailedResponse:
        """Synchronous entry point. Safe to call from many threads at once.

        The work is submitted to this object's own loop rather than run on a
        per-call loop, because NeMo holds loop-bound asyncio primitives and a
        second thread with a second loop raises `Stale event loop` mid-request.
        """
        if _in_running_loop():
            raise RuntimeError(
                "Guardrails.run() was called from inside a running event loop. "
                "Use `await guardrails.arun(question)` instead -- blocking the "
                "loop here would deadlock the very requests it is serving.")
        # rails is built here, under the lock, rather than inside the loop
        # thread, so a construction failure surfaces to the caller intact.
        _ = self.rails
        loop = self._owned_loop()
        return asyncio.run_coroutine_threadsafe(self.arun(question), loop).result()

    def _assemble(self, question: str, text: str, ledger: RunLedger,
                  error: str | None, latency_ms: float) -> GuardrailedResponse:
        agent = ledger.agent_response
        blocked_by = ledger.blocked_by
        blocked_decision = next(
            (d for d in ledger.decisions if d.triggered and d.blocking), None)

        cd = ledger.citation_decision
        if cd is not None:
            citations, provenance = cd.citations, cd.provenance
        else:
            # Blocked before check_citations ran. A blocked answer carries no
            # citations by construction; what retrieval saw stays as provenance.
            raw = list(getattr(agent, "citations", []) or []) if agent else []
            if blocked_by:
                citations, provenance = [], [
                    {**c, "supports_answer": False,
                     "demoted_because": f"blocked_by_{blocked_by}"} for c in raw]
            else:
                citations, provenance = raw, []

        if error:
            blocked_by = blocked_by or "runner_error"

        return GuardrailedResponse(
            question=question,
            answer=text,
            masked_question=ledger.masked_question,
            citations=citations,
            provenance=provenance,
            blocked=bool(blocked_by),
            blocked_by=blocked_by,
            blocked_stage=blocked_decision.stage if blocked_decision else (
                "runner" if error else None),
            rails=[d.to_dict() for d in ledger.decisions],
            pii_input=ledger.pii_input,
            pii_output=ledger.pii_output,
            grounding=ledger.grounding.to_dict() if ledger.grounding else None,
            agent=agent,
            agent_ran=agent is not None,
            tools_used=list(getattr(agent, "tools_used", []) or []) if agent else [],
            latency_ms=latency_ms,
            agent_latency_ms=getattr(ledger, "agent_latency_ms", 0.0) or 0.0,
            error=error,
        )


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


_DEFAULT: Guardrails | None = None
_DEFAULT_LOCK = threading.Lock()


def get_guardrails(settings: Settings | None = None) -> Guardrails:
    """Process-wide instance. Phase 6 imports this rather than constructing one."""
    global _DEFAULT
    if _DEFAULT is not None:
        return _DEFAULT
    # Locked: two threads racing here would each build an LLMRails, and NeMo's
    # process-global framework registry raises on the second.
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Guardrails(settings=settings)
    return _DEFAULT


def run_guarded(question: str, settings: Settings | None = None) -> GuardrailedResponse:
    return get_guardrails(settings).run(question)
