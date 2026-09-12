# Chapter 10 — The API and UI

## Why this / what's the need

Everything so far is Python functions. Users need a URL, and other programs (the evaluation
harnesses in Chapter 13, the Streamlit page) need a stable contract. **FastAPI** provides
one HTTP boundary — `POST /chat` plus `/health`, `/sessions/{id}` and `/metrics` — and a
thin **Streamlit** page talks to it.

The rule that keeps the system safe is stated in the UI's docstring:

```python
This file speaks HTTP to the API and nothing else. It does not import the agent,
the retrieval layer or the guardrails, and that is a boundary worth stating: the
moment a UI can call `run_agent` directly, there is a second code path that
skips the rails, and the rails stop being a property of the system and become a
property of one caller. Everything here goes through `POST /chat`.
```

> 🔑 **New word — API (Application Programming Interface):** a doorway that lets one
> program call another in a predictable way. Here: send JSON, get JSON.

> 🔑 **New word — event loop:** the single thread that runs all `async` code in a Python
> service, switching between requests while they wait on the network.

---

## Three decisions in `app/api/main.py`

### 1. Everything awaits `guardrails.arun()` on the service's own loop

```python
**1. Everything is `async`, and every rails call is `await guardrails.arun(...)`.**

Not a style preference -- a correctness constraint measured in Phase 5. NeMo
caches loop-bound `asyncio` primitives inside `LLMRails`, so driving one
instance from two different event loops makes its LLM client fail with
`Stale event loop: <asyncio.locks.Event> is bound to a different event loop`
and retry. Mixing a sync `run()` path with an async one produced 7 retries on a
single request; keeping every request on the server's own loop via `arun`
produced 0.
```

- `Guardrails.run()` (the sync entry point) raises if called from inside a running loop, so
  this cannot regress silently. Across 88 real requests the server log contained zero stale
  loop warnings.

### 2. Warmup in the lifespan handler, not on the first request

```python
async def _warmup(state: AppState) -> None:
    t0 = time.perf_counter()
    try:
        rails = await state.rails()

        def _load() -> None:
            from app.ingestion.embedding_model import get_embedder, get_reranker

            get_embedder(state.settings)          # bi-encoder, ~2s
            get_reranker(state.settings)          # cross-encoder, ~14s cold
            rails.warmup()                        # Colang + flows + spaCy, ~7-9s
            ...
        await asyncio.to_thread(_load)
        state.warmup_complete = True
```

- Measured: the first request without warmup took 19.6 s against 4.5 s for the second. A
  15.1 s penalty would have landed on the first user. Paid at startup instead, off the loop
  so `/health` can already answer.
- `/health` returns **503** until warmup completes and both Qdrant and Neo4j are reachable:

```python
        # 200 only when the system can actually answer a question. A green
        # check over a dead Qdrant is worse than no check at all: it tells a
        # load balancer to keep sending traffic that cannot succeed.
        return JSONResponse(status_code=200 if ready else 503,
                            content=body.model_dump())
```

### 3. Meta questions are answered before the rails — `app/api/system_card.py`

Chapter 09 left three benign false positives open: "what can you do?", "which documents do
you have access to?", "can you cite your sources?". The chain was: topic-scope allowed them,
the agent answered from its own knowledge without a tool call, and the grounding rail
correctly blocked `answered_without_retrieval`. Teaching the rail to excuse them would put
routing logic inside a safety check — one broad pattern away from excusing "what do you
know about our password policy?". So the question never reaches the rails:

```python
    meta = system_card.classify(question)
    ...
            if meta is not None:
                # Answered before retrieval, before the rails, and before the
                # cache -- it is already ~0.1ms, and caching a constant string
                # would add a network round-trip to make it slower.
                response = ChatResponse(
                    answer=system_card.answer(question),
                    trace_id=trace_id, session_id=session_id, route="system_card",
                    ...
```

The classifier is deliberately precision-biased — two rules, both narrow:

```python
def classify(question: str) -> MetaMatch | None:
    if not question or not question.strip():
        return None
    q = question.strip()
    if len(q.split()) > MAX_META_WORDS:
        return None
    if _CORPUS_RE.search(q):
        # Talks about ACME's world. Not a question about me.
        return None
    for intent, rx in _COMPILED:
        if rx.search(q):
            return MetaMatch(intent=intent, pattern=rx.pattern)
    return None
```

- An anchored pattern about *the assistant* must match, **and** no corpus vocabulary
  (`pol-\d+`, `password`, `team`, `owns`, `acme`…) may appear. A missed meta question just
  takes the normal path. Verified: 6/6 meta questions routed, 0/10 corpus traps
  misrouted, including "can you tell me who owns Payment-Service?". Benign
  false-positive count for the meta class: 3 → 0.

## The request contract — `app/api/schemas.py`

```python
class ChatRequest(BaseModel):
    # extra="forbid": a typo'd field name ("sessionId") silently creating a new
    # session is a worse failure than a 422 that names the mistake.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=128)
```

- Malformed payloads return **422** from pydantic before any handler runs — never a 500.

The response carries the rails' telemetry verbatim:

```python
class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    trace_id: str
    trace_url: str | None = None
    session_id: str
    latency_ms: float
    cached: bool = False
    blocked: bool = False
    blocked_by: str | None = None
    blocked_stage: str | None = None
    route: Literal["system_card", "guardrails"] = "guardrails"
    rails_fired: list[str] = Field(default_factory=list)
    ...
```

- `blocked_by` and `rails_fired` are *copied* from the ledger, never recomputed. That is why
  Chapter 13's Promptfoo cases can assert "which rail fired" instead of "the answer looked
  like a refusal".

## Sessions — `app/api/sessions.py`

```python
**Volatility is the design, and it is documented rather than hidden.** Sessions
live in this process's memory and nowhere else: restarting the API loses every
session, and two API processes behind a load balancer would not share them.
```

```python
DEFAULT_TTL_SECONDS = 60 * 60          # one hour idle
DEFAULT_MAX_SESSIONS = 500
DEFAULT_MAX_TURNS = 40                 # per session; oldest turns drop first
```

- TTL- and LRU-capped, every mutation under one `asyncio.Lock`, reads return deep copies.
  The comment ties it back to Chapter 09: a dict of sessions touched by concurrent handlers
  is the same hazard as the ledger bug wearing different clothes.
- `GET /sessions/{id}` for an id from before a restart returns 404 with a note that says so.
- Note: the session transcript is stored, but `run_agent` is called without history — the
  answer is a pure function of the masked question. That matters for the cache in
  Chapter 11.

## Metrics — `app/api/metrics.py`

```python
def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. `values` need not be sorted."""
    if not values:
        return None
    ordered = sorted(values)
    # Nearest rank = ceil(pct/100 * N), 1-indexed. math.ceil, not round(x+0.5):
    # round() is banker's rounding and put p95 of 1..100 at 96.
    k = max(0, min(len(ordered) - 1,
                   math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[k]
```

- Counters and latency percentiles over a bounded ring of the last 5,000 samples. Chapter
  13's load report reads `/metrics` for the agent-latency split.

## The Streamlit page — `ui/streamlit_app.py`

```python
def render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("No citations - this answer claims no corpus support.")
        return
    docs = [c for c in citations if c.get("doc_id")]
    graph = [c for c in citations if not c.get("doc_id") and c.get("graph_template")]
```

- Document citations and graph citations are rendered as separate lists, because a chunk a
  human can open and a graph traversal are different kinds of evidence.

```python
    trace_url = data.get("trace_url")
    if trace_url:
        st.markdown(f"[Open trace in Langfuse]({trace_url})")
```

- Every answer carries a clickable link to its Langfuse trace (Chapter 11), plus a "Rail
  detail" expander showing the full `rails`, `grounding` and `provenance` structures.

## Concurrency, verified at this layer

6 simultaneous requests through the real API: every citation naming its own question, 6
distinct sessions, no cross-contamination — the Phase 5 bug class, re-tested where it
would actually bite.

---

## ✅ You just learned
- Four endpoints and the one guarded path into the system.
- Why every request awaits `arun()` on one loop, and why warmup happens at startup.
- The system-card router: precision-biased, outside the rails, and why.
- Sessions are in-memory by design; malformed input is a 422.

## ▶️ Run this now
```bash
make serve        # terminal 1; wait for "warmup finished"
curl -s localhost:8000/health | python -m json.tool
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"What can you do?"}' | python -c "import json,sys; d=json.load(sys.stdin); print(d['route'], d['latency_ms'])"
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"Who leads the Infrastructure team?"}' | python -m json.tool
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"sessionId":"x"}'   # expect 422
```

## 🧠 Check yourself
1. Why does `/health` return 503 during warmup instead of 200 with a flag?
2. "What do you know about our password policy?" — which route does it take, and which rule
   in `classify` decides?
3. What would go wrong if the Streamlit page imported `run_agent`?

---

Next: the cache, the two model tiers, and the traces →
[11-cache-tiering-tracing.md](11-cache-tiering-tracing.md)
