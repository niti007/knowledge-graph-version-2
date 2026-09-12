# Chapter 11 — Caching, Tiering, Tracing

## Why this / what's the need

Three production concerns, one chapter:

- **Cost and speed.** Many people ask the same questions. A cache that recognises "we
  already answered this" skips the agent entirely.
- **Cost, again.** Not every LLM call needs the expensive model. The rails ask yes/no
  questions; `gpt-4o-mini` answers those as well as `gpt-4o` at a fraction of the price.
- **Visibility.** When an answer is wrong or slow, you need to see every step — which tool
  ran, what each LLM call cost, which rail fired. That is **Langfuse**.

The cache section contains the project's second most instructive negative result: the plan
said "semantic cache, cosine ≥ 0.95". Measured on this corpus, **0.95 serves the wrong
answer**, and no threshold fixes it.

> 🔑 **New word — semantic cache:** a cache keyed on *meaning* rather than exact text, so
> "Who leads Infrastructure?" and "Who is the Infrastructure lead?" share an entry.

> 🔑 **New word — trace:** a tree of timed, nested records for one request — the root is the
> HTTP call, the children are rails, cache lookup, agent, each LLM call and tool call.

---

## The semantic cache — `app/llm/cache.py`

### What was measured

The module docstring is the evidence. Pairs that need **different** answers:

```
0.9929  "Which systems depend on Auth-DB?"    | "Which systems does Auth-DB depend on?"
0.9920  "Can I use ACME laptops for personal email?" | "Can I use personal laptops for ACME email?"
0.9777  "Who manages Marcus Lee?"             | "Who does Marcus Lee manage?"
0.9558  "What depends on DataWarehouse?"      | "What does DataWarehouse depend on?"
```

Pairs that mean **the same** thing:

```
0.9799  "Who leads the Infrastructure team?"  | "Who is the lead of the Infrastructure team?"
0.9736  "Who owns Payment-Service?"           | "Payment-Service is owned by whom?"
0.9519  "How long are backups retained?"      | "How long do we retain backups?"
```

- The worst near-miss (0.9929) outscores every non-trivial paraphrase. The distributions
  overlap; no threshold separates them. At the planned 0.95, four of nine near-miss pairs
  would have been served the opposite question's answer.
- The cross-encoder was tried as a second gate and is *worse*: it scores all four inversions
  at exactly 1.000, because an inverted question is maximally relevant to its own inversion.
  Relevance is not equivalence — the same lesson as Chapter 07 and Chapter 09.

### The structural guard

The failure mode is structural: **the same content words in a different order**. So the
guard is structural too.

```python
def content_tokens(text: str) -> list[str]:
    """Stemmed non-stopword tokens, in order. Order is the point."""
    return [_stem(w) for w in normalize(text).split() if w not in STOPWORDS]


def guard_verdict(question: str, candidate: str) -> GuardVerdict:
    if normalize(question) == normalize(candidate):
        return GuardVerdict(True, "exact_normalized")
    a, b = content_tokens(question), content_tokens(candidate)
    if a and b and sorted(a) == sorted(b) and a != b:
        return GuardVerdict(False, "argument_order_differs")
    return GuardVerdict(True, "distinct_wording")
```

- `normalize` folds case, punctuation, possessives, hyphens and whitespace. Identical after
  that: always safe.
- Same multiset of content tokens, different order: the inversion signature. **Refused.**
- Different multisets: not an inversion; cosine (threshold 0.97) decides.

The cost is stated plainly: the guard also refuses passive-voice paraphrases ("Who owns
Payment-Service?" / "Payment-Service is owned by whom?"), since they are structurally
identical to an inversion. Paraphrase hit rate is 3/6, and the wins are normalization-level.
*On this corpus a safe semantic cache is close to a normalization-tolerant exact cache.* The
design fails toward misses: a refused hit costs one cache miss; a wrong hit serves a
confidently-cited answer to the opposite question. Result: 0/9 near-miss pairs hit, with
about 0.06 of margin to the highest guard-uncaught near-miss.

### Correctness properties

**Blocked requests are structurally uncacheable.** The lookup lives inside
`Guardrails._generate` (Chapter 09), which only runs after the input rails pass. A test
asserts zero Qdrant queries for a jailbreak. And `store()` re-checks:

```python
        if blocked or error:
            # A rail decision is per-request. Cached, one attacker's refusal
            # would be served to a benign user asking a nearby question, and a
            # benign answer stored under a blocked turn would be served to the
            # next attacker whose input rails should have fired.
            return False
```

**The key is the PII-masked question**, so no PII reaches the cache collection.

**A namespace covers everything that changes the answer:**

```python
    def namespace(self) -> str:
        s = self.settings
        raw = "|".join([
            str(CACHE_SCHEMA_VERSION), s.qdrant_collection, s.embedding_model,
            s.llm_fast_model, s.llm_smart_model,
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

- Re-tier from `gpt-4o` to `gpt-4o-mini`, or re-ingest into a new collection, and old
  answers are not served under the new configuration.

**A hit replays the original evidence into the ledger**, so the output rails re-evaluate it:

```python
            if lookup.hit:
                # The ORIGINAL run's citations, provenance and abstention
                # evidence go back into the ledger, so check_grounding and
                # check_citations evaluate the same evidence they saw on the
                # miss and reach the same verdict. A hit is not a citation-free
                # answer, and it is not an unchecked one either.
                cached = lookup.as_agent_response()
                ledger.agent_response = cached
```

**TTL is enforced in the query**, not after it, so an expired entry cannot even be a
candidate the guard could accept; and the lookup fetches `cache_search_limit = 5`
candidates, because the nearest neighbour may be an inverted near-miss while the true
paraphrase sits at rank 2.

**A cache failure never fails a request.** `lookup` returns a miss and `store` returns
`False` on any exception, with the reason kept in the trace.

### What a hit saves

Cold 6.3 s → warm 3.5 s (1.8×). Agent time goes to zero; the remaining 3.4 s is the rails,
which is the architectural floor of caching *behind* them. The load test (Chapter 13) saw
the same shape: cache-on p50 3.0 s, not 0.3 s.

---

## Tiering — is it real?

Chapter 08 introduced `pick_model(task)`. Phase 7 confirmed from **provider token counts**
that `gpt-4o` runs the agent and `gpt-4o-mini` runs all five rail calls. The runner
harvests NeMo's own per-call log so the rails tier is costed from the provider's numbers:

```python
    @staticmethod
    def _llm_calls_from(result: Any) -> list[dict]:
        log_obj = getattr(result, "log", None)
        calls = getattr(log_obj, "llm_calls", None) or []
        out = []
        for c in calls:
            out.append({
                "task": getattr(c, "task", None),
                "model": getattr(c, "llm_model_name", None),
                "prompt_tokens": getattr(c, "prompt_tokens", None),
                "completion_tokens": getattr(c, "completion_tokens", None),
                ...
```

Tiering saves about **25% of request cost** — but 98% of spend is the agent, so it is a
secondary lever. The larger lever, running early tool-selection turns on the fast tier, is
something this design does not do; the commit message says so.

Measured unit costs (Phase 9): **$0.0038** per full answer, **$0.0016** per input-blocked
refusal. Refusals are not free — the jailbreak prompt is long and runs on every request.

---

## Tracing — `app/observability/langfuse_client.py`

### The invariant

```python
**The one invariant: tracing never changes the outcome of a request.** Langfuse
Cloud is a third-party HTTP dependency on the hot path of every answer, and the
failure modes are real -- an expired key, a DNS blip, a slow export. So every
entry point here is wrapped: constructing the client, opening a span, updating
one, resolving the trace URL, flushing. If any of it raises, the wrapper falls
back to `_NullSpan` and the request proceeds exactly as if tracing were off.
```

- Tested against a client that raises on construction, span open, update, close and flush.
- What is *not* wrapped: the caller's own `with` body. `_safe_span` marks the span as
  errored and **re-raises**, because an observability layer that swallows the application's
  exceptions hides the very failure it exists to record.

### Trace ids are ours

```python
    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, state: AppState = Depends(_state)) -> ChatResponse:
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        ...
        with state.tracing.trace(
                "chat", trace_id=trace_id, input=question,
                metadata={"session_id": session_id,
                          "route": "system_card" if meta else "guardrails"},
        ) as root:
```

- `/chat` mints a `uuid4().hex` — already a valid 32-hex W3C trace id — and passes it in as
  the trace context. The `trace_id` in the API response, the session transcript and the
  Langfuse UI are one string. No join table.

### Cost is computed here

```python
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    ...
}
```

- Langfuse prices known model ids itself but does not know OpenRouter's `openai/` prefix,
  so `record_generation` sends `usage_details` and `cost_details` explicitly. Wrong-but-
  visible beats absent; the numbers are one edit from current.

### The trace URL is resolved once, at warmup

```python
    def resolve_trace_url_template(self) -> str | None:
        """Fetch the project id ONCE, off the request path. Called at warmup."""
        ...
            url = client.get_trace_url(trace_id="0" * 32)
            if url:
                self._url_template = url.replace("0" * 32, "{trace_id}")
```

- `get_trace_url` does an API round-trip the first time. Done at warmup, `trace_url()` is
  pure string formatting on every request. Unresolved, the UI shows no link — correct
  degraded behaviour.

### What a trace looks like

Nested `chat > guardrails > agent > llm/tool` spans, with observation types Langfuse renders
distinctly: `guardrail` for each rail decision, `retriever` for `retrieval.hybrid_rerank`
(carrying Phase 3's own stage timings as metadata so the trace and the ablation harness
report the same numbers), `tool` for each tool call, `generation` for each LLM call with
tokens and cost. Phase 7 confirmed 21 observations arriving for one request. Chapter 14
confirmed traces arriving from inside the deployed container.

Two bugs found on the way, both recorded: the default tracer made the API test suite open a
live exporter and *hang* on shutdown rather than fail (fixed with a `conftest.py` that
installs a disabled tracer before every test), and `langfuse` was pinned to a range that
would have installed the v2 SDK.

---

## ✅ You just learned
- Why cosine ≥ 0.95 is unsafe on this corpus, what the structural guard does, and its
  honestly-stated cost.
- Four cache correctness properties: blocked-uncacheable, masked key, namespace, replayed
  evidence.
- Tiering is measured (25% saving, 98% of spend is the agent).
- Tracing never breaks a request, trace ids are the API's own, and cost is attached per call.

## ▶️ Run this now
With the API running (`make serve`):
```bash
Q='{"question":"Who owns Payment-Service?"}'
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d "$Q" | python -c "import json,sys; d=json.load(sys.stdin); print('cached', d['cached'], d['latency_ms'], d['trace_url'])"
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d "$Q" | python -c "import json,sys; d=json.load(sys.stdin); print('cached', d['cached'], d['latency_ms'])"
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"question":"Which systems does Auth-DB depend on?"}' | python -c "import json,sys; d=json.load(sys.stdin); print('cached', d['cached'])"
```
Second call: `cached True`, roughly half the latency. Open the `trace_url` from the first
call in Langfuse and expand the tree. Then test the guard directly:
```bash
.venv/bin/python -c "from app.llm.cache import guard_verdict as g; print(g('Which systems depend on Auth-DB?','Which systems does Auth-DB depend on?')); print(g('who owns payment-service','Who owns Payment-Service?'))"
```

## 🧠 Check yourself
1. Why can raising the threshold not fix the Auth-DB inversion?
2. What does the guard refuse that it should ideally allow, and why is that the right
   direction to fail?
3. Why is the cache lookup inside the rails' generation step rather than in the API handler?
4. Why does `_safe_span` re-raise the caller's exception?

---

Next: run the whole thing →
[12-run-it.md](12-run-it.md)
