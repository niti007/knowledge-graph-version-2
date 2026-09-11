# Load Test Report

Enterprise Knowledge Assistant — Phase 9. Locust 2.46 against `POST /chat`, two bounded
3-minute runs of 3 concurrent users (1–3 s think time), one with the semantic cache off
and one with it on. Each run gets a fresh uvicorn process on its own port so `/metrics`
counters are per-run, and each starts only after `/health` is 200 — **warmup (12.9 s /
13.5 s) is excluded** from every number below. Raw outputs in `results/`
(`*_stats.csv`, `*_stats_history.csv`, `*.app_stats.json`, `runs.json`). Driver:
`python -m evals.load.run_load`.

## What is being measured

This is a **latency and correctness-under-concurrency probe, not a capacity benchmark.**
One uvicorn worker on a laptop, with every request making 2–4 round trips to OpenRouter.
Requests/second here is a function of how fast the provider answers, not of what this
code can absorb. The numbers that mean something are the latency distribution, the
error rate, the cache delta, and the split between our code and upstream time.

Traffic mix (weights 6/3/1): four questions asked repeatedly (cache-eligible), ten asked
at random (agent path), and one input-rail-blocked jailbreak (so refusals — real traffic
— are in the distribution and the p50 is not computed over agent runs only).

## Headline

| | cache OFF | cache ON | delta |
|---|---:|---:|---:|
| requests completed in 3 min | 66 | 102 | **+55%** |
| throughput | 0.37 req/s | 0.57 req/s | +55% |
| errors (non-200) | **0** | **0** | — |
| p50 | 4.9 s | **3.0 s** | −39% |
| p95 | 7.8 s | 6.0 s | −23% |
| p99 | 60.0 s | 6.7 s | see below |
| max | 59.9 s | 20.9 s | |
| mean | 6.0 s | 3.3 s | −45% |
| cache hits | 0 | 78 / 102 (76%) | |
| agent actually ran (`/metrics` agent_latency count) | 58 | 14 | |
| rails-blocked | 8 | 10 | |

Zero failures in 168 requests under 3-way concurrency. Phase 5's event-loop fix
(ledger in a `ContextVar`, one loop for all rails traffic) is what this run was really
testing — the Phase 5 concurrency bug produced cross-request citation leakage under
exactly this load, and there was none.

### Per request type

| type | cache OFF n / p50 / p95 / p99 | cache ON n / p50 / p95 / p99 |
|---|---|---|
| `[repeat]` (cacheable) | 45 / 4.8 s / 7.8 s / 60.0 s | 63 / **3.0 s** / 5.2 s / 21.0 s |
| `[varied]` (agent path) | 13 / 5.7 s / 25.0 s / 25.0 s | 29 / 3.7 s / 6.6 s / 6.6 s |
| `[blocked]` (input rail) | 8 / 1.1 s / 1.2 s / 1.2 s | 10 / 1.1 s / 1.3 s / 1.3 s |

`[varied]` also got faster with the cache on — the ten "varied" questions repeat across
three users over three minutes, so most of them were cached by the second pass. The
mix is honest about that: 76% of a 3-minute window hitting cache is what an internal
assistant with a short list of hot questions looks like, and is the optimistic case.

## The p99: what the 60 s actually was

One `[repeat]` request in the cache-off run took 59.9 s. `/metrics` shows agent p99 =
56.0 s on that run, so ~56 s of the 60 s was inside the agent — a single stalled
OpenRouter round-trip, not a queue and not our code. No error, no retry, no log line;
the provider just held the connection. Phase 6 saw the same shape (p99 23.8 s from
queueing behind upstream). The cache-on run had one at 20.9 s. With n = 66 the p99 *is*
the single worst request, so it should be read as "the tail is upstream and
unbounded", not as a 1-in-100 rate. There is no client-side timeout on the OpenRouter
call today; Phase 10 should say whether that is intentional.

## Our overhead vs. upstream time

The rails ledger records per-rail latency on every request. Across the 102 scorecard
probes (same server build, same day):

| rail | kind | n | p50 | p95 |
|---|---|---:|---:|---:|
| `pii_input` (Presidio) | local | 99 | 9.6 ms | 20.3 ms |
| `self_check_input` | LLM call | 99 | 673 ms | 1248 ms |
| `topic_scope` | LLM call | 50 | 735 ms | 1203 ms |
| `self_check_output` | LLM call | 47 | 890 ms | 1301 ms |
| `check_grounding` | local | 46 | 0.1 ms | 0.2 ms |
| `pii_output` (Presidio) | local | 43 | 24 ms | 58 ms |
| `check_citations` | local | 43 | 0.0 ms | 0.0 ms |

So for a full request at cache-off p50 (4.8 s):

| component | p50 | share |
|---|---:|---:|
| agent (retrieval + rerank + 1–2 agent LLM calls + synthesis) | 2.4 s (`/metrics` agent_latency p50 2425 ms) | 50% |
| three rails LLM calls (2 input + 1 output, gpt-4o-mini) | ≈ 2.3 s | 48% |
| **everything that is ours and not an LLM** — Presidio ×2, grounding, citations, FastAPI, JSON | **≈ 35 ms** | **< 1%** |

The unassigned remainder (~50 ms) is serialisation and loop scheduling. In other words,
**the rails cost as much wall time as the agent**, and it is all provider round-trips:
three sequential ~700 ms classifications on the fast tier. The code this project wrote
is under 1% of the request.

Inside the 2.4 s agent, the cross-encoder re-ranker is the dominant retrieval cost and
the only large local one. Phase 8 measured **`bge-reranker-base` p50 1401 ms vs
`MiniLM-L-6-v2` p50 232 ms** on the same 30-question set with no demonstrated quality
difference (RAGAS faithfulness 0.933 vs 0.967, context precision 0.904 vs 0.836 — mixed,
inside noise at n = 30). It is swappable via one setting (`RERANKER_MODEL`); switching
would take ~1.2 s off every cache miss, roughly a quarter of p50. Not switched here —
this phase measures, it does not tune.

## What the cache does and does not save

Cache hit p50 is **3.0 s, not 0.3 s**. The cache sits *after* the input rails and
*before* the agent (Phase 5 ordering: PII-mask → jailbreak → topic → cache → agent →
output rails), so a hit still pays `self_check_input` + `topic_scope` (~1.4 s) and the
output policy check (~0.9 s). What it skips is the agent (~2.4 s). The floor for any
answered request is therefore the ~2.3 s of rails LLM time, and the cache can only ever
bring an answer down to that floor. Whether cached answers should re-run output rails at
all is a Phase 10 design question; the argument for it is that the rail prompts can
change after an answer was cached.

Telemetry note: the response's `agent_ran` flag is `true` on cache hits (92 of 92
answered requests in the cache-on run) while `/metrics` counts only 14 agent
invocations. `cached: true` is the reliable signal; `agent_ran` means "the generation
step was entered", which includes the cache lookup. Worth renaming; not changed here.

## Cost

Measured directly from the OpenRouter key's usage meter, after letting it settle
(it lags by 1–3 minutes), on the same server build:

| request type | measured cost | how |
|---|---:|---|
| full answer (input rails + agent + output rails) | **$0.0038** | 10 requests, $0.0383 |
| input-rail-blocked | **$0.0016** | 10 requests, $0.0156 |
| cache hit (input rails + output policy, no agent) | ≈ $0.0016–0.0025 | inferred, not measured separately |

A blocked request costs 41% of a full answer: the `self_check_input` prompt is long
(it carries the full false-positive guidance) and runs on every request. Abuse is not
free to serve.

Phase 9 spend, derived from those unit costs and the request counts above:

| run | requests | est. cost |
|---|---:|---:|
| safety scorecard (47 blocked, 55 full) | 102 | $0.28 |
| load, cache off (8 blocked, 58 full) | 66 | $0.24 |
| load, cache on (10 blocked, 14 full, 78 hits) | 102 | $0.19 |
| cost calibration runs | 38 | $0.09 |
| **Phase 9 total** | **308** | **≈ $0.80** |

Cumulative key usage after Phase 9: **$3.18** of a $10 limit, for the entire project
including all Phase 8 evaluation runs.

## Caveats

- n = 66 and n = 102. Percentiles above p90 are one or two requests.
- Single laptop, single worker, provider on the other side of the public internet.
  Absolute latencies will not transfer; the split between local and upstream time will.
- The cache-on run was not pre-warmed; 76% is the hit rate a fresh deploy would see on
  this mix over three minutes, and would rise in a longer window.
- Think time of 1–3 s means 3 users generate ~0.6 req/s of *offered* load; the system
  was never saturated, which is why the error rate is a statement about correctness
  under concurrency and not about capacity.
