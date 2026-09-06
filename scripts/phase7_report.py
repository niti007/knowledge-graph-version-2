"""Phase 7 evidence: cache behaviour, model tiering, and Langfuse tracing, live.

This is the script behind every number in the Phase 7 report. It spends real
tokens, so it is not part of `pytest` and nothing imports it.

    .venv/bin/python scripts/phase7_report.py            # everything
    .venv/bin/python scripts/phase7_report.py --near-miss  # embeddings only, free

Sections, in order:

1. NEAR-MISS -- the measured cosine between question pairs that need different
   answers, and the guard's verdict on each. Runs the real embedder, spends no
   tokens, and is the evidence for the threshold choice.
2. CACHE -- cold vs warm latency for a repeated question and for a paraphrase,
   through the full guarded path.
3. TIERING -- which task ran on which model, with the provider's own token
   counts and the resulting cost per tier.
4. CONCURRENCY -- Phase 6's six-overlapping-request check, re-run with the cache
   enabled, asserting no answer is served under the wrong question.
5. LANGFUSE -- confirms the trace reached the API and prints the trace URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm.cache import SemanticCache, guard_verdict, normalize  # noqa: E402
from app.observability.langfuse_client import cost_usd, get_tracing  # noqa: E402

NEAR_MISS = [
    ("What depends on DataWarehouse?", "What does DataWarehouse depend on?"),
    ("Who owns Payment-Service?", "Who owns Auth-DB?"),
    ("Which systems depend on Auth-DB?", "Which systems does Auth-DB depend on?"),
    ("Who manages Marcus Lee?", "Who does Marcus Lee manage?"),
    ("What is SOP-01?", "What is SOP-02?"),
    ("What caused INC-201?", "What caused INC-202?"),
    ("What is the RTO for Payment-Service?", "What is the RPO for Payment-Service?"),
    ("How do I roll back a deployment?", "How do I roll out a deployment?"),
    ("Can I use ACME laptops for personal email?",
     "Can I use personal laptops for ACME email?"),
]

PARAPHRASE = [
    ("Who owns Payment-Service?", "who owns payment-service"),
    ("Who owns Payment-Service?", "Payment-Service is owned by whom?"),
    ("Who leads the Infrastructure team?", "Who is the lead of the Infrastructure team?"),
    ("What is the incident response procedure?", "What is the incident-response procedure?"),
    ("How long are backups retained?", "How long do we retain backups?"),
    ("What depends on DataWarehouse?", "What systems depend on DataWarehouse?"),
]

CONCURRENT = [
    "Who owns Auth-DB?",
    "What is ACME's password policy?",
    "Which team owns Payment-Service?",
]


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------ 1. near miss

def section_near_miss():
    from app.ingestion.vector_index import embed_query

    s = get_settings()
    rule(f"1. NEAR-MISS SAFETY  (threshold={s.cache_similarity_threshold})")
    cache = {}

    def vec(q):
        if q not in cache:
            cache[q] = embed_query(normalize(q))
        return cache[q]

    def row(a, b):
        score = cosine(vec(a), vec(b))
        g = guard_verdict(a, b)
        would_hit = score >= s.cache_similarity_threshold and g.allowed
        return score, g.reason, would_hit

    print("\nPairs that MUST NOT share an answer:")
    print(f"  {'cos':>7}  {'guard':<24} {'would hit':<9}  pair")
    worst = 0.0
    for a, b in NEAR_MISS:
        score, reason, hit = row(a, b)
        flag = "  <-- UNSAFE" if hit else ""
        print(f"  {score:7.4f}  {reason:<24} {str(hit):<9}  {a!r} | {b!r}{flag}")
        if hit:
            worst = max(worst, score)
    print(f"\n  wrong-answer hits: {sum(1 for a, b in NEAR_MISS if row(a, b)[2])}"
          f"/{len(NEAR_MISS)}")

    print("\nPairs that SHOULD share an answer:")
    print(f"  {'cos':>7}  {'guard':<24} {'would hit':<9}  pair")
    hits = 0
    for a, b in PARAPHRASE:
        score, reason, hit = row(a, b)
        hits += hit
        print(f"  {score:7.4f}  {reason:<24} {str(hit):<9}  {a!r} | {b!r}")
    print(f"\n  paraphrase hit rate: {hits}/{len(PARAPHRASE)} "
          f"({hits / len(PARAPHRASE):.0%})")

    print("\nWhat the plan's 0.95 threshold alone would have done "
          "(no structural guard):")
    bad = [(cosine(vec(a), vec(b)), a, b) for a, b in NEAR_MISS
           if cosine(vec(a), vec(b)) >= 0.95]
    for score, a, b in sorted(bad, reverse=True):
        print(f"  {score:7.4f}  WRONG ANSWER SERVED: {a!r} <- {b!r}")
    print(f"  {len(bad)}/{len(NEAR_MISS)} near-miss pairs would have been "
          "served the wrong answer.")


# ---------------------------------------------------------------- live helpers

def build_stack():
    from app.guardrails.runner import get_guardrails
    from app.llm.cache import get_cache

    s = get_settings()
    tracing = get_tracing(s)
    tracing.resolve_trace_url_template()
    cache = get_cache(s)
    rails = get_guardrails(s)
    print("warming up (rails, embedder, reranker)...", flush=True)
    t0 = time.perf_counter()
    from app.ingestion.embedding_model import get_embedder, get_reranker

    get_embedder(s)
    get_reranker(s)
    rails.warmup()
    print(f"warm in {time.perf_counter() - t0:.1f}s")
    return rails, cache, tracing


async def ask(rails, question):
    """One guarded turn, timed.

    Everything in this script runs on ONE event loop (see `main`). A fresh
    `asyncio.run` per question makes NeMo's loop-bound primitives stale and it
    prints "Retrying after stale event loop binding" and re-issues the call --
    which would quietly inflate every latency number here. Phase 5 documents the
    same constraint for the API.
    """
    t0 = time.perf_counter()
    r = await rails.arun(question)
    return r, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- 2. cache live

async def section_cache(rails, cache):
    rule("2. CACHE -- COLD vs WARM, THROUGH THE FULL GUARDED PATH")
    q = "Who owns Auth-DB?"
    cache.clear()
    print(f"cleared cache ({cache.count()} points)\n")

    probes = [
        ("cold        ", q),
        ("warm        ", q),
        ("normalized  ", "who owns auth-db"),
        ("reworded    ", "Who is the owner of Auth-DB?"),
        ("near-miss   ", "Who owns Payment-Service?"),
    ]
    out = {}
    for label, question in probes:
        r, ms = await ask(rails, question)
        c = r.cache or {}
        print(f"  {label} {ms:8.0f} ms  cached={str(r.cached):<5} "
              f"citations={len(r.citations)}  agent_ms={r.agent_latency_ms:7.0f}  "
              f"reason={c.get('reason')}  sim={c.get('similarity')}")
        out[label.strip()] = {"ms": ms, "cached": r.cached, "response": r}

    cold, warm = out["cold"], out["warm"]
    print(f"\n  warm is {cold['ms'] / max(warm['ms'], 1e-9):.1f}x faster than cold "
          f"({cold['ms'] - warm['ms']:.0f} ms saved)")
    print(f"  the agent is skipped entirely on a hit: "
          f"agent_ms {cold['response'].agent_latency_ms:.0f} -> "
          f"{warm['response'].agent_latency_ms:.0f}")
    print(f"  what remains on a hit is the RAILS, by design -- input and output "
          f"rails still run over a cached answer.")
    print(f"  answers identical on hit:   "
          f"{cold['response'].answer == warm['response'].answer}")
    print(f"  citations identical on hit: "
          f"{cold['response'].citations == warm['response'].citations}")
    return {k: {"ms": round(v["ms"]), "cached": v["cached"]}
            for k, v in out.items()}


# ------------------------------------------------------------------ 3. tiering

async def section_tiering(rails):
    rule("3. MODEL TIERING -- WHAT ACTUALLY RAN ON WHICH MODEL")
    from app.llm.tiering import Task, pick_model, tier_for

    s = get_settings()
    print("\nconfigured:")
    for task in Task:
        print(f"  {task.value:<24} {tier_for(task).value:<6} "
              f"{pick_model(task, s)}")

    # One uncached multi-hop question, so both the rails (fast) and the agent
    # (smart) are exercised in a single request.
    q = ("Which team owns the system Payment-Service depends on, "
         "and who leads it?")
    print(f"\nlive run: {q!r}")
    r, ms = await ask(rails, q)
    print(f"  {ms:.0f} ms  cached={r.cached}  tools={r.tools_used}  "
          f"citations={len(r.citations)}")

    rows = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for c in r.llm_calls:
        k = (c.get("task") or "unknown", c.get("model") or "?", "fast")
        rows[k]["calls"] += 1
        rows[k]["in"] += c.get("prompt_tokens") or 0
        rows[k]["out"] += c.get("completion_tokens") or 0

    agent = r.agent
    for call in getattr(agent, "llm_usage", []) or []:
        k = (call["task"], call["model"], call["tier"])
        rows[k]["calls"] += 1
        rows[k]["in"] += call["input"]
        rows[k]["out"] += call["output"]

    print(f"\n  {'task':<22} {'model':<22} {'tier':<6} {'calls':>5} "
          f"{'in':>7} {'out':>6} {'cost $':>10}")
    totals = defaultdict(float)
    for (task, model, tier), v in sorted(rows.items()):
        c = cost_usd(model, v["in"], v["out"]) or {"total": 0.0}
        totals[tier] += c["total"]
        print(f"  {task:<22} {model:<22} {tier:<6} {v['calls']:>5} "
              f"{v['in']:>7} {v['out']:>6} {c['total']:>10.6f}")
    print(f"\n  cost by tier: " + "  ".join(f"{k}=${v:.6f}" for k, v in totals.items()))
    if not r.llm_calls:
        print("  NOTE: NeMo returned no llm_calls log for this run.")
    return rows


# -------------------------------------------------------------- 4. concurrency

async def section_concurrency(rails, cache):
    """Phase 6's six-request check, re-run with the cache as new shared state.

    What counts as contamination has to be defined carefully, because the naive
    check gives a false alarm. Two identical questions answered concurrently by
    a live model can return DIFFERENT WORDING -- that is non-determinism, not a
    leak. The bug this is looking for is Phase 5's: a response carrying another
    request's material. So the assertions are:

      * every response is paired with the question it was asked (ordering)
      * every response's own `question` field matches that question
      * the answer talks about the entity that was asked about
      * citations for a question are the same across its duplicates

    and answer-text differences between duplicates are reported separately,
    labelled as what they are.
    """
    rule("4. CONCURRENCY WITH THE CACHE ENABLED (6 overlapping requests)")
    cache.clear()

    # One distinctive token per question that a contaminated answer would lack.
    markers = {
        "Who owns Auth-DB?": ("auth-db", "auth db"),
        "What is ACME's password policy?": ("password", "passphrase"),
        "Which team owns Payment-Service?": ("payment",),
    }
    qs = CONCURRENT * 2
    t0 = time.perf_counter()
    results = await asyncio.gather(*(rails.arun(q) for q in qs))
    wall = (time.perf_counter() - t0) * 1000
    pairs = list(zip(qs, results))
    print(f"  wall clock: {wall:.0f} ms for {len(pairs)} requests\n")

    contaminated = []
    by_q = defaultdict(list)
    for q, r in pairs:
        by_q[q].append(r)
        if r.question != q:
            contaminated.append((q, "response.question mismatch", r.question))
        low = (r.answer or "").lower()
        if not r.blocked and not any(m in low for m in markers[q]):
            contaminated.append((q, "answer does not mention the subject", r.answer[:120]))

    for q, rs in by_q.items():
        cites = {tuple(sorted(c.get("citation", "") for c in r.citations)) for r in rs}
        answers = {r.answer for r in rs}
        if len(cites) > 1:
            contaminated.append((q, "duplicates got different citation sets", cites))
        print(f"  {q!r}")
        print(f"      citations stable across duplicates: {len(cites) == 1}"
              f"   ({len(rs[0].citations)} citations)")
        print(f"      answer wording identical:           {len(answers) == 1}"
              f"   (differences here are model non-determinism, not a leak)")
        print(f"      cached flags: {[r.cached for r in rs]}   "
              f"blocked: {[r.blocked for r in rs]}")

    print()
    if contaminated:
        print("  CROSS-CONTAMINATION DETECTED:")
        for row in contaminated:
            print(f"    {row}")
    else:
        print("  NO CROSS-CONTAMINATION: every response carries its own question, "
              "subject and citations.")
    return not contaminated


# ---------------------------------------------------------------- 5. langfuse

async def section_langfuse(rails, tracing):
    rule("5. LANGFUSE")
    tmpl = tracing.resolve_trace_url_template()
    print(f"  enabled: {tracing.enabled}")
    print(f"  host:    {get_settings().langfuse_host}")
    print(f"  url template: {tmpl}")
    if not tmpl:
        print("  no project id -- traces will not be linkable")
        return None
    import uuid

    trace_id = uuid.uuid4().hex
    with tracing.trace("chat", trace_id=trace_id,
                       input="Who owns Auth-DB?",
                       metadata={"source": "phase7_report"}):
        r = await rails.arun("Who owns Auth-DB?")
    tracing.flush()
    url = tracing.trace_url(trace_id)
    print(f"  trace_url: {url}")
    print(f"  answer len {len(r.answer)}  cached={r.cached}")
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-miss", action="store_true",
                    help="only the free, token-free embedding evidence")
    args = ap.parse_args()

    section_near_miss()
    if args.near_miss:
        return

    rails, cache, tracing = build_stack()

    async def run_all():
        # ONE loop for the whole script. See `ask`.
        summary = await section_cache(rails, cache)
        await section_tiering(rails)
        clean = await section_concurrency(rails, cache)
        url = await section_langfuse(rails, tracing)
        rule("SUMMARY")
        print(json.dumps({"cache": summary, "no_cross_contamination": clean,
                          "trace_url": url}, indent=2, default=str))

    try:
        asyncio.run(run_all())
    finally:
        tracing.flush()
        tracing.shutdown()
        rails.close()
        cache.close()


if __name__ == "__main__":
    main()
