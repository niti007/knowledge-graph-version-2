"""Retrieval ablation probe: vector-only vs hybrid vs hybrid+re-rank.

Scoring is answer-containment, not document identity. A graph fact and a
document chunk can both legitimately answer "who owns Payment-Service", and a
doc-id metric would score the graph branch as a miss for returning the right
answer in the wrong shape. A probe is a hit@n when every required answer string
appears somewhere in the concatenated text of the top-n items.

    python -m evals.retrieval_probe
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from app.retrieval.hybrid import ALL_MODES, HybridRetriever, Mode


@dataclass
class Probe:
    category: str
    query: str
    answers: list[str] = field(default_factory=list)   # all must appear
    gold_docs: list[str] = field(default_factory=list)  # secondary signal
    answerable: bool = True


PROBES: list[Probe] = [
    # --- factual lookup ---
    Probe("factual", "Who owns the Payment-Service?",
          ["Billing"], ["FAQ", "manual_payment_service"]),
    Probe("factual", "What is the connection pool ceiling for Payment-Service?",
          ["MAX_CONNECTIONS", "100"], ["manual_payment_service"]),
    Probe("factual", "What is the circuit breaker threshold for Auth-DB?",
          ["CIRCUIT_BREAKER_THRESHOLD"], ["manual_auth_db"]),

    # --- policy / compliance ---
    Probe("policy", "What is the data retention policy?",
          ["retention"], ["POL-002"]),
    Probe("policy", "Which policy governs acceptable use of company systems?",
          ["Acceptable Use"], ["POL-001"]),
    Probe("policy", "What does the information security policy require?",
          ["Information Security"], ["POL-003"]),

    # --- multi-hop relational ---
    Probe("multi_hop", "Which team owns the system Payment-Service depends on?",
          ["Auth-DB", "Infrastructure"], ["manual_auth_db", "FAQ"]),
    Probe("multi_hop", "What breaks if Auth-DB goes down?",
          ["Payment-Service", "UserProfile-API"], ["FAQ"]),
    Probe("multi_hop", "Who should I contact if Payment-Service is failing "
                       "because of its authentication dependency?",
          ["Marcus Lee"], ["manual_payment_service", "FAQ"]),

    # --- people / teams ---
    Probe("people", "Who leads the Security team?", ["Natalia Voss"], ["users"]),
    Probe("people", "Who is on the Data Engineering team?",
          ["Chen Wei"], ["users"]),

    # --- incident ---
    Probe("incident", "What caused the March 2026 Payment-Service outage?",
          ["INC-204"], ["INC-204", "FAQ"]),
    Probe("incident", "Which incidents involved Auth-DB?",
          ["INC-201"], ["INC-201", "INC-204", "INC-205"]),

    # --- SOP ---
    Probe("sop", "What are the steps for incident response?",
          ["SOP-01"], ["SOP-01"]),
    Probe("sop", "How do I restore the payment service after an outage?",
          ["SOP-17"], ["SOP-17"]),

    # --- genuinely unanswerable from this corpus ---
    Probe("unanswerable", "What is ACME Corp's parental leave policy?",
          [], [], answerable=False),
    Probe("unanswerable", "Who is the CEO of ACME Corp and what is their salary?",
          [], [], answerable=False),
]


def _hit(items, answers: list[str], n: int) -> bool:
    blob = " ".join(i.text for i in items[:n]).lower()
    return all(a.lower() in blob for a in answers)


def _doc_hit(items, gold: list[str], n: int) -> bool:
    if not gold:
        return False
    seen: set[str] = set()
    for it in items[:n]:
        if it.doc_id:
            seen.add(it.doc_id)
        seen.update(it.metadata.get("evidence_docs", []) or [])
    return bool(seen & set(gold))


def main() -> int:
    answerable = [p for p in PROBES if p.answerable]
    unanswerable = [p for p in PROBES if not p.answerable]

    with HybridRetriever() as r:
        # Warm up both models and both connections; otherwise the first
        # measured query carries ~13s of bi-encoder load and ~35s of
        # cross-encoder load and every latency number is fiction.
        print("warming up models and connections ...")
        t0 = time.perf_counter()
        for m in ALL_MODES:
            r.retrieve("warm up the models", m)
        print(f"warm-up took {time.perf_counter() - t0:.1f}s\n")

        results: dict[str, dict] = {}
        per_probe: dict[str, dict[str, dict]] = {}

        for mode in ALL_MODES:
            top1 = top3 = d1 = d3 = 0
            lat: list[float] = []
            rerank_lat: list[float] = []
            per_probe[mode.value] = {}
            for p in PROBES:
                res = r.retrieve(p.query, mode)
                lat.append(res.timings_ms["total_ms"])
                if "rerank_ms" in res.timings_ms:
                    rerank_lat.append(res.timings_ms["rerank_ms"])
                rec = {
                    "top1": _hit(res.items, p.answers, 1) if p.answerable else None,
                    "top3": _hit(res.items, p.answers, 3) if p.answerable else None,
                    "doc1": _doc_hit(res.items, p.gold_docs, 1),
                    "doc3": _doc_hit(res.items, p.gold_docs, 3),
                    "graph_used": any(i.branch == "graph" for i in res.items),
                    "graph_declined": res.graph_declined,
                    "top_doc": res.items[0].citation if res.items else "-",
                    "top_score": (res.items[0].rerank_score
                                  if res.items and res.items[0].rerank_score is not None
                                  else (res.items[0].branch_score if res.items else 0.0)),
                }
                per_probe[mode.value][p.query] = rec
                if p.answerable:
                    top1 += bool(rec["top1"])
                    top3 += bool(rec["top3"])
                    d1 += bool(rec["doc1"])
                    d3 += bool(rec["doc3"])
            n = len(answerable)
            results[mode.value] = {
                "top1": top1 / n, "top3": top3 / n,
                "doc1": d1 / n, "doc3": d3 / n,
                "p50": statistics.median(lat),
                "p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)],
                "mean": statistics.mean(lat),
                "rerank_p50": statistics.median(rerank_lat) if rerank_lat else None,
            }

    # ------------------------------------------------------------- report
    n = len(answerable)
    print(f"ABLATION over {n} answerable probes "
          f"({len(unanswerable)} unanswerable reported separately)\n")
    head = (f"{'mode':<18} {'top-1':>7} {'top-3':>7} {'doc@1':>7} {'doc@3':>7} "
            f"{'p50 ms':>9} {'p95 ms':>9} {'rerank p50':>11}")
    print(head)
    print("-" * len(head))
    for mode in ALL_MODES:
        m = results[mode.value]
        print(f"{mode.value:<18} {m['top1']:>6.0%} {m['top3']:>7.0%} "
              f"{m['doc1']:>6.0%} {m['doc3']:>7.0%} "
              f"{m['p50']:>9.1f} {m['p95']:>9.1f} "
              # A value that was never computed prints as "-", not as a
              # plausible-looking 0.0 -- the same discipline rerank_score keeps.
              f"{(f"{m['rerank_p50']:.1f}" if m['rerank_p50'] is not None else '-'):>11}")

    print("\nPer-probe top-1 by mode (answerable only):")
    print(f"{'category':<13} {'query':<52} " +
          " ".join(f"{m.value[:9]:>10}" for m in ALL_MODES))
    for p in answerable:
        cells = " ".join(
            f"{('HIT' if per_probe[m.value][p.query]['top1'] else 'miss'):>10}"
            for m in ALL_MODES)
        print(f"{p.category:<13} {p.query[:52]:<52} {cells}")

    print("\nGraph branch usage (hybrid_rerank):")
    for p in PROBES:
        rec = per_probe[Mode.HYBRID_RERANK.value][p.query]
        state = ("declined" if rec["graph_declined"]
                 else ("in top-5" if rec["graph_used"] else "built, not ranked"))
        print(f"   {p.query[:56]:<56} {state}")

    print("\nUnanswerable probes (no abstention mechanism in retrieval yet):")
    for p in unanswerable:
        for mode in ALL_MODES:
            rec = per_probe[mode.value][p.query]
            print(f"   [{mode.value:<16}] {p.query[:44]:<44} "
                  f"top={rec['top_doc'][:34]:<34} score={rec['top_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
