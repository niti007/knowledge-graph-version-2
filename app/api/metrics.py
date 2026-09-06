"""Request counters and latency percentiles, in process.

Phase 9's scorecard reads `/metrics` rather than re-running the corpus, so what
is counted here has to be the same thing the rails actually decided -- block
counts come from `GuardrailedResponse.blocked_by` and rail names from
`.fired`, never from matching refusal text.

Percentiles are computed over a bounded ring of the most recent samples. An
unbounded list would be a slow leak in a long-lived process, and a p99 over the
last 5,000 requests is the number anyone actually wants anyway. The estimator is
nearest-rank on a sorted copy: exact for the sample it holds, no interpolation
to explain, and cheap enough at this size that a streaming sketch would be
complexity spent on nothing.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, deque

MAX_SAMPLES = 5000


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


class Metrics:
    """Counters + latency ring. Guarded by a plain lock; every op is O(1)."""

    def __init__(self, max_samples: int = MAX_SAMPLES):
        self._lock = threading.Lock()
        self.requests_total = 0
        self.by_route: Counter[str] = Counter()
        self.by_status: Counter[str] = Counter()
        self.blocked_total = 0
        self.blocks_by_rail: Counter[str] = Counter()
        self.rails_fired: Counter[str] = Counter()
        self.tools_used: Counter[str] = Counter()
        self.cache_hits = 0
        self._latency: deque[float] = deque(maxlen=max_samples)
        self._agent_latency: deque[float] = deque(maxlen=max_samples)

    def record_chat(self, *, route: str, latency_ms: float,
                    agent_latency_ms: float = 0.0, blocked: bool = False,
                    blocked_by: str | None = None,
                    rails_fired: list[str] | None = None,
                    tools_used: list[str] | None = None,
                    cached: bool = False, status: str = "ok") -> None:
        with self._lock:
            self.requests_total += 1
            self.by_route[route] += 1
            self.by_status[status] += 1
            self._latency.append(float(latency_ms))
            if agent_latency_ms:
                self._agent_latency.append(float(agent_latency_ms))
            if blocked:
                self.blocked_total += 1
                # Attributed to the rail that made the call. "unknown" would be
                # a lie of omission, so an unattributed block is named as such.
                self.blocks_by_rail[blocked_by or "unattributed"] += 1
            for r in rails_fired or []:
                self.rails_fired[r] += 1
            for t in tools_used or []:
                self.tools_used[t] += 1
            if cached:
                self.cache_hits += 1

    def record_error(self, route: str = "error") -> None:
        with self._lock:
            self.requests_total += 1
            self.by_route[route] += 1
            self.by_status["error"] += 1

    def _stats(self, samples: deque[float]) -> dict:
        vals = list(samples)
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "p50_ms": round(percentile(vals, 50), 2),
            "p95_ms": round(percentile(vals, 95), 2),
            "p99_ms": round(percentile(vals, 99), 2),
            "mean_ms": round(sum(vals) / len(vals), 2),
            "max_ms": round(max(vals), 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_by_route": dict(self.by_route),
                "requests_by_status": dict(self.by_status),
                "blocked_total": self.blocked_total,
                "blocks_by_rail": dict(self.blocks_by_rail),
                "rails_fired": dict(self.rails_fired),
                "tools_used": dict(self.tools_used),
                "cache_hits": self.cache_hits,
                "latency": self._stats(self._latency),
                "agent_latency": self._stats(self._agent_latency),
            }
