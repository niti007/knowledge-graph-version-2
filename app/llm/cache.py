"""Semantic cache over Qdrant, with a structural guard the cosine score cannot provide.

The plan specified "cosine >= 0.95 in collection `acme_cache`". Phase 7 measured
that and **0.95 is not safe on this corpus**, so this module ships a different
design and the evidence for it.

## What was measured

`embed_query` (bge-small-en-v1.5, the same asymmetric bi-encoder retrieval uses)
on pairs that need *different* answers:

```
0.9929  "Which systems depend on Auth-DB?"    | "Which systems does Auth-DB depend on?"
0.9920  "Can I use ACME laptops for personal email?" | "Can I use personal laptops for ACME email?"
0.9777  "Who manages Marcus Lee?"             | "Who does Marcus Lee manage?"
0.9558  "What depends on DataWarehouse?"      | "What does DataWarehouse depend on?"
```

and on pairs that mean the same thing:

```
0.9799  "Who leads the Infrastructure team?"  | "Who is the lead of the Infrastructure team?"
0.9736  "Who owns Payment-Service?"           | "Payment-Service is owned by whom?"
0.9519  "How long are backups retained?"      | "How long do we retain backups?"
```

The distributions **overlap**, and not narrowly: the worst near-miss (0.9929)
outscores every non-trivial paraphrase. No threshold separates them, so
"raise the threshold" is not a fix -- at 0.99 the cache would still serve the
Auth-DB inversion and would have stopped hitting on anything real long before.

A cross-encoder was tried as the precision gate (bge-reranker-base is already
loaded in-process for Phase 3, so it was nearly free). It is *worse*: it scores
all four inversions above at 1.000. It is a relevance model, and an inverted
question is maximally relevant to its own inversion.

## What this module does instead

The failure mode is specific and it is structural, not semantic: **the same
content words in a different argument order.** "X depends on Y" and "Y depends
on X" are the same bag of words and, to a bi-encoder trained for retrieval,
nearly the same point in space. So the guard is structural too:

    if the two questions have the SAME content-token multiset
    but a DIFFERENT content-token order  ->  refuse the hit.

`guard_verdict` implements exactly that, and it catches all four inversions
above. Cosine still has to clear `cache_similarity_threshold` (0.97) first; the
guard only removes what cosine cannot see.

**Cost, stated plainly.** The guard cannot distinguish an inversion from a
passive-voice paraphrase -- "Who owns Payment-Service?" and "Payment-Service is
owned by whom?" have the same words in a different order too, and that real
paraphrase is refused. That is the deliberate direction to fail in: a refused
hit costs one cache miss, a wrong hit serves a confidently-cited answer to the
opposite question. The honest summary is that on this corpus a *safe* semantic
cache is close to a normalization-tolerant exact cache. Section "Hit rate" in
the Phase 7 report has the numbers.

## Correctness properties

- **Blocked requests never touch the cache.** The lookup runs inside the rails'
  generation step (`Guardrails._generate`), which is only reached *after* the
  input rails pass. A jailbreak that would be blocked is blocked before this
  module is consulted, and `store()` refuses any response that a rail blocked or
  that errored. Both directions of requirement (a) are therefore structural
  rather than a check that could be forgotten.
- **The key is the masked question, not the raw one.** By the time `_generate`
  runs, Presidio has replaced any PII, and the masked text is exactly what the
  agent would have been given. Keying on it means (i) no user's PII is ever
  written to the cache collection, and (ii) the key is precisely the agent's
  input, so two questions sharing a key would have produced the same answer
  anyway.
- **No session in the key, on purpose.** `Guardrails.arun(question)` takes a
  question and nothing else; `run_agent` is called with no history, so the
  answer is a pure function of (masked question, corpus, model tier). There is
  no session-dependent state that could differ between two callers, which is
  why keying on session would only fragment the cache without removing a hazard.
  `namespace()` covers what *does* change the answer -- corpus collection,
  embedding model, and both model tiers -- so a re-tier or a re-ingest cannot
  serve a stale answer.
- **A hit carries the original provenance.** The entry stores the retrieved
  records and the citations that were actually served, and `as_agent_response`
  replays them into the ledger, so `check_grounding` and `check_citations` see
  the same evidence they saw on the miss and reach the same verdict. A cache hit
  is never a citation-free answer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1

# Namespace for deterministic point ids. Fixed -- changing it orphans the cache.
_CACHE_NAMESPACE = uuid.UUID("2b7a4d16-9c3e-4c0f-8a51-6d0f2a7e91c4")

# --------------------------------------------------------------- normalization

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")
_POSSESSIVE_RE = re.compile(r"\b(\w+)'s\b")

# Function words only. Nothing here can change which entity a question is about
# or which direction a relation runs -- those live in the content tokens, which
# is what the guard compares.
STOPWORDS: frozenset[str] = frozenset("""
a an the is are was were be been being am do does did doing done
of in on at to for from by with about into over under as
i we you he she it they me us them my our your their its
what which who whom whose when where why how
can could will would shall should may might must
there here that this these those and or but if then than
please tell show give list me about
""".split())

# Crude, deliberate, and only ever applied to both sides of a comparison: the
# guard needs "depends" and "depend" to be the same token, and needs no more
# linguistic machinery than that. A real stemmer would be a dependency and a
# source of surprises for a decision this conservative.
_SUFFIXES = ("ies", "es", "s", "ing", "ed")


def normalize(text: str) -> str:
    """Case, punctuation, possessives and whitespace folded away.

    Two questions with the same `normalize()` are the same question by any
    reading, so they take the fast path in `guard_verdict`.
    """
    t = (text or "").strip().lower()
    t = _POSSESSIVE_RE.sub(r"\1", t)
    t = t.replace("-", " ").replace("_", " ")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _stem(token: str) -> str:
    """Fold inflections hard, and only ever in the safe direction.

    Over-stemming makes two different words collide, which can only make the
    guard refuse MORE hits -- a refusal costs a cache miss. Under-stemming is
    what actually hurts: "manages" and "manage" as distinct tokens give the
    inversion pair different multisets, the guard sees no inversion, and the
    wrong answer is served. Hence the trailing-`e` strip after suffix removal,
    which is what makes manages/manage/managed one token.
    """
    if len(token) <= 3:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            token = token[: -len(suf)]
            break
    if len(token) > 3 and token[-1] in "ey":
        # `e` folds manage/manages/managed; `y` folds policy/policies and
        # apply/applies, which the "ies" rule leaves half-done otherwise.
        token = token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Stemmed non-stopword tokens, in order. Order is the point."""
    return [_stem(w) for w in normalize(text).split() if w not in STOPWORDS]


# ---------------------------------------------------------------- the guard

@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    reason: str


def guard_verdict(question: str, candidate: str) -> GuardVerdict:
    """May a cached answer for `candidate` be served for `question`?

    Three outcomes, in order:

    1. `exact_normalized` -- identical once cased/punctuated down. Always safe.
    2. `argument_order_differs` -- same content tokens, different order. This is
       the inversion signature ("what depends on X" / "what does X depend on"),
       and it is refused. It also refuses passive-voice paraphrases, which is
       the accepted cost documented in the module docstring.
    3. `distinct_wording` -- the token multisets differ, so this is not an
       inversion and the cosine threshold is left to decide.
    """
    if normalize(question) == normalize(candidate):
        return GuardVerdict(True, "exact_normalized")
    a, b = content_tokens(question), content_tokens(candidate)
    if a and b and sorted(a) == sorted(b) and a != b:
        return GuardVerdict(False, "argument_order_differs")
    return GuardVerdict(True, "distinct_wording")


# ------------------------------------------------------------------- records

@dataclass
class CacheEntry:
    """One cached turn. Everything a hit needs to be indistinguishable from a miss."""

    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    abstention: dict = field(default_factory=dict)
    grounding: dict | None = None
    stored_at: float = 0.0
    expires_at: float = 0.0
    namespace: str = ""
    source_trace_id: str | None = None

    def to_payload(self) -> dict:
        return {
            "question": self.question,
            "norm_question": normalize(self.question),
            "answer": self.answer,
            "citations": self.citations,
            "provenance": self.provenance,
            "retrieved": self.retrieved,
            "tools_used": self.tools_used,
            "abstention": self.abstention,
            "grounding": self.grounding,
            "stored_at": self.stored_at,
            "expires_at": self.expires_at,
            "namespace": self.namespace,
            "source_trace_id": self.source_trace_id,
            "schema": CACHE_SCHEMA_VERSION,
        }

    @classmethod
    def from_payload(cls, p: dict) -> "CacheEntry":
        return cls(
            question=p.get("question") or "",
            answer=p.get("answer") or "",
            citations=list(p.get("citations") or []),
            provenance=list(p.get("provenance") or []),
            retrieved=list(p.get("retrieved") or []),
            tools_used=list(p.get("tools_used") or []),
            abstention=dict(p.get("abstention") or {}),
            grounding=p.get("grounding"),
            stored_at=float(p.get("stored_at") or 0.0),
            expires_at=float(p.get("expires_at") or 0.0),
            namespace=p.get("namespace") or "",
            source_trace_id=p.get("source_trace_id"),
        )


@dataclass
class CachedAgentResponse:
    """Stands in for `AgentResponse` on a hit.

    Duck-typed rather than the real class: the output rails read `.abstention`,
    `.citations` and `.answer`, and building the real object would mean
    importing the agent module into the cache for no gain. Every field the rails
    or `_assemble` touch is present and carries the ORIGINAL run's values, which
    is what makes a hit reach the same rail verdict as the miss did.
    """

    answer: str
    citations: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    abstention: dict = field(default_factory=dict)
    iterations: int = 0
    hit_iteration_limit: bool = False
    latency_ms: float = 0.0
    messages: list = field(default_factory=list)
    from_cache: bool = True


@dataclass
class CacheLookup:
    hit: bool
    reason: str
    similarity: float | None = None
    entry: CacheEntry | None = None
    candidates: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_agent_response(self) -> CachedAgentResponse | None:
        if not self.hit or self.entry is None:
            return None
        e = self.entry
        return CachedAgentResponse(
            answer=e.answer, citations=list(e.citations),
            tools_used=list(e.tools_used), retrieved=list(e.retrieved),
            abstention=dict(e.abstention))

    def to_dict(self) -> dict:
        return {
            "hit": self.hit,
            "reason": self.reason,
            "similarity": self.similarity,
            "source_question": self.entry.question if self.entry else None,
            "stored_at": self.entry.stored_at if self.entry else None,
            "n_candidates": len(self.candidates),
            "lookup_ms": round(self.latency_ms, 2),
        }


# --------------------------------------------------------------- the cache

class SemanticCache:
    """Qdrant-backed. Every public method is failure-tolerant by contract.

    A cache that can take the request path down is worse than no cache, so
    `lookup` returns a miss and `store` returns False on *any* exception. The
    reason string keeps the failure visible in the trace instead of silent.
    """

    def __init__(self, settings: Settings | None = None, *, client: Any = None,
                 threshold: float | None = None, ttl_seconds: int | None = None,
                 enabled: bool | None = None, clock=time.time):
        self.settings = settings or get_settings()
        self.threshold = (self.settings.cache_similarity_threshold
                          if threshold is None else threshold)
        self.ttl_seconds = (self.settings.cache_ttl_seconds
                            if ttl_seconds is None else ttl_seconds)
        self.enabled = (self.settings.cache_enabled if enabled is None else enabled)
        self._clock = clock
        self._client = client
        self._owns_client = client is None
        self._ensured = False
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.errors = 0

    # -- identity ---------------------------------------------------------
    def namespace(self) -> str:
        """Everything that would change the answer for an unchanged question.

        The corpus collection, the embedder, and BOTH model tiers. Re-tiering
        from 4o to 4o-mini changes the answers; without this in the key the old
        tier's answers would keep being served under the new configuration.
        """
        s = self.settings
        raw = "|".join([
            str(CACHE_SCHEMA_VERSION), s.qdrant_collection, s.embedding_model,
            s.llm_fast_model, s.llm_smart_model,
        ])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def point_id(self, question: str) -> str:
        """Deterministic per (namespace, normalized question).

        Re-answering the same question overwrites its entry rather than adding a
        second one, so the collection cannot grow a duplicate whose stale answer
        outranks the fresh one.
        """
        return str(uuid.uuid5(_CACHE_NAMESPACE,
                              f"{self.namespace()}::{normalize(question)}"))

    # -- plumbing ---------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from app.ingestion.vector_index import get_client

            self._client = get_client(self.settings)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def ensure_collection(self) -> bool:
        """Create `acme_cache` and its filter indexes once per process."""
        from qdrant_client import models

        if self._ensured:
            return False
        s = self.settings
        created = False
        if not self.client.collection_exists(s.qdrant_cache_collection):
            self.client.create_collection(
                collection_name=s.qdrant_cache_collection,
                vectors_config=models.VectorParams(
                    size=s.embedding_dim, distance=models.Distance.COSINE),
            )
            created = True
        for field_name, schema in (("namespace", models.PayloadSchemaType.KEYWORD),
                                   ("expires_at", models.PayloadSchemaType.FLOAT)):
            try:
                self.client.create_payload_index(
                    collection_name=s.qdrant_cache_collection,
                    field_name=field_name, field_schema=schema, wait=True)
            except Exception as exc:  # noqa: BLE001
                if "already exists" not in str(exc).lower():
                    raise
        self._ensured = True
        return created

    def _embed(self, question: str) -> list[float]:
        # embed_query, NEVER embed_passages. bge-small-en-v1.5 is asymmetric:
        # queries carry the "Represent this sentence..." instruction prefix and
        # passages carry none. A cache keyed with the passage encoding would sit
        # in a different region of the space than the questions searching it.
        # tests/test_cache.py::test_cache_embeds_with_embed_query_not_passages
        # is mutation-proof against this.
        from app.ingestion.vector_index import embed_query

        return embed_query(normalize(question), self.settings)

    # -- read -------------------------------------------------------------
    def lookup(self, question: str) -> CacheLookup:
        """Find a servable cached answer, or explain why there is none."""
        t0 = time.perf_counter()
        if not self.enabled:
            return CacheLookup(False, "disabled")
        if not (question or "").strip():
            return CacheLookup(False, "empty_question")
        try:
            result = self._lookup(question)
        except Exception as exc:  # noqa: BLE001 - a cache must never 500 a request
            self.errors += 1
            log.warning("cache lookup failed: %s: %s", type(exc).__name__, exc)
            result = CacheLookup(False, f"error:{type(exc).__name__}")
        result.latency_ms = (time.perf_counter() - t0) * 1000
        if result.hit:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def _lookup(self, question: str) -> CacheLookup:
        from qdrant_client import models

        self.ensure_collection()
        now = self._clock()
        hits = self.client.query_points(
            collection_name=self.settings.qdrant_cache_collection,
            query=self._embed(question),
            limit=self.settings.cache_search_limit,
            score_threshold=self.threshold,
            with_payload=True,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="namespace",
                                      match=models.MatchValue(value=self.namespace())),
                # TTL enforced in the query, not after it: an expired entry must
                # not even be a candidate the guard could then accept.
                models.FieldCondition(key="expires_at",
                                      range=models.Range(gt=now)),
            ]),
        ).points

        candidates: list[dict] = []
        for p in hits:
            entry = CacheEntry.from_payload(p.payload or {})
            verdict = guard_verdict(question, entry.question)
            candidates.append({"question": entry.question,
                               "score": round(float(p.score), 4),
                               "guard": verdict.reason,
                               "allowed": verdict.allowed})
            if not verdict.allowed:
                continue
            if not entry.answer.strip():
                continue
            return CacheLookup(True, verdict.reason, float(p.score), entry, candidates)

        if candidates:
            return CacheLookup(False, "guard_rejected", float(hits[0].score),
                               None, candidates)
        return CacheLookup(False, "no_candidate_above_threshold", None, None, [])

    # -- write ------------------------------------------------------------
    def store(self, question: str, *, answer: str,
              citations: Sequence[dict] | None = None,
              provenance: Sequence[dict] | None = None,
              retrieved: Sequence[dict] | None = None,
              tools_used: Sequence[str] | None = None,
              abstention: dict | None = None,
              grounding: dict | None = None,
              blocked: bool = False,
              error: str | None = None,
              trace_id: str | None = None) -> bool:
        """Write one entry. Returns False -- never raises -- if it must not or cannot.

        `blocked` and `error` are parameters rather than something the caller is
        trusted to filter on, so requirement (a) is enforced at the only place
        that writes to the collection.
        """
        if not self.enabled:
            return False
        if blocked or error:
            # A rail decision is per-request. Cached, one attacker's refusal
            # would be served to a benign user asking a nearby question, and a
            # benign answer stored under a blocked turn would be served to the
            # next attacker whose input rails should have fired.
            return False
        if not (question or "").strip() or not (answer or "").strip():
            return False
        try:
            from qdrant_client import models

            self.ensure_collection()
            now = self._clock()
            entry = CacheEntry(
                question=question, answer=answer,
                citations=list(citations or []), provenance=list(provenance or []),
                retrieved=list(retrieved or []), tools_used=list(tools_used or []),
                abstention=dict(abstention or {}), grounding=grounding,
                stored_at=now, expires_at=now + self.ttl_seconds,
                namespace=self.namespace(), source_trace_id=trace_id)
            self.client.upsert(
                collection_name=self.settings.qdrant_cache_collection,
                points=[models.PointStruct(id=self.point_id(question),
                                           vector=self._embed(question),
                                           payload=entry.to_payload())],
                wait=True,
            )
            self.stores += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            log.warning("cache store failed: %s: %s", type(exc).__name__, exc)
            return False

    # -- maintenance ------------------------------------------------------
    def purge_expired(self) -> int:
        """Drop entries past their TTL. Lookups already ignore them; this reclaims space."""
        from qdrant_client import models

        self.ensure_collection()
        before = self.count()
        self.client.delete(
            collection_name=self.settings.qdrant_cache_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="expires_at",
                                      range=models.Range(lte=self._clock())),
            ])), wait=True)
        return before - self.count()

    def count(self) -> int:
        self.ensure_collection()
        return self.client.count(self.settings.qdrant_cache_collection,
                                 exact=True).count

    def clear(self) -> int:
        """Empty the collection. Used by the eval harness for cache-off runs."""
        from qdrant_client import models

        self.ensure_collection()
        n = self.count()
        self.client.delete(
            collection_name=self.settings.qdrant_cache_collection,
            points_selector=models.FilterSelector(filter=models.Filter()),
            wait=True)
        return n

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"enabled": self.enabled, "threshold": self.threshold,
                "ttl_seconds": self.ttl_seconds, "namespace": self.namespace(),
                "hits": self.hits, "misses": self.misses, "stores": self.stores,
                "errors": self.errors,
                "hit_rate": round(self.hits / total, 4) if total else None}


_DEFAULT: SemanticCache | None = None


def get_cache(settings: Settings | None = None) -> SemanticCache:
    """Process-wide instance, mirroring `get_guardrails`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SemanticCache(settings=settings)
    return _DEFAULT


def reset_cache() -> None:
    """Drop the process-wide instance. Tests only."""
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.close()
    _DEFAULT = None
