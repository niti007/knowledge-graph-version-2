"""Request and response contracts for the HTTP layer.

Validation lives here so that a malformed payload is rejected by pydantic
before any handler code runs -- a bad request must cost a 422, not a traceback
and a 500. `question` therefore has both a floor and a ceiling: an empty string
is a client bug, and an unbounded one is a way to spend somebody else's tokens.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 2000


class ChatRequest(BaseModel):
    # extra="forbid": a typo'd field name ("sessionId") silently creating a new
    # session is a worse failure than a 422 that names the mistake.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class Citation(BaseModel):
    """One source. Document and graph citations are the same shape by design.

    A graph citation has `graph_template` and `graph_path` and no `doc_id`; a
    document citation is the reverse. The UI distinguishes them on exactly that,
    so `source_type` is reported but never load-bearing.
    """

    model_config = ConfigDict(extra="allow")

    citation: str
    source_type: str | None = None
    branch: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    graph_template: str | None = None
    graph_path: str | None = None
    title: str | None = None
    url: str | None = None
    rerank_score: float | None = None
    final_rank: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # Retrieval the system saw but is NOT claiming as support -- populated when
    # an answer is blocked or its citations are demoted. Kept distinct from
    # `citations` so a refusal can never look sourced.
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    trace_id: str
    # Populated in Phase 7, when Langfuse assigns the trace an addressable id.
    # The field is wired end to end now -- API contract, response model and UI
    # slot -- so turning tracing on is a value change, not a schema change.
    trace_url: str | None = None
    session_id: str
    latency_ms: float
    # Always False in Phase 6. The field exists now so the contract does not
    # change when Phase 7 puts the semantic cache behind it.
    cached: bool = False
    blocked: bool = False
    blocked_by: str | None = None
    blocked_stage: str | None = None
    # Which path answered: the static system card, or the guarded agent.
    route: Literal["system_card", "guardrails"] = "guardrails"
    rails_fired: list[str] = Field(default_factory=list)
    rails: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] | None = None
    pii_masked: bool = False
    agent_ran: bool = False
    error: str | None = None


class DependencyHealth(BaseModel):
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    # "ok" only when warmup finished AND every dependency answered. Anything
    # else is served with a 503: a green health check on a system that cannot
    # answer is worse than no health check at all.
    status: Literal["ok", "degraded", "starting"]
    ready: bool
    warmup_complete: bool
    warmup_seconds: float | None = None
    warmup_error: str | None = None
    uptime_seconds: float
    dependencies: list[DependencyHealth] = Field(default_factory=list)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    at: float
    trace_id: str | None = None
    blocked: bool = False
    blocked_by: str | None = None
    route: str | None = None
    citations: int = 0
    latency_ms: float | None = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: float
    last_seen_at: float
    expires_at: float
    turns: list[Turn] = Field(default_factory=list)
    n_turns: int = 0


class LatencyStats(BaseModel):
    count: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None
    max_ms: float | None = None


class MetricsResponse(BaseModel):
    uptime_seconds: float
    requests_total: int
    requests_by_route: dict[str, int] = Field(default_factory=dict)
    requests_by_status: dict[str, int] = Field(default_factory=dict)
    blocked_total: int = 0
    blocks_by_rail: dict[str, int] = Field(default_factory=dict)
    rails_fired: dict[str, int] = Field(default_factory=dict)
    tools_used: dict[str, int] = Field(default_factory=dict)
    cache_hits: int = 0
    latency: LatencyStats = Field(default_factory=LatencyStats)
    agent_latency: LatencyStats = Field(default_factory=LatencyStats)
    sessions_active: int = 0
    sessions_created: int = 0
    sessions_evicted: int = 0
