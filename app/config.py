"""Single source of truth for configuration.

Every runtime knob is read from the environment (.env) exactly once, via a
cached `get_settings()`. No module should read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenRouter / LLM ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_fast_model: str = "openai/gpt-4o-mini"
    llm_smart_model: str = "openai/gpt-4o"

    # --- Qdrant ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "acme_docs"
    qdrant_cache_collection: str = "acme_cache"

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- App ---
    app_env: str = "local"
    log_level: str = "INFO"
    api_port: int = 8000

    # --- Retrieval / embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dim: int = 384
    # 300, not 700: bge-small-en-v1.5 truncates at 512 tokens without warning, and
    # a 700-token budget produced chunks up to 1084 real tokens whose tails never
    # reached the vector. See tests/test_ingestion.py::test_no_chunk_exceeds_model_limit.
    chunk_size: int = 300
    chunk_overlap: int = 100
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF
    # paper (Cormack et al. 2009); larger flattens the contribution of top
    # ranks, smaller makes rank 1 dominate.
    # Torch device for the local models. None lets sentence-transformers pick
    # (mps here, cuda on a GPU box, cpu on CI) -- which means the shipped
    # latency silently depends on the host: bge-reranker-base is 634ms on mps
    # and 1863ms on cpu for 20 pairs. Pin it for reproducible load numbers.
    torch_device: str | None = None
    rrf_k: int = 60
    # Cap on how many rendered fact blocks the graph branch may contribute.
    graph_max_facts: int = 6
    # Rank penalty applied to graph facts the query did not actually ask for.
    # An uncued fact is speculative context, and RRF rank 1 is the strongest
    # endorsement the fuser can give -- a guess must never land there. The
    # offset must be >= 1 for the demotion to hold; see
    # tests/test_retrieval.py::test_uncued_graph_fact_cannot_outrank_a_vector_hit
    graph_uncued_rank_offset: int = 10
    # --- Semantic cache (Phase 7) ---
    cache_enabled: bool = True
    # 0.97, NOT the 0.95 the plan proposed. Phase 7 measured bge-small on this
    # corpus's real near-miss pairs and 0.95 conflated two of them outright:
    # "What depends on DataWarehouse?" vs "What does DataWarehouse depend on?"
    # scores 0.9558, and "Which systems depend on Auth-DB?" vs "Which systems
    # does Auth-DB depend on?" scores 0.9929 -- higher than genuine paraphrases.
    # Raising the threshold alone therefore CANNOT fix argument inversion; the
    # structural guard in app/llm/cache.py does that, and this number carries
    # the remaining margin (highest guard-uncaught near-miss measured: 0.9125).
    # See tests/test_cache.py::TestNearMissSafety.
    cache_similarity_threshold: float = 0.97
    # 24h. The corpus is re-ingested by hand, so nothing invalidates an entry
    # implicitly; the TTL is the only bound on how stale a served answer can be.
    cache_ttl_seconds: int = 86400
    # How many ANN candidates the structural guard gets to reject before the
    # lookup gives up. >1 matters: the nearest neighbour may be an inverted
    # near-miss while the true paraphrase sits at rank 2.
    cache_search_limit: int = 5

    # --- Observability (Phase 7) ---
    langfuse_enabled: bool = True
    # Seconds the request path will wait on Langfuse before giving up on it.
    # Tracing is never allowed to be the reason a request fails or hangs.
    langfuse_timeout: int = 5
    langfuse_flush_at: int = 20
    langfuse_flush_interval: float = 1.0

    # --- Agent ---
    # Hard ceiling on tool-calling rounds. At the ceiling the agent is re-invoked
    # with NO tools bound, so termination is structural rather than a request the
    # model may decline: an adversarial "keep searching until you find it" cannot
    # buy a fifth round. 4 covers the deepest real chain in this corpus
    # (dependency -> owner -> lead) with one spare.
    agent_max_iterations: int = 4
    agent_temperature: float = 0.0

    # --- Client / batching knobs ---
    qdrant_timeout: int = 60
    embed_batch_size: int = 32
    upsert_batch_size: int = 64
    # Fallback only, for the no-model token estimate. The chunker uses the real
    # tokenizer for every size decision.
    tokens_per_word: float = 2.2

    # --- Paths ---
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    @property
    def documents_jsonl(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def chunks_jsonl(self) -> Path:
        return self.processed_dir / "chunks.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
