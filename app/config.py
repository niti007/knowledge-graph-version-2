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
    cache_similarity_threshold: float = 0.95

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
