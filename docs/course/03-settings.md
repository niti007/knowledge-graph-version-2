# Chapter 03 — Settings (`app/config.py`)

## Why this / what's the need

A system with a dozen moving parts has a dozen places that need a URL, a key, a model name
or a threshold. If each module reads the environment on its own, you end up with the same
value spelled three ways, a typo that only fails in production, and no single place to look
when a number seems wrong.

`app/config.py` is that single place. Its docstring is the rule:

```python
"""Single source of truth for configuration.

Every runtime knob is read from the environment (.env) exactly once, via a
cached `get_settings()`. No module should read os.environ directly.
"""
```

> 🔑 **New word — pydantic-settings:** a library that reads environment variables (and a
> `.env` file) into a typed Python object, converting `"300"` to the integer `300` and
> `"true"` to `True`, and failing loudly if a value cannot be converted.

---

## The `Settings` class

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
```

- `env_file=PROJECT_ROOT / ".env"` — the file from Chapter 02. `PROJECT_ROOT` is computed
  from this file's location, so the settings load correctly no matter which directory you
  run a command from.
- `extra="ignore"` — unknown variables in `.env` are ignored rather than rejected, so a
  leftover from another project does not break startup.
- `case_sensitive=False` — `OPENROUTER_API_KEY` in the environment maps to the field
  `openrouter_api_key`.

Each field has a typed default:

```python
    # --- OpenRouter / LLM ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_fast_model: str = "openai/gpt-4o-mini"
    llm_smart_model: str = "openai/gpt-4o"
```

- Two model names, not one. Chapter 11 explains *tiering*: the fast model handles the
  rails' yes/no checks; the smart one runs the agent. Callers never name a model directly —
  they name a *task* and `app/llm/tiering.py` picks the tier.

## Numbers that carry their own evidence

The interesting part of this file is that several defaults are not round numbers, and each
one says why.

### `chunk_size: 300`, not 700

```python
    # 300, not 700: bge-small-en-v1.5 truncates at 512 tokens without warning, and
    # a 700-token budget produced chunks up to 1084 real tokens whose tails never
    # reached the vector. See tests/test_ingestion.py::test_no_chunk_exceeds_model_limit.
    chunk_size: int = 300
    chunk_overlap: int = 100
```

- The plan proposed ~700-token chunks. Phase 1 discovered that the estimate used to size
  chunks undershot badly, producing chunks of up to 1084 *real* tokens, and the embedding
  model silently drops everything after 512. Chapter 05 tells the whole story.

### `cache_similarity_threshold: 0.97`, not 0.95

```python
    # 0.97, NOT the 0.95 the plan proposed. Phase 7 measured bge-small on this
    # corpus's real near-miss pairs and 0.95 conflated two of them outright:
    # "What depends on DataWarehouse?" vs "What does DataWarehouse depend on?"
    # scores 0.9558, and "Which systems depend on Auth-DB?" vs "Which systems
    # does Auth-DB depend on?" scores 0.9929 -- higher than genuine paraphrases.
    # Raising the threshold alone therefore CANNOT fix argument inversion; the
    # structural guard in app/llm/cache.py does that, and this number carries
    # the remaining margin (highest guard-uncaught near-miss measured: 0.9125).
    cache_similarity_threshold: float = 0.97
```

- Two questions with *opposite* answers scored 0.9929 similarity. No threshold can separate
  them. Chapter 11 shows the structural guard that does.

### `agent_max_iterations: 4`

```python
    # Hard ceiling on tool-calling rounds. At the ceiling the agent is re-invoked
    # with NO tools bound, so termination is structural rather than a request the
    # model may decline: an adversarial "keep searching until you find it" cannot
    # buy a fifth round. 4 covers the deepest real chain in this corpus
    # (dependency -> owner -> lead) with one spare.
    agent_max_iterations: int = 4
```

- The deepest real question needs three hops; four gives one spare. How the bound is
  *enforced* (rather than politely requested) is in Chapter 08.

### `graph_uncued_rank_offset: 10` and `rrf_k: 60`

```python
    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF
    # paper (Cormack et al. 2009); larger flattens the contribution of top
    # ranks, smaller makes rank 1 dominate.
    ...
    rrf_k: int = 60
    # Cap on how many rendered fact blocks the graph branch may contribute.
    graph_max_facts: int = 6
    # Rank penalty applied to graph facts the query did not actually ask for.
    # An uncued fact is speculative context, and RRF rank 1 is the strongest
    # endorsement the fuser can give -- a guess must never land there.
    graph_uncued_rank_offset: int = 10
```

- Chapter 07 covers both. The point here is the pattern: a magic number with a reference
  or a measurement next to it.

### `torch_device: None`

```python
    # Torch device for the local models. None lets sentence-transformers pick
    # (mps here, cuda on a GPU box, cpu on CI) -- which means the shipped
    # latency silently depends on the host: bge-reranker-base is 634ms on mps
    # and 1863ms on cpu for 20 pairs. Pin it for reproducible load numbers.
    torch_device: str | None = None
```

- On an Apple laptop the re-ranker uses the GPU ("mps"); in the Docker image it is pinned
  to `cpu`. The three-fold difference is why load-test numbers state their hardware.

## The singleton

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- `@lru_cache(maxsize=1)` means the `.env` file is parsed once per process. Every module
  calls `get_settings()` and gets the same object.
- Almost every function in the codebase also accepts an optional `settings` argument
  (`def chunk_document(doc, settings: Settings | None = None, ...)`). That is the injection
  seam tests use: pass a `Settings(chunk_size=64)` and the function obeys it without touching
  the environment.

## Derived paths

```python
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    @property
    def documents_jsonl(self) -> Path:
        return self.processed_dir / "documents.jsonl"
```

- The corpus lives in `data/raw/`; ingestion writes `documents.jsonl` and `chunks.jsonl` to
  `data/processed/`, which is gitignored because it is regenerated by `make ingest`.

---

## ✅ You just learned
- All configuration flows through one cached `Settings` object; no module reads
  `os.environ` directly.
- Several defaults (300, 0.97, 4, 10) are measured values, and the file records the
  measurement next to the number.
- The optional `settings` parameter everywhere is how tests inject configuration.

## ▶️ Run this now
```bash
.venv/bin/python -c "from app.config import get_settings; s=get_settings(); print(s.chunk_size, s.cache_similarity_threshold, s.llm_smart_model, s.neo4j_uri)"
```
Expected: `300 0.97 openai/gpt-4o bolt://localhost:7688`.

## 🧠 Check yourself
1. Why was `chunk_size` lowered from the planned 700 to 300?
2. What is the danger in leaving `torch_device` unset when you publish latency numbers?
3. Why do functions take `settings: Settings | None = None` when `get_settings()` already
   exists?

---

Next: loading the 28 files →
[04-corpus-and-ingestion.md](04-corpus-and-ingestion.md)
