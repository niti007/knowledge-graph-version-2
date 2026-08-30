"""Shared access to the local embedding model, its tokenizer, and its hard limits.

Both the chunker and the vector index need the *same* notion of "how many
tokens is this" and "how many tokens fit". Keeping one cached model here means
a chunk sized by the chunker is measured with the exact tokenizer that will
later embed it -- the mismatch that previously let 9 chunks be silently
truncated at embedding time.
"""

from __future__ import annotations

import re

from app.config import Settings, get_settings

# Rough sub-token units: WordPiece splits on case/digit/punctuation boundaries,
# so counting those approximates it far better than counting whitespace words.
_SUBTOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]")

_MODEL_CACHE: dict[str, object] = {}


class ChunkTooLongError(ValueError):
    """A chunk exceeds the embedding model's max sequence length.

    This is fatal, never a warning: sentence-transformers truncates silently,
    so everything past the limit would vanish from the vector while the
    pipeline still reported success.
    """


def get_embedder(settings: Settings | None = None):
    """Load (once per process) the local sentence-transformers encoder."""
    s = settings or get_settings()
    if s.embedding_model not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[s.embedding_model] = SentenceTransformer(
            s.embedding_model, device=s.torch_device)
    return _MODEL_CACHE[s.embedding_model]


def get_reranker(settings: Settings | None = None):
    """Load (once per process) the BGE cross-encoder used for re-ranking.

    Cached exactly like the bi-encoder: the model is ~1.1s to construct and the
    ablation harness builds it per query otherwise.
    """
    s = settings or get_settings()
    key = f"cross::{s.reranker_model}"
    if key not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[key] = CrossEncoder(s.reranker_model, device=s.torch_device)
    return _MODEL_CACHE[key]


def get_max_seq_length(settings: Settings | None = None) -> int:
    """The model's real token limit, read from the model, never hardcoded."""
    return int(get_embedder(settings).max_seq_length)


def count_tokens(text: str, settings: Settings | None = None) -> int:
    """Exact token count, including the special tokens the model will add."""
    tokenizer = get_embedder(settings).tokenizer
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def make_token_counter(settings: Settings | None = None):
    """Return a `str -> int` counter bound to the configured model.

    Counting a not-yet-split body legitimately exceeds max_seq_length -- that is
    the whole point of measuring it -- so the tokenizer's "longer than the
    specified maximum" warning is suppressed here. Nothing is encoded; the real
    limit is enforced on finished chunks by `chunker.enforce_model_limit`.
    """
    s = settings or get_settings()
    tokenizer = get_embedder(s).tokenizer
    import logging

    def _count(text: str) -> int:
        tf_log = logging.getLogger("transformers.tokenization_utils_base")
        prev = tf_log.level
        tf_log.setLevel(logging.ERROR)
        try:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])
        finally:
            tf_log.setLevel(prev)

    return _count


def estimate_tokens(text: str, settings: Settings | None = None) -> int:
    """Model-free UPPER BOUND on the token count.

    Never used for chunk sizing -- the chunker uses the real tokenizer -- but
    where an estimate is unavoidable it must never undershoot, because
    undershooting is precisely what let over-limit chunks reach the model and be
    silently truncated. A plain words x 1.3 heuristic undershot list-structured
    text by 1.6x and URL-bearing lines by 5.75x, so this takes the max of three
    bounds: whitespace words, characters, and WordPiece-ish sub-tokens.
    Verified to never undershoot on any line or document of this corpus
    (mean overshoot 1.76x).
    """
    s = settings or get_settings()
    words = len(text.split()) * s.tokens_per_word
    chars = len(text) / 2.0
    subtokens = len(_SUBTOKEN_RE.findall(text)) * 1.4
    return int(max(words, chars, subtokens)) + 2
