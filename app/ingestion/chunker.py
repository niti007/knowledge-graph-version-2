"""Header-aware chunking, sized with the embedding model's real tokenizer.

Three rules matter for retrieval quality here:

1. The section heading is kept *inside* every chunk's text, not only in the
   payload. A chunk that reads "rotate the credential every 90 days" is
   ambiguous alone; "# Auth-DB Manual\n\n## Credential Rotation\n\nrotate the
   credential ..." is not, and the embedding sees the difference.
2. Splits happen at section boundaries first and only fall back to a sliding
   window inside a section that is too long, so a chunk rarely straddles two
   unrelated topics.
3. Every size decision uses the ACTUAL tokenizer of the configured embedding
   model. A word-count heuristic previously undershot list-structured text by
   1.6x, which pushed chunks past the model's 512-token limit; because
   sentence-transformers truncates without warning, the tails were dropped from
   the vectors while the pipeline reported success. `enforce_model_limit()`
   now makes that failure loud.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from app.config import Settings, get_settings
from app.ingestion.embedding_model import (
    ChunkTooLongError,
    estimate_tokens,
    get_embedder,
    get_max_seq_length,
    make_token_counter,
)

# "## Heading" (markdown) or "3. Heading" / "3.1 Heading" (PDF policy sections)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$")

__all__ = [
    "Chunk", "Section", "ChunkTooLongError", "chunk_document", "chunk_documents",
    "split_sections", "pack_sections", "make_chunk_id", "enforce_model_limit",
    "estimate_tokens",
]


@dataclass
class Section:
    heading_path: list[str]
    body: str

    @property
    def heading(self) -> str:
        return self.heading_path[-1] if self.heading_path else ""

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading: str
    heading_path: list[str]
    section_headings: list[str]
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


def split_sections(text: str) -> list[Section]:
    """Split a document into sections, tracking the nesting path of headings."""
    lines = text.split("\n")
    # Numbered headings ("3. Scope") only exist in the PDFs, which carry no
    # markdown headings at all. In a markdown document the same pattern is an
    # ordered list -- an SOP's "8. Write the post-mortem" is a step, not a
    # section, and treating it as one shatters the procedure across chunks.
    allow_numbered = not any(_MD_HEADING.match(ln) for ln in lines)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in stack], body))
        buf.clear()

    for line in lines:
        md = _MD_HEADING.match(line)
        num = None if (md or not allow_numbered) else _NUM_HEADING.match(line.strip())
        if md or num:
            flush()
            if md:
                level, heading = len(md.group(1)), md.group(2).strip()
            else:
                level = num.group(1).count(".") + 1
                heading = f"{num.group(1)}. {num.group(2).strip()}"
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()

    if not sections and text.strip():
        sections = [Section([], text.strip())]
    return sections


def _split_oversized(text: str, size: int, tok: Callable[[str], int]) -> list[str]:
    """Split a single over-budget unit without ever leaving the source text.

    Splitting on token ids and decoding back would be exact on counts but lossy
    on text: BGE uses a lowercase WordPiece vocab, so a decode round-trip
    returns "david. bradley48 @ acme. int" and "##0043". Chunks are quoted in
    citations, so they must remain verbatim source. Lines first (this corpus is
    list-structured, one record per line), then words for a single huge line.
    """
    def pack(units: list[str], joiner: str) -> list[str]:
        out: list[str] = []
        cur: list[str] = []
        for u in units:
            if cur and tok(joiner.join(cur + [u])) > size:
                out.append(joiner.join(cur))
                cur = [u]
            else:
                cur.append(u)
        if cur:
            out.append(joiner.join(cur))
        return out

    pieces: list[str] = []
    for block in pack([ln for ln in text.split("\n") if ln.strip()], "\n"):
        if tok(block) <= size:
            pieces.append(block)
        else:
            pieces.extend(pack(block.split(), " "))
    return [p for p in pieces if p.strip()] or [text]


def _window(body: str, size: int, overlap: int, tok: Callable[[str], int]) -> list[str]:
    """Sliding window over paragraphs, falling back to tokens for huge paragraphs."""
    units = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    expanded: list[str] = []
    for u in units:
        if tok(u) <= size:
            expanded.append(u)
        else:
            expanded.extend(_split_oversized(u, size, tok))

    chunks: list[str] = []
    cur: list[str] = []
    for unit in expanded:
        candidate = cur + [unit]
        if cur and tok("\n\n".join(candidate)) > size:
            chunks.append("\n\n".join(cur))
            # carry back trailing units worth ~`overlap` tokens
            carry: list[str] = []
            for prev in reversed(cur):
                if tok("\n\n".join([prev, *carry])) > overlap:
                    break
                carry.insert(0, prev)
            cur = [*carry, unit]
        else:
            cur = candidate
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [body]


def make_chunk_id(doc_id: str, text: str) -> str:
    """Stable id from doc + content. Identical content re-ingests to the same id,
    which is what makes the Qdrant upsert idempotent."""
    h = hashlib.sha256(f"{doc_id}::{text}".encode("utf-8")).hexdigest()
    return h[:32]


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _common_prefix(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []
    out: list[str] = []
    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        out.append(parts[0])
    return out


def pack_sections(sections: list[Section], budget: int,
                  tok: Callable[[str], int]) -> list[list[Section]]:
    """Group consecutive small sections into one chunk each.

    This corpus is written in many short sections (a five-line "Impact", a
    three-line "Resolution"). One chunk per section would produce 40-token
    chunks whose embeddings carry almost no signal, so consecutive sections are
    packed up to the token budget. A section too large to pack stands alone and
    is windowed by the caller.
    """
    groups: list[list[Section]] = []
    cur: list[Section] = []

    def cur_tokens(extra: Section | None = None) -> int:
        items = cur + ([extra] if extra else [])
        return sum(tok(s.breadcrumb) + tok(s.body) for s in items)

    for sec in sections:
        if tok(sec.body) > budget:
            if cur:
                groups.append(cur)
                cur = []
            groups.append([sec])
            continue
        if cur and cur_tokens(sec) > budget:
            groups.append(cur)
            cur = [sec]
        else:
            cur.append(sec)
    if cur:
        groups.append(cur)
    return groups


def enforce_model_limit(chunks: list[Chunk], settings: Settings | None = None) -> None:
    """Fail loudly if any chunk would be truncated by the embedding model."""
    limit = get_max_seq_length(settings)
    bad = [c for c in chunks if c.token_count > limit]
    if bad:
        detail = ", ".join(f"{c.doc_id}#{c.chunk_index}={c.token_count}tok" for c in bad[:8])
        raise ChunkTooLongError(
            f"{len(bad)} chunk(s) exceed the embedding model limit of {limit} tokens "
            f"and would be silently truncated: {detail}"
        )


def chunk_document(doc: dict, settings: Settings | None = None,
                   token_fn: Callable[[str], int] | None = None) -> list[Chunk]:
    """Chunk one normalized document dict (from normalize.build_documents)."""
    s = settings or get_settings()
    tok = token_fn or make_token_counter(s)
    doc_id = doc["doc_id"]
    title = doc.get("title") or doc_id
    meta = {k: v for k, v in doc.items() if k != "text"}

    sections = split_sections(doc["text"])
    # Documents usually open with an H1 restating the title; keeping both would
    # waste budget repeating the same words in every chunk.
    for sec in sections:
        if sec.heading_path and _norm_title(sec.heading_path[0]) == _norm_title(title):
            sec.heading_path = sec.heading_path[1:]

    title_prefix = f"# {title}"
    # A chunk_size above the model's context is a misconfiguration, not something
    # to quietly absorb: silently clamping here is how content got truncated
    # without anyone noticing. Both entry points must fail the same loud way.
    model_limit = get_max_seq_length(s)
    if s.chunk_size > model_limit:
        raise ChunkTooLongError(
            f"chunk_size={s.chunk_size} exceeds the embedding model limit of "
            f"{model_limit} tokens; text past the limit would be silently "
            f"truncated at embedding time. Lower CHUNK_SIZE in .env."
        )
    budget = max(64, s.chunk_size - tok(title_prefix))

    chunks: list[Chunk] = []
    for group in pack_sections(sections, budget, tok):
        if len(group) == 1 and tok(group[0].body) > budget:
            sec = group[0]
            head = f"{title_prefix} > {sec.breadcrumb}" if sec.breadcrumb else title_prefix
            pieces = _window(sec.body, max(64, budget - tok(sec.breadcrumb)),
                             s.chunk_overlap, tok)
            entries = [(f"{head}\n\n{p}".strip(), [sec]) for p in pieces]
        else:
            parts = [title_prefix]
            for sec in group:
                if sec.breadcrumb:
                    parts.append(f"## {sec.breadcrumb}")
                parts.append(sec.body)
            entries = [("\n\n".join(parts).strip(), group)]

        for text, secs in entries:
            headings = [sec.heading for sec in secs if sec.heading]
            # A packed chunk spans several sections; name the range rather than
            # only its first, so a citation points at what the chunk contains.
            if len(headings) > 1:
                heading = f"{headings[0]} … {headings[-1]}"
            else:
                heading = headings[0] if headings else ""
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, text),
                    doc_id=doc_id,
                    text=text,
                    heading=heading,
                    heading_path=_common_prefix([sec.heading_path for sec in secs]),
                    section_headings=[sec.breadcrumb for sec in secs if sec.breadcrumb],
                    chunk_index=len(chunks),
                    token_count=tok(text),
                    metadata=meta,
                )
            )
    if token_fn is None:
        enforce_model_limit(chunks, s)
    return chunks


def chunk_documents(documents: list[dict], settings: Settings | None = None) -> list[Chunk]:
    s = settings or get_settings()
    tok = make_token_counter(s)
    out: list[Chunk] = []
    for doc in documents:
        out.extend(chunk_document(doc, s, tok))
    enforce_model_limit(out, s)
    return out
