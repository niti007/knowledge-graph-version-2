# Chapter 05 — Chunking and the Vector Index

## Why this / what's the need

An embedding model reads a limited window of text — for `bge-small-en-v1.5`, **512 tokens**
— and squashes it into one vector. A whole manual is far longer than that, and even if it
fit, one vector for "everything about Auth-DB" would match every Auth-DB question equally
and none of them well. So documents are cut into **chunks** small enough to embed and
focused enough to search.

This chapter contains the most instructive bug in the project. The first chunker sized
chunks with a word-count estimate. The estimate undershot by up to **5.75×** on lines
containing URLs, so nine chunks were longer than 512 real tokens. The embedding library
truncates silently, so the tails of those chunks — including the Security team roster,
INC-204's root cause, and three manual sections — **never reached the index at all**, while
the pipeline reported success. The fix is the theme of the chapter: measure with the real
tokenizer, and make the failure loud.

> 🔑 **New word — token:** the unit an LLM or embedding model actually reads. Roughly a
> word or a word-piece; `Payment-Service` is several tokens. Models have a hard maximum.

> 🔑 **New word — chunk:** a piece of a document, cut to fit the model's window while
> keeping enough context to be searchable on its own.

---

## `app/ingestion/embedding_model.py` — one model, one tokenizer

```python
def get_embedder(settings: Settings | None = None):
    """Load (once per process) the local sentence-transformers encoder."""
    s = settings or get_settings()
    if s.embedding_model not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[s.embedding_model] = SentenceTransformer(
            s.embedding_model, device=s.torch_device)
    return _MODEL_CACHE[s.embedding_model]
```

- Loaded once per process and cached in a module dict, so the chunker and the vector index
  measure and embed with the very same model object.

```python
def get_max_seq_length(settings: Settings | None = None) -> int:
    """The model's real token limit, read from the model, never hardcoded."""
    return int(get_embedder(settings).max_seq_length)


def count_tokens(text: str, settings: Settings | None = None) -> int:
    """Exact token count, including the special tokens the model will add."""
    tokenizer = get_embedder(settings).tokenizer
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])
```

- The limit (512) is read off the model object, so swapping `EMBEDDING_MODEL` in `.env`
  changes the limit automatically.
- `add_special_tokens=True` — the model prepends `[CLS]` and appends `[SEP]`; those count
  toward the 512, so the counter includes them.

The word-count estimator still exists, but its docstring now says what it is for:

```python
def estimate_tokens(text: str, settings: Settings | None = None) -> int:
    """Model-free UPPER BOUND on the token count.

    Never used for chunk sizing -- the chunker uses the real tokenizer -- but
    where an estimate is unavoidable it must never undershoot, because
    undershooting is precisely what let over-limit chunks reach the model and be
    silently truncated. A plain words x 1.3 heuristic undershot list-structured
    text by 1.6x and URL-bearing lines by 5.75x, so this takes the max of three
    bounds: whitespace words, characters, and WordPiece-ish sub-tokens.
    """
```

## `app/ingestion/chunker.py` — header-aware, tokenizer-sized

### Rule 1: headings stay inside the chunk

```python
    title_prefix = f"# {title}"
    ...
            parts = [title_prefix]
            for sec in group:
                if sec.breadcrumb:
                    parts.append(f"## {sec.breadcrumb}")
                parts.append(sec.body)
            entries = [("\n\n".join(parts).strip(), group)]
```

- Every chunk's text begins with `# <document title>` and then `## <section breadcrumb>`.
  The module docstring gives the reason: *"rotate the credential every 90 days"* is
  ambiguous alone; *"# Auth-DB Manual > ## Credential Rotation > rotate the credential…"* is
  not, and the embedding sees the difference.

### Rule 2: split on sections first

```python
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$")
```

```python
    # Numbered headings ("3. Scope") only exist in the PDFs, which carry no
    # markdown headings at all. In a markdown document the same pattern is an
    # ordered list -- an SOP's "8. Write the post-mortem" is a step, not a
    # section, and treating it as one shatters the procedure across chunks.
    allow_numbered = not any(_MD_HEADING.match(ln) for ln in lines)
```

- Markdown files use `##` headings. The policy PDFs use `3. Scope`-style numbered headings.
  But an SOP's numbered *steps* look identical to numbered headings — so numbered headings
  are only honoured in documents that have no Markdown headings at all.

### Rule 3: pack small sections, window big ones

This corpus is written in many tiny sections (a five-line "Impact", a three-line
"Resolution"). One chunk per section would give 40-token chunks whose embeddings carry
almost nothing, so consecutive small sections are packed together up to the budget:

```python
def pack_sections(sections: list[Section], budget: int,
                  tok: Callable[[str], int]) -> list[list[Section]]:
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
```

- `tok` is the real token counter. Every size decision goes through it.
- A section too large for the budget stands alone and is then windowed (a sliding window
  over paragraphs with `chunk_overlap` tokens carried between windows), so a fact on a
  boundary appears whole in at least one chunk.

The windowing helper contains one more subtle decision:

```python
def _split_oversized(text: str, size: int, tok: Callable[[str], int]) -> list[str]:
    """Split a single over-budget unit without ever leaving the source text.

    Splitting on token ids and decoding back would be exact on counts but lossy
    on text: BGE uses a lowercase WordPiece vocab, so a decode round-trip
    returns "david. bradley48 @ acme. int" and "##0043". Chunks are quoted in
    citations, so they must remain verbatim source.
    """
```

- The obvious approach — tokenize, cut the token list, decode — would produce mangled
  lowercase text. Chunks are what citations point at, so they must be the original bytes.
  The splitter cuts on lines and then words instead, checking sizes with the tokenizer.

### Rule 4: fail loudly

```python
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
```

And a misconfiguration is refused at the door:

```python
    model_limit = get_max_seq_length(s)
    if s.chunk_size > model_limit:
        raise ChunkTooLongError(
            f"chunk_size={s.chunk_size} exceeds the embedding model limit of "
            f"{model_limit} tokens; text past the limit would be silently "
            f"truncated at embedding time. Lower CHUNK_SIZE in .env."
        )
```

- Both entry points (`chunk_document`, `chunk_documents`) raise the same error. The test
  `tests/test_ingestion.py::test_no_chunk_exceeds_model_limit` runs the real corpus through
  the real tokenizer and asserts zero chunks over the limit.

### Stable ids

```python
def make_chunk_id(doc_id: str, text: str) -> str:
    """Stable id from doc + content. Identical content re-ingests to the same id,
    which is what makes the Qdrant upsert idempotent."""
    h = hashlib.sha256(f"{doc_id}::{text}".encode("utf-8")).hexdigest()
    return h[:32]
```

- The id is a hash of the content. Re-ingesting unchanged text produces the same id, so the
  upsert overwrites in place instead of duplicating. Citations show the first 8 characters
  of this hash: `SOP-01#8ab9023d`.

> 🔑 **New word — idempotent:** an operation you can run twice and get the same end state as
> running it once. `make ingest` twice leaves the same 66 points, not 132.

---

## `app/ingestion/vector_index.py` — embed and upsert

### BGE's asymmetry: prefix the query, not the passage

```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""  # BGE v1.5 passages take no instruction prefix.
```

```python
def embed_passages(texts: Sequence[str], settings: Settings | None = None,
                   batch_size: int | None = None) -> list[list[float]]:
    s = settings or get_settings()
    model = get_embedder(s)
    vecs = model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=batch_size or s.embed_batch_size,
        normalize_embeddings=True,   # cosine distance on unit vectors
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    s = settings or get_settings()
    vec = get_embedder(s).encode(
        QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
    )
    return vec.tolist()
```

- The model card for `bge-small-en-v1.5` specifies this: short *queries* get an instruction
  prefix, *passages* get none. Getting it backwards does not error — recall just quietly
  gets worse. Both functions live in one module so nobody has to remember which is which,
  and `app/retrieval/vector.py` calls `embed_query` only.
- `normalize_embeddings=True` — vectors are scaled to length 1 so cosine similarity is a
  plain dot product.

> 🔑 **New word — cosine similarity:** a measure of how aligned two vectors are, from −1 to
> 1. For normalized embeddings, 1.0 means "same meaning".

### Deterministic point ids and payload indexes

```python
def point_id(chunk_id: str) -> str:
    """Deterministic point id -- the same chunk content always maps here."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
```

```python
PAYLOAD_INDEXES = {
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "dept": models.PayloadSchemaType.KEYWORD,
    "system_refs": models.PayloadSchemaType.KEYWORD,
    "sop_refs": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}
```

- Qdrant needs UUIDs as point ids; `uuid5` derives one deterministically from the chunk id.
- Payload indexes let the vector search be filtered ("only `doc_type=policy`") efficiently.
  `ensure_collection` creates them and then **verifies** they exist — a comment records that
  swallowing errors here once produced a collection with no indexes and a clean "Done."

### Sync = upsert + purge

```python
def sync_chunks(chunks: Iterable[Chunk], settings: Settings | None = None,
                client: QdrantClient | None = None) -> SyncResult:
    """Make the collection exactly match `chunks`: upsert all, purge the rest."""
    ...
    upserted = upsert_chunks(chunks, s, client)
    purged = purge_orphans({c.chunk_id for c in chunks}, s, client)
    after = count_points(client, s)
    if after != len(chunks):
        raise RuntimeError(
            f"index out of sync: {after} points for {len(chunks)} chunks"
        )
```

- Upsert alone is not idempotent in the way that matters: edit a document and its *old*
  chunks stay in the index forever, retrievable and citable. `purge_orphans` deletes every
  point whose `chunk_id` is not in the current set, and the final assertion enforces
  `points == chunks`.

### Why an unreadable file aborts the run

In `app/ingestion/run.py`:

```python
    elif skipped and not args.allow_skips:
        print(f"\n[4/5] ABORTED: {len(skipped)} source file(s) could not be read.")
        print("      Syncing now would purge those documents from the index, so a")
        print("      transient read error would silently delete real content.")
        print("      Fix the file(s), or re-run with --allow-skips to accept the loss.")
        return 2
```

- Put the two previous facts together: the purge removes anything not in the current chunk
  set, and a skipped file contributes no chunks. So "skip the unreadable file and carry on"
  would *delete a policy document from the index* because of a transient read error. Exit
  code 2 instead.

---

## ✅ You just learned
- Why chunks are sized with the model's own tokenizer, and what silently vanished when they
  were not.
- Header-aware, section-first chunking; packing small sections; verbatim text in chunks.
- BGE's query/passage prefix asymmetry and why both functions live in one module.
- Content-hash ids, deterministic point ids, and why sync = upsert + purge + assert.
- Why a skipped source file aborts ingestion.

## ▶️ Run this now
```bash
.venv/bin/python -m app.ingestion.run --no-graph
```
Read the `[3/5]` block: 66 chunks, a token-length histogram, and `over limit: 0`. Run it a
second time and confirm `[4/5]` reports `upserted 66 points, purged 0 orphaned points` and
the collection still holds 66.

Then search:
```bash
.venv/bin/python -c "from app.retrieval.vector import search; [print(f'{h.score:.3f} {h.doc_id}  {h.text[:70]!r}') for h in search('what is the password rotation policy', 5)]"
```

## 🧠 Check yourself
1. Why did the old estimator's undershoot cause data *loss* rather than an error?
2. Why are chunks split on lines and words rather than on token ids?
3. Explain in one sentence why `--allow-skips` exists and why it is off by default.
4. What goes wrong if `embed_passages` is used for a query?

---

Next: the knowledge graph →
[06-knowledge-graph.md](06-knowledge-graph.md)
