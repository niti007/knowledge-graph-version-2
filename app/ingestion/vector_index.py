"""Embedding + Qdrant upsert for the document collection.

Prefix convention: BGE v1.5 is *not* an E5-style "query: " / "passage: " model.
The BAAI model card for bge-small-en-v1.5 specifies an asymmetric setup where
**passages are embedded with no prefix at all**, and short retrieval **queries**
are prefixed with "Represent this sentence for searching relevant passages: ".
Getting this backwards -- prefixing passages, or using "passage: " -- quietly
degrades recall, so both sides live in this one module and retrieval must call
`embed_query`.

Sync semantics: a point's id is a UUID5 of the chunk's content hash, so
re-ingesting unchanged content overwrites in place. Upsert alone is not enough
to call the index idempotent, though -- an edited document leaves its previous
chunks behind forever, retrievable and citable. `sync_chunks` therefore upserts
and then purges every point whose chunk_id is absent from the current set.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk
from app.ingestion.embedding_model import get_embedder  # re-exported for callers

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""  # BGE v1.5 passages take no instruction prefix.

# Namespace for deterministic point ids. Fixed constant -- changing it would
# orphan every existing point.
_POINT_NAMESPACE = uuid.UUID("6f1d5a8e-3f1b-4c9d-8a20-9c9b6a2f4e11")

PAYLOAD_INDEXES = {
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "dept": models.PayloadSchemaType.KEYWORD,
    "system_refs": models.PayloadSchemaType.KEYWORD,
    "sop_refs": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}


class PayloadIndexError(RuntimeError):
    """A payload index could not be created or verified."""


@dataclass
class SyncResult:
    upserted: int
    purged: int
    before: int
    after: int


# ---------------------------------------------------------------- embedding

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


# ------------------------------------------------------------------ client

def get_client(settings: Settings | None = None) -> QdrantClient:
    """Create a client. Prefer `qdrant_client()` so it gets closed."""
    s = settings or get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=s.qdrant_timeout)


@contextmanager
def qdrant_client(settings: Settings | None = None) -> Iterator[QdrantClient]:
    """Context-managed client that is always closed."""
    client = get_client(settings)
    try:
        yield client
    finally:
        client.close()


def point_id(chunk_id: str) -> str:
    """Deterministic point id -- the same chunk content always maps here."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


# -------------------------------------------------------------- collection

def _is_already_exists(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "already exists" in msg or "already indexed" in msg


def ensure_collection(client: QdrantClient, settings: Settings | None = None) -> bool:
    """Create the collection and payload indexes if absent, then VERIFY them.

    Payload-index creation is not blanket-excepted: swallowing every error here
    once produced a collection with no indexes at all and a clean "Done." Only
    the already-exists case is tolerated, and the result is checked against the
    collection's reported payload_schema.
    """
    s = settings or get_settings()
    created = False
    if not client.collection_exists(s.qdrant_collection):
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=models.VectorParams(
                size=s.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        created = True

    for field_name, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=s.qdrant_collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except (UnexpectedResponse, ValueError) as exc:
            if not _is_already_exists(exc):
                raise PayloadIndexError(
                    f"could not create payload index on {field_name!r}: {exc}"
                ) from exc

    verify_payload_indexes(client, s)
    return created


def verify_payload_indexes(client: QdrantClient, settings: Settings | None = None) -> list[str]:
    """Confirm every expected payload index actually exists. Returns their names.

    Qdrant's local/in-memory mode filters without materialized indexes and
    reports an empty payload_schema; there is nothing to verify there, so that
    case is skipped rather than faked.
    """
    s = settings or get_settings()
    schema = client.get_collection(s.qdrant_collection).payload_schema or {}
    if not schema and getattr(client, "_client", None).__class__.__name__ == "QdrantLocal":
        return []
    missing = [f for f in PAYLOAD_INDEXES if f not in schema]
    if missing:
        raise PayloadIndexError(
            f"payload indexes missing after creation: {', '.join(missing)}"
        )
    return list(schema)


# ------------------------------------------------------------------ payload

def chunk_payload(chunk: Chunk) -> dict:
    m = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "heading": chunk.heading,
        "heading_path": chunk.heading_path,
        "section_headings": chunk.section_headings,
        "chunk_index": chunk.chunk_index,
        "filename": m.get("filename", ""),
        "title": m.get("title", ""),
        "doc_type": m.get("doc_type", "general"),
        "dept": m.get("dept", "All"),
        "date": m.get("date", ""),
        "author": m.get("author", ""),
        "system_refs": m.get("system_refs", []),
        "sop_refs": m.get("sop_refs", []),
        "source_format": m.get("source_format", ""),
        "token_count": chunk.token_count,
    }


# ------------------------------------------------------------------- upsert

def upsert_chunks(chunks: Iterable[Chunk], settings: Settings | None = None,
                  client: QdrantClient | None = None,
                  batch_size: int | None = None) -> int:
    s = settings or get_settings()
    client = client or get_client(s)
    ensure_collection(client, s)

    chunks = list(chunks)
    size = batch_size or s.upsert_batch_size
    total = 0
    for i in range(0, len(chunks), size):
        batch = chunks[i: i + size]
        vectors = embed_passages([c.text for c in batch], s)
        client.upsert(
            collection_name=s.qdrant_collection,
            points=[
                models.PointStruct(
                    id=point_id(c.chunk_id), vector=v, payload=chunk_payload(c)
                )
                for c, v in zip(batch, vectors)
            ],
            wait=True,
        )
        total += len(batch)
    return total


def iter_point_ids(client: QdrantClient, settings: Settings | None = None):
    """Yield (point_id, chunk_id) for every point currently in the collection."""
    s = settings or get_settings()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=s.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for p in points:
            yield p.id, (p.payload or {}).get("chunk_id")
        if offset is None:
            return


def purge_orphans(current_chunk_ids: set[str], settings: Settings | None = None,
                  client: QdrantClient | None = None) -> int:
    """Delete points whose chunk_id is not in the current set.

    Without this the index only ever grows: edit a source document and its
    superseded chunk stays retrievable and citable indefinitely.
    """
    s = settings or get_settings()
    client = client or get_client(s)
    stale = [pid for pid, cid in iter_point_ids(client, s)
             if cid is None or cid not in current_chunk_ids]
    if stale:
        client.delete(
            collection_name=s.qdrant_collection,
            points_selector=models.PointIdsList(points=stale),
            wait=True,
        )
    return len(stale)


def sync_chunks(chunks: Iterable[Chunk], settings: Settings | None = None,
                client: QdrantClient | None = None) -> SyncResult:
    """Make the collection exactly match `chunks`: upsert all, purge the rest."""
    s = settings or get_settings()
    client = client or get_client(s)
    chunks = list(chunks)
    ensure_collection(client, s)
    before = count_points(client, s)
    upserted = upsert_chunks(chunks, s, client)
    purged = purge_orphans({c.chunk_id for c in chunks}, s, client)
    after = count_points(client, s)
    if after != len(chunks):
        raise RuntimeError(
            f"index out of sync: {after} points for {len(chunks)} chunks"
        )
    return SyncResult(upserted=upserted, purged=purged, before=before, after=after)


def count_points(client: QdrantClient | None = None,
                 settings: Settings | None = None) -> int:
    s = settings or get_settings()
    client = client or get_client(s)
    return client.count(s.qdrant_collection, exact=True).count
