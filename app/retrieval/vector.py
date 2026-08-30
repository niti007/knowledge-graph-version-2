"""Vector branch: dense retrieval over the Qdrant document collection.

The one rule that matters here is BGE's asymmetry. bge-small-en-v1.5 embeds
passages with NO prefix and queries with an instruction prefix, so this module
calls `embed_query` and never `embed_passages`. Swapping them degrades recall
silently -- nothing errors, results just get worse -- which is why
tests/test_retrieval.py pins the behaviour rather than the constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion.vector_index import embed_query, get_client


@dataclass
class VectorHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    payload: dict


def search(query: str, top_k: int | None = None, *,
           settings: Settings | None = None,
           client: QdrantClient | None = None,
           doc_type: str | None = None,
           system: str | None = None) -> list[VectorHit]:
    """Dense top-k search. Optional payload filters use the Phase 1 indexes."""
    s = settings or get_settings()
    client = client or get_client(s)
    k = top_k or s.retrieval_top_k

    conditions = []
    if doc_type:
        conditions.append(models.FieldCondition(
            key="doc_type", match=models.MatchValue(value=doc_type)))
    if system:
        conditions.append(models.FieldCondition(
            key="system_refs", match=models.MatchValue(value=system)))
    query_filter = models.Filter(must=conditions) if conditions else None

    points = client.query_points(
        collection_name=s.qdrant_collection,
        query=embed_query(query, s),   # NOT embed_passages -- see module docstring
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    ).points

    return [
        VectorHit(
            chunk_id=(p.payload or {}).get("chunk_id", str(p.id)),
            doc_id=(p.payload or {}).get("doc_id", ""),
            text=(p.payload or {}).get("text", ""),
            score=float(p.score),
            payload=dict(p.payload or {}),
        )
        for p in points
    ]
