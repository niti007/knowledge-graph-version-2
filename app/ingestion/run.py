"""Ingestion pipeline entrypoint: load -> normalize -> chunk -> embed -> Qdrant.

    python -m app.ingestion.run [--no-upsert]

Idempotent: running twice leaves the Qdrant point count unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from app.config import get_settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedding_model import get_max_seq_length
from app.ingestion.loaders import load_corpus
from app.ingestion.normalize import (
    CANONICAL_SYSTEMS,
    build_documents,
    write_documents_jsonl,
)
from app.ingestion import vector_index as vi


def _bar(n: int, scale: int) -> str:
    return "#" * max(1, round(n / scale)) if n else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the ACME corpus into Qdrant.")
    ap.add_argument("--no-upsert", action="store_true",
                    help="build documents.jsonl and chunks only; skip Qdrant")
    ap.add_argument("--allow-skips", action="store_true",
                    help="sync to Qdrant even if some source files failed to load. "
                         "Without this, unreadable files abort the run, because the "
                         "orphan purge would otherwise evict those documents from "
                         "the index on a merely transient read error.")
    args = ap.parse_args()
    s = get_settings()

    print("=" * 72)
    print("ACME corpus ingestion")
    print("=" * 72)
    print(f"raw dir     : {s.raw_dir}")
    print(f"collection  : {s.qdrant_collection}")
    print(f"embedding   : {s.embedding_model} ({s.embedding_dim}d, cosine)")
    print(f"chunking    : ~{s.chunk_size} tokens / {s.chunk_overlap} overlap")
    print()

    # 1. Load -------------------------------------------------------------
    raw = load_corpus(s.raw_dir)
    skipped: list[tuple[str, str]] = list(getattr(load_corpus, "last_skipped", []))
    print(f"[1/4] Loaded {len(raw)} source files")
    for fmt, n in sorted(Counter(r.source_format for r in raw).items()):
        print(f"        {fmt:<10} {n}")
    if skipped:
        print(f"\n      !! {len(skipped)} source file(s) FAILED to load:")
        for name, why in skipped:
            print(f"         - {name}: {why}")

    # 2. Normalize + metadata join ---------------------------------------
    docs = build_documents(raw, s.raw_dir / "doc_metadata.csv")
    write_documents_jsonl(docs, s.documents_jsonl)
    missing = [d["doc_id"] for d in docs if not d["has_metadata"]]
    print(f"\n[2/4] Normalized {len(docs)} documents -> {s.documents_jsonl}")
    print("      by doc_type:")
    for dt, n in sorted(Counter(d["doc_type"] for d in docs).items()):
        print(f"        {dt:<18} {n}")
    print("      by dept:")
    for dp, n in sorted(Counter(d["dept"] for d in docs).items()):
        print(f"        {dp:<18} {n}")
    seen_systems = Counter(sys for d in docs for sys in d["system_refs"])
    print(f"      canonical systems referenced: {len(seen_systems)}/{len(CANONICAL_SYSTEMS)}")
    for sysname in CANONICAL_SYSTEMS:
        print(f"        {sysname:<22} {seen_systems.get(sysname, 0)} docs")
    if missing:
        print(f"      WARNING: no metadata row for: {', '.join(missing)}")

    # 3. Chunk ------------------------------------------------------------
    chunks = chunk_documents(docs, s)
    lens = [c.token_count for c in chunks]
    print(f"\n[3/4] Created {len(chunks)} chunks "
          f"({len(set(c.chunk_id for c in chunks))} unique chunk_ids)")
    limit = get_max_seq_length(s)
    print(f"      tokens  min={min(lens)}  p50={int(statistics.median(lens))}  "
          f"mean={int(statistics.mean(lens))}  max={max(lens)}  "
          f"(model limit {limit}, over limit: {sum(1 for t in lens if t > limit)})")
    buckets = Counter(int(t // 50) * 50 for t in lens)
    print("      length distribution:")
    for lo in sorted(buckets):
        print(f"        {lo:>4}-{lo + 49:<4} tok  {buckets[lo]:>3}  {_bar(buckets[lo], 1)}")
    with s.chunks_jsonl.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                "heading": c.heading, "heading_path": c.heading_path,
                "section_headings": c.section_headings,
                "chunk_index": c.chunk_index, "token_count": c.token_count,
                "metadata": c.metadata,
            }, ensure_ascii=False) + "\n")
    print(f"      wrote {s.chunks_jsonl}")

    # 4. Embed + upsert ---------------------------------------------------
    if args.no_upsert:
        print("\n[4/4] --no-upsert: skipping Qdrant.")
        return 0

    if skipped and not args.allow_skips:
        print(f"\n[4/4] ABORTED: {len(skipped)} source file(s) could not be read.")
        print("      Syncing now would purge those documents from the index, so a")
        print("      transient read error would silently delete real content.")
        print("      Fix the file(s), or re-run with --allow-skips to accept the loss.")
        return 2

    with vi.qdrant_client(s) as client:
        created = vi.ensure_collection(client, s)
        before = vi.count_points(client, s)
        print(f"\n[4/4] Qdrant collection '{s.qdrant_collection}' "
              f"{'created' if created else 'exists'}; {before} points before sync")
        result = vi.sync_chunks(chunks, s, client)
        print(f"      upserted {result.upserted} points, "
              f"purged {result.purged} orphaned points")
        print(f"      collection now holds {result.after} points "
              f"for {len(chunks)} chunks (enforced equal)")
        indexes = vi.verify_payload_indexes(client, s)
        print(f"      payload indexes verified: "
              f"{', '.join(indexes) if indexes else '(local mode: n/a)'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
