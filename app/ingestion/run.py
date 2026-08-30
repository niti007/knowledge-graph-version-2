"""Ingestion entrypoint: load -> normalize -> chunk -> embed -> Qdrant -> Neo4j.

    python -m app.ingestion.run [--no-upsert] [--no-graph] [--reset-graph]

Idempotent end to end: running twice leaves both the Qdrant point count and the
Neo4j node/edge counts unchanged.
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
from app.ingestion.graph_builder import build_graph


def _bar(n: int, scale: int) -> str:
    return "#" * max(1, round(n / scale)) if n else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the ACME corpus into Qdrant.")
    ap.add_argument("--no-graph", action="store_true",
                    help="skip the Neo4j knowledge graph build")
    ap.add_argument("--reset-graph", action="store_true",
                    help="delete all graph nodes before rebuilding (clean teardown)")
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
    print(f"graph       : {s.neo4j_uri}")
    print()

    # 1. Load -------------------------------------------------------------
    raw = load_corpus(s.raw_dir)
    skipped: list[tuple[str, str]] = list(getattr(load_corpus, "last_skipped", []))
    print(f"[1/5] Loaded {len(raw)} source files")
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
    print(f"\n[2/5] Normalized {len(docs)} documents -> {s.documents_jsonl}")
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
    print(f"\n[3/5] Created {len(chunks)} chunks "
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
    # The two stores are independent: --no-upsert skips Qdrant only, and the
    # graph below still builds.
    if args.no_upsert:
        print("\n[4/5] --no-upsert: skipping Qdrant.")
    elif skipped and not args.allow_skips:
        print(f"\n[4/5] ABORTED: {len(skipped)} source file(s) could not be read.")
        print("      Syncing now would purge those documents from the index, so a")
        print("      transient read error would silently delete real content.")
        print("      Fix the file(s), or re-run with --allow-skips to accept the loss.")
        return 2
    else:
        _sync_qdrant(s, chunks)

    return _build_graph_stage(s, args)


def _sync_qdrant(s, chunks) -> None:
    with vi.qdrant_client(s) as client:
        created = vi.ensure_collection(client, s)
        before = vi.count_points(client, s)
        print(f"\n[4/5] Qdrant collection '{s.qdrant_collection}' "
              f"{'created' if created else 'exists'}; {before} points before sync")
        result = vi.sync_chunks(chunks, s, client)
        print(f"      upserted {result.upserted} points, "
              f"purged {result.purged} orphaned points")
        print(f"      collection now holds {result.after} points "
              f"for {len(chunks)} chunks (enforced equal)")
        indexes = vi.verify_payload_indexes(client, s)
        print(f"      payload indexes verified: "
              f"{', '.join(indexes) if indexes else '(local mode: n/a)'}")
def _build_graph_stage(s, args) -> int:
    # 5. Knowledge graph --------------------------------------------------
    if args.no_graph:
        print("\n[5/5] --no-graph: skipping Neo4j.")
        print("\nDone.")
        return 0

    print(f"\n[5/5] Building knowledge graph in Neo4j ({s.neo4j_uri})")
    gs = build_graph(s, reset=args.reset_graph)
    if args.reset_graph:
        print("      (--reset-graph: existing nodes deleted first)")
    print(f"      nodes: {gs.total_nodes}")
    for label, n in gs.nodes.items():
        print(f"        {label:<10} {n:>4}")
    print(f"      relationships: {gs.total_relationships}")
    for rel, n in gs.relationships.items():
        print(f"        {rel:<12} {n:>4}")

    origins = Counter(o for row in gs.dependency_rows for o in row["origins"])
    print(f"      DEPENDS_ON edges: {len(gs.dependency_rows)} loaded "
          f"(all from deterministic parsing, 0 from LLM)")
    for origin, n in sorted(origins.items()):
        print(f"        via {origin:<22} {n:>3} edge(s)")
    conf = Counter(row["confidence"] for row in gs.dependency_rows)
    print(f"        confidence: " + ", ".join(f"{k}={v}" for k, v in sorted(conf.items())))
    if gs.rejected_dependency_rows:
        print(f"      {len(gs.rejected_dependency_rows)} diagram-only edge(s) NOT "
              "loaded (the diagram contradicts each manual's own")
        print("        'downstream event emission' sentence, so an uncorroborated "
              "arrow is a reading, not a fact):")
        for row in gs.rejected_dependency_rows:
            print(f"          {row['source']} -> {row['target']}  {row['origins']}")
    if gs.self_loops:
        loops = ", ".join(f"{e.source} ({e.doc_id})" for e in gs.self_loops)
        print(f"      dropped {len(gs.self_loops)} self-loop(s) in the corpus: {loops}")
    if gs.unknown_prose_names:
        names = sorted({n for _, n in gs.unknown_prose_names})
        print(f"      non-system names skipped in prose: {', '.join(names)}")
    if gs.unmatched_departments:
        print(f"      WARNING: dept values matching no Team: "
              f"{dict(gs.unmatched_departments)}")
    if gs.orphan_documents:
        print(f"      WARNING: {len(gs.orphan_documents)} document(s) have no "
              f"graph edges: {', '.join(gs.orphan_documents)}")
    else:
        print("      every Document is reachable in the graph (no orphans)")
    if gs.systems_without_owner:
        print(f"      WARNING: no owning team in the corpus for: "
              f"{', '.join(gs.systems_without_owner)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
