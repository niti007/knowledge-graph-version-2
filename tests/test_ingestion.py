"""Phase 1 ingestion tests: canonicalization, chunking, metadata joining."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.ingestion.chunker import (
    Chunk,
    ChunkTooLongError,
    chunk_document,
    chunk_documents,
    enforce_model_limit,
    make_chunk_id,
    pack_sections,
    split_sections,
)
from app.ingestion.embedding_model import (
    count_tokens,
    estimate_tokens,
    get_max_seq_length,
    make_token_counter,
)
from app.ingestion.loaders import CorpusSchemaError, SKIP_FILES, load_corpus
from app.ingestion.normalize import (
    CANONICAL_SYSTEMS,
    UnknownSystemError,
    build_documents,
    canonicalize_system,
    canonicalize_systems,
    load_doc_metadata,
    normalize_text,
)

RAW = get_settings().raw_dir
META_CSV = RAW / "doc_metadata.csv"


# ---------------------------------------------------------------- canonicalize

@pytest.mark.parametrize(
    "variant,expected",
    [
        # exact canonical forms round-trip
        ("Payment-Service", "Payment-Service"),
        ("Auth-DB", "Auth-DB"),
        ("UserProfile-API", "UserProfile-API"),
        ("DataWarehouse", "DataWarehouse"),
        ("APIGateway", "APIGateway"),
        ("Notification-Service", "Notification-Service"),
        ("ReportingPortal", "ReportingPortal"),
        # the actual corrupted spellings present in doc_metadata.csv
        ("Auth-Db", "Auth-DB"),
        ("Apigateway", "APIGateway"),
        ("Userprofile-Api", "UserProfile-API"),
        ("Datawarehouse", "DataWarehouse"),
        # separator and casing variants seen in prose
        ("auth_db", "Auth-DB"),
        ("AUTH-DB", "Auth-DB"),
        ("auth db", "Auth-DB"),
        ("api gateway", "APIGateway"),
        ("API-Gateway", "APIGateway"),
        ("data warehouse", "DataWarehouse"),
        ("user profile api", "UserProfile-API"),
        ("reporting portal", "ReportingPortal"),
        ("payment service", "Payment-Service"),
        ("notification-service", "Notification-Service"),
        # surrounding whitespace
        ("  Auth-Db  ", "Auth-DB"),
    ],
)
def test_canonicalize_system_variants(variant, expected):
    assert canonicalize_system(variant) == expected


def test_canonicalize_is_idempotent():
    for s in CANONICAL_SYSTEMS:
        assert canonicalize_system(canonicalize_system(s)) == s


def test_canonicalize_rejects_empty():
    with pytest.raises(UnknownSystemError):
        canonicalize_system("   ")


def test_canonicalize_unknown_passes_through_unless_strict():
    assert canonicalize_system("Billing-Ledger") == "Billing-Ledger"
    with pytest.raises(UnknownSystemError):
        canonicalize_system("Billing-Ledger", strict=True)


def test_metadata_canonicalization_is_strict():
    """An unlisted spelling in doc_metadata.csv must fail, not pass through:
    silent pass-through is exactly the node-fragmentation hazard."""
    with pytest.raises(UnknownSystemError):
        canonicalize_systems(["Auth-DB", "Mystery-System"], strict=True)
    assert canonicalize_systems(["Auth-DB", "Mystery-System"]) == [
        "Auth-DB", "Mystery-System"
    ]


def test_canonicalize_systems_dedupes_across_casings():
    # This is the whole point: three spellings, one system, one node.
    assert canonicalize_systems(["Auth-DB", "Auth-Db", "auth_db", ""]) == ["Auth-DB"]


def test_no_duplicate_slugs_in_corpus_metadata():
    """Every system_ref in the real corpus collapses onto the 7 canonical names."""
    meta = load_doc_metadata(META_CSV)
    seen = {s for m in meta.values() for s in m.system_refs}
    assert seen <= set(CANONICAL_SYSTEMS)
    assert seen == set(CANONICAL_SYSTEMS), "corpus should reference all 7 systems"


# ------------------------------------------------------------------- normalize

def test_normalize_text_collapses_whitespace_but_keeps_paragraphs():
    out = normalize_text("A  line\t here\n\n\n\nSecond para")
    assert out == "A line here\n\nSecond para"


def test_normalize_text_repairs_pdf_hyphenation_and_ligatures():
    assert "compliance" in normalize_text("compli-\nance")
    assert normalize_text("oﬃce") == "office"


# ------------------------------------------------------------ metadata joining

def test_build_documents_joins_metadata():
    docs = {d["doc_id"]: d for d in build_documents(load_corpus(RAW), META_CSV)}

    inc = docs["INC-201"]
    assert inc["doc_type"] == "incident_report"
    assert inc["dept"] == "Infrastructure"
    assert inc["author"] == "Marcus Lee"
    assert inc["date"] == "2026-01-15"
    assert inc["system_refs"] == ["Auth-DB", "UserProfile-API"]
    assert inc["sop_refs"] == ["SOP-01"]
    assert inc["has_metadata"] is True

    # the manual whose CSV row says "Auth-Db" must land on the canonical name
    assert docs["manual_auth_db"]["system_refs"] == ["Auth-DB"]
    assert docs["manual_apigateway"]["system_refs"] == ["APIGateway"]
    assert docs["manual_userprofile_api"]["system_refs"] == ["UserProfile-API"]
    assert docs["manual_datawarehouse"]["system_refs"] == ["DataWarehouse"]


def test_every_document_has_metadata_and_text():
    for d in build_documents(load_corpus(RAW), META_CSV):
        assert d["has_metadata"], f"{d['doc_id']} missing a doc_metadata.csv row"
        assert d["text"].strip(), f"{d['doc_id']} produced empty text"


# ---------------------------------------------------------------------- loaders

def test_loader_covers_expected_corpus_and_skips_meta_files():
    docs = load_corpus(RAW)
    names = {d.filename for d in docs}
    assert SKIP_FILES.isdisjoint(names)
    assert {"users.csv", "products.csv", "transactions.csv"} <= names
    assert len([d for d in docs if d.source_format == "pdf"]) == 4
    assert len(docs) == 25


def test_csv_documents_render_as_prose_not_raw_rows():
    users = next(d for d in load_corpus(RAW) if d.filename == "users.csv")
    assert "Priya Sharma" in users.text
    assert "Billing" in users.text
    # rendered, not dumped: no raw CSV header line
    assert "user_id,name,email" not in users.text


# ---------------------------------------------------------------------- chunker

MD_DOC = {
    "doc_id": "TEST-01",
    "title": "Test Manual",
    "doc_type": "technical_manual",
    "dept": "Infrastructure",
    "system_refs": ["Auth-DB"],
    "text": (
        "# Test Manual\n\n"
        "## Overview\n\nThe service handles logins.\n\n"
        "## Dependencies\n\nIt depends on Auth-DB.\n\n"
        "### Failover\n\nFail over to the replica.\n"
    ),
}


def test_split_sections_tracks_heading_nesting():
    secs = split_sections(MD_DOC["text"])
    paths = [s.heading_path for s in secs]
    assert ["Test Manual", "Overview"] in paths
    assert ["Test Manual", "Dependencies", "Failover"] in paths


def test_split_sections_handles_numbered_pdf_headings():
    secs = split_sections("1. Purpose\n\nThis policy applies.\n\n2. Scope\n\nCovers all.\n")
    headings = [s.heading for s in secs]
    assert headings == ["1. Purpose", "2. Scope"]


def test_chunks_retain_heading_text_in_body():
    chunks = chunk_document(MD_DOC)
    joined = "\n".join(c.text for c in chunks)
    assert "Test Manual" in joined
    assert "Dependencies" in joined
    assert "Failover" in joined
    for c in chunks:
        assert c.text.startswith("# Test Manual"), "title must lead every chunk"


def test_chunks_carry_parent_doc_metadata():
    for c in chunk_document(MD_DOC):
        assert c.metadata["doc_type"] == "technical_manual"
        assert c.metadata["dept"] == "Infrastructure"
        assert c.metadata["system_refs"] == ["Auth-DB"]
        assert c.doc_id == "TEST-01"


def test_chunk_id_is_stable_across_runs():
    a = chunk_document(MD_DOC)
    b = chunk_document(MD_DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_id_changes_with_content():
    edited = dict(MD_DOC, text=MD_DOC["text"].replace("logins", "logouts"))
    assert {c.chunk_id for c in chunk_document(MD_DOC)} != {
        c.chunk_id for c in chunk_document(edited)
    }


def test_chunk_id_is_content_derived_not_positional():
    assert make_chunk_id("D", "same text") == make_chunk_id("D", "same text")
    assert make_chunk_id("D", "same text") != make_chunk_id("E", "same text")


def test_chunk_ids_unique_across_real_corpus():
    chunks = []
    for d in build_documents(load_corpus(RAW), META_CSV):
        chunks.extend(chunk_document(d))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_document_splits_with_overlap():
    """A body far over budget must window, and consecutive windows must share text."""
    # distinct words per paragraph so overlap is detectable by set intersection
    paras = [" ".join(f"w{i}x{j}" for j in range(60)) for i in range(30)]
    doc = dict(MD_DOC, text="# Big\n\n## Section\n\n" + "\n\n".join(paras))
    small = Settings(chunk_size=300, chunk_overlap=100)
    chunks = chunk_document(doc, small)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= small.chunk_size * 1.6
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = set(a.text.split()[-80:])
        if len(tail & set(b.text.split()[:120])) > 5:
            overlaps += 1
    assert overlaps >= 1, "expected sliding-window overlap between adjacent chunks"


def test_pack_sections_merges_small_sections_but_respects_budget():
    secs = split_sections(
        "".join(f"## S{i}\n\nshort body {i}\n\n" for i in range(10))
    )
    groups = pack_sections(secs, budget=1000, tok=estimate_tokens)
    assert len(groups) == 1, "small sections should pack into one chunk"
    tight = pack_sections(secs, budget=20, tok=estimate_tokens)
    assert len(tight) > 1, "a tight budget must still split"


def test_real_corpus_chunk_sizes_are_sane():
    s = get_settings()
    for d in build_documents(load_corpus(RAW), META_CSV):
        for c in chunk_document(d, s):
            assert c.token_count <= s.chunk_size + 8, (
                f"{c.doc_id} chunk {c.chunk_index} is {c.token_count} tokens"
            )


def test_no_chunk_exceeds_model_limit():
    """THE regression guard. bge-small-en-v1.5 truncates at max_seq_length
    silently, so any chunk over the limit loses its tail from the vector while
    the pipeline still reports success. Counted with the model's own tokenizer,
    against the model's own reported limit -- never a hardcoded 512."""
    limit = get_max_seq_length()
    chunks = chunk_documents(build_documents(load_corpus(RAW), META_CSV))
    over = [(c.doc_id, c.token_count) for c in chunks if c.token_count > limit]
    assert not over, f"chunks exceed the {limit}-token model limit: {over}"


def test_chunk_token_count_matches_the_real_tokenizer():
    """The stored count must be the tokenizer's, not an estimate -- the whole
    truncation bug came from trusting a word-count heuristic."""
    for c in chunk_documents(build_documents(load_corpus(RAW), META_CSV))[:12]:
        assert c.token_count == count_tokens(c.text)


def test_enforce_model_limit_raises_loudly():
    limit = get_max_seq_length()
    oversized = Chunk(
        chunk_id="x", doc_id="D", text="word " * (limit * 2), heading="",
        heading_path=[], section_headings=[], chunk_index=0,
        token_count=limit + 1,
    )
    with pytest.raises(ChunkTooLongError):
        enforce_model_limit([oversized])
    # and the pipeline path enforces it too
    huge = {"doc_id": "D", "title": "T", "text": "alpha bravo charlie " * 4000}
    chunks = chunk_document(huge)
    assert all(c.token_count <= limit for c in chunks)


def test_word_estimator_never_undershoots_the_real_count():
    """The old estimator undershot list-structured text by 1.6x, which is how
    over-limit chunks slipped through. The fallback must err high."""
    samples = [
        "- Priya Sharma (user id U0001) is a Team Lead on the Billing team. "
        "Email: priya.sharma@acme.internal. Joined 2023-01-15. Account is active.",
        "The Payment-Service depends on Auth-DB and the APIGateway.",
        "| col | col |\n|---|---|\n| a | b |",
    ]
    for text in samples:
        assert estimate_tokens(text) >= count_tokens(text), text[:40]


def test_packed_chunk_attribution_spans_all_its_sections():
    """A chunk covering several sections must not be labelled with only the
    first, or citations point at the wrong section."""
    doc = {
        "doc_id": "P", "title": "Packed",
        "text": "## Alpha\n\nfirst body\n\n## Beta\n\nsecond body\n\n"
                "## Gamma\n\nthird body\n",
    }
    chunks = chunk_document(doc)
    packed = [c for c in chunks if len(c.section_headings) > 1]
    assert packed, "expected the small sections to pack into one chunk"
    c = packed[0]
    assert c.section_headings == ["Alpha", "Beta", "Gamma"]
    assert "Alpha" in c.heading and "Gamma" in c.heading


def test_ordered_list_steps_are_not_treated_as_headings():
    """An SOP's numbered steps are content, not sections -- treating them as
    headings shatters the procedure across chunks."""
    secs = split_sections(
        "# SOP-01\n\n## Procedure\n\n1. Page the on-call.\n2. Open a bridge.\n"
        "3. Post an update.\n"
    )
    assert [x.heading for x in secs] == ["Procedure"]
    assert "2. Open a bridge." in secs[0].body


def test_oversized_split_keeps_text_verbatim():
    """Splitting must not round-trip through the tokenizer: BGE's lowercase
    WordPiece decode returns 'david. bradley48 @ acme. int'."""
    body = "\n".join(
        f"- David Bradley (user id U00{i:02d}) email david.bradley{i}@acme.internal"
        for i in range(60)
    )
    doc = {"doc_id": "U", "title": "Directory", "text": f"## Team\n\n{body}\n"}
    joined = " ".join(c.text for c in chunk_document(doc))
    assert "david.bradley7@acme.internal" in joined
    assert "@ acme. int" not in joined
    assert "##" not in joined.replace("## Team", "")


# ------------------------------------------------------- vector index (offline)

def test_point_id_is_deterministic_uuid():
    from app.ingestion.vector_index import point_id

    assert point_id("abc") == point_id("abc")
    assert point_id("abc") != point_id("abd")
    assert len(point_id("abc")) == 36


def test_bge_prefix_convention_is_applied_by_the_real_functions():
    """Asserting the two string constants was worthless: swapping them at the
    call sites left the test green while recall degraded. This pins the actual
    behaviour of embed_passages/embed_query."""
    import numpy as np

    from app.ingestion import vector_index as vi
    from app.ingestion.embedding_model import get_embedder

    model = get_embedder()
    text = "Auth-DB credential rotation runs every 90 days."

    bare = model.encode(text, normalize_embeddings=True)
    instructed = model.encode(vi.QUERY_PREFIX + text, normalize_embeddings=True)

    # passages carry NO prefix
    assert np.allclose(vi.embed_passages([text])[0], bare, atol=1e-5)
    # queries carry the BGE instruction prefix
    assert np.allclose(vi.embed_query(text), instructed, atol=1e-5)
    # and the two sides are genuinely different, so a swap cannot pass
    assert not np.allclose(vi.embed_query(text), vi.embed_passages([text])[0], atol=1e-4)


def test_embeddings_are_unit_normalized_for_cosine():
    import numpy as np

    from app.ingestion import vector_index as vi

    v = vi.embed_passages(["a short passage", "another passage"])
    for vec in v:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4
    assert len(v[0]) == get_settings().embedding_dim


# ------------------------------------------------------ vector index (in-memory)

@pytest.fixture()
def mem_setup():
    """A real QdrantClient against an in-memory store, with its own collection."""
    from qdrant_client import QdrantClient

    from app.ingestion import vector_index as vi

    settings = Settings(qdrant_collection="test_docs", upsert_batch_size=8)
    client = QdrantClient(":memory:")
    try:
        yield client, settings, vi
    finally:
        client.close()


def _mk_chunks(texts):
    return [
        Chunk(
            chunk_id=make_chunk_id("D", t), doc_id="D", text=t, heading="H",
            heading_path=["H"], section_headings=["H"], chunk_index=i,
            token_count=count_tokens(t),
            metadata={"doc_type": "sop", "dept": "Infrastructure",
                      "system_refs": ["Auth-DB"], "sop_refs": ["SOP-01"],
                      "filename": "D.md", "title": "D"},
        )
        for i, t in enumerate(texts)
    ]


def test_ensure_collection_creates_with_correct_vector_params(mem_setup):
    client, settings, vi = mem_setup
    assert vi.ensure_collection(client, settings) is True
    assert vi.ensure_collection(client, settings) is False  # idempotent
    info = client.get_collection(settings.qdrant_collection)
    params = info.config.params.vectors
    assert params.size == settings.embedding_dim
    assert params.distance.lower() == "cosine"


def test_upsert_then_reupsert_keeps_point_count_stable(mem_setup):
    client, settings, vi = mem_setup
    chunks = _mk_chunks(["alpha body", "beta body", "gamma body"])
    vi.upsert_chunks(chunks, settings, client)
    first = vi.count_points(client, settings)
    assert first == 3
    vi.upsert_chunks(chunks, settings, client)
    assert vi.count_points(client, settings) == first, "re-ingest must not duplicate"


def test_sync_purges_orphans_from_edited_documents(mem_setup):
    client, settings, vi = mem_setup
    old = _mk_chunks(["alpha body", "beta body", "gamma body"])
    vi.sync_chunks(old, settings, client)
    assert vi.count_points(client, settings) == 3

    # the document is edited: one chunk survives, one changes, one disappears
    new = _mk_chunks(["alpha body", "beta body EDITED"])
    result = vi.sync_chunks(new, settings, client)
    assert result.purged == 2
    assert vi.count_points(client, settings) == 2
    live = {cid for _, cid in vi.iter_point_ids(client, settings)}
    assert live == {c.chunk_id for c in new}
    assert make_chunk_id("D", "gamma body") not in live


def test_purge_removes_a_foreign_orphan_point(mem_setup):
    """An injected point that belongs to no current chunk must not survive."""
    from qdrant_client import models as qm

    client, settings, vi = mem_setup
    chunks = _mk_chunks(["alpha body"])
    vi.sync_chunks(chunks, settings, client)
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[qm.PointStruct(id=vi.point_id("orphan"), vector=[0.0] * 384,
                               payload={"chunk_id": "orphan"})],
        wait=True,
    )
    assert vi.count_points(client, settings) == 2
    vi.sync_chunks(chunks, settings, client)
    assert vi.count_points(client, settings) == 1


def test_sync_enforces_count_equals_chunk_count(mem_setup):
    client, settings, vi = mem_setup
    chunks = _mk_chunks(["alpha", "beta"])
    result = vi.sync_chunks(chunks, settings, client)
    assert result.after == len(chunks)


def test_payload_is_filterable_and_carries_doc_metadata(mem_setup):
    from qdrant_client import models as qm

    client, settings, vi = mem_setup
    vi.sync_chunks(_mk_chunks(["auth db rotation", "unrelated"]), settings, client)
    hits = client.query_points(
        settings.qdrant_collection,
        query=vi.embed_query("credential rotation"),
        query_filter=qm.Filter(must=[qm.FieldCondition(
            key="system_refs", match=qm.MatchValue(value="Auth-DB"))]),
        limit=5,
    ).points
    assert hits
    payload = hits[0].payload
    for key in ("chunk_id", "doc_id", "text", "doc_type", "dept",
                "system_refs", "sop_refs", "heading", "section_headings"):
        assert key in payload
    assert payload["system_refs"] == ["Auth-DB"]


def test_payload_index_fields_are_declared():
    from app.ingestion import vector_index as vi

    assert {"doc_type", "dept", "system_refs"} <= set(vi.PAYLOAD_INDEXES)


def test_estimator_never_undershoots_anywhere_in_the_real_corpus():
    """Corpus-wide version of the guard: the fallback bound must hold on every
    line and every whole document, not just hand-picked samples."""
    docs = build_documents(load_corpus(RAW), META_CSV)
    texts = [d["text"] for d in docs]
    texts += [ln for d in docs for ln in d["text"].split("\n") if ln.strip()]
    bad = [t[:60] for t in texts if estimate_tokens(t) < count_tokens(t)]
    assert not bad, f"estimator undershot on {len(bad)} text(s): {bad[:3]}"


# ------------------------------------------------------------ loader robustness

def test_corrupt_file_is_skipped_not_fatal(tmp_path):
    """One bad file must not abort the whole ingest with a traceback."""
    (tmp_path / "good.md").write_text("# Good\n\nreal content here\n")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf at all")

    docs = load_corpus(tmp_path)
    assert [d.filename for d in docs] == ["good.md"]
    assert any(name == "broken.pdf" for name, _ in load_corpus.last_skipped)

    with pytest.raises(Exception):
        load_corpus(tmp_path, strict=True)


def test_csv_missing_column_raises_named_schema_error(tmp_path):
    """A dropped column must name itself, not surface as a bare KeyError from
    inside a groupby."""
    (tmp_path / "users.csv").write_text(
        "user_id,name,email,role,active,created_date\n"
        "U1,A,a@x.com,Lead,True,2023-01-01\n"      # no `team` column
    )
    with pytest.raises(CorpusSchemaError) as exc:
        load_corpus(tmp_path, strict=True)
    assert "team" in str(exc.value)
    assert load_corpus(tmp_path) == []  # non-strict: skipped, not fatal


def test_empty_document_is_skipped(tmp_path):
    (tmp_path / "empty.md").write_text("   \n\n")
    assert load_corpus(tmp_path) == []
