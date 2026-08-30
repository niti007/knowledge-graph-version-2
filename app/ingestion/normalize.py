"""Text cleanup, system-name canonicalization, and metadata joining.

The corpus has one silent-corruption hazard: `system_refs` in doc_metadata.csv
uses inconsistent casing ("Auth-DB" vs "Auth-Db", "APIGateway" vs "Apigateway").
Left alone, the knowledge graph fragments into duplicate System nodes and
filtered retrieval misses documents. Everything funnels through
`canonicalize_system()`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

# The 7 canonical system names. These are the spellings written to Qdrant
# payloads and used as Neo4j System node ids.
CANONICAL_SYSTEMS: tuple[str, ...] = (
    "Payment-Service",
    "Auth-DB",
    "UserProfile-API",
    "DataWarehouse",
    "APIGateway",
    "Notification-Service",
    "ReportingPortal",
)


def _slug(name: str) -> str:
    """Casing/separator-insensitive key: 'Auth-Db' / 'auth_db' / 'AUTH DB' -> 'authdb'."""
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


# slug -> canonical. Built from the canonical list, then extended with the
# real-world aliases seen in the corpus prose and filenames.
_SYSTEM_LOOKUP: dict[str, str] = {_slug(s): s for s in CANONICAL_SYSTEMS}
_SYSTEM_LOOKUP.update(
    {
        _slug("auth db"): "Auth-DB",
        _slug("authentication-db"): "Auth-DB",
        _slug("auth database"): "Auth-DB",
        _slug("userprofile api"): "UserProfile-API",
        _slug("user-profile-api"): "UserProfile-API",
        _slug("user profile api"): "UserProfile-API",
        _slug("api-gateway"): "APIGateway",
        _slug("api gateway"): "APIGateway",
        _slug("data-warehouse"): "DataWarehouse",
        _slug("data warehouse"): "DataWarehouse",
        _slug("reporting-portal"): "ReportingPortal",
        _slug("reporting portal"): "ReportingPortal",
        _slug("notification service"): "Notification-Service",
        _slug("payment service"): "Payment-Service",
    }
)


class UnknownSystemError(ValueError):
    """Raised when a system name cannot be mapped to a canonical spelling."""


def canonicalize_system(name: str, *, strict: bool = False) -> str:
    """Map any casing/separator variant of a system name to its canonical form.

    Unknown names are returned title-cased and unchanged in shape rather than
    dropped, so new systems surface in the data instead of vanishing; pass
    strict=True to raise instead.
    """
    if name is None:
        raise UnknownSystemError("system name is None")
    cleaned = name.strip()
    if not cleaned:
        raise UnknownSystemError("system name is empty")
    key = _slug(cleaned)
    if key in _SYSTEM_LOOKUP:
        return _SYSTEM_LOOKUP[key]
    if strict:
        raise UnknownSystemError(f"unknown system name: {name!r}")
    return cleaned


def canonicalize_systems(names: Iterable[str], *, strict: bool = False) -> list[str]:
    """Canonicalize a list, de-duplicating while preserving first-seen order."""
    out: list[str] = []
    for n in names:
        if not n or not str(n).strip():
            continue
        c = canonicalize_system(str(n), strict=strict)
        if c not in out:
            out.append(c)
    return out


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

_PDF_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ", "​": "",
}


def normalize_text(text: str) -> str:
    """Strip PDF artifacts and collapse whitespace without destroying structure.

    Blank lines are meaningful (paragraph and markdown-block boundaries), so
    runs of them collapse to exactly one rather than disappearing.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _PDF_LIGATURES.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # de-hyphenate words split across PDF line breaks: "compli-\nance" -> "compliance"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # drop bare page-number lines left by pypdf
    text = re.sub(r"\n[ \t]*\d{1,3}[ \t]*\n", "\n", text)
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Metadata join
# --------------------------------------------------------------------------

@dataclass
class DocMetadata:
    filename: str
    doc_type: str = "general"
    dept: str = "All"
    date: str = ""
    author: str = ""
    system_refs: list[str] = field(default_factory=list)
    sop_refs: list[str] = field(default_factory=list)


def _split_refs(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def load_doc_metadata(metadata_csv: Path) -> dict[str, DocMetadata]:
    """Read doc_metadata.csv into filename -> DocMetadata, systems canonicalized."""
    df = pd.read_csv(metadata_csv)
    out: dict[str, DocMetadata] = {}
    for row in df.to_dict("records"):
        fn = str(row["filename"]).strip()
        out[fn] = DocMetadata(
            filename=fn,
            doc_type=str(row.get("doc_type") or "general").strip(),
            dept=str(row.get("dept") or "All").strip(),
            date=str(row.get("date") or "").strip(),
            author=str(row.get("author") or "").strip(),
            # strict: doc_metadata.csv is the authoritative index. An unmapped
            # spelling here is the exact fragmentation hazard this module exists
            # to prevent, so it must fail loudly rather than pass through.
            system_refs=canonicalize_systems(
                _split_refs(row.get("system_refs")), strict=True
            ),
            sop_refs=_split_refs(row.get("sop_refs")),
        )
    return out


def build_documents(raw_docs: list, metadata_csv: Path) -> list[dict]:
    """Join loaded RawDocuments with doc_metadata.csv into normalized doc dicts.

    A document with no metadata row still ingests, with defaults, so corpus
    additions are never silently dropped.
    """
    meta = load_doc_metadata(metadata_csv)
    documents: list[dict] = []
    for raw in raw_docs:
        m = meta.get(raw.filename, DocMetadata(filename=raw.filename))
        doc = asdict(m)
        doc["doc_id"] = Path(raw.filename).stem
        doc["source_format"] = raw.source_format
        doc["title"] = raw.title
        doc["text"] = normalize_text(raw.text)
        doc["has_metadata"] = raw.filename in meta
        documents.append(doc)
    return documents


def write_documents_jsonl(documents: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in documents:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path
