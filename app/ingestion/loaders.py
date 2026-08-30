"""Corpus loaders: PDF, Markdown, and CSV -> RawDocument.

CSVs are part of the enterprise corpus (the people directory, product
catalogue, and transaction ledger), so they are rendered into readable prose
rather than skipped -- an embedding of "user_id,name,email" rows retrieves
nothing useful, but "Priya Sharma (U0001) is a Team Lead on the Billing team"
does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

# Meta-files that describe the dataset rather than being enterprise content.
# doc_metadata.csv is consumed by normalize.py as the metadata index.
SKIP_FILES = {"manifest.json", "raw_data_explanation.md", "doc_metadata.csv"}

log = logging.getLogger(__name__)

# Columns each renderer reads. Checked up front so a schema drift fails with a
# named column rather than a bare KeyError from inside a groupby.
CSV_REQUIRED_COLUMNS = {
    "users.csv": ["user_id", "name", "email", "team", "role", "active", "created_date"],
    "products.csv": ["product_id", "name", "owner_team", "category", "price_usd"],
    "transactions.csv": ["txn_id", "date", "user_id", "product_id", "amount_usd",
                         "status", "payment_system"],
}


class CorpusSchemaError(ValueError):
    """A CSV in the corpus is missing columns its renderer needs."""


@dataclass
class RawDocument:
    filename: str
    title: str
    text: str
    source_format: str  # pdf | markdown | csv


# --------------------------------------------------------------------------

def load_pdf(path: Path) -> RawDocument:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    title = _first_heading(text) or path.stem
    return RawDocument(path.name, title, text, "pdf")


def load_markdown(path: Path) -> RawDocument:
    text = path.read_text(encoding="utf-8")
    title = _first_heading(text) or path.stem
    return RawDocument(path.name, title, text, "markdown")


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:200]
    return ""


# --------------------------------------------------------------------------
# CSV renderers -- one readable paragraph per row, grouped under headings so
# the header-aware chunker keeps context on every chunk.
# --------------------------------------------------------------------------

def _render_users(df: pd.DataFrame) -> str:
    lines = ["# ACME Employee Directory", "", f"The directory lists {len(df)} people across "
             f"{df['team'].nunique()} teams: {', '.join(sorted(df['team'].unique()))}.", ""]
    for team, grp in df.groupby("team", sort=True):
        lines.append(f"## Team: {team}")
        leads = grp[grp["role"].str.contains("Lead", case=False, na=False)]["name"].tolist()
        lines.append(
            f"The {team} team has {len(grp)} members."
            + (f" Team lead: {', '.join(leads)}." if leads else "")
        )
        for r in grp.to_dict("records"):
            status = "active" if str(r["active"]).lower() == "true" else "inactive"
            lines.append(
                f"- {r['name']} (user id {r['user_id']}) is a {r['role']} on the {r['team']} "
                f"team. Email: {r['email']}. Joined {r['created_date']}. Account is {status}."
            )
        lines.append("")
    return "\n".join(lines)


def _render_products(df: pd.DataFrame) -> str:
    lines = ["# ACME Product Catalogue", "",
             f"The catalogue contains {len(df)} products.", "", "## Products"]
    for r in df.to_dict("records"):
        lines.append(
            f"- {r['name']} (product id {r['product_id']}) is a {r['category']} product owned "
            f"by the {r['owner_team']} team, priced at ${r['price_usd']:.2f} USD."
        )
    lines.append("")
    lines.append("## Product ownership by team")
    for team, grp in df.groupby("owner_team", sort=True):
        lines.append(f"- The {team} team owns {len(grp)} products: "
                     f"{', '.join(grp['name'].tolist())}.")
    return "\n".join(lines)


def _render_transactions(df: pd.DataFrame) -> str:
    total = df["amount_usd"].sum()
    lines = [
        "# ACME Transaction Ledger Summary", "",
        f"The ledger holds {len(df)} transactions totalling ${total:,.2f} USD, dated from "
        f"{df['date'].min()} to {df['date'].max()}.", "",
        "## Transactions by status",
    ]
    for status, grp in df.groupby("status", sort=True):
        lines.append(f"- {len(grp)} transactions are {status}, worth "
                     f"${grp['amount_usd'].sum():,.2f} USD "
                     f"({len(grp) / len(df) * 100:.1f}% of all transactions).")
    lines += ["", "## Transactions by payment system"]
    for sysname, grp in df.groupby("payment_system", sort=True):
        failed = (grp["status"] == "failed").sum()
        lines.append(
            f"- {sysname} processed {len(grp)} transactions worth "
            f"${grp['amount_usd'].sum():,.2f} USD, of which {failed} failed."
        )
    lines += ["", "## Transactions by product"]
    for pid, grp in df.groupby("product_id", sort=True):
        lines.append(f"- Product {pid}: {len(grp)} transactions, "
                     f"${grp['amount_usd'].sum():,.2f} USD.")
    lines += ["", "## Monthly transaction volume"]
    months = df.assign(month=df["date"].str.slice(0, 7)).groupby("month")
    for month, grp in months:
        lines.append(f"- {month}: {len(grp)} transactions, ${grp['amount_usd'].sum():,.2f} USD.")
    return "\n".join(lines)


_CSV_RENDER_FN = {
    "users.csv": _render_users,
    "products.csv": _render_products,
    "transactions.csv": _render_transactions,
}


def load_csv(path: Path) -> RawDocument:
    df = pd.read_csv(path)
    required = CSV_REQUIRED_COLUMNS.get(path.name)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise CorpusSchemaError(
                f"{path.name} is missing required column(s): {', '.join(missing)}"
            )
    renderer = _CSV_RENDER_FN.get(path.name)
    text = renderer(df) if renderer else df.to_string(index=False)
    return RawDocument(path.name, _first_heading(text) or path.stem, text, "csv")


# --------------------------------------------------------------------------

def load_corpus(raw_dir: Path, strict: bool = False) -> list[RawDocument]:
    """Load every ingestable file in raw_dir, sorted for deterministic output.

    A single unreadable file (a corrupt PDF, a CSV missing a column) is skipped
    with a warning rather than aborting the whole ingest with a traceback --
    except under strict=True, which tests use to make such damage fatal.
    """
    loaders = {".pdf": load_pdf, ".md": load_markdown, ".csv": load_csv}
    docs: list[RawDocument] = []
    skipped: list[tuple[str, str]] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.name in SKIP_FILES or path.name.startswith("."):
            continue
        loader = loaders.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            doc = loader(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            if strict:
                raise
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))
            log.warning("skipping %s -- %s: %s", path.name, type(exc).__name__, exc)
            continue
        if not doc.text.strip():
            if strict:
                raise ValueError(f"{path.name} produced no extractable text")
            skipped.append((path.name, "no extractable text"))
            log.warning("skipping %s -- no extractable text", path.name)
            continue
        docs.append(doc)
    if skipped:
        log.warning("%d file(s) skipped during load", len(skipped))
    load_corpus.last_skipped = skipped  # type: ignore[attr-defined]
    return docs


load_corpus.last_skipped = []  # type: ignore[attr-defined]
