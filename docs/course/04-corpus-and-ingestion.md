# Chapter 04 — The Corpus and Ingestion

## Why this / what's the need

An assistant is only as good as what it has read. This chapter is about turning 28 files
of three different formats into a clean, uniform list of documents with consistent metadata
— the raw material for both the vector index (Chapter 05) and the knowledge graph
(Chapter 06).

Two things go wrong with real corpora, and both happen in this one:

1. **Names are spelled inconsistently.** The master metadata file spells the same system
   `Auth-DB` in one row and `Auth-Db` in another, `APIGateway` and `Apigateway`. Left alone,
   the graph would grow two nodes for one system and every multi-hop query would silently
   miss half the facts.
2. **Spreadsheets are not prose.** An embedding of the CSV row `U0001,Priya Sharma,...`
   retrieves nothing useful. "Priya Sharma (user id U0001) is a Team Lead on the Billing
   team" does.

> 🔑 **New word — ingestion:** the pipeline that reads raw files and turns them into the
> structured form a search system needs. Runs once (or whenever the documents change), not
> on every question.

> 🔑 **New word — canonicalization:** mapping every spelling variant of a name to one agreed
> ("canonical") spelling, so that the same thing is always the same string.

---

## The corpus (`data/raw/`)

| Group | Files | Format |
|---|---|---|
| Policies | POL-001 … POL-004 | PDF |
| Incident reports | INC-201 … INC-205 | Markdown |
| SOPs | SOP-01 … SOP-05, SOP-17, SOP-22 | Markdown |
| Technical manuals | 5 × `manual_*.md` | Markdown |
| FAQ | FAQ.md | Markdown |
| People / products / transactions | users.csv, products.csv, transactions.csv | CSV |
| Metadata index | doc_metadata.csv | CSV |

`doc_metadata.csv` is the master index: for each document it records `doc_type`, `dept`,
`date`, `author`, the systems it references (`system_refs`) and the SOPs it references
(`sop_refs`). Three files are skipped as *about* the dataset rather than *part* of it:

```python
SKIP_FILES = {"manifest.json", "raw_data_explanation.md", "doc_metadata.csv"}
```

After ingestion: **25 documents → 66 chunks → 66 Qdrant points**.

---

## `app/ingestion/loaders.py` — three formats, one shape

Everything becomes a `RawDocument`:

```python
@dataclass
class RawDocument:
    filename: str
    title: str
    text: str
    source_format: str  # pdf | markdown | csv
```

### PDF and Markdown

```python
def load_pdf(path: Path) -> RawDocument:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    title = _first_heading(text) or path.stem
    return RawDocument(path.name, title, text, "pdf")
```

- `PdfReader` (from `pypdf`) reads each page; `extract_text() or ""` guards against a page
  with no text layer returning `None`.
- Pages are joined with a blank line so paragraph structure survives.
- The title is the first non-empty line, or the filename if there is none.

### CSVs are rendered to prose

This is the decision that makes people questions answerable at all. Here is the users
renderer:

```python
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
```

- Line 1: a Markdown H1 heading. This matters in Chapter 05, where the chunker keeps
  headings *inside* every chunk so an embedding knows what document it came from.
- `df.groupby("team")` — one `## Team: X` section per team, with a headcount sentence and
  the lead named up front. "How many people are on the Billing team?" is now a sentence in
  the text, not a count someone has to compute.
- One bullet per person, as a full English sentence.

The same pattern renders `products.csv` (per product, then "the X team owns N products")
and `transactions.csv` as a **summary** — totals by status, by payment system, by product,
by month — because nobody asks a knowledge assistant for row 4,217 of a ledger; they ask
"how many transactions failed on Payment-Service?".

Before rendering, the loader checks the columns it is about to read:

```python
CSV_REQUIRED_COLUMNS = {
    "users.csv": ["user_id", "name", "email", "team", "role", "active", "created_date"],
    ...
}
```

- A missing column raises `CorpusSchemaError` naming the column, instead of a bare
  `KeyError` from inside a `groupby`.

### Loading the whole folder

```python
def load_corpus(raw_dir: Path, strict: bool = False) -> list[RawDocument]:
    loaders = {".pdf": load_pdf, ".md": load_markdown, ".csv": load_csv}
    docs: list[RawDocument] = []
    skipped: list[tuple[str, str]] = []
    for path in sorted(raw_dir.iterdir()):
        ...
        try:
            doc = loader(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            if strict:
                raise
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))
```

- `sorted(...)` — deterministic order, so two runs produce identical output.
- A corrupt file is recorded in `skipped` rather than crashing the loop. But — and this is
  the important bit — the *ingestion entrypoint* treats a non-empty `skipped` list as fatal
  by default. Chapter 05 explains why (a skipped file would be *purged* from the index).

---

## `app/ingestion/normalize.py` — one spelling per system

### The canonical list

```python
CANONICAL_SYSTEMS: tuple[str, ...] = (
    "Payment-Service",
    "Auth-DB",
    "UserProfile-API",
    "DataWarehouse",
    "APIGateway",
    "Notification-Service",
    "ReportingPortal",
)
```

These seven strings are what gets written to Qdrant payloads and used as Neo4j node ids.

### The lookup

```python
def _slug(name: str) -> str:
    """Casing/separator-insensitive key: 'Auth-Db' / 'auth_db' / 'AUTH DB' -> 'authdb'."""
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())

_SYSTEM_LOOKUP: dict[str, str] = {_slug(s): s for s in CANONICAL_SYSTEMS}
_SYSTEM_LOOKUP.update(
    {
        _slug("auth db"): "Auth-DB",
        _slug("authentication-db"): "Auth-DB",
        _slug("auth database"): "Auth-DB",
        _slug("api-gateway"): "APIGateway",
        ...
    }
)
```

- `_slug` strips everything that is not a letter or digit and lowercases: `Auth-DB`,
  `Auth-Db`, `auth_db` and `AUTH DB` all become `authdb`.
- The lookup maps slug → canonical, seeded from the canonical list and extended with the
  aliases seen in the prose ("auth database", "api gateway").

### Strict where it matters

```python
def canonicalize_system(name: str, *, strict: bool = False) -> str:
    ...
    key = _slug(cleaned)
    if key in _SYSTEM_LOOKUP:
        return _SYSTEM_LOOKUP[key]
    if strict:
        raise UnknownSystemError(f"unknown system name: {name!r}")
    return cleaned
```

And in the metadata loader:

```python
            # strict: doc_metadata.csv is the authoritative index. An unmapped
            # spelling here is the exact fragmentation hazard this module exists
            # to prevent, so it must fail loudly rather than pass through.
            system_refs=canonicalize_systems(
                _split_refs(row.get("system_refs")), strict=True
            ),
```

- The metadata CSV is authoritative, so an unknown spelling there is a **hard error**.
  Prose (Chapter 06) is not authoritative — it mentions "Clients" and "InventoryEngine" —
  so unknown names in prose are counted and skipped instead.

### Text cleanup

```python
def normalize_text(text: str) -> str:
    ...
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _PDF_LIGATURES.items():
        text = text.replace(bad, good)
    # de-hyphenate words split across PDF line breaks: "compli-\nance" -> "compliance"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # drop bare page-number lines left by pypdf
    text = re.sub(r"\n[ \t]*\d{1,3}[ \t]*\n", "\n", text)
    ...
    text = re.sub(r"\n{3,}", "\n\n", text)
```

- PDFs produce ligature characters (`ﬁ`), curly quotes, hyphenated line breaks and stray
  page numbers. Each line here removes one class of artefact. Runs of blank lines collapse
  to exactly one — not zero — because a blank line is a paragraph boundary the chunker
  needs.

### Joining metadata

```python
def build_documents(raw_docs: list, metadata_csv: Path) -> list[dict]:
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
```

- `doc_id` is the filename without extension: `INC-204`, `SOP-01`, `manual_auth_db`. This
  is the id you will see in every citation.
- A document with no metadata row still ingests with defaults and `has_metadata=False`, so
  a newly dropped-in file surfaces as a warning instead of vanishing.

The result is written to `data/processed/documents.jsonl`, one JSON object per line — the
input to both the chunker and the graph builder.

---

## ✅ You just learned
- The 28-file corpus, what `doc_metadata.csv` carries, and the 25 → 66 → 66 pipeline count.
- Why CSVs are rendered as English sentences under Markdown headings.
- The slug-based canonicalization that stops `Auth-DB`/`Auth-Db` fragmenting the graph, and
  why the metadata CSV is strict while prose is lenient.
- The PDF cleanups and why blank lines are preserved.

## ▶️ Run this now
Build the documents without touching Qdrant or Neo4j:
```bash
.venv/bin/python -m app.ingestion.run --no-upsert --no-graph
```
Then inspect one:
```bash
.venv/bin/python -c "import json; d=[json.loads(l) for l in open('data/processed/documents.jsonl')]; x=[i for i in d if i['doc_id']=='manual_auth_db'][0]; print(x['system_refs'], x['sop_refs'], x['dept']); print(x['text'][:400])"
```
Note that `system_refs` reads `['Auth-DB']` even though the CSV row says `Auth-Db`.

## 🧠 Check yourself
1. What would happen to the multi-hop question if `Auth-Db` were not canonicalized?
2. Why is `canonicalize_system` strict for `doc_metadata.csv` but lenient for prose?
3. Why does `normalize_text` collapse three blank lines to one rather than to zero?

---

Next: cutting documents into chunks and embedding them →
[05-chunking-and-vector-index.md](05-chunking-and-vector-index.md)
