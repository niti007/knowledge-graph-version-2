# Chapter 02 — Project Setup and the Phase 0 Gate

## Why this / what's the need

Most projects discover a broken credential three hours in, buried under a stack trace from
some unrelated library. This project refuses to start building until every external service
has been contacted and has answered. That is "Phase 0": a script, `scripts/check_env.py`,
that makes a **real network call** to each of the five services and prints a pass/fail table.
Nothing in the plan proceeds until it is all green.

> 🔑 **New word — gate:** a check that must pass before the next step is allowed to begin.

---

## Step 1 — clone and install

```bash
git clone <the repository url> genai-capstone
cd genai-capstone
make install
```

`make install` creates `.venv/` with Python 3.11 and installs everything in
`pyproject.toml`. The dependency list is **pinned, not floored** — every version is exact.
The comment at the top of that file explains why:

```toml
# Pinned, not floored. The declared floors (langgraph>=0.2.28,
# langchain-openai>=0.2) resolved to langgraph 1.2 / langchain-openai 1.6 --
# two majors up, across the 0.x-to-1.x API break. A floor that wide does not
# describe a reproducible environment, and the grader who clones this in six
# months would get something else again.
```

Two pins carry a story you will meet again:

- `presidio-analyzer==2.2.364` — the PII rail's threshold is 0.40 and an un-cued phone
  number scores *exactly* 0.40. A version bump that nudges the score down would switch the
  rail off silently. A test pins the boundary (Chapter 09).
- `langfuse==4.15.1` — the earlier range `>=2.50,<3` would have installed the v2 SDK, whose
  API the code does not use. It would not have degraded; it would have failed to import.

## Step 2 — create `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in the five credentials from Chapter 01. Everything else already has a
working default. Two things to notice in the template:

```dotenv
# 0.97, not 0.95: Phase 7 measured bge-small conflating near-miss pairs with
# opposite answers at 0.95 (see app/llm/cache.py). The structural guard in that
# module does the rest of the work; this number carries the margin.
CACHE_SIMILARITY_THRESHOLD=0.97
```

Even the config template records *why* a number is what it is. You'll see where 0.97 came
from in Chapter 11.

`.env` is listed in `.gitignore` (`.env`, `.env.*`, `!.env.example`) and in `.dockerignore`.
The Docker build in Chapter 14 goes further and **fails** if any `.env` file is inside the
image.

## Step 3 — start Neo4j

```bash
make up
```

This runs `docker compose up -d && docker compose ps`. The compose file starts one service:

```yaml
services:
  neo4j:
    image: neo4j:5.20-community
    container_name: genai_capstone_neo4j
    ports:
      - "7475:7474"   # HTTP / Browser  -> http://localhost:7475
      - "7688:7687"   # Bolt            -> bolt://localhost:7688
```

- `image: neo4j:5.20-community` — the free edition; nothing here needs Enterprise.
- `"7475:7474"` — host port 7475 forwards to the container's 7474 (the browser UI).
- `"7688:7687"` — host port 7688 forwards to Bolt, the protocol the Python driver speaks.

There is a `healthcheck` block too, so `docker compose ps` shows `healthy` once the database
is actually accepting connections rather than merely started.

## Step 4 — run the gate

```bash
make check      # runs: .venv/bin/python scripts/check_env.py
```

Let's read the important parts of `scripts/check_env.py`.

### Secrets are never printed

```python
def mask(v: str | None) -> str:
    if not v:
        return f"{DIM}(unset){RST}"
    return f"{v[:6]}…{v[-4:]}" if len(v) > 14 else f"{v[:3]}…"
```

- Prints the first six and last four characters of a key, so you can tell *which* key is
  loaded without exposing it. This habit matters: a terminal scrollback that contains a full
  key is a leak waiting to be screenshotted.

### The OpenRouter check does four things, in a deliberate order

```python
            # Check an AUTHENTICATED endpoint first. /models is public and
            # returns 200 for a revoked key or a deleted account, which made a
            # dead credential look healthy here once already.
            r = c.get(f"{base}/key", headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 401:
                return Result(
                    "OpenRouter", False,
                    f"401 on /key - {r.json().get('error', {}).get('message', 'rejected')} "
                    "(revoked key, or a deleted/suspended account)", fix)
```

- `GET /key` is authenticated; `GET /models` is public. The comment records a real
  mistake: checking only `/models` once made a dead key look healthy.

```python
            # Live completion on the cheap model: proves the key can actually
            # spend, not just read the catalogue. A valid key with no credit
            # passes /models but fails here - better to learn that now.
            r = c.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": fast,
                    "messages": [{"role": "user", "content": "Reply with: ok"}],
                    "max_tokens": 5,
                },
            )
            if r.status_code in (402, 403):
                return Result(
                    "OpenRouter", False,
                    f"{r.status_code} - key valid but no credit available",
                    "Add credit at https://openrouter.ai/credits",
                )
```

- A five-token completion on `gpt-4o-mini`. It costs a fraction of a cent and is the only
  way to prove the account can spend.

```python
            # NeMo Guardrails compatibility probe (see plan, Phase 5 risk).
            # NeMo drives OpenRouter through its OpenAI-compatible engine and
            # needs non-streaming tool-calling to work through the proxy.
```

- The fourth call sends a request with a `tools` array and `"stream": False`, and checks that
  a `tool_calls` field comes back. NeMo Guardrails (Chapter 09) needs exactly this to work
  through a proxy. Finding out on day one beats finding out in Phase 5.

### The Neo4j check gives a *specific* fix

```python
        if "authentication" in msg.lower() or "unauthorized" in msg.lower():
            hint = ("Password mismatch between .env and the container's stored auth. "
                    "Reset with: docker compose down -v && make up")
        elif "could not connect" in msg.lower() or "refused" in msg.lower():
            hint = f"Nothing listening on {uri}. Start it with: make up"
```

- Neo4j stores its password in the data volume on first boot. If you change
  `NEO4J_PASSWORD` in `.env` later, the container still has the old one — hence `down -v`
  (delete the volume) is the fix, and the script tells you so instead of printing a driver
  traceback.

### Expected output

```
PHASE 0 - credential and connectivity check

  [PASS] OpenRouter  sk-or-…xxxx - openai/gpt-4o-mini + openai/gpt-4o reachable
           | tool-calling OK (NeMo-compatible)
  [PASS] Qdrant      connected - no collections yet
  [PASS] Langfuse    pk-lf-…xxxx @ https://cloud.langfuse.com
  [PASS] Tavily      tvly-…xxxx - search returned 1 result(s)
  [PASS] Neo4j       bolt://localhost:7688 - empty (expected before `make ingest`)

All 5 checks passed. Phase 0 complete - ready for Phase 1.
```

"no collections yet" and "empty" are correct at this point — Chapter 04–06 fill them.

---

## ✅ You just learned
- `make install`, `.env`, `make up`, `make check` — the four setup commands.
- Why dependency versions are pinned exactly, with two examples where a loose pin would have
  broken something silently.
- What "the key works" means for each service, and why the OpenRouter check makes a real
  paid call.

## ▶️ Run this now
```bash
make install
cp .env.example .env     # then fill in the five keys
make up
make check
```
Do not continue until all five lines say `PASS`.

## 🧠 Check yourself
1. Why is `GET /models` not sufficient to verify an OpenRouter key?
2. You changed `NEO4J_PASSWORD` in `.env` and the check now fails with "unauthorized". What
   is the fix, and why does it involve deleting a volume?
3. What is the NeMo compatibility probe testing, and why is it run in Phase 0 rather than
   Phase 5?

---

Next: the one file every other module reads its knobs from →
[03-settings.md](03-settings.md)
