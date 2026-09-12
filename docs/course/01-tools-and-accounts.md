# Chapter 01 — Tools and Accounts

## Why this / what's the need

Every later chapter assumes four things are installed and five credentials work. Getting a
key wrong is the single most common reason a build stalls, and it is also the most boring —
so this project front-loads all of it into one step with a script that *proves* each key
works before a single line of the real system runs. This chapter is the shopping list; the
next chapter runs the proof.

> 🔑 **New word — API key:** a long secret string that identifies you to a service. Anyone
> who has it can spend your money, so it lives only in a local file called `.env` that is
> never committed to git.

---

## Tools to install

### Python 3.11 — via `uv`
The project pins `requires-python = ">=3.11,<3.12"` in `pyproject.toml`. `uv` is a fast
Python package and environment manager; the `Makefile` uses it to create the virtual
environment:

```make
install:  ## Create venv and install dependencies
	uv venv --python 3.11 && uv pip install -e ".[dev]"
```

- `uv venv --python 3.11` — creates `.venv/` using Python 3.11 specifically (uv will download
  it if you don't have it).
- `uv pip install -e ".[dev]"` — installs the project in *editable* mode (so code changes are
  picked up without reinstalling) plus the `dev` extras (pytest, ruff).

Install `uv` from https://docs.astral.sh/uv/ (one shell command on macOS/Linux, one
PowerShell command on Windows).

> 🔑 **New word — virtual environment:** a private folder of installed Python packages for
> one project, so that this project's pinned versions cannot collide with another project's.

### Docker
Neo4j runs locally in Docker (`docker-compose.yml`). Install Docker Desktop from
https://www.docker.com/products/docker-desktop/. This project pulls `neo4j:5.20-community`.

### Node.js
Only needed for one thing: the Promptfoo evaluation suite in Chapter 13 runs with
`npx -y promptfoo@0.118.11`. Install an LTS release from https://nodejs.org/.

### `make`
Every command in this course is a `make` target. macOS and Linux have it; on Windows use
WSL or read the target's body out of the `Makefile` and run it by hand.

---

## Accounts and keys — exactly where to find each one

All of these go in `.env` (Chapter 02). The instructions below are copied from the comments
in `.env.example`, which is the file you will fill in.

### 1. OpenRouter — `OPENROUTER_API_KEY`
- Go to https://openrouter.ai → sign in → **Keys** → *Create Key*.
- Format: starts with `sk-or-v1-`.
- **Needs credit on the account.** `gpt-4o-mini` costs about $0.15 per million input tokens;
  the whole build including every evaluation run in this course used **$3.18**.
- What the check verifies: the key is *authenticated* (not just that the public model list
  loads), both `openai/gpt-4o-mini` and `openai/gpt-4o` are available, a tiny completion
  actually succeeds (a valid key with zero credit fails here), and non-streaming
  tool-calling works — which NeMo Guardrails needs in Chapter 09.

### 2. Qdrant Cloud — `QDRANT_URL` + `QDRANT_API_KEY`
- Go to https://cloud.qdrant.io → create the **free cluster** → **API Keys** → *Create*.
- URL looks like `https://xxxxxxxx.us-east.aws.cloud.qdrant.io:6333`.
- The free tier is 1 GB RAM / 4 GB disk, free forever. This corpus is 66 vectors.
- Two collections will be created for you: `acme_docs` (the index) and `acme_cache` (the
  semantic cache).

### 3. Langfuse Cloud — `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_HOST`
- Go to https://cloud.langfuse.com → new project → **Settings → API Keys**.
- Public key starts `pk-lf-`, secret key starts `sk-lf-`.
- `LANGFUSE_HOST` is `https://cloud.langfuse.com` (EU) or `https://us.cloud.langfuse.com`
  (US) — it must match the region you picked, or the health check returns 401.
- Optional: set `LANGFUSE_ENABLED=false` to run without tracing. The system never fails a
  request because tracing failed (Chapter 11).

### 4. Tavily — `TAVILY_API_KEY`
- Go to https://tavily.com → sign up → dashboard → copy the key.
- Format: starts `tvly-`. 1,000 free searches per month.
- Optional: without it the `web_search` tool reports itself unavailable and the agent carries
  on with the other two tools.

### 5. Neo4j — nothing to sign up for
- Runs locally via `make up`. You choose a password and put it in `NEO4J_PASSWORD`;
  `docker-compose.yml` reads the same variable:
  ```yaml
  environment:
    NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD:?NEO4J_PASSWORD must be set in .env}"
  ```
  The `:?` syntax makes Compose refuse to start with an empty password rather than starting
  a database with no auth.
- Ports: this project binds **7475** (browser) and **7688** (Bolt) instead of the usual
  7474/7687, so it can run beside any other local Neo4j without a port collision.
  `NEO4J_URI=bolt://localhost:7688`.

---

## ✅ You just learned
- The four tools (uv/Python 3.11, Docker, Node, make) and what each is used for.
- The five credentials, the exact page each comes from, and what "working" means for each.
- Which services are optional (Tavily, Langfuse) and which are not (OpenRouter, Qdrant, Neo4j).

## ▶️ Run this now
Create the four accounts and copy each key somewhere safe (a password manager, not a chat
window). Install `uv`, Docker Desktop and Node. Confirm:

```bash
uv --version
docker --version
node --version
make --version
```

## 🧠 Check yourself
1. Why does the OpenRouter check make a real (tiny) completion instead of only listing models?
2. Why does this project use ports 7475/7688 for Neo4j?
3. Which two services can be missing without stopping the system from answering?

---

Next: put the keys in place and run the gate →
[02-project-setup.md](02-project-setup.md)
