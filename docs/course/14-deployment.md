# Chapter 14 — Deployment

## Why this / what's the need

A system that only runs on the author's laptop is a demo, not a product. The requirement
here was a public URL a grader could open a week later and get the multi-hop answer. That
sounds simple and was cut from the plan once, for a specific reason:

> The earlier cut was because a Space cannot reach Docker on a laptop and Neo4j Aura's free
> tier pauses after 3 days idle — a graded demo breaking exactly when a grader opens it.

The resolution is the whole chapter: run **Neo4j Community inside the container** and
rebuild the 98-node graph from the corpus on every boot. The graph is deterministic
(Chapter 06), so a rebuild is seconds, needs no persistent disk, no cloud database, and no
new credentials. Qdrant, Langfuse, OpenRouter and Tavily were already cloud services.

**Live:** https://nitishgalat-enterprise-knowledge-assistant.hf.space

> 🔑 **New word — Hugging Face Space:** free hosting for ML demos. A *Docker* Space builds
> your `Dockerfile` and runs the container, exposing one port (7860 here).

> 🔑 **New word — cold start:** the time from "container starts" to "can answer a
> question". Free Spaces sleep after 48 hours idle, so the first visitor after that pays it.

---

## The image — `Dockerfile`

One image: `python:3.11-slim` + a headless Java runtime + Neo4j + the app, with the ML
models downloaded at **build** time.

```dockerfile
FROM python:3.11-slim-trixie AS base

ARG NEO4J_VERSION=5.26.30

# --- System: headless Java 21 (Neo4j 5.26 supports 17 or 21; trixie ships 21) --
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless ca-certificates curl procps \
 && rm -rf /var/lib/apt/lists/*

# --- Neo4j Community (tarball: arch-independent, no systemd, no extra user) --
ENV NEO4J_HOME=/opt/neo4j
RUN set -eux; \
    curl -fsSL "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz" \
        -o /tmp/neo4j.tgz; \
    ...
COPY docker/neo4j.conf $NEO4J_HOME/conf/neo4j.conf
```

- Neo4j is installed from the tarball rather than a package, so there is no init system to
  fight and no second user.

```dockerfile
# --- Non-root user. HF Spaces run the container as uid 1000; create that user
#     now so every path below is owned by it whether or not HF overrides USER.
RUN useradd -m -u 1000 -s /bin/bash user \
 && chown -R user:user $NEO4J_HOME /var/lib/neo4j /var/log/neo4j
```

- Hugging Face enforces uid 1000. Every directory Neo4j and the app write to is owned by it.

```dockerfile
# --- Python deps: pinned top-level (pyproject) + pinned transitive (constraints)
#     torch comes from the CPU wheel index: same version as the tested host,
#     without the ~2 GB of CUDA libraries the default wheel drags in.
COPY pyproject.toml constraints.txt ./
RUN mkdir -p app && touch app/__init__.py \
 && pip install -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        . \
 && rm -rf app
```

- `constraints.txt` pins the transitive dependencies too. The CPU-only torch wheel avoids
  the ~2 GB of CUDA libraries the default wheel drags in; the image is 4.02 GB.

```dockerfile
# --- Pre-download models at BUILD time (embedding, reranker, spaCy) ----------
USER user
COPY --chown=user:user docker/warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py
```

- `docker/warm_models.py` loads `bge-small-en-v1.5`, `bge-reranker-base` and
  `en_core_web_sm` once, so the ~1.2 GB of weights are in the image and cold start never
  hits the network. `HF_HUB_OFFLINE=1` at runtime keeps the Hub client from phoning home.

```dockerfile
 # Assert no secret file entered the image. Build FAILS if one did.
 && if find / -xdev \( -name ".env" -o -name ".env.*" \) -not -name ".env.example" \
        -not -path "*/site-packages/*" 2>/dev/null | grep -q .; then \
        echo "FATAL: a .env file is inside the image" >&2; exit 1; fi
```

- `.dockerignore` excludes `.env`; this line makes sure. Secrets come only from the Space's
  settings at runtime.

### Neo4j inside the container — `docker/neo4j.conf`

```
dbms.security.auth_enabled=false

# Loopback only. No HTTP/HTTPS connector at all -- the app speaks bolt.
server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=127.0.0.1:7687
server.http.enabled=false
server.https.enabled=false
```

- Auth is off, and the file says why: the instance listens on loopback only with HTTP off,
  so anyone who can open a socket to it is already inside the container. A password would
  protect nothing and be one more secret to manage.
- Heap 1 GB, page cache 128 MB — sized for the free tier with the torch models sharing the
  box. Steady-state RAM measured at 1.9 GiB.

## The boot sequence — `start.sh`

Four timed steps with readiness gates, so cold start is measurable from the logs and a
failure names its stage:

```bash
# --- [1] Neo4j ----------------------------------------------------------------
"${NEO4J_HOME:-/opt/neo4j}/bin/neo4j" console > /var/log/neo4j/console.log 2>&1 &
NEO4J_PID=$!
...
until python - "$NEO4J_URI" <<'PY' 2>/dev/null
import sys
from neo4j import GraphDatabase
with GraphDatabase.driver(sys.argv[1], auth=None) as d:
    d.verify_connectivity()
PY
do
  kill -0 "$NEO4J_PID" 2>/dev/null || { tail -n 40 /var/log/neo4j/console.log; die "neo4j process exited"; }
  [ "$(date +%s)" -lt "$deadline" ] || { ...; die "neo4j did not accept bolt ... within ${NEO4J_WAIT}s"; }
  sleep 1
done
```

- Polls Bolt with the real driver until it connects. If the process dies or the deadline
  passes, the last 40 lines of the Neo4j log are printed before exiting — a Space stuck at
  "Starting" with no output is the one failure that cannot be debugged from outside.

```bash
# --- [2] Graph rebuild ---------------------------------------------------------
# Qdrant is skipped on purpose: the 66 vectors already live in Qdrant Cloud and
# re-upserting on every boot would be redundant work plus a run of the orphan
# purge against a live index. Only the (ephemeral) Neo4j graph is rebuilt.
python -m app.ingestion.run --no-upsert --reset-graph 2>&1 | sed -u 's/^/    | /'
```

- The same ingestion entrypoint as Chapter 12, Neo4j-only. A post-check asserts
  `nodes > 0 and rels > 0` and logs `98 nodes / 236 relationships`.

```bash
# --- [3] FastAPI ---------------------------------------------------------------
python -m uvicorn app.api.main:app --host 127.0.0.1 --port "$API_PORT" ... &
...
until curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; do
```

- `/health` returns 503 until warmup completes (Chapter 10), so `curl -f` is the readiness
  gate. The API binds to 127.0.0.1 — only Streamlit can reach it.

```bash
# --- [4] Streamlit (foreground) -----------------------------------------------
exec python -m streamlit run ui/streamlit_app.py \
    --server.address 0.0.0.0 --server.port "$UI_PORT" ...
```

- `exec` replaces the shell, so the container lives exactly as long as the UI does.

### The stale-secret incident

The first live boot failed after 120 s, waiting on `neo4j+s://…databases.neo4j.io`. The
Space had been reused from an earlier project, and a leftover Aura secret **overrode the
Dockerfile's `ENV NEO4J_URI`** — Space secrets win over image ENV. The embedded Neo4j had
come up fine on localhost; boot was simply waiting on the wrong host. The fix is at the top
of `start.sh`:

```bash
# This container OWNS its Neo4j. An external NEO4J_URI in the environment is
# never right here -- and it happened: a leftover Aura secret on the reused
# Space overrode the Dockerfile ENV, the embedded instance came up fine on
# localhost, and boot spent 120s waiting on a hostname it should never have
# looked at. Force the embedded endpoint and say so loudly if we overrode.
EMBEDDED_URI="bolt://127.0.0.1:7687"
if [ -n "${NEO4J_URI:-}" ] && [ "${NEO4J_URI}" != "${EMBEDDED_URI}" ]; then
  log "WARNING: ignoring external NEO4J_URI=${NEO4J_URI}; this image runs its own Neo4j"
fi
export NEO4J_URI="${EMBEDDED_URI}"
```

Eighteen stale secrets from the old project (Aura, OpenAI, Redis, Chroma) were deleted from
the Space at the same time. Lesson: a reused environment carries its history; the container
should own the invariants it depends on.

## Measured

| Where | Neo4j | graph | API ready | total cold start |
|---|---:|---:|---:|---:|
| local, 2-core container | ~8 s | ~14 s | ~12 s | 32 s |
| HF `cpu-basic`, live | 9.9 s | 16.2 s (98/236) | Qdrant + Neo4j ok | **41.2 s** |

Validated locally as uid 1000 and then on the live URL: graph 98/236 every boot, `/health`
ready, injection blocked at input, out-of-corpus declined, multi-hop answered via
`graph_query`, repeat served from cache, and a trace from inside the container fetched
back from Langfuse Cloud.

## Operating it

- **Free Spaces sleep after 48 hours idle.** The first visit after that pays the ~41 s cold
  start; the UI shows "Not ready" until the logs print `[4/4]`. The vector index is
  unaffected (it lives in Qdrant Cloud); the graph is rebuilt.
- **Required secrets** (Settings → Variables and secrets): `OPENROUTER_API_KEY`,
  `QDRANT_URL`, `QDRANT_API_KEY`; optional `LANGFUSE_*` and `TAVILY_API_KEY`. `start.sh`
  refuses to boot without the first three and says which is missing.
- **Each answer costs real tokens** on the owner's OpenRouter account.
- `README_HF.md` is the Space's own README, with the `sdk: docker` / `app_port: 7860`
  frontmatter Hugging Face reads.

## Build and run locally

```make
docker-build:  ## Build the HF Space image (embedded Neo4j + pre-downloaded models)
	docker build -t $(IMAGE) .

docker-run:    ## Run the image locally on :7860 with cloud creds from .env (never baked in)
	docker run --rm --name genai-capstone -p 7860:7860 --user 1000:1000 \
	  --env-file .env -e NEO4J_URI=bolt://127.0.0.1:7687 -e NEO4J_PASSWORD=embedded-no-auth \
	  -e TORCH_DEVICE=cpu -e APP_ENV=hf-space $(IMAGE)
```

- `--env-file .env` passes credentials at *run* time; they are never in the image.
- `--user 1000:1000` reproduces the Space's uid so permission problems show up locally.

---

## ✅ You just learned
- Why the deployment was cut and how embedding Neo4j un-cut it.
- What is baked into the image (models, Neo4j, code) and what is not (secrets, the vector
  index).
- The four-step boot with readiness gates, and the stale-secret incident.
- The measured cold start and the 48-hour sleep.

## ▶️ Run this now
```bash
make docker-build          # ~10+ minutes the first time; 4.02 GB image
make docker-run            # watch [1/4] … [4/4] in the log; then open http://localhost:7860
```
Ask the multi-hop question. Then open the live Space and ask it there.

## 🧠 Check yourself
1. Why is Neo4j auth disabled in the container, and why is that safe?
2. Why does `start.sh` skip the Qdrant upsert on boot?
3. A Space secret and a Dockerfile `ENV` set the same variable. Which wins, and what did
   that cost here?
4. What happens on the first visit after 48 idle hours, and what is *not* lost?

---

Next, for instructors →
[15-teacher-notes.md](15-teacher-notes.md)
