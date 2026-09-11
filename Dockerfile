# syntax=docker/dockerfile:1.6
# =============================================================================
# Enterprise Knowledge Assistant -- single image for Hugging Face Docker Spaces
#
# One container runs, in order (see start.sh):
#   Neo4j Community 5 (embedded, loopback only)  ->  graph rebuild from data/raw
#   ->  FastAPI on 127.0.0.1:8000                  ->  Streamlit on 0.0.0.0:7860
#
# Qdrant, Langfuse, OpenRouter and Tavily stay in the cloud and are reached with
# credentials supplied at RUNTIME (HF Space secrets / --env-file). Nothing
# secret is baked in: .env is excluded by .dockerignore and the build asserts it.
# =============================================================================
FROM python:3.11-slim-trixie AS base

ARG NEO4J_VERSION=5.26.30
ARG DEBIAN_FRONTEND=noninteractive

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
    mkdir -p /opt; tar -xzf /tmp/neo4j.tgz -C /opt; \
    mv "/opt/neo4j-community-${NEO4J_VERSION}" "$NEO4J_HOME"; rm /tmp/neo4j.tgz; \
    rm -rf "$NEO4J_HOME/data" "$NEO4J_HOME/logs" "$NEO4J_HOME/run" "$NEO4J_HOME/import"; \
    mkdir -p /var/lib/neo4j/data /var/lib/neo4j/run /var/lib/neo4j/import /var/log/neo4j
COPY docker/neo4j.conf $NEO4J_HOME/conf/neo4j.conf

# --- Non-root user. HF Spaces run the container as uid 1000; create that user
#     now so every path below is owned by it whether or not HF overrides USER.
RUN useradd -m -u 1000 -s /bin/bash user \
 && chown -R user:user $NEO4J_HOME /var/lib/neo4j /var/log/neo4j
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$NEO4J_HOME/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model cache lives inside the image (populated below) and is read-only at
    # runtime; HF_HUB_OFFLINE keeps the Hub client from phoning home on boot.
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# --- Python deps: pinned top-level (pyproject) + pinned transitive (constraints)
#     torch comes from the CPU wheel index: same version as the tested host,
#     without the ~2 GB of CUDA libraries the default wheel drags in.
COPY pyproject.toml constraints.txt ./
RUN mkdir -p app && touch app/__init__.py \
 && pip install -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        . \
 && rm -rf app

# --- Pre-download models at BUILD time (embedding, reranker, spaCy) ----------
USER user
COPY --chown=user:user docker/warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py
USER root

# --- Application code --------------------------------------------------------
COPY --chown=user:user app ./app
COPY --chown=user:user ui ./ui
COPY --chown=user:user data/raw ./data/raw
COPY --chown=user:user start.sh ./start.sh
# Ingestion writes documents.jsonl / chunks.jsonl here on every boot.
# Re-install the package (editable) so `app` resolves from /app, not the
# placeholder used for the dependency layer above.
RUN pip install --no-deps --no-build-isolation -e . \
 && mkdir -p /app/data/processed && chmod +x /app/start.sh && chown -R user:user /app \
 # Assert no secret file entered the image. Build FAILS if one did.
 && if find / -xdev \( -name ".env" -o -name ".env.*" \) -not -name ".env.example" \
        -not -path "*/site-packages/*" 2>/dev/null | grep -q .; then \
        echo "FATAL: a .env file is inside the image" >&2; exit 1; fi

# --- Runtime configuration (non-secret). Secrets arrive from the environment. --
ENV NEO4J_URI=bolt://127.0.0.1:7687 \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=embedded-no-auth \
    TORCH_DEVICE=cpu \
    API_PORT=8000 \
    API_URL=http://127.0.0.1:8000 \
    APP_ENV=hf-space \
    HF_HUB_OFFLINE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/_stcore/health || exit 1
CMD ["/app/start.sh"]
