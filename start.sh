#!/usr/bin/env bash
# =============================================================================
# Container boot sequence for the Hugging Face Space.
#
#   [1] Neo4j (embedded, loopback)      wait for bolt
#   [2] rebuild the knowledge graph     Neo4j-only path of app.ingestion.run
#   [3] FastAPI on 127.0.0.1:8000       wait for /health ready
#   [4] Streamlit on 0.0.0.0:7860       FOREGROUND: the container lives as long
#                                       as the UI does
#
# Every step is timed and logged so cold start is measurable from the Space
# logs alone. Every wait has a timeout and fails LOUDLY (log tail included),
# because a Space stuck at "Starting" with no output is the one failure mode
# that cannot be debugged from outside.
# =============================================================================
set -euo pipefail

T0=$(date +%s.%N)
since(){ awk -v a="$(date +%s.%N)" -v b="$1" 'BEGIN{printf "%.1f", a-b}'; }
log()  { printf '[start.sh +%6ss] %s\n' "$(since "$T0")" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

NEO4J_WAIT=${NEO4J_WAIT_SECONDS:-120}
API_WAIT=${API_WAIT_SECONDS:-300}
API_PORT=${API_PORT:-8000}
UI_PORT=${PORT:-7860}
export API_URL=${API_URL:-http://127.0.0.1:${API_PORT}}

cd /app
log "boot as uid=$(id -u) gid=$(id -g)  NEO4J_URI=${NEO4J_URI}  TORCH_DEVICE=${TORCH_DEVICE:-auto}"

# Refuse to start without the cloud credentials the app cannot run without.
# (Tavily is optional: the web-search tool degrades to 'unavailable'.)
for v in OPENROUTER_API_KEY QDRANT_URL QDRANT_API_KEY; do
  [ -n "${!v:-}" ] || die "$v is not set. On Hugging Face add it under Settings -> Variables and secrets."
done

cleanup() {
  log "shutting down"
  [ -n "${API_PID:-}" ]   && kill "$API_PID"   2>/dev/null || true
  [ -n "${NEO4J_PID:-}" ] && kill "$NEO4J_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- [1] Neo4j ----------------------------------------------------------------
S=$(date +%s.%N)
mkdir -p /var/lib/neo4j/data /var/lib/neo4j/run /var/log/neo4j
"${NEO4J_HOME:-/opt/neo4j}/bin/neo4j" console > /var/log/neo4j/console.log 2>&1 &
NEO4J_PID=$!
log "[1/4] neo4j starting (pid $NEO4J_PID), waiting up to ${NEO4J_WAIT}s for bolt"

BOLT_HOST=$(python -c "from urllib.parse import urlparse;u=urlparse('${NEO4J_URI}');print(u.hostname or '127.0.0.1')")
BOLT_PORT=$(python -c "from urllib.parse import urlparse;u=urlparse('${NEO4J_URI}');print(u.port or 7687)")
deadline=$(( $(date +%s) + NEO4J_WAIT ))
until python - "$NEO4J_URI" <<'PY' 2>/dev/null
import sys
from neo4j import GraphDatabase
with GraphDatabase.driver(sys.argv[1], auth=None) as d:
    d.verify_connectivity()
PY
do
  kill -0 "$NEO4J_PID" 2>/dev/null || { tail -n 40 /var/log/neo4j/console.log; die "neo4j process exited"; }
  [ "$(date +%s)" -lt "$deadline" ] || { tail -n 40 /var/log/neo4j/console.log; die "neo4j did not accept bolt on ${BOLT_HOST}:${BOLT_PORT} within ${NEO4J_WAIT}s"; }
  sleep 1
done
log "[1/4] neo4j ready on bolt://${BOLT_HOST}:${BOLT_PORT} in $(since "$S")s"

# --- [2] Graph rebuild ---------------------------------------------------------
# Qdrant is skipped on purpose: the 66 vectors already live in Qdrant Cloud and
# re-upserting on every boot would be redundant work plus a run of the orphan
# purge against a live index. Only the (ephemeral) Neo4j graph is rebuilt.
S=$(date +%s.%N)
log "[2/4] building knowledge graph (Neo4j only, --no-upsert --reset-graph)"
python -m app.ingestion.run --no-upsert --reset-graph 2>&1 | sed -u 's/^/    | /'
COUNTS=$(python - <<'PY'
from app.config import get_settings
from neo4j import GraphDatabase
s = get_settings()
with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
    n = d.execute_query("MATCH (n) RETURN count(n) AS c").records[0]["c"]
    r = d.execute_query("MATCH ()-[r]->() RETURN count(r) AS c").records[0]["c"]
print(f"{n} {r}")
PY
)
NODES=${COUNTS% *}; RELS=${COUNTS#* }
[ "$NODES" -gt 0 ] && [ "$RELS" -gt 0 ] || die "graph build produced ${NODES} nodes / ${RELS} relationships"
log "[2/4] graph ready: ${NODES} nodes / ${RELS} relationships in $(since "$S")s"

# --- [3] FastAPI ---------------------------------------------------------------
S=$(date +%s.%N)
python -m uvicorn app.api.main:app --host 127.0.0.1 --port "$API_PORT" \
    --log-level "${UVICORN_LOG_LEVEL:-info}" > >(sed -u 's/^/    | api: /') 2>&1 &
API_PID=$!
log "[3/4] api starting on 127.0.0.1:${API_PORT} (pid $API_PID); warmup loads models, waiting up to ${API_WAIT}s"
deadline=$(( $(date +%s) + API_WAIT ))
until curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; do
  kill -0 "$API_PID" 2>/dev/null || die "api process exited during warmup"
  [ "$(date +%s)" -lt "$deadline" ] || {
    curl -sS "http://127.0.0.1:${API_PORT}/health" || true
    die "api not ready within ${API_WAIT}s"; }
  sleep 1
done
HEALTH_SUMMARY=$(curl -sS "http://127.0.0.1:${API_PORT}/health" | python -c '
import json, sys
h = json.load(sys.stdin)
deps = " ".join("%s=%s" % (d["name"], "ok" if d["ok"] else "FAIL") for d in h["dependencies"])
print(h["status"], "warmup=%ss" % h["warmup_seconds"], deps)')
log "[3/4] api ready in $(since "$S")s: ${HEALTH_SUMMARY}"

# --- [4] Streamlit (foreground) -----------------------------------------------
log "[4/4] streamlit on 0.0.0.0:${UI_PORT} -> ${API_URL}; total cold start so far $(since "$T0")s"
exec python -m streamlit run ui/streamlit_app.py \
    --server.address 0.0.0.0 --server.port "$UI_PORT" \
    --server.headless true --browser.gatherUsageStats false \
    --server.fileWatcherType none
