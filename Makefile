PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: help install up down logs check ingest serve ui eval test fmt docker-build docker-run docker-stop

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## Create venv and install dependencies
	uv venv --python 3.11 && uv pip install -e ".[dev]"

up:       ## Start Neo4j (ports 7475/7688)
	docker compose up -d && docker compose ps

down:     ## Stop Neo4j
	docker compose down

logs:     ## Tail Neo4j logs
	docker compose logs -f neo4j

check:    ## PHASE 0 GATE: verify every credential works
	$(PY) scripts/check_env.py

ingest:   ## Build Qdrant index + Neo4j graph
	$(PY) -m app.ingestion.run

serve:    ## Run FastAPI on :8000
	.venv/bin/uvicorn app.api.main:app --reload --port 8000

ui:       ## Run Streamlit on :8501
	.venv/bin/streamlit run ui/streamlit_app.py

test:     ## Run pytest
	$(PYTEST) tests/ -v

eval:     ## Safety scorecard against a running API (see evals/*/README for RAGAS, promptfoo, load)
	PYTHONPATH=. $(PY) -m evals.safety.run_scorecard --base http://localhost:8000
	PYTHONPATH=. $(PY) -m evals.safety.report

# --- Hugging Face Space image (Phase 10) --------------------------------------
IMAGE ?= genai-capstone:latest

docker-build:  ## Build the HF Space image (embedded Neo4j + pre-downloaded models)
	docker build -t $(IMAGE) .
	@docker image inspect $(IMAGE) --format 'image size: {{.Size}} bytes'

docker-run:    ## Run the image locally on :7860 with cloud creds from .env (never baked in)
	docker run --rm --name genai-capstone -p 7860:7860 --user 1000:1000 \
	  --env-file .env -e NEO4J_URI=bolt://127.0.0.1:7687 -e NEO4J_PASSWORD=embedded-no-auth \
	  -e TORCH_DEVICE=cpu -e APP_ENV=hf-space $(IMAGE)

docker-stop:   ## Stop the local container
	docker stop genai-capstone
