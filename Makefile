.PHONY: help install up down logs check ingest serve ui eval test fmt

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
	python scripts/check_env.py

ingest:   ## Build Qdrant index + Neo4j graph
	python -m app.ingestion.run

serve:    ## Run FastAPI on :8000
	uvicorn app.api.main:app --reload --port 8000

ui:       ## Run Streamlit on :8501
	streamlit run ui/streamlit_app.py

test:     ## Run pytest
	pytest tests/ -v

eval:     ## Run the full evaluation suite
	python -m evals.run_all
