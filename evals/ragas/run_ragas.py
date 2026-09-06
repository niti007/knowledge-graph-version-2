"""Score the collected retrieval runs with RAGAS. Runs in `.venv-ragas` ONLY.

    ../../.venv-ragas/bin/python run_ragas.py

Why a second venv: ragas 0.2.15 pins the langchain 0.3 line, and Phases 4-7 of
this app run on langchain-core 1.6.1 / langgraph 1.2.11. Installing ragas into
the main venv would silently downgrade the agent's runtime. The two processes
share nothing but the JSONL files in ./data.

Judge: gpt-4o-mini through OpenRouter's OpenAI-compatible endpoint. RAGAS did
not fight the base URL -- it takes any LangChain BaseChatModel through
LangchainLLMWrapper, so pointing ChatOpenAI at OpenRouter is all it needed.

Embeddings: answer_relevancy needs an embedding model, and OpenRouter serves no
embeddings endpoint. Rather than reach for a second provider, this uses the
LOCAL BAAI/bge-small-en-v1.5 -- the same model the corpus was indexed with, so
the relevancy metric measures similarity in the same space the retriever uses.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                           faithfulness)
from ragas.run_config import RunConfig

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

CONFIGS = ["vector_only", "hybrid_no_rerank",
           "hybrid_rerank@bge", "hybrid_rerank@minilm"]
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise SystemExit(f"{name} is not set. Run: set -a && . ../../.env && set +a")
    return v


def build_judge():
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=_env("OPENROUTER_API_KEY"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.0,
        timeout=120,
        max_retries=3,
    )
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def load(name: str) -> Dataset:
    rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    return Dataset.from_dict({
        "question": [r["user_input"] for r in rows],
        "answer": [r["response"] for r in rows],
        "contexts": [r["retrieved_contexts"] for r in rows],
        "ground_truth": [r["reference"] for r in rows],
    }), [r["qid"] for r in rows], [r["category"] for r in rows]


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    llm, emb = build_judge()
    # Low concurrency and a generous timeout: OpenRouter rate-limits harder than
    # the OpenAI endpoint RAGAS defaults assume, and a throttled judge returns
    # NaN, which averages into a silently better-looking score.
    run_config = RunConfig(timeout=180, max_workers=4, max_retries=5)

    summary = {}
    for name in CONFIGS:
        ds, qids, cats = load(name)
        print(f"\n=== {name} ({len(ds)} rows) ===", flush=True)
        t0 = time.perf_counter()
        res = evaluate(ds, metrics=METRICS, llm=llm, embeddings=emb,
                       run_config=run_config, raise_exceptions=False)
        elapsed = time.perf_counter() - t0
        df: pd.DataFrame = res.to_pandas()
        df.insert(0, "qid", qids)
        df.insert(1, "category", cats)
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        cols = [m.name for m in METRICS]
        # NaN means the judge failed on that row, not that the row scored zero.
        # Reported separately so a run with holes cannot pass as a clean one.
        summary[name] = {
            "n": len(df),
            "elapsed_s": round(elapsed, 1),
            "nan_cells": int(df[cols].isna().sum().sum()),
            "overall": {c: (None if df[c].isna().all() else round(float(df[c].mean()), 4))
                        for c in cols},
            "by_category": {
                cat: {c: (None if g[c].isna().all() else round(float(g[c].mean()), 4))
                      for c in cols}
                for cat, g in df.groupby("category")
            },
        }
        print(json.dumps(summary[name]["overall"], indent=1), flush=True)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------- report
    cols = [m.name for m in METRICS]
    w = max(len(c) for c in CONFIGS) + 2
    print("\n\nRAGAS ABLATION  (30 questions, judge = gpt-4o-mini via OpenRouter)\n")
    head = f"{'configuration':<{w}}" + "".join(f"{c[:18]:>20}" for c in cols) + f"{'NaN':>6}"
    print(head); print("-" * len(head))
    for name in CONFIGS:
        s = summary[name]
        ov = s["overall"]
        cells = "".join(
            "{:>20}".format("-" if ov[c] is None else "{:.3f}".format(ov[c]))
            for c in cols)
        print(f"{name:<{w}}" + cells + f"{s['nan_cells']:>6}")

    print("\n\nMULTI-HOP SUBSET ONLY (the category the Phase 3 re-ranker choice turned on)\n")
    head = f"{'configuration':<{w}}" + "".join(f"{c[:18]:>20}" for c in cols)
    print(head); print("-" * len(head))
    for name in CONFIGS:
        bc = summary[name]["by_category"].get("multi_hop", {})
        print(f"{name:<{w}}" + "".join(
            "{:>20}".format("-" if bc.get(c) is None else "{:.3f}".format(bc[c]))
            for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
