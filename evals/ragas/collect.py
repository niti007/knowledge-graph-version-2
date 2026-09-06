"""Collect (question, contexts, answer, ground_truth) rows for RAGAS.

Runs in the MAIN venv, because it needs the app's retrieval stack. RAGAS itself
runs in `.venv-ragas` -- its pins conflict with the langchain range Phases 4-7
depend on -- and reads the JSONL this writes. The two never share a process.

One generator, one prompt, one model across every configuration: the answer is
produced from the retrieved context alone, so the ONLY variable between rows of
different configurations is what retrieval put in front of the model. That is
what makes the ablation an ablation. Using the full LangGraph agent instead
would let tool-choice and multi-turn reasoning paper over a weak retriever, and
the resulting table would measure the agent, not the retrieval mode.

    PYTHONPATH=. .venv/bin/python -m evals.ragas.collect
    PYTHONPATH=. .venv/bin/python -m evals.ragas.collect --only hybrid_rerank@bge
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.llm.client import build_chat_model
from app.retrieval.hybrid import HybridRetriever, Mode
from evals.ragas.golden_set import GOLDEN_SET

OUT_DIR = Path(__file__).parent / "data"

# Judge and generator are both gpt-4o-mini: cost control, and a generator the
# ablation can afford to run four times over. A stronger generator would raise
# every row's faithfulness and compress the gap the ablation is trying to see.
GENERATOR_MODEL = "openai/gpt-4o-mini"

RERANKERS = {
    "bge": "BAAI/bge-reranker-base",
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


@dataclass(frozen=True)
class Config:
    name: str
    mode: Mode
    reranker: str | None      # key into RERANKERS; None when the mode never re-ranks

    @property
    def reranker_model(self) -> str | None:
        return RERANKERS[self.reranker] if self.reranker else None


CONFIGS: list[Config] = [
    # The three-mode ablation, all on the incumbent re-ranker.
    Config("vector_only", Mode.VECTOR_ONLY, None),
    Config("hybrid_no_rerank", Mode.HYBRID_NO_RERANK, None),
    Config("hybrid_rerank@bge", Mode.HYBRID_RERANK, "bge"),
    # The deferred Phase 3 decision: same mode, other cross-encoder.
    Config("hybrid_rerank@minilm", Mode.HYBRID_RERANK, "minilm"),
]

SYSTEM = (
    "You answer questions about ACME Corp using ONLY the context provided. "
    "If the context does not contain the answer, say so plainly and do not guess. "
    "Be specific and concise: name the systems, teams, people, document ids and "
    "numeric values that the context actually states. Do not add information "
    "from outside the context."
)

USER = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _settings_for(cfg: Config):
    s = get_settings()
    if cfg.reranker_model and s.reranker_model != cfg.reranker_model:
        # A copy, not a mutation: get_settings() is cached and shared with the
        # rest of the process, and get_reranker keys its cache on the model
        # name, so both cross-encoders can live in one run.
        s = s.model_copy(update={"reranker_model": cfg.reranker_model})
    return s


def collect(cfg: Config) -> Path:
    settings = _settings_for(cfg)
    # build_chat_model, not get_chat_model: pick_model(SYNTHESIS) would route to
    # gpt-4o. The ablation pins the generator so the four runs differ in
    # retrieval and in nothing else.
    llm = build_chat_model(GENERATOR_MODEL, settings=settings, temperature=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{cfg.name}.jsonl"
    rows = []
    t0 = time.perf_counter()
    with HybridRetriever(settings=settings) as r:
        r.retrieve("warm up the models", cfg.mode)   # exclude model load from timings
        for q in GOLDEN_SET:
            t1 = time.perf_counter()
            res = r.retrieve(q.question, cfg.mode)
            retrieval_ms = (time.perf_counter() - t1) * 1000
            contexts = [i.text for i in res.items]
            t2 = time.perf_counter()
            answer = llm.invoke([
                ("system", SYSTEM),
                ("user", USER.format(context=res.context_text(), question=q.question)),
            ]).content.strip()
            rows.append({
                "qid": q.qid,
                "category": q.category,
                "user_input": q.question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": q.ground_truth,
                "evidence": q.evidence,
                "citations": [i.citation for i in res.items],
                "branches": [i.branch for i in res.items],
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round((time.perf_counter() - t2) * 1000, 1),
            })
            print(f"  {cfg.name:<22} {q.qid} {retrieval_ms:7.0f}ms "
                  f"n_ctx={len(contexts)} graph={sum(b == 'graph' for b in rows[-1]['branches'])}")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{cfg.name}: {len(rows)} rows -> {out}  ({time.perf_counter() - t0:.0f}s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None,
                    help="config name; repeatable. Default: all four.")
    args = ap.parse_args()
    todo = [c for c in CONFIGS if not args.only or c.name in args.only]
    if not todo:
        raise SystemExit(f"no config matched {args.only}; "
                         f"known: {[c.name for c in CONFIGS]}")
    for cfg in todo:
        collect(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
