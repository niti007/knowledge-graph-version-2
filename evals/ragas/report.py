"""Render the ablation tables from results/summary.json. Runs in `.venv-ragas`.

Split out from run_ragas.py so the tables can be regenerated (and the deltas
recomputed) without paying for another judged run.

    ../../.venv-ragas/bin/python report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
CONFIGS = ["vector_only", "hybrid_no_rerank", "hybrid_rerank@bge", "hybrid_rerank@minilm"]


def cell(v, width=20):
    return "{:>{w}}".format("-" if v is None else "{:.3f}".format(v), w=width)


def table(title: str, get) -> None:
    w = max(len(c) for c in CONFIGS) + 2
    print(f"\n{title}\n")
    head = f"{'configuration':<{w}}" + "".join(f"{m[:18]:>20}" for m in METRICS)
    print(head)
    print("-" * len(head))
    for name in CONFIGS:
        print(f"{name:<{w}}" + "".join(cell(get(name, m)) for m in METRICS))


def main() -> int:
    s = json.loads((RESULTS / "summary.json").read_text())

    table("OVERALL (30 questions)", lambda n, m: s[n]["overall"][m])

    cats = sorted({c for n in CONFIGS for c in s[n]["by_category"]})
    for cat in cats:
        table(f"CATEGORY: {cat}",
              lambda n, m, cat=cat: s[n]["by_category"].get(cat, {}).get(m))

    print("\nDELTA vs vector_only (percentage points)\n")
    w = max(len(c) for c in CONFIGS) + 2
    head = f"{'configuration':<{w}}" + "".join(f"{m[:18]:>20}" for m in METRICS)
    print(head)
    print("-" * len(head))
    base = s["vector_only"]["overall"]
    for name in CONFIGS[1:]:
        row = ""
        for m in METRICS:
            a, b = s[name]["overall"][m], base[m]
            row += "{:>20}".format("-" if a is None or b is None
                                   else "{:+.1f}pp".format((a - b) * 100))
        print(f"{name:<{w}}" + row)

    print("\nRUN METADATA\n")
    for name in CONFIGS:
        print(f"  {name:<22} n={s[name]['n']:<4} judge_seconds={s[name]['elapsed_s']:<8} "
              f"nan_cells={s[name]['nan_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
