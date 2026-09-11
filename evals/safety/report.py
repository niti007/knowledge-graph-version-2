"""Turn `results.json` into the tables in `scorecard.md`.

    PYTHONPATH=. .venv/bin/python -m evals.safety.report

Prints markdown to stdout. The narrative in `scorecard.md` -- the findings, the
analysis of which attacks got through and why -- is written by hand against a
specific run and is NOT generated here: a script cannot tell you that a
fabricated citation is the most serious result on the page. This module
produces only the numbers that narrative refers to, so a re-run can be checked
against it line by line.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from evals.safety.attacks import ALL_ATTACKS, DEEP_ATTACKS, INPUT_ATTACKS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"

STAGES = ["input", "output", "none"]


def load(path: Path = RESULTS) -> dict:
    return json.loads(path.read_text())


def _rails_for(rows) -> str:
    c = collections.Counter()
    for r in rows:
        if r["blocked"] and r["blocked_by"]:
            c[r["blocked_by"]] += 1
    return ", ".join(f"`{k}` {v}" for k, v in c.most_common()) or "—"


def table_by_category(rows) -> str:
    attacks = [r for r in rows if r["kind"] == "attack"]
    out = ["| category | n | input-blocked | output-blocked | not blocked | block rate | rails that fired |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for cat in sorted({r["category"] for r in attacks}):
        sub = [r for r in attacks if r["category"] == cat]
        c = collections.Counter(r["stage"] for r in sub)
        blocked = c["input"] + c["output"]
        out.append(f"| {cat} | {len(sub)} | {c['input']} | {c['output']} | {c['none']} | "
                   f"{blocked / len(sub):.0%} | {_rails_for(sub)} |")
    c = collections.Counter(r["stage"] for r in attacks)
    blocked = c["input"] + c["output"]
    out.append(f"| **all** | **{len(attacks)}** | **{c['input']}** | **{c['output']}** | "
               f"**{c['none']}** | **{blocked / len(attacks):.0%}** | |")
    return "\n".join(out)


def table_by_half(rows) -> str:
    ids_input = {a.id for a in INPUT_ATTACKS}
    ids_deep = {a.id for a in DEEP_ATTACKS}
    out = ["| probe set | n | input-blocked | output-blocked | not blocked | agent ran |",
           "|---|---:|---:|---:|---:|---:|"]
    for name, ids in (("conventional (INPUT_ATTACKS)", ids_input),
                      ("input-rail-evading (DEEP_ATTACKS)", ids_deep)):
        sub = [r for r in rows if r["id"] in ids]
        c = collections.Counter(r["stage"] for r in sub)
        ran = sum(1 for r in sub if r["agent_ran"])
        out.append(f"| {name} | {len(sub)} | {c['input']} | {c['output']} | {c['none']} | {ran} |")
    return "\n".join(out)


def table_rails(rows) -> str:
    fired = collections.Counter()
    blocked = collections.Counter()
    for r in rows:
        for name in r["rails_fired"]:
            fired[name] += 1
        if r["blocked"] and r["blocked_by"]:
            blocked[r["blocked_by"]] += 1
    stage = {}
    for r in rows:
        for d in r["rails"]:
            stage[d["rail"]] = d["stage"]
    order = ["pii_input", "self_check_input", "topic_scope",
             "self_check_output", "check_grounding", "pii_output", "check_citations"]
    out = ["| rail | stage | times fired | times it was the blocking rail |",
           "|---|---|---:|---:|"]
    for name in order:
        out.append(f"| `{name}` | {stage.get(name, '?')} | {fired[name]} | {blocked[name]} |")
    return "\n".join(out)


def table_pairs(rows) -> str:
    out = ["| # | lookalike (must be allowed) | result | attack (must be blocked) | result |",
           "|---|---|---|---|---|"]
    a = [r for r in rows if r["kind"] == "pair_allowed"]
    b = [r for r in rows if r["kind"] == "pair_blocked"]
    for i, (x, y) in enumerate(zip(a, b), 1):
        xs = "allowed ✓" if not x["blocked"] else f"BLOCKED by `{x['blocked_by']}` ✗"
        ys = f"blocked by `{y['blocked_by']}` ✓" if y["blocked"] else "ALLOWED ✗"
        out.append(f"| {i} | {x['prompt'][:60]} | {xs} | {y['prompt'][:60]} | {ys} |")
    return "\n".join(out)


def table_pii(data) -> str:
    rows = data["rows"]
    leaks = sum(r["pii"]["n_emails"] for r in rows)
    phones = sum(len(r["pii"]["phones"]) for r in rows)
    names = sum(1 for r in rows if r["pii"]["n_names"])
    masks = sum(r["answer"].count("<EMAIL_ADDRESS>") for r in rows)
    cp = data["corpus_pii"]
    return "\n".join([
        "| measure | value |",
        "|---|---:|",
        f"| responses scanned | {len(rows)} |",
        f"| real corpus email addresses in the corpus | {cp['n_emails']} |",
        f"| **real corpus email addresses leaked** | **{leaks}** |",
        f"| responses where an address was masked to `<EMAIL_ADDRESS>` | {masks} |",
        f"| phone-shaped strings in any response | {phones} |",
        f"| phone numbers present anywhere in the corpus | {cp['n_phones_in_corpus']} |",
        f"| responses naming an employee (directory content, not a leak) | {names} |",
    ])


def main() -> None:
    data = load()
    rows = data["rows"]
    print("## Attacks by category\n")
    print(table_by_category(rows))
    print("\n## Attacks by probe half\n")
    print(table_by_half(rows))
    print("\n## Rail activity\n")
    print(table_rails(rows))
    print("\n## Regression pairs\n")
    print(table_pairs(rows))
    print("\n## PII\n")
    print(table_pii(data))
    ben = [r for r in rows if r["kind"] == "benign"]
    blocked = [r for r in ben if r["blocked"]]
    print(f"\n## Benign\n\n{len(blocked)}/{len(ben)} blocked "
          f"({len(blocked) / len(ben):.1%})")
    for r in blocked:
        print(f"- {r['id']}: {r['prompt']!r} -> `{r['blocked_by']}` ({r['blocked_stage']})")


if __name__ == "__main__":
    main()
