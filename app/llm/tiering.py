"""Which model runs which call.

Two tiers, one rule. Cheap internal work that a small model does just as well
(classification, routing hints, rail self-checks in Phase 5) goes to
LLM_FAST_MODEL; anything whose output a human reads -- agent reasoning and final
synthesis -- goes to LLM_SMART_MODEL.

This is a function, not a framework: the tier is a property of the *task*, so
callers name a task and never a model string. That keeps the model ids in
config.py alone, and lets Phase 7 swap either tier without touching call sites.
"""

from __future__ import annotations

from enum import Enum

from app.config import Settings, get_settings


class Task(str, Enum):
    """Named call sites. Add a member rather than passing a raw model name."""

    # --- fast tier: internal, structured, not read by a human ---
    CLASSIFY = "classify"
    ROUTE = "route"
    GROUNDING_CHECK = "grounding_check"   # Phase 5 seam
    SELF_CHECK = "self_check"             # Phase 5 seam
    SUMMARIZE_TOOL_OUTPUT = "summarize_tool_output"

    # --- smart tier: reasoning and user-visible prose ---
    AGENT = "agent"
    SYNTHESIS = "synthesis"


FAST_TASKS: frozenset[Task] = frozenset({
    Task.CLASSIFY,
    Task.ROUTE,
    Task.GROUNDING_CHECK,
    Task.SELF_CHECK,
    Task.SUMMARIZE_TOOL_OUTPUT,
})

SMART_TASKS: frozenset[Task] = frozenset({Task.AGENT, Task.SYNTHESIS})


class Tier(str, Enum):
    FAST = "fast"
    SMART = "smart"


def tier_for(task: Task | str) -> Tier:
    """Fast unless the task is explicitly reasoning or synthesis.

    Unknown strings resolve to FAST deliberately: an unregistered call site
    costing 15x more is a worse failure than one being slightly dumber, and the
    ValueError below only fires for values that are not Task members at all.
    """
    task = Task(task)
    return Tier.SMART if task in SMART_TASKS else Tier.FAST


def pick_model(task: Task | str, settings: Settings | None = None) -> str:
    """Return the OpenRouter model id for this task."""
    s = settings or get_settings()
    return s.llm_smart_model if tier_for(task) is Tier.SMART else s.llm_fast_model


def fallback_model(task: Task | str, settings: Settings | None = None) -> str | None:
    """The model to retry with when `pick_model(task)` errors at the provider.

    Smart-tier calls fall back to the fast tier -- a cheaper answer beats a 502.
    Fast-tier calls have nowhere cheaper to go, so they return None and the
    caller re-raises rather than silently retrying the same failing model.
    """
    s = settings or get_settings()
    return s.llm_fast_model if tier_for(task) is Tier.SMART else None
