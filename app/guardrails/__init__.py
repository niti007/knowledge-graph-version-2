"""NeMo Guardrails wrapped around the LangGraph agent.

Public surface is deliberately small -- Phase 6 needs a runner and a response
type, and nothing else:

    from app.guardrails import get_guardrails, run_guarded, GuardrailedResponse
"""

from app.guardrails.runner import (  # noqa: F401
    CONFIG_PATH,
    GuardrailedResponse,
    Guardrails,
    get_guardrails,
    run_guarded,
)

__all__ = ["Guardrails", "GuardrailedResponse", "get_guardrails", "run_guarded",
           "CONFIG_PATH"]
