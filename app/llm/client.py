"""OpenRouter chat models, via langchain-openai's OpenAI-compatible client.

OpenRouter speaks the OpenAI wire protocol, so ChatOpenAI with a different
base_url is the whole integration -- no provider shim, no adapter layer.

Non-streaming on purpose: Phase 5 wraps the agent in NeMo Guardrails, whose
output rails must see a complete answer before it reaches the user. Streaming
tokens past a grounding check would make the rail decorative.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.llm.tiering import Task, fallback_model, pick_model

log = logging.getLogger(__name__)

# OpenRouter attributes traffic with these; harmless elsewhere.
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/niti007/knowledge-graph-version-2",
    "X-Title": "ACME Enterprise Knowledge Assistant",
}


def build_chat_model(model: str, settings: Settings | None = None,
                     temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    """A ChatOpenAI pointed at OpenRouter. One place that knows the wiring."""
    from langchain_openai import ChatOpenAI

    s = settings or get_settings()
    return ChatOpenAI(
        model=model,
        api_key=s.openrouter_api_key or "missing",
        base_url=s.openrouter_base_url,
        temperature=temperature,
        streaming=False,
        timeout=60,
        max_retries=2,
        default_headers=DEFAULT_HEADERS,
        **kwargs,
    )


@lru_cache(maxsize=8)
def _cached_model(model: str, temperature: float) -> BaseChatModel:
    return build_chat_model(model, temperature=temperature)


def get_chat_model(task: Task | str = Task.AGENT, temperature: float = 0.0,
                   settings: Settings | None = None, **kwargs: Any) -> BaseChatModel:
    """The model for `task`, per app.llm.tiering. Cached when built from config."""
    model = pick_model(task, settings)
    if settings is None and not kwargs:
        return _cached_model(model, temperature)
    return build_chat_model(model, settings=settings, temperature=temperature, **kwargs)


def invoke_with_fallback(messages, task: Task | str = Task.AGENT,
                         tools: list | None = None,
                         settings: Settings | None = None,
                         temperature: float = 0.0):
    """Invoke the tier for `task`; on a provider error retry once on the fast tier.

    Only the *provider* call is retried. A tool-schema error or a bad message
    list will fail identically on the fallback model, so this catches the
    transient case (502/timeout/model unavailable) and lets everything else
    surface. Fast-tier tasks have no fallback and re-raise immediately.
    """
    def _run(model_name: str):
        llm = get_chat_model(task, temperature=temperature, settings=settings) \
            if model_name == pick_model(task, settings) \
            else build_chat_model(model_name, settings=settings, temperature=temperature)
        if tools:
            llm = llm.bind_tools(tools)
        return llm.invoke(messages)

    primary = pick_model(task, settings)
    try:
        return _run(primary)
    except Exception as exc:  # noqa: BLE001 - provider errors are not a fixed type
        alt = fallback_model(task, settings)
        if alt is None or alt == primary:
            raise
        log.warning("LLM call on %s failed (%s); falling back to %s",
                    primary, type(exc).__name__, alt)
        return _run(alt)
