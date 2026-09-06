"""Langfuse tracing, wrapped so that it can fail without taking a request with it.

**The one invariant: tracing never changes the outcome of a request.** Langfuse
Cloud is a third-party HTTP dependency on the hot path of every answer, and the
failure modes are real -- an expired key, a DNS blip, a slow export. So every
entry point here is wrapped: constructing the client, opening a span, updating
one, resolving the trace URL, flushing. If any of it raises, the wrapper falls
back to `_NullSpan` and the request proceeds exactly as if tracing were off.
`tests/test_observability.py::TestTracingNeverBreaksARequest` pins that with a
client that raises on every method, including one that raises on `__init__`.

Note what is *not* wrapped: the caller's own `with` body. A `try/except` around
the yield would swallow the application's exceptions, which is the opposite of
what an observability layer should do -- it would hide the very failure it
exists to record. `_safe_span` re-raises the body's exception after marking the
span as errored.

**Trace ids are ours, not Langfuse's.** `/chat` mints a `uuid4().hex`, which is
already a valid 32-hex W3C trace id, and passes it in as the trace context. The
`trace_id` in the API response, the id in the Langfuse UI and the id in the
session transcript are therefore the same string, so a support question ("what
happened on request X?") is answerable without a join table.

**The trace URL is built from a cached project id.** `Langfuse.get_trace_url`
does an API round-trip the first time it is called, and the first time would
otherwise be inside somebody's request. `resolve_trace_url_template()` is called
during API warmup, off the loop; after that `trace_url()` is pure string
formatting and costs nothing. Unresolved, it returns None and the UI simply
shows no link -- which is the correct degraded behaviour.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# Langfuse's own observation types. `span` is the default; the rest render with
# distinct icons in the UI, which is the whole reason to bother naming them.
GENERATION = "generation"
TOOL = "tool"
RETRIEVER = "retriever"
GUARDRAIL = "guardrail"
AGENT = "agent"

# OpenRouter list prices, USD per 1M tokens, as of the Phase 7 run. Langfuse
# prices known model ids itself, but it does not know OpenRouter's `openai/`
# prefixes, so the cost that appears on a generation is computed here and sent
# explicitly. Wrong-but-visible beats absent: these are the numbers the tiering
# table in the Phase 7 report is built from, and they are one edit from current.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def cost_usd(model: str | None, prompt_tokens: int | None,
             completion_tokens: int | None) -> dict[str, float] | None:
    """Cost split into input/output/total, or None for an unpriced model."""
    if not model:
        return None
    price = MODEL_PRICES_USD_PER_MTOK.get(model)
    if price is None:
        price = MODEL_PRICES_USD_PER_MTOK.get(model.split("/")[-1])
    if price is None:
        return None
    pin, pout = price
    cin = (prompt_tokens or 0) / 1_000_000 * pin
    cout = (completion_tokens or 0) / 1_000_000 * pout
    return {"input": round(cin, 8), "output": round(cout, 8),
            "total": round(cin + cout, 8)}


# ------------------------------------------------------------------ handles

class _NullSpan:
    """What every failure degrades to. Accepts everything, records nothing."""

    id = None
    trace_id = None
    enabled = False

    def update(self, **_: Any) -> "_NullSpan":
        return self

    def event(self, **_: Any) -> "_NullSpan":
        return self

    def score(self, **_: Any) -> "_NullSpan":
        return self

    def end(self, **_: Any) -> "_NullSpan":
        return self


class _Span:
    """Thin adapter over a Langfuse span object. Swallows SDK errors only."""

    enabled = True

    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def id(self) -> str | None:
        return getattr(self._inner, "id", None)

    @property
    def trace_id(self) -> str | None:
        return getattr(self._inner, "trace_id", None)

    def update(self, **kwargs: Any) -> "_Span":
        try:
            self._inner.update(**kwargs)
        except Exception:  # noqa: BLE001
            log.debug("langfuse span.update failed", exc_info=True)
        return self

    def event(self, **kwargs: Any) -> "_Span":
        try:
            self._inner.create_event(**kwargs)
        except Exception:  # noqa: BLE001
            log.debug("langfuse span.event failed", exc_info=True)
        return self

    def score(self, **kwargs: Any) -> "_Span":
        try:
            self._inner.score(**kwargs)
        except Exception:  # noqa: BLE001
            log.debug("langfuse span.score failed", exc_info=True)
        return self

    def end(self, **kwargs: Any) -> "_Span":
        try:
            self._inner.end(**kwargs)
        except Exception:  # noqa: BLE001
            log.debug("langfuse span.end failed", exc_info=True)
        return self


NULL_SPAN = _NullSpan()


# ------------------------------------------------------------------ tracing

class Tracing:
    """The app's whole view of Langfuse. Construct once, use from anywhere."""

    def __init__(self, settings: Settings | None = None, *, client: Any = None,
                 enabled: bool | None = None):
        self.settings = settings or get_settings()
        self._client = client
        self._explicit_client = client is not None
        self._url_template: str | None = None
        self._init_failed = False
        s = self.settings
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = bool(
                s.langfuse_enabled and (client is not None
                                        or (s.langfuse_public_key and s.langfuse_secret_key)))

    # -- client -----------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is not None or self._init_failed or not self.enabled:
            return self._client
        try:
            from langfuse import Langfuse

            s = self.settings
            # Passed explicitly rather than left to the SDK's env lookup: config
            # comes from `Settings` everywhere else in this app, and an env var
            # that only Langfuse reads is a second source of truth.
            self._client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
                timeout=s.langfuse_timeout,
                flush_at=s.langfuse_flush_at,
                flush_interval=s.langfuse_flush_interval,
                environment=s.app_env,
                tracing_enabled=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Once, not per request: a broken SDK install would otherwise pay
            # the construction cost and log a traceback on every single call.
            self._init_failed = True
            self.enabled = False
            log.warning("langfuse disabled: %s: %s", type(exc).__name__, exc)
        return self._client

    # -- spans ------------------------------------------------------------
    @contextmanager
    def _safe_span(self, **kwargs: Any) -> Iterator[Any]:
        client = self.client
        if client is None:
            yield NULL_SPAN
            return
        cm = None
        try:
            cm = client.start_as_current_observation(**kwargs)
            inner = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            log.debug("langfuse span open failed: %s", exc, exc_info=True)
            yield NULL_SPAN
            return
        span = _Span(inner)
        try:
            yield span
        except Exception as exc:
            # Mark, then RE-RAISE. Recording a failure is not the same as
            # absorbing one, and the caller's error must reach the caller.
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:  # noqa: BLE001
                pass
            raise
        else:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                log.debug("langfuse span close failed", exc_info=True)

    def trace(self, name: str, *, trace_id: str | None = None,
              input: Any = None, metadata: dict | None = None,
              as_type: str = "span", session_id: str | None = None,
              user_id: str | None = None) -> Any:
        """Open the ROOT observation of a request, bound to our own trace id."""
        kwargs: dict[str, Any] = {"name": name, "as_type": as_type,
                                  "input": input, "metadata": metadata}
        if trace_id:
            try:
                from langfuse.types import TraceContext

                kwargs["trace_context"] = TraceContext(trace_id=trace_id)
            except Exception:  # noqa: BLE001
                log.debug("langfuse trace context unavailable", exc_info=True)
        return self._safe_span(**kwargs)

    def span(self, name: str, *, as_type: str = "span", input: Any = None,
             metadata: dict | None = None, **kwargs: Any) -> Any:
        return self._safe_span(name=name, as_type=as_type, input=input,
                               metadata=metadata, **kwargs)

    def generation(self, name: str, *, model: str | None = None,
                   input: Any = None, metadata: dict | None = None,
                   model_parameters: dict | None = None) -> Any:
        """A span Langfuse renders as an LLM call, so tokens and cost roll up."""
        return self._safe_span(name=name, as_type=GENERATION, model=model,
                               input=input, metadata=metadata,
                               model_parameters=model_parameters)

    def record_generation(self, span: Any, *, model: str | None,
                          prompt_tokens: int | None, completion_tokens: int | None,
                          output: Any = None, task: str | None = None,
                          tier: str | None = None, extra: dict | None = None) -> None:
        """Attach usage and cost to an open generation span.

        `usage_details` and `cost_details` are the fields Langfuse aggregates on,
        so the per-tier cost table in the Phase 7 report is a UI query rather
        than something reconstructed from logs.
        """
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        usage = {"input": prompt_tokens or 0, "output": completion_tokens or 0,
                 "total": total}
        meta = {"task": task, "tier": tier, **(extra or {})}
        payload: dict[str, Any] = {"output": output, "usage_details": usage,
                                   "metadata": {k: v for k, v in meta.items()
                                                if v is not None}}
        cost = cost_usd(model, prompt_tokens, completion_tokens)
        if cost:
            payload["cost_details"] = cost
        span.update(**payload)

    # -- ids and urls -----------------------------------------------------
    def resolve_trace_url_template(self) -> str | None:
        """Fetch the project id ONCE, off the request path. Called at warmup."""
        if self._url_template is not None or not self.enabled:
            return self._url_template
        client = self.client
        if client is None:
            return None
        try:
            url = client.get_trace_url(trace_id="0" * 32)
            if url:
                self._url_template = url.replace("0" * 32, "{trace_id}")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not resolve langfuse trace url: %s: %s",
                        type(exc).__name__, exc)
        return self._url_template

    def trace_url(self, trace_id: str | None) -> str | None:
        """Pure string formatting once the template is resolved. Never blocks."""
        if not trace_id or not self._url_template:
            return None
        return self._url_template.format(trace_id=trace_id)

    def flush(self) -> None:
        """Force the batch out. Langfuse batches, so a short-lived process
        (a test, an eval run, a container being stopped) loses its spans without
        this. Called from the API's lifespan shutdown."""
        client = self._client
        if client is None:
            return
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            log.debug("langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001
            log.debug("langfuse shutdown failed", exc_info=True)


_DEFAULT: Tracing | None = None


def get_tracing(settings: Settings | None = None) -> Tracing:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Tracing(settings=settings)
    return _DEFAULT


def set_tracing(tracing: Tracing | None) -> None:
    """Install a specific instance (or None to reset). Tests and warmup only."""
    global _DEFAULT
    _DEFAULT = tracing


def tracing_disabled() -> Tracing:
    """An instance that is off. Used by tests that must not touch the network."""
    return Tracing(enabled=False)


def _env_flag(name: str) -> bool:  # pragma: no cover - trivial
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}
