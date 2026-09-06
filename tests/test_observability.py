"""Phase 7 tracing tests. No tokens, no network, no Langfuse account needed.

The suite is built around one claim, and it is the claim that matters for an
observability layer sitting on the hot path: **tracing can fail in any way it
likes and the request still succeeds.** So the fakes here are not "a Langfuse
that works" -- they are a Langfuse that raises on construction, on opening a
span, on updating one, on closing one, and on flushing. Every one of those must
be invisible to the caller.

The second claim is narrower and easy to get backwards: an exception raised by
the *application* inside a traced block must still reach the application. A
`try/except` in the wrong place would turn the observability layer into a
silent swallower of the very bugs it exists to surface.
"""

from __future__ import annotations

import pytest

from app.observability.langfuse_client import (
    NULL_SPAN,
    Tracing,
    cost_usd,
    get_tracing,
    set_tracing,
    tracing_disabled,
)


# ------------------------------------------------------------------- fakes

class FakeSpan:
    def __init__(self, name, recorder):
        self.name = name
        self.id = f"span-{name}"
        self.trace_id = recorder.trace_id
        self.updates = []
        self._recorder = recorder

    def update(self, **kw):
        self.updates.append(kw)
        self._recorder.updates.append((self.name, kw))


class _Ctx:
    def __init__(self, span, recorder):
        self.span, self._recorder = span, recorder

    def __enter__(self):
        self._recorder.open.append(self.span.name)
        return self.span

    def __exit__(self, *exc):
        self._recorder.closed.append(self.span.name)
        return False


class FakeLangfuse:
    """Records the tree instead of shipping it."""

    def __init__(self, trace_id="deadbeef" * 4):
        self.trace_id = trace_id
        self.spans: list[FakeSpan] = []
        self.open: list[str] = []
        self.closed: list[str] = []
        self.updates: list[tuple[str, dict]] = []
        self.flushed = 0
        self.shutdowns = 0
        self.project_calls = 0

    def start_as_current_observation(self, *, name, **kw):
        span = FakeSpan(name, self)
        span.kwargs = kw
        self.spans.append(span)
        return _Ctx(span, self)

    def get_trace_url(self, *, trace_id=None):
        self.project_calls += 1
        return f"https://cloud.langfuse.com/project/proj-1/traces/{trace_id}"

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.shutdowns += 1

    @property
    def names(self):
        return [s.name for s in self.spans]


class ExplodingLangfuse:
    """Fails at every seam. Nothing here may reach a caller."""

    def start_as_current_observation(self, **kw):
        raise RuntimeError("langfuse unreachable")

    def get_trace_url(self, **kw):
        raise RuntimeError("langfuse unreachable")

    def flush(self):
        raise RuntimeError("langfuse unreachable")

    def shutdown(self):
        raise RuntimeError("langfuse unreachable")


class HalfBrokenLangfuse(FakeLangfuse):
    """Opens spans fine, then fails on update and on close."""

    def start_as_current_observation(self, *, name, **kw):
        span = FakeSpan(name, self)

        def boom(**_):
            raise RuntimeError("export failed")

        span.update = boom
        self.spans.append(span)

        class Ctx:
            def __enter__(_self):
                return span

            def __exit__(_self, *exc):
                raise RuntimeError("close failed")

        return Ctx()


def _trace_id_of(ctx):
    """langfuse's TraceContext is a TypedDict; be tolerant of either shape."""
    if isinstance(ctx, dict):
        return ctx.get("trace_id")
    return getattr(ctx, "trace_id", None)


@pytest.fixture
def fake():
    return FakeLangfuse()


@pytest.fixture
def tracing(fake):
    return Tracing(client=fake, enabled=True)


@pytest.fixture(autouse=True)
def _no_global_tracing():
    """Never let a test leave a live tracer installed for the next one.

    Restores the DISABLED tracer rather than clearing the slot: clearing it
    would let the next `get_tracing()` build a credentialed client, which is
    exactly the network call `tests/conftest.py` exists to prevent.
    """
    set_tracing(tracing_disabled())
    yield
    set_tracing(tracing_disabled())


# ------------------------------------------------------------ the invariant

class TestTracingNeverBreaksARequest:

    @pytest.mark.parametrize("client", [ExplodingLangfuse(), HalfBrokenLangfuse()])
    def test_a_broken_langfuse_does_not_stop_the_work(self, client):
        t = Tracing(client=client, enabled=True)
        done = []
        with t.span("work") as sp:
            sp.update(output="anything")
            done.append("ran")
        assert done == ["ran"]

    def test_a_client_that_raises_on_construction_disables_tracing_once(self, monkeypatch):
        import app.observability.langfuse_client as mod

        calls = []

        class Boom:
            def __init__(self, **kw):
                calls.append(kw)
                raise RuntimeError("bad credentials")

        monkeypatch.setitem(__import__("sys").modules, "langfuse",
                            type("M", (), {"Langfuse": Boom}))
        t = Tracing(enabled=True)
        t.settings = t.settings.model_copy(update={
            "langfuse_public_key": "pk-lf-x", "langfuse_secret_key": "sk-lf-x"})
        with t.span("work") as sp:
            sp.update(output="x")
        with t.span("work2"):
            pass
        # Constructed ONCE, then given up on. A retry per request would pay the
        # failure cost on every single call and log a traceback each time.
        assert len(calls) == 1
        assert t.enabled is False

    def test_a_body_exception_is_recorded_and_RE_RAISED(self, tracing, fake):
        """Marking a failure is not the same as absorbing one."""
        with pytest.raises(ValueError, match="the real bug"):
            with tracing.span("work"):
                raise ValueError("the real bug")
        levels = [kw.get("level") for _, kw in fake.updates]
        assert "ERROR" in levels

    def test_a_disabled_tracer_yields_the_null_span_and_touches_nothing(self):
        t = tracing_disabled()
        with t.span("work") as sp:
            assert sp is NULL_SPAN
            sp.update(output="x").event(name="y").end()
        assert t.client is None

    def test_flush_and_shutdown_swallow_failures(self):
        t = Tracing(client=ExplodingLangfuse(), enabled=True)
        t.flush()
        t.shutdown()  # no raise is the assertion


# ------------------------------------------------------------------ shape

class TestSpanShape:

    def test_the_root_span_is_bound_to_our_own_trace_id(self, tracing, fake):
        with tracing.trace("chat", trace_id="a" * 32, input="hello"):
            pass
        # TraceContext is a TypedDict in langfuse 4.x, so read it as a mapping.
        ctx = fake.spans[0].kwargs.get("trace_context")
        assert ctx is not None and _trace_id_of(ctx) == "a" * 32

    def test_spans_open_and_close_in_order(self, tracing, fake):
        with tracing.trace("chat", trace_id="b" * 32):
            with tracing.span("guardrails"):
                with tracing.span("agent"):
                    pass
        assert fake.open == ["chat", "guardrails", "agent"]
        assert fake.closed == ["agent", "guardrails", "chat"]

    def test_a_generation_carries_model_usage_and_cost(self, tracing, fake):
        with tracing.generation("llm.agent.turn_1", model="openai/gpt-4o") as sp:
            tracing.record_generation(sp, model="openai/gpt-4o",
                                      prompt_tokens=1000, completion_tokens=500,
                                      output="hi", task="agent", tier="smart")
        (_, kw), = [u for u in fake.updates if "usage_details" in u[1]]
        assert kw["usage_details"] == {"input": 1000, "output": 500, "total": 1500}
        assert kw["cost_details"]["total"] == pytest.approx(0.0025 + 0.005)
        assert kw["metadata"]["tier"] == "smart"

    def test_an_unpriced_model_gets_usage_but_no_invented_cost(self, tracing, fake):
        with tracing.generation("llm.x", model="acme/mystery-7b") as sp:
            tracing.record_generation(sp, model="acme/mystery-7b",
                                      prompt_tokens=10, completion_tokens=2)
        (_, kw), = [u for u in fake.updates if "usage_details" in u[1]]
        assert "cost_details" not in kw


class TestCostTable:

    def test_the_two_tiers_are_priced_and_mini_is_far_cheaper(self):
        smart = cost_usd("openai/gpt-4o", 1_000_000, 1_000_000)["total"]
        fast = cost_usd("openai/gpt-4o-mini", 1_000_000, 1_000_000)["total"]
        assert smart == pytest.approx(12.50)
        assert fast == pytest.approx(0.75)
        # The tiering claim in one assertion: the fast tier is >15x cheaper, so
        # routing internal calls to it is a real lever and not a rounding error.
        assert smart / fast > 15

    def test_a_bare_model_id_prices_the_same_as_the_openrouter_prefixed_one(self):
        assert cost_usd("gpt-4o-mini", 1000, 1000) == cost_usd(
            "openai/gpt-4o-mini", 1000, 1000)

    def test_an_unknown_model_returns_none_rather_than_zero(self):
        # Zero would show up in the UI as a free call, which is a lie a cost
        # dashboard should never be told.
        assert cost_usd("who/knows", 100, 100) is None
        assert cost_usd(None, 100, 100) is None


class TestTraceUrl:

    def test_the_project_id_is_fetched_once_then_the_url_is_pure_formatting(
            self, tracing, fake):
        assert tracing.resolve_trace_url_template() is not None
        url = tracing.trace_url("c" * 32)
        tracing.trace_url("d" * 32)
        assert url == f"https://cloud.langfuse.com/project/proj-1/traces/{'c' * 32}"
        assert fake.project_calls == 1, "the trace url hit the network per request"

    def test_no_url_before_the_template_is_resolved(self, tracing):
        assert tracing.trace_url("e" * 32) is None

    def test_an_unreachable_langfuse_yields_no_url_rather_than_an_error(self):
        t = Tracing(client=ExplodingLangfuse(), enabled=True)
        assert t.resolve_trace_url_template() is None
        assert t.trace_url("f" * 32) is None


# ------------------------------------------------------------ api wiring

def _app(**kw):
    from app.api.main import create_app

    from tests.test_api import StubRails  # reuse Phase 6's stub

    return create_app(guardrails=StubRails(), probes=[], warm=False, **kw)


class TestApiTracingWiring:
    """The response contract, driven through the real FastAPI app."""

    def test_trace_url_is_populated_when_tracing_is_up(self, fake):
        from fastapi.testclient import TestClient

        t = Tracing(client=fake, enabled=True)
        t.resolve_trace_url_template()
        with TestClient(_app(tracing=t)) as client:
            body = client.post("/chat", json={"question": "Who owns Auth-DB?"}).json()
        assert body["trace_url"].endswith(body["trace_id"])
        assert "chat" in fake.names

    def test_trace_url_is_null_and_the_request_still_answers_when_tracing_is_down(self):
        from fastapi.testclient import TestClient

        t = Tracing(client=ExplodingLangfuse(), enabled=True)
        with TestClient(_app(tracing=t)) as client:
            r = client.post("/chat", json={"question": "Who owns Auth-DB?"})
        assert r.status_code == 200
        assert r.json()["trace_url"] is None
        assert r.json()["answer"]

    def test_the_response_trace_id_is_the_langfuse_trace_id(self, fake):
        from fastapi.testclient import TestClient

        t = Tracing(client=fake, enabled=True)
        with TestClient(_app(tracing=t)) as client:
            body = client.post("/chat", json={"question": "Who owns Auth-DB?"}).json()
        root = fake.spans[0]
        assert _trace_id_of(root.kwargs["trace_context"]) == body["trace_id"]

    def test_shutdown_flushes_so_a_short_lived_process_still_traces(self, fake):
        from fastapi.testclient import TestClient

        t = Tracing(client=fake, enabled=True)
        with TestClient(_app(tracing=t)) as client:
            client.post("/chat", json={"question": "Who owns Auth-DB?"})
        assert fake.flushed >= 1 and fake.shutdowns >= 1


class TestMetaQuestionsBypassTheCache:
    """System-card answers are ~0.1ms already; a cache would only slow them down.

    The assertion is on the cache object: it was never asked anything. Checking
    `cached is False` would pass even if the lookup ran and missed, which is the
    thing this is meant to rule out.
    """

    def test_a_meta_question_never_reaches_the_cache_or_the_rails(self):
        from fastapi.testclient import TestClient

        from app.api.main import create_app
        from tests.test_api import StubRails

        class ExplodingCache:
            enabled = True
            threshold = 0.97

            def lookup(self, *a, **k):  # pragma: no cover - must not run
                raise AssertionError("a meta question consulted the cache")

            def store(self, *a, **k):  # pragma: no cover - must not run
                raise AssertionError("a meta question was written to the cache")

        rails = StubRails()
        rails.cache = ExplodingCache()
        app = create_app(guardrails=rails, probes=[], warm=False,
                         tracing=tracing_disabled())
        with TestClient(app) as client:
            body = client.post("/chat", json={"question": "What can you do?"}).json()
        assert body["route"] == "system_card"
        assert body["cached"] is False
        assert body["citations"] == []

    def test_a_corpus_question_does_not_take_the_system_card_route(self):
        from fastapi.testclient import TestClient

        with TestClient(_app(tracing=tracing_disabled())) as client:
            body = client.post("/chat",
                               json={"question": "Who owns Auth-DB?"}).json()
        assert body["route"] == "guardrails"


class TestGlobalTracer:

    def test_get_tracing_is_a_singleton_and_set_tracing_replaces_it(self):
        # `Tracing.__init__` builds no client -- the SDK is constructed lazily
        # on first span -- so clearing the slot here costs no network call.
        set_tracing(None)
        a = get_tracing()
        assert get_tracing() is a
        replacement = tracing_disabled()
        set_tracing(replacement)
        assert get_tracing() is replacement
