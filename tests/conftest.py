"""Suite-wide guarantees. The important one: no test talks to Langfuse.

Phase 7 put a `Tracing` instance inside `create_app`, and its default is the
real, credentialed client -- which is correct for the service and wrong for a
test suite. Left alone, `tests/test_api.py` opened a live exporter, and the
Langfuse shutdown at the end of each `TestClient` context blocked on the
network; the suite did not fail, it hung, which is the worse outcome because it
looks like a slow machine rather than a bug.

So a disabled tracer is installed before every test. A test that wants to
observe tracing installs its own (see `tests/test_observability.py`) and this
fixture puts the disabled one back afterwards.
"""

from __future__ import annotations

import pytest

from app.observability.langfuse_client import set_tracing, tracing_disabled


@pytest.fixture(autouse=True)
def _tracing_off():
    set_tracing(tracing_disabled())
    yield
    set_tracing(tracing_disabled())
