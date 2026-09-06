"""Streamlit demo client. Deliberately thin.

This file speaks HTTP to the API and nothing else. It does not import the agent,
the retrieval layer or the guardrails, and that is a boundary worth stating: the
moment a UI can call `run_agent` directly, there is a second code path that
skips the rails, and the rails stop being a property of the system and become a
property of one caller. Everything here goes through `POST /chat`.

Sessions are held by the API in memory, so **restarting the API clears every
conversation** -- the sidebar says so, and a stale `session_id` simply starts a
fresh session rather than erroring.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("UI_TIMEOUT", "120"))

st.set_page_config(page_title="ACME Knowledge Assistant", page_icon="📚",
                   layout="wide")


# ------------------------------------------------------------------ client

def post_chat(question: str, session_id: str | None) -> tuple[int, dict]:
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    try:
        r = httpx.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
        return r.status_code, (r.json() if r.content else {})
    except httpx.HTTPError as exc:
        return 0, {"detail": f"could not reach the API at {API_URL}: {exc}"}


def get_health() -> tuple[int, dict]:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=20)
        return r.status_code, r.json()
    except httpx.HTTPError as exc:
        return 0, {"detail": str(exc)}


# ------------------------------------------------------------- rendering

def render_citations(citations: list[dict]) -> None:
    """Document and graph citations are different claims and read differently.

    A document citation points at text a human can open. A graph citation points
    at a traversal -- a named Cypher template and the path it walked -- which is
    evidence of a different kind, and collapsing the two into one list of blue
    links would misrepresent what the second one is.
    """
    if not citations:
        st.caption("No citations - this answer claims no corpus support.")
        return
    docs = [c for c in citations if c.get("doc_id")]
    graph = [c for c in citations if not c.get("doc_id") and c.get("graph_template")]
    other = [c for c in citations if c not in docs and c not in graph]

    if docs:
        st.markdown("**Document sources**")
        for c in docs:
            score = c.get("rerank_score")
            bits = [f"`{c['doc_id']}`", c.get("title") or c.get("citation", "")]
            if c.get("chunk_id"):
                bits.append(f"chunk `{c['chunk_id']}`")
            if score is not None:
                bits.append(f"rerank {score:.3f}")
            st.markdown("- " + " · ".join(str(b) for b in bits if b))
    if graph:
        st.markdown("**Graph sources**")
        for c in graph:
            st.markdown(
                f"- template `{c.get('graph_template')}` · path "
                f"`{c.get('graph_path') or '-'}`")
    if other:
        st.markdown("**Other sources**")
        for c in other:
            st.markdown(f"- {c.get('citation')}")


def render_rails(data: dict) -> None:
    fired = data.get("rails_fired") or []
    if data.get("blocked"):
        st.error(
            f"Blocked by **{data.get('blocked_by')}** "
            f"at the *{data.get('blocked_stage') or 'unknown'}* stage.")
    elif fired:
        st.info("Rails fired (non-blocking): " + ", ".join(fired))
    else:
        st.caption("No rail fired.")
    if data.get("pii_masked"):
        st.warning("PII was detected in the question and masked before retrieval.")
    with st.expander("Rail detail"):
        st.json({"route": data.get("route"),
                 "rails": data.get("rails"),
                 "grounding": data.get("grounding"),
                 "provenance": data.get("provenance")})


def render_footer(data: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Latency", f"{data.get('latency_ms', 0)/1000:.2f}s")
    cols[1].metric("Cached", "yes" if data.get("cached") else "no")
    cols[2].metric("Tools", ", ".join(data.get("tools_used") or []) or "-")
    cols[3].metric("Route", data.get("route", "-"))
    trace_id = data.get("trace_id")
    trace_url = data.get("trace_url")
    if trace_url:
        st.markdown(f"[Open trace in Langfuse]({trace_url})")
    elif trace_id:
        # The slot is wired; Phase 7 fills trace_url in when Langfuse is on.
        st.caption(f"Trace `{trace_id}` — Langfuse link appears here once "
                   "tracing is enabled (Phase 7).")


# ----------------------------------------------------------------- layout

if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

with st.sidebar:
    st.header("ACME Knowledge Assistant")
    st.caption(f"API: `{API_URL}`")
    code, health = get_health()
    if code == 200:
        st.success("Ready")
    elif code == 503:
        st.warning(f"Not ready: {health.get('status')}")
    else:
        st.error("API unreachable")
    with st.expander("Health detail"):
        st.json(health)
    st.divider()
    # A placeholder, not a plain caption: the session id is only known after
    # /chat replies, which happens further down the script. Writing into a slot
    # reserved here keeps the sidebar correct on the same run instead of one
    # interaction behind.
    session_slot = st.empty()
    session_slot.caption(f"Session: `{st.session_state.session_id or 'new'}`")
    if st.button("New session"):
        st.session_state.history = []
        st.session_state.session_id = None
        st.rerun()
    st.caption("Sessions live in the API's memory. Restarting the API clears "
               "them; a stale id just starts a new one.")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("data"):
            render_citations(turn["data"].get("citations") or [])
            render_rails(turn["data"])
            render_footer(turn["data"])

question = st.chat_input("Ask about ACME's policies, SOPs, incidents or systems")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            status, data = post_chat(question, st.session_state.session_id)
        if status != 200:
            msg = f"Request failed ({status}): {data.get('detail') or data}"
            st.error(msg)
            st.session_state.history.append({"role": "assistant", "content": msg})
        else:
            st.session_state.session_id = data.get("session_id")
            st.markdown(data.get("answer") or "*(empty answer)*")
            render_citations(data.get("citations") or [])
            render_rails(data)
            render_footer(data)
            st.session_state.history.append(
                {"role": "assistant", "content": data.get("answer") or "",
                 "data": data})
            session_slot.caption(f"Session: `{st.session_state.session_id}`")
