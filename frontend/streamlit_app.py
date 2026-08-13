"""OmniCare customer assistant — Streamlit chat UI.

Kept deliberately plain. The brief values working features over UI, and every
minute spent on CSS here is a minute not spent on the agent. What it does do
carefully:

* **Shows its work.** Citations and tool calls sit under each answer, so a
  reader can see the answer came from a named policy section and which backend
  operations ran — the two things a reviewer most wants to verify.
* **Fails with directions.** Every error state says what happened and what to
  do about it, naming the environment variable or command where relevant. A
  spinner that ends in "Error" teaches nobody anything.
* **Keeps a real session.** ``user_id`` is a per-session UUID sent with every
  request; the backend keys conversation history off it. "Start over" mints a
  new one, which is what actually clears the server-side history.
"""

from __future__ import annotations

import uuid

import streamlit as st

from api_client import BACKEND_URL, STATUS_OK, STATUS_STARTING, ask, backend_status

SAMPLE_QUESTIONS = [
    "Is water damage from a burst pipe covered?",
    "Am I covered if my basement floods?",
    "What's the status of claim CLM-8821?",
    "I need to file a water damage claim for $4,200 on POL-1092.",
]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _escape_currency(text: str) -> str:
    """Stop Markdown reading a pair of dollar signs as LaTeX.

    "$25,000 with a $500 deductible" is valid Markdown for an inline equation,
    so the amounts render as italic maths instead of currency. Escaping the
    dollar sign keeps policy figures looking like money.
    """
    return text.replace("$", r"\$")

def render_turn(turn: dict) -> None:
    """Draw one assistant turn: the answer, then how it was produced."""
    st.markdown(_escape_currency(turn["content"]))

    sources = turn.get("sources") or []
    tool_calls = turn.get("tool_calls") or []
    if not sources and not tool_calls:
        return

    label = "Sources and actions" if sources else "Actions"
    with st.expander(label, expanded=bool(sources)):
        if sources:
            st.caption("Policy sections this answer came from")
            for source in sources:
                st.markdown(f"- `{source}`")
        if tool_calls:
            if sources:
                st.divider()
            st.caption("Backend operations")
            for call in tool_calls:
                mark = "✅" if call.get("ok") else "⚠️"
                summary = call.get("summary") or ""
                st.markdown(f"{mark} **{call.get('name')}** — {summary}")
                if call.get("arguments"):
                    st.json(call["arguments"], expanded=False)


def submit(prompt: str) -> None:
    """Append the user turn, call the backend, append the answer."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    body = ask(st.session_state.user_id, prompt)

    if "error" in body:
        st.session_state.messages.append(
            {"role": "assistant", "content": f":warning: {body['error']}"}
        )
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": body.get("response", ""),
            "sources": body.get("sources", []),
            "tool_calls": body.get("tool_calls", []),
        }
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="OmniCare Assistant", page_icon="🏠", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"usr_{uuid.uuid4().hex[:8]}"
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.subheader("Session")
    st.caption(f"User ID `{st.session_state.user_id}`")
    st.caption("Sent with every request. The backend keys your history off it.")

    if st.button("Start over", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_id = f"usr_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.divider()
    st.subheader("Backend")
    state, detail = backend_status()
    if state == STATUS_OK:
        st.success("Connected")
    elif state == STATUS_STARTING:
        st.warning(detail)
    else:
        st.error(detail)
    st.caption(f"`{BACKEND_URL}`")

st.title("OmniCare Financial")
st.caption("Check your coverage, look up a claim, or file a new one.")

if not st.session_state.messages:
    st.write("Try one of these:")
    for index, question in enumerate(SAMPLE_QUESTIONS):
        if st.button(question, key=f"sample_{index}", use_container_width=True):
            st.session_state.pending = question
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_turn(message)
        else:
            st.markdown(message["content"])

prompt = st.chat_input("Ask about your coverage or a claim")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Checking your policy…"):
        submit(prompt)
    st.rerun()
