#!/usr/bin/env python3
"""Simple chatbot UI for the Researcher AI Bot."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Research Bot",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal custom CSS for a clean chat look
st.markdown("""
<style>
    .stChatMessage { padding: 0.75rem 1rem; }
    [data-testid="stChatMessage"] { border-radius: 12px; }
    .block-container { max-width: 720px; padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


def get_agent():
    """Create or reuse the research agent (cached in session state)."""
    if "agent" not in st.session_state:
        try:
            from research_bot import ResearchAgent
            st.session_state.agent = ResearchAgent(
                model=st.session_state.get("model", os.environ.get("OLLAMA_MODEL", "llama3.2")),
                temperature=st.session_state.get("temperature", 0.2),
                base_url=os.environ.get("OLLAMA_BASE_URL"),
            )
        except ImportError:
            st.error("Could not import research_bot. Install dependencies: `pip install -r requirements.txt`")
            st.stop()
    return st.session_state.agent


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Header
    st.title("🔬 Research Bot")
    st.caption("Ask anything — I'll search the web and cite sources.")

    # Sidebar (optional settings)
    with st.sidebar:
        st.subheader("Settings")
        model = st.text_input(
            "Ollama model",
            value=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            help="Model must support tool calling (e.g. llama3.2, mistral)",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        st.session_state["model"] = model
        st.session_state["temperature"] = temperature
        # Invalidate agent when settings change so next query uses new config
        if st.button("Apply and reset agent"):
            if "agent" in st.session_state:
                del st.session_state["agent"]
            st.rerun()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a research question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Researching…"):
                    agent = get_agent()
                    answer = agent.research(prompt)
            except Exception as e:
                answer = f"Something went wrong: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if not st.session_state.messages:
        st.info("Type a question above. I'll search the web and give you an answer with citations.")


if __name__ == "__main__":
    main()
