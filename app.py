import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from research_agent import research_agent
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Research Agent")
    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        1. You ask any question
        2. Agent searches the **live web**
        3. Reads & synthesizes results
        4. Gives you a clean, cited answer
        """
    )
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown(
        """
        - 🦜 **LangChain** — tool orchestration
        - 🔀 **LangGraph** — agent state graph
        - ⚡ **Groq** — ultra-fast LLM (Llama 3.3)
        - 🦆 **DuckDuckGo** — private web search
        """
    )
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 AI Research Agent")
st.markdown(
    "_Ask anything — I'll search the web in real-time and give you "
    "a comprehensive, up-to-date answer._"
)
st.markdown("---")

# ── Chat History ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("searches"):
            with st.expander("🔍 Web searches performed", expanded=False):
                for q in msg["searches"]:
                    st.code(q, language="text")

# ── Chat Input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything to research…"):

    # Show user message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent with live status updates
    with st.chat_message("assistant"):
        status = st.empty()
        searches_done = []
        final_answer = ""

        status.info("🧠 Thinking…")

        try:
            for event in research_agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="values",
            ):
                last = event["messages"][-1]

                # Agent is about to call a tool
                if hasattr(last, "tool_calls") and last.tool_calls:
                    for tc in last.tool_calls:
                        query = tc["args"].get("query", "")
                        searches_done.append(query)
                        status.info(f"🔍 Searching: *{query}*")

                # Final AI answer
                elif isinstance(last, AIMessage) and last.content:
                    final_answer = last.content

        except Exception as e:
            final_answer = f"⚠️ Error: {e}"

        status.empty()

        # Show searches
        if searches_done:
            with st.expander("🔍 Web searches performed", expanded=False):
                for q in searches_done:
                    st.code(q, language="text")

        st.markdown(final_answer)

    st.session_state.history.append(
        {"role": "assistant", "content": final_answer, "searches": searches_done}
    )
