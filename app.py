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
    st.session_state.history = []        # for display (dicts)

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []    # for agent (LangChain objects) ← MEMORY

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
    st.session_state.lc_messages.append(HumanMessage(content=prompt))  # ← add to memory
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent — pass FULL conversation history so it remembers context
    with st.chat_message("assistant"):
        status = st.empty()
        searches_done = []
        final_answer = ""

        status.info("🧠 Thinking…")

        try:
            for event in research_agent.stream(
                {"messages": st.session_state.lc_messages},  # ← full history
                stream_mode="values",
                config={"recursion_limit": 8},  # ← hard stop: max 8 steps
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

        # Fallback: if agent ended without a text answer (edge case), ask LLM directly
        if not final_answer and not final_answer.strip():
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage
            fallback_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0,
                                    groq_api_key=os.getenv("GROQ_API_KEY"))
            fallback_resp = fallback_llm.invoke(
                [SystemMessage(content="Answer the user's question concisely based on what you know.")]
                + st.session_state.lc_messages
            )
            final_answer = fallback_resp.content

        status.empty()

        # Show searches
        if searches_done:
            with st.expander("🔍 Web searches performed", expanded=False):
                for q in searches_done:
                    st.code(q, language="text")

        st.markdown(final_answer)

    # Save assistant reply to memory too
    st.session_state.lc_messages.append(AIMessage(content=final_answer))  # ← add to memory
    st.session_state.history.append(
        {"role": "assistant", "content": final_answer, "searches": searches_done}
    )
