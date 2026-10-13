from typing import Annotated
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ── Tools ─────────────────────────────────────────────────────────────────────
# Use html backend — avoids external API calls (Wikipedia, Wiktionary) that
# fail on restricted networks like Streamlit Cloud
search_tool = DuckDuckGoSearchResults(name="web_search", backend="html", num_results=5)
tools = [search_tool]

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState):
    """LLM decides: answer directly or call a tool."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Route to tool execution or finish."""
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

# ── Graph ─────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

research_agent = graph.compile()
