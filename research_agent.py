from typing import Annotated
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
search_tool = DuckDuckGoSearchResults(name="web_search", backend="html", num_results=3)
tools = [search_tool]

# ── LLM (two versions) ────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)   # can search
llm_no_tools   = llm                     # cannot search → must answer

SYSTEM = SystemMessage(content=(
    "You are a helpful AI research assistant. "
    "Answer questions clearly and concisely using markdown."
))

FORCE_MSG = HumanMessage(content=(
    "You have finished searching. "
    "Now give me your final answer based on the search results above. "
    "Do NOT call any tools."
))

# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState):
    """LLM with tools — may decide to search or answer directly."""
    response = llm_with_tools.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}

def force_answer_node(state: AgentState):
    """LLM WITHOUT tools — forced to produce a final text answer."""
    response = llm_no_tools.invoke([SYSTEM] + state["messages"] + [FORCE_MSG])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """
    After agent_node runs:
      - Agent wants to search AND < 2 searches done  → go to tools
      - Agent wants to search BUT ≥ 2 searches done  → force_answer (no tool access)
      - Agent gave a text answer                     → END
    """
    last = state["messages"][-1]
    completed = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))

    if last.tool_calls:
        return "tools" if completed < 2 else "force_answer"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent",        agent_node)
graph.add_node("tools",        tool_node)
graph.add_node("force_answer", force_answer_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
graph.add_edge("force_answer", END)

research_agent = graph.compile()
