from typing import Annotated
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage
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

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(content="""You are a helpful AI research assistant.
When the user asks a question:
- Search the web MAXIMUM 2 times to gather information.
- After searching, ALWAYS provide a final answer immediately. Do NOT keep searching.
- Be concise and clear. Format your answer in markdown with bullet points where helpful.
- If you already have enough information, answer directly without searching again.""")

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
    # Always prepend system prompt
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Route to tool execution or finish. Stop after 6 messages to prevent loops."""
    last = state["messages"][-1]
    # Count tool calls made so far to enforce max 2 searches
    tool_call_count = sum(
        1 for m in state["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    if last.tool_calls and tool_call_count < 2:
        return "tools"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# recursion_limit stops infinite loops as a hard safety net
research_agent = graph.compile()
