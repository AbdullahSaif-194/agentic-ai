from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
load_dotenv()
def travel_agent():
    return Agent(
        model = Groq(id="openai/gpt-oss-120b"),
        tools = [DuckDuckGoTools()],
        instructions ="Act like you are a travel agency agent which has to give intructions about the specific place and areaa",
        markdown = True,
        add_datetime_to_context = True
    )
agent = travel_agent()
agent.print_response("is it safe to travel Iran today ?")