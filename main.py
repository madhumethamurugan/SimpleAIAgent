from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


tools = [search_tool, wiki_tool, save_tool]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a research assistant that will help generate a research paper.

    Answer the user's query and use the necessary tools.

    Provide the final answer in this format:

    {
        "topic": "...",
        "summary": "...",
        "sources": ["..."],
        "tools_used": ["..."]
    }
    """
)


query = input("What can i help you research? ")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)

print("\nFinal Response:\n")

for message in result["messages"]:
    if message.type == "ai" and message.content:
        print(message.content)