from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def search_web(query: str) -> str:
#     """
#     Tool that searches over the internet for the given query and returns the results as a string.
#     args:
#         query (str): The search query.
#     returns:
#         str: The search results.
#     """
#     print(f"Searching the web for: {query}")
#     return tavily.search(query=query)

llm=ChatOpenAI(model="gpt-5", temperature=0)
# tools=[search_web]
tools=[TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    user_input = "Search for the latest jobs for Data Scientist in Bangalore with 3 plus years experience professionals and sort based on the most recent jobs posted."
    result = agent.invoke({'messages': [HumanMessage(content=user_input)]})
    print(result)

if __name__ == "__main__":
    main()  