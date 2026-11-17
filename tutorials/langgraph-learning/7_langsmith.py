from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_openai import ChatOpenAI
import os


@tool
def get_stock_price(symbol: str) -> str:
    '''Return the current price of a stock given the stock symbol.
    :param symbol: The stock symbol
    :return: The current price of the stock as a string.
    '''
    return {
        "MSFT": "200.3",
        "AAPL": "100.4",
        "AMZN": "150.0",
        "RIL": "87.6"
    }.get(symbol, 0.0)


tools = [get_stock_price]

llm = ChatOpenAI(
    base_url=os.getenv("OCI_LLM_BASE_URL"),
    api_key=os.getenv("OCI_LLM_API_KEY"),
    model=os.getenv("MODEL_GEMMA"),
    temperature=0.2
)

llm_with_tools = llm.bind_tools(tools=tools)

from langgraph.graph import StateGraph, START, END


def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile()

from langsmith import traceable


@traceable
def call_graph(query: str):
    state = graph.invoke({"messages": [{"role": "user", "content": query}]})
    return state["messages"][-1].content


if __name__ == "__main__":
    print(call_graph("I want to buy 15 MSFT stocks and 20 AMZN stocks. What will be the total price?"))
