import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import (
    load_dotenv,
)  ## used to store secret stuff like API keys or configuration values

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(model="gpt-4o")


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":

    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


##Normally you need to lof the history of conversation to vector database, but
##for prototype agent, textfile is enough
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")

"""
There is also one problem here regarding current state, the history of conversation. It keeps increasing, and as it increases
the number of tokens used are get increased which is not a good thing from the perspective of finance, for this situation,
we can define some limit number, lets say 5, after after 5 human messages, the very first human message will be removed, smth
like this, not the latest one cause it could be more relevant but the first one, the hsitorical one."""
