from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
)  # The foundational class for all message types in LangGraph
from langchain_core.messages import (
    ToolMessage,
)  # Passes data back to LLM after it calls a tool such as the content and
from langchain_core.messages import (
    SystemMessage,
)  # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

"""
Annotated - provides additional context without affecting teh type itself
email = Annotated(str, "This has to be a valid email format!")
print(email.__metadata__)

Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history

add_messages - Reducer function
Reducer function - rule that controls how updates from nodes are combined with the existing state.
Tell us how to merge new data into the current state
Without a reducer, updates would have replaced the existing value entirely!

#Without reducer:
state = {"message": ["Hi!"]}
update = {"message": ["Nice to meet you!"]}
new_state = {"message": ["Nice to meet you!"]}

#With reducer:
state = {"message": ["Hi!"]}
update = {"message": ["Nice to meet you!"]}
new_state = {"message": [""Hi!", Nice to meet you!"]}

"""

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage], add_messages
    ]  ## preserve the state by appending it, Sequence[BaseMessage] is the data type and
    # add_message is the metadata, that is these two are annotated


@tool  # decorator, defines that this function is special, it is a tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""  ## docstring

    return a + b


@tool
def subtract(a: int, b: int):
    """Subtraction function"""  ## docstring

    return a - b


@tool
def multiply(a: int, b: int):
    """Multiplication function"""  ## docstring

    return a * b


tools = [add, subtract, multiply]  # list of tools for infusing it into LLM

model = ChatOpenAI(model="gpt-4o").bind_tools(
    tools
)  # we tell llm that use these tools with this built-in function, now llm has access to all these tools


def model_call(state: AgentState) -> AgentState:  ## my node
    # response = model.invoke(
    #     ["You are my AI assistant, please answer my query to the best of your ability."]
    # ) ## System message for LLM, but below is more professional approach

    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke(
        [system_prompt] + state["messages"]
    )  ## human message in latter part
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please")]}
# inputs = {"messages": [("user", "Add 3 + 4")]}
# inputs = {"messages": [("user", "Add 3 + 4, then add 7 + 9")]}
inputs = {"messages": [("user", "Add 3 + 4 and then multiply the result by 6")]}
print_stream(app.stream(inputs, stream_mode="values"))
