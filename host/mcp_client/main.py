"""
This module defines the main logic for the MCP client application.

It includes the definition of the agent state, initialization logic, user input handling,
orchestration, and tool execution. The module also sets up a state graph to manage
the flow of the application.
"""

from langgraph.graph import StateGraph, START, END
from typing import Optional, Dict, TypedDict, Union, List, Any, Annotated, Sequence, Literal
# from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from .client import MCPClient
from host.core.config import settings
# from langgraph.graph.message import add_messages
from host.llm.client import OllamaClient
# from host.llm.models import OllamaResponseModel
from mcp import Tool
import logging
from rich import print
from pprint import pprint

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

_llm: OllamaClient = None

class AgentState(TypedDict):
    """
    Represents the state of the agent during execution.

    Attributes:
        messages (Sequence[BaseMessage]): A sequence of messages exchanged.
        tools (List[MCPTool]): A list of available tools.
        tool_calls (Optional[List[Dict[str, Any]]]): A list of tool calls to be executed.
        last_node (Optional[NodeType]): The last executed node.

    Note:
        tool_calls is a list of dictionaries where each dictionary contains:
        - 'name': The name of the tool to be called.
        - 'args': The arguments to pass to the tool.
        - 'id': A unique identifier for the tool call.
    """
    message: Dict
    tools: List[Tool]
    state: Optional[Literal["tool_call", "agent", "orchestrator", "human"]] = None
    tool_calls: Optional[List[Any]] = None

async def initialize_state(agent_state: AgentState) -> AgentState:
    """
    Initializes the agent state with the necessary tools and messages.

    Args:
        agent_state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with initialized tools and messages.
    """
    logger.debug("Initializing agent state with tools and messages.")
    global _llm

    async with MCPClient(settings.MCP_SERVER_URL) as client:
        agent_state['tools'] = await client.list_tools()
        print("Available tools:\n", agent_state['tools'])

    _llm = OllamaClient(model=settings.LLM_MODEL, tools=agent_state.get('tools'))
    logger.debug("Agent state initialized")

    return agent_state

# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response['data']

def user_input(state: AgentState) -> AgentState:
    """
    Handles user input and updates the agent state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with user input.
    """
    logger.debug("Prompting user for input.")
    user_input = input("Enter your query: ")
    state['message'] = {
        'role': 'user',
        'content': user_input
    }
    state['state'] = 'agent'
    logger.debug("User input received:")
    pprint(user_input)
    return state

def orchestrator(state: AgentState) -> AgentState:
    """
    Orchestrates the flow of the agent.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state.
    """
    logger.debug("Orchestrating the agent flow with state: %s", state)
    print("Current node: orchestrator    state: ", state.get('state'))
    return state

def next_node(state: AgentState) -> AgentState:
    """
    Decides the next node to execute based on the current state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with the next node.
    """
    return state.get('state', 'end')

async def call_tool(state: AgentState) -> AgentState:
    """
    Calls the specified tool and updates the state with the result.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state after tool execution.

    Raises:
        ValueError: If no tool is specified to call.
    """
    logger.info("In call_tool node.")

    state['message'] = {
        'role': 'tool',
        'content': '',
    }

    async with MCPClient(settings.MCP_SERVER_URL) as client:
        if state.get('tool_calls'):
            for tool in state.get('tool_calls'):
                function = tool.function
                result = await client.call_tool(function.name, function.arguments)
                state['message']['content'] += 'Result for {} is {}'.format(function.name, result.content[0].text)
            print("Tool result: ", state['message']['content'])
    state['tool_calls'] = None

    state['state'] = 'agent'

    return state

async def agent(state: AgentState) -> AgentState:
    """
    Executes the agent logic and updates the state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state after agent execution.
    """
    logger.info("In agent node.")

    response = await _llm.ainvoke(state.get('message'))
    
    print("LLM response:", response)

    tool_calls = response.message.get('tool_calls', [])
    if tool_calls:
        state['tool_calls'] = tool_calls
        state['state'] = 'tool_call'
    else:
        state['state'] = 'human'

    print("LLM response content:", response.message.content, " tool calls:", tool_calls)

    return state


graph = StateGraph(state_schema=AgentState)

logger.debug("Setting up the state graph.")
graph.add_edge(START, "initialize_state")

graph.add_node("initialize_state", initialize_state)
graph.add_node("human", user_input)
graph.add_node("orchestrator", orchestrator)
graph.add_node("agent", agent)
graph.add_node("tool_call", call_tool)

graph.add_edge("initialize_state", "human")
graph.add_edge("human", "orchestrator")
graph.add_edge("tool_call", "orchestrator")
graph.add_edge("agent", "orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    next_node,
    {
        "tool_call": "tool_call",
        "agent": "agent",
        "human": "human",
        "end": END
    }
)

app = graph.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

if __name__ == '__main__':
    """
    Main entry point for the application.

    Initializes the agent state and runs the application.
    """
    logger.info("Starting the application.")
    import asyncio

    agent_state: AgentState = {}
    logger.debug("Initial agent state:")
    pprint(agent_state)
    asyncio.run(app.ainvoke(agent_state, config={"recursion_limit": 100}))