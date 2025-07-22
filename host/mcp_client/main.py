"""
This module defines the main logic for the MCP client application.

It includes the definition of the agent state, initialization logic, user input handling,
orchestration, and tool execution. The module also sets up a state graph to manage
the flow of the application.
"""

from langgraph.graph import StateGraph, START, END
from typing import Optional, Dict, TypedDict, Union, List, Any, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from .client import MCPClient
from host.core.config import settings
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from host.llm.client import OllamaClient
from host.llm.models import OllamaResponseModel
from mcp import Tool as MCPTool
import logging
from rich import print
from pprint import pprint
from IPython.display import Image, display

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
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools: List[MCPTool]
    last_node: Optional[Literal["call_tool", "agent", "orchestrator", "human_assistance"]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

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
    agent_state['messages'] = []
    agent_state['tool_calls'] = []
    logger.debug("Agent state initialized")

    system_message = SystemMessage(
        content=
        """
        You are a helpful AI assistant. Your main goal is to respond to the user's request.

        **Decision Making Guidelines:**
        * **Prioritize direct answers:** If the user's query is a greeting, a general question, or a conversational remark, **DO NOT call any tool. Instead, respond using the format described in guideline 1.
        * **Call a function ONLY when explicitly necessary:** A function call is required only when the user's intent clearly and unambiguously requests a data operation (e.g., "Get me the latest sales data", "Find users in New York").
        * **Validate all arguments:** Ensure every argument you provide to a tool is meaningful, correctly formatted, and directly relevant to the tool's purpose. **NEVER** provide empty or nonsensical arguments.
        * Do not respond with thinking or reasoning messages or any meta-commentary about the decision-making process. Provide a direct response only.

        **Reasoning Sequence:**
        1. *Attempt Direct Answer*: First, try to answer the user's query directly without making any tool calls. If the content of the query can be fully and accurately addressed using your existing knowledge, provide a direct response.

        2. *Evaluate Tool Necessity*: If a direct answer is not possible or insufficient, assess whether any available tools can resolve the query.

        3. *Conditional Tool Call*: Call a tool only if the query cannot be answered without making a call and it is clear from the tool description that using the tool will directly resolve the query.

        4. *Inform User*: If the query cannot be answered directly and no suitable tool is available or clearly applicable, inform the user that the query cannot be answered.


        **Output Structure and Field Usage Guidelines:**

        1.  **For conversational/direct answers (NO tool needed):**
            * No need to provide response in json format. Directly respond with a string.
            * Example: "Hello! How can I assist you today?"

        2.  **For tool calls (tool IS needed):**
            * The **"content" field MUST be an empty string `""`**.
            * Example: `{"tool_name": "execute_sql_query", "arguments": {"query": "SELECT * FROM users", "params": {}}, "content": ""}`
        """
    )

    agent_state['messages'].append(system_message)

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
    state['messages'].append(HumanMessage(content=user_input))
    state['last_node'] = 'human_assistance'
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
    print("Current node: orchestrator    last_node: ", state.get('last_node'))
    return state

def decide_next_node(state: AgentState) -> AgentState:
    """
    Decides the next node to execute based on the current state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with the next node.
    """
    logger.info("Entered decide_next_node")
    last_node = state.get('last_node', None)
    if state.get('tool_calls') and state['tool_calls'] and last_node == 'agent':
        logger.debug("Next node decided: tool_executor")
        return "tool_executor"
    elif last_node == 'call_tool' or last_node == 'human_assistance':
        logger.debug("Next node decided: agent")
        return "agent"
    elif last_node == 'agent':
        logger.debug("Next node decided: end")
        return "user_input"

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

    state['last_node'] = 'call_tool'

    async with MCPClient(settings.MCP_SERVER_URL) as client:
        if state.get('tool_calls'):
            for tool in state.get('tool_calls'):
                result = await client.call_tool(tool['name'], tool['args'])
                print("Tool result: ", result)
                state['messages'].append(ToolMessage(content=result.content[0].text, tool_call_id=tool['id']))
    
    state['tool_calls'] = None

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
    state["last_node"] = "agent"

    response = await _llm.ainvoke(state.get('messages'))  
    
    content, tool_calls = response.get('content'), response.get('tool_calls', None)

    state['tool_calls'] = tool_calls
    state['messages'].append(AIMessage(content=response['raw_response'].message.content or "", tool_calls=tool_calls))

    print("LLM response content:", content, " tool calls:", tool_calls)

    return state


graph = StateGraph(state_schema=AgentState)

logger.debug("Setting up the state graph.")
graph.add_edge(START, "initialize_state")

graph.add_node("initialize_state", initialize_state)
graph.add_node("user_input", user_input)
graph.add_node("orchestrator", orchestrator)
graph.add_node("agent", agent)
graph.add_node("tools", call_tool)

graph.add_edge("initialize_state", "user_input")
graph.add_edge("user_input", "orchestrator")
graph.add_edge("tools", "orchestrator")
graph.add_edge("agent", "orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    decide_next_node,
    {
        "tool_executor": "tools",
        "agent": "agent",
        "end": END,
        "user_input": "user_input"
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
    asyncio.run(app.ainvoke(agent_state))