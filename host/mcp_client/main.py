from langgraph.graph import StateGraph, START, END
from typing import Optional, Dict, TypedDict, Union, List, Any, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from .client import MCPClient
from host.core.config import settings
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from host.llm.client import OllamaClient
from host.exceptions import LLMGenerationError
from host.llm.models import OllamaResponseModel
import json
from mcp import Tool as MCPTool
from .utils import parse_json_string_to_dict
import logging
from rich import print

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

_llm: OllamaClient = None

NodeType = Literal[
    "call_tool",
    "agent",
    "orchestrator",
    "human_assistance"
]

class AgentState(TypedDict):
    """
    Represents the state of the agent during execution.

    Attributes:
        messages (Sequence[BaseMessage]): A sequence of messages exchanged.
        tools (List[MCPTool]): A list of available tools.
        arguments (Dict[str, Any]): Arguments for the next tool.
        next_tool (Optional[NodeType]): The next tool to be executed.
        last_node (Optional[NodeType]): The last executed node.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools: List[MCPTool]
    arguments: Dict[str, Any]
    next_tool: Optional[NodeType] = None
    last_node: Optional[NodeType] = None

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
    _llm = OllamaClient(url=settings.LLM_URL, model=settings.LLM_MODEL, tools=agent_state.get('tools'), format=OllamaResponseModel.model_json_schema())
    # _llm = OllamaClient(url=settings.LLM_URL, model=settings.LLM_MODEL, format=OllamaResponseModel.model_json_schema())
    agent_state['messages'] = []
    logger.debug("Agent state initialized: %s", agent_state)

    system_message = SystemMessage(
        content="""
        You are a helpful AI assistant. Your main objective is to understand the user's request and respond appropriately.

        You have access to a set of functions to help you answer questions or perform actions.

        **Guidelines for Responding:**
        - **Prioritize direct answers:** If a question can be answered from your general knowledge or is a simple conversational greeting (like "hi", "hello", "how are you?"), **do NOT use any functions**. Provide a direct, natural language response.
        - **Function use is for specific tasks:** Only call a function when the user's intent clearly requires its specific functionality.
        - **Validate arguments:** If you decide to call a function, ensure all necessary arguments are correctly identified and populated based on the user's request. **Do not provide empty or irrelevant arguments.**
        - **If you do not need to call a function, provide a conversational response in plain text.**
        - **If you need to call a function, respond strictly with the function call in the specified JSON format.**
        """)

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
    logger.debug("User input received: %s", user_input)
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
    return state

def decide_next_node(state: AgentState) -> AgentState:
    """
    Decides the next node to execute based on the current state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with the next node.
    """
    logger.debug("Deciding the next node with state: %s", state)
    last_node = state.get('last_node', None)
    if state.get('next_tool') and state.get('next_tool') in state.get('tools') and last_node == 'agent':
        logger.debug("Next node decided: tool_executor")
        return "tool_executor"
    elif last_node == 'call_tool' or last_node == 'human_assistance':
        logger.debug("Next node decided: agent")
        return "agent"
    else:
        logger.debug("Next node decided: end")
        return "end"

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
    logger.debug("Calling tool with state: %s", state)
    state['last_node'] = 'call_tool'
    if not state.get('next_tool'):
        logger.error("No tool specified to call.")
        raise ValueError("No tool specified to call.")
    
    async with MCPClient(settings.MCP_SERVER_URL) as client:
        result = await client.call_tool(state.get('next_tool'), state.get('arguments'))
    logger.debug("Tool result: %s", result)
    state['messages'].append(ToolMessage(content=json.dumps(result)))
    state['next_tool'] = None
    state['arguments'] = None
    logger.debug("State updated after tool call: %s", state)

async def agent(state: AgentState) -> AgentState:
    """
    Executes the agent logic and updates the state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state after agent execution.
    """
    logger.debug("Executing agent logic with state: %s", state)
    state["last_node"] = "agent"
    response = await _llm.ainvoke(state.get('messages'))  
    print(f"LLM Response: {response}")  
    state['next_tool'] = response.get('content').get('tool_calls')[0].get('function').get('name', None)
    state['arguments'] = response.get('content').get('tool_calls')[0].get('function').get('arguments', None)
    logger.debug("Tool name and arguments extracted: %s, %s", state.get('next_tool'), state.get('arguments'))
    state['messages'].append(AIMessage(response['content']['content']))
    logger.info("LLM Response: %s", response['content']['content'])


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
        "end": END
    }
)

app = graph.compile()

if __name__ == '__main__':
    """
    Main entry point for the application.

    Initializes the agent state and runs the application.
    """
    logger.info("Starting the application.")
    import asyncio

    agent_state: AgentState = {}
    logger.debug("Initial agent state: %s", agent_state)
    asyncio.run(app.ainvoke(agent_state))