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
from pprint import pprint
from ollama import chat
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
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
    next_tool_id: Optional[str] = None
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
    # _llm = OllamaClient(url=settings.LLM_URL, model=settings.LLM_MODEL, tools=agent_state.get('tools'), format=OllamaResponseModel.model_json_schema(by_alias=False))
    agent_state['messages'] = []
    logger.debug("Agent state initialized:")
    # pprint(agent_state)

    system_message = SystemMessage(
        content=
        """
        You are a helpful AI assistant. Your main goal is to respond to the user's request by strictly adhering to a specific JSON output format.

        You have access to a set of functions to help you with data-related tasks.

        **Output Structure and Field Usage Guidelines (MUST always be a valid JSON object):**

        1.  **For conversational/direct answers (NO tool needed):**
            * Set the **"tool_name" field to `null`**.
            * Set the **"arguments" field to `null`**.
            * The **"content" field MUST contain your natural language response** to the user (e.g., greetings, general facts, conversational remarks).
            * Example: `{"tool_name": null, "arguments": null, "content": "Hello! How can I assist you today?"}`

        2.  **For tool calls (tool IS needed):**
            * The **"tool_name" field MUST contain the exact name of the tool** to be called (e.g., "execute_sql_query").
            * The **"arguments" field MUST be a JSON object containing all required parameters** for the tool.
            * The **"content" field MUST be an empty string `""`**.
            * Example: `{"tool_name": "execute_sql_query", "arguments": {"query": "SELECT * FROM users", "params": {}}, "content": ""}`

        **Specific details for the `execute_sql_query` tool:**
        * The `query` argument **MUST** be a valid SQL statement directly related to the user's data request. It **MUST NOT** contain conversational text or non-SQL commands.
        * The `params` argument **MUST** be a JSON object (`{}`) containing any necessary query parameters. Use `{}` if no parameters are needed.

        **Decision Making Guidelines:**
        * **Prioritize direct answers:** If the user's query is a greeting, a general question, or a conversational remark, **DO NOT call any tool. Instead, respond using the format described in guideline 1.
        * **Call a function ONLY when explicitly necessary:** A function call is required only when the user's intent clearly and unambiguously requests a data operation (e.g., "Get me the latest sales data", "Find users in New York").
        * **Validate all arguments:** Ensure every argument you provide to a tool is meaningful, correctly formatted, and directly relevant to the tool's purpose. **NEVER** provide empty or nonsensical arguments.
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
    logger.debug("Deciding the next node with state:")
    # pprint(state)
    last_node = state.get('last_node', None)
    if state.get('next_tool') and last_node == 'agent':
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
    logger.debug("Calling tool with state:")
    state['last_node'] = 'call_tool'
    if not state.get('next_tool'):
        logger.error("No tool specified to call.")
        raise ValueError("No tool specified to call.")
    
    async with MCPClient(settings.MCP_SERVER_URL) as client:
        result = await client.call_tool(state.get('next_tool'), state.get('arguments'))
    print("Tool result:")
    print(result)
    print("type of content", type(result.content[0].text))
    state['messages'].append(ToolMessage(content=result.content[0].text, tool_call_id=state['next_tool_id']))
    state['next_tool'] = None
    state['next_tool_id'] = None
    state['arguments'] = None

    return state

async def agent(state: AgentState) -> AgentState:
    """
    Executes the agent logic and updates the state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state after agent execution.
    """
    logger.debug("Executing agent logic with state:")
    # pprint(state)
    state["last_node"] = "agent"

    # response = await _llm.ainvoke(state.get('messages'))  
    raw_response = chat(
        messages=OllamaClient.convert_lc_messages_to_ollama_messages(state.get('messages')),
        model=settings.LLM_MODEL,
        tools=OllamaClient.convert_mcp_tools_to_ollama_tools(state.get('tools')),
        format=OllamaResponseModel.model_json_schema(by_alias=False)
    )

    print("LLM raw Response: ", raw_response)

    try:
        if raw_response.message.content.strip():
            response = OllamaResponseModel.model_validate_json(raw_response.message.content)
            print("LLM Response:")
            pprint(response)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response: %s", e)
        state['next_tool'] = None
        state['arguments'] = None
        # print("Response: ", response.get('content'))
        print("Entered json.JsonDecodeError")
    
    if raw_response.message.tool_calls:
        next_tool = raw_response.message.tool_calls[0].function
        state['next_tool'] = next_tool.name
        state['arguments'] = next_tool.arguments
        state['next_tool_id'] = str(uuid.uuid4())

    print("The response from the LLM: ", raw_response.message.content)
    print(f"Tool calls: {state['next_tool']} with arguments: {state['arguments']}")

    args = {}
    if state.get('next_tool'):
        args["tool_calls"] = [{
            "name": state['next_tool'],
            "args": state['arguments'],
            "id": state['next_tool_id']
        }]

    state['messages'].append(AIMessage(content=raw_response.message.content, **args))

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
    logger.debug("Initial agent state:")
    pprint(agent_state)
    asyncio.run(app.ainvoke(agent_state))