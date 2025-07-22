"""
This module provides the OllamaClient class for interacting with the Ollama API.

The OllamaClient class is designed to manage communication with the Ollama API,
including sending messages, converting tools and messages, and handling responses.
It also demonstrates the conversion of MCP Tool objects to OllamaTool instances.
"""

import httpx
from host.exceptions import LLMGenerationError
from shared.utils import log_duration
import logging
from typing import List, Dict, Any, Sequence, Optional, Union
from mcp import ListToolsResult
from mcp.types import Tool
from host.core.config import settings
from .models import OllamaTool, FunctionTool, FunctionParameters, ParameterProperty
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage
from langchain_core.messages.tool import ToolCall 
from pprint import pprint
from ollama import AsyncClient, chat
from pydantic import BaseModel
import uuid, json

logger = logging.getLogger(__name__)

class OllamaClient():
    """
    A class for interacting with the Ollama API to generate responses based on prompts.

    Attributes:
        url (Optional[str]): The API endpoint URL.
        headers (dict): HTTP headers for the API request.
        model (str): The model to use for generation.
        default_params (dict): Additional parameters for the API request.
        async_client (AsyncClient): The asynchronous client for API communication.
        format (Optional[BaseModel]): The format for validating responses.
        tools (Optional[List[Dict[str, Any]]]): A list of tools converted to Ollama format.
    """

    def __init__(self, url: Optional[str] = None, headers: dict = None, model: str = None, tools: Optional[Sequence[Tool]] = None, format: Optional[BaseModel] = None, **kwargs):
        """
        Initializes the Ollama instance.

        Args:
            url (Optional[str]): The API endpoint URL.
            headers (dict, optional): HTTP headers for the API request. Defaults to None.
            model (str, optional): The model to use for generation. Defaults to "phi3:mini".
            tools (Optional[Sequence[Tool]]): A list of tools to convert to Ollama format.
            format (Optional[BaseModel]): The format for validating responses.
            **kwargs: Additional parameters for the API request.
        """
        self.url = url.strip() if url else None
        headers = headers or { "Content-Type": "application/json" }
        self.model = model or settings.LLM_MODEL.strip()
        self.default_params = kwargs
        self.async_client = AsyncClient(url, headers=headers)
        self.format = format 

        if tools:
            self.tools = self.convert_mcp_tools_to_ollama_tools(tools)
            self.default_params['tools'] = self.tools

        logger.info(f"Ollama instance initialized with model: {self.model}")

    @log_duration
    async def ainvoke(self, messages: Union[Dict[str, Any], Sequence[BaseMessage]]) -> Dict[str, Any]:
        """
        Asynchronously generates a response from the Ollama chat model using a list of messages.

        Args:
            messages (Union[Dict[str, Any], Sequence[BaseMessage]]): A list of messages for the conversation or a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response, tool calls, and raw response.

        Raises:
            LLMGenerationError: If there is an error during the generation process.

        NOTE: Currently, streaming is not supported in this method.
        """
        if isinstance(messages, Sequence) and messages and isinstance(messages[0], BaseMessage):
            logger.debug("Converting LangChain messages to Ollama format.")
            messages = self.convert_lc_messages_to_ollama_messages(messages)

        try:
            raw_response = chat(
                model=self.model,
                messages=messages,
                format=self.format.model_json_schema(by_alias=False) if self.format else None,
                **self.default_params
            )

            response = ''
            try:
                if self.format and raw_response.message.content.strip():
                    response = self.format.model_validate_json(raw_response.message.content)
                else:
                    response = raw_response.message.content
            except json.JSONDecodeError as e:
                response = raw_response.message.content

            tool_calls = self._get_tool_calls(raw_response.message)

            print(f"Raw response: {raw_response}")
            print(f"raw.message.content : ", raw_response.message.content)
            
            return {
                'content': response,
                'tool_calls': tool_calls,
                'raw_response': raw_response
            }

        except Exception as e:
            logger.error(f"Unexpected error during query: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Unexpected error during query: {str(e)}")
        
    def _get_tool_calls(self, ollama_result) -> List[Dict[str, Any]]:
        """
        Extracts tool calls from the Ollama result.

        Args:
            ollama_result: The result object from the Ollama API.

        Returns:
            List[Dict[str, Any]]: A list of tool calls with their names, arguments, and IDs.
        """
        if not ollama_result.tool_calls:
            return []
        
        tool_calls = []
        for tool_call in ollama_result.tool_calls:
            tool = tool_call.function
            tool_calls.append({
                "name": tool.name,
                "args": tool.arguments,
                "id": str(uuid.uuid4())
            })
        
        return tool_calls

    @staticmethod
    def convert_mcp_tools_to_ollama_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Converts a list of MCP Tool objects to Ollama-compatible tool definitions.
        """
        ollama_tools = []

        for tool in tools.__getattribute__('tools'):
            props = {}
            for name, details in tool.inputSchema.get('properties', {}).items():
                extra = {
                    k: v for k, v in details.items()
                    if k in {"title", "default", "additionalProperties"} and v is not None
                }
                props[name] = ParameterProperty(
                    type=details['type'],
                    description=details.get('description') or details.get('title'),
                    json_schema_extra=extra or None
                )

            func_params = FunctionParameters(
                properties=props,
                required=tool.inputSchema.get('required', []),
                additionalProperties=tool.inputSchema.get('additionalProperties')
            )

            ollama_tools.append(OllamaTool(
                function=FunctionTool(
                    name=tool.name,
                    description=tool.description,
                    parameters=func_params
                )
            ).model_dump())

        return ollama_tools

    @staticmethod
    def convert_lc_messages_to_ollama_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Converts LangChain messages to Ollama chat message format.
        """
        ollama_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})

            elif isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})

            elif isinstance(msg, AIMessage):
                entry = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        name, args = tc.get("name"), tc.get("args")
                        if not name or not args:
                            logger.warning(f"Malformed tool call: {tc}")
                            continue
                        tool_calls.append({
                            "type": "function",
                            "function": {"name": name, "arguments": args}
                        })
                    entry["tool_calls"] = tool_calls
                ollama_messages.append(entry)

            elif isinstance(msg, ToolMessage):
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })

            elif isinstance(msg, FunctionMessage):
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, 'tool_call_id', None) or msg.name
                })

            else:
                logger.warning(f"Unsupported message type: {type(msg)}")
                ollama_messages.append({"role": "user", "content": str(msg)})

        return ollama_messages

if __name__ == "__main__":
    """
    Demonstrates the conversion of MCP Tool objects to OllamaTool instances.

    This script creates a test MCP Tool object, converts it to an OllamaTool instance,
    and prints the resulting JSON representation.
    """
    from .models import OllamaResponseModel

    test_tools = [Tool(
        name="test_tool",
        description="A test tool for demonstration purposes.",
        inputSchema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "title": "Parameter 1"},
                "param2": {"type": "integer", "title": "Parameter 2"}
            },
            "required": ["param1"],
        }
    ), Tool(
        name="test_tool2",
        description="A test tool for demonstration purposes 22222.",
        inputSchema={
            "type": "object",
            "properties": {
                "param1222222": {"type": "string", "title": "Parameter 222221"},
                "param22222222": {"type": "integer", "title": "Parameter 22222222"}
            },
            "required": ["param1222222"],
        }
    )]

    ollama_tool = OllamaClient.convert_mcp_tools_to_ollama_tools(test_tools)
    print(ollama_tool.model_dump_json(indent=2))