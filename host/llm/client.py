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
from typing import List, Dict, Any, Optional
from mcp.types import Tool
from host.core.config import settings
# from .models import OllamaTool, FunctionTool, FunctionParameters, ParameterProperty
# from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage
# from langchain_core.messages.tool import ToolCall 
# from langgraph.graph.message import add_messages
from rich import print
from ollama import AsyncClient, chat
# from pydantic import BaseModel

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

    def __init__(self, url: Optional[str] = None, headers: dict = None, model: str = None, tools: Optional[List[Tool]] = None):
        """
        Initializes the Ollama's LLM instance.

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
        self.async_client = AsyncClient(url, headers=headers)
        self.messages: List[dict] = []
        self.tools : List[Dict[Any, Any]] = None

        if tools:
            self.tools = self.convert_mcp_tools_to_ollama_tools(tools)

        self.messages.append(
            {
                "role": "system",
                "content": """
                    You are a helpful assistant who can use available tools to solve problems.
                    Only use tools when necessary. 
                """
            }
        )

        logger.info(f"Ollama instance initialized with model: {self.model}")

    def set_tools(self, tools: List[Tool]):
        """Set the tools for the model."""
        self.tools = tools

    # @log_duration
    async def ainvoke(self, message: Dict) -> Dict[str, Any]:
        """
        Asynchronously generates a response from the Ollama chat model using a list of messages.

        Args:
            messages (str): A JSON string representing the messages for the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response, tool calls, and raw response.

        Raises:
            LLMGenerationError: If there is an error during the generation process.

        NOTE: Currently, streaming is not supported in this method.
        """
        # self.messages.append(message)

        try:
            self.messages.append(message)

            response = chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                stream=False,
            )

            if response.message.content:
                self.messages.append({
                    "role": "assistant",
                    "content": response.message.content,
                })
            
            return response

        except Exception as e:
            logger.error(f"Unexpected error during query: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Unexpected error during query: {str(e)}")

    @staticmethod
    def convert_mcp_tools_to_ollama_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Converts a list of Tool objects to a list of dictionaries conforming to the specified JSON schema.

        Args:
            tools: A list of Tool objects.

        Returns:
            A list of dictionaries, each representing a tool in the target JSON schema.
        """
        ollama_tools = []
        for tool in tools:
            converted_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.inputSchema.get("properties", {}),
                        "required": tool.inputSchema.get("required", [])
                    }
                }
            }
            ollama_tools.append(converted_tool)
        return ollama_tools

# async def main():
#     client = OllamaClient(model="qwen2.5:7b")
#     newMessage = {"role": "user", "content": "Hi"}
#     response = await client.ainvoke(newMessage)
#     print(response)

# if __name__ == "__main__":
#     # Example usage
#     import asyncio
#     asyncio.run(main())