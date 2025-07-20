import httpx
from host.exceptions import LLMGenerationError
from shared.utils import log_duration
import logging
from typing import List, Dict, Any, Sequence
from mcp import Tool, ListToolsResult
from host.core.config import settings
from .models import OllamaTool, FunctionTool, FunctionParameters, ParameterProperty

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage
from langchain_core.messages.tool import ToolCall 

logger = logging.getLogger(__name__)

class OllamaClient():
    """
    A class for interacting with the Ollama API to generate responses based on prompts.

    Attributes:
        url (str): The API endpoint URL.
        headers (dict): HTTP headers for the API request.
        model (str): The model to use for generation.
        default_params (dict): Additional parameters for the API request.
    """

    def __init__(self, url: str, headers: dict = None, model: str = None, **kwargs):
        """
        Initializes the Ollama instance.

        Args:
            url (str): The API endpoint URL.
            headers (dict, optional): HTTP headers for the API request. Defaults to None.
            model (str, optional): The model to use for generation. Defaults to "phi3:mini".
            **kwargs: Additional parameters for the API request.
        """
        self.url = url.strip() if url else settings.NL2SQL_LLM_URL.strip()
        self.headers = headers or { "Content-Type": "application/json" }
        self.model = model or settings.NL2SQL_LLM_MODEL.strip()
        self.default_params = kwargs
        logger.info(f"Ollama instance initialized with model: {self.model}")

    @log_duration
    async def _make_ollama_request(self, json: dict):
        """
        Makes a request to the Ollama API with the given JSON payload.

        Args:
            json (dict): The JSON payload for the API request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            LLMGenerationError: If there is an error during the API request.
        """
        try:
            logger.debug(f"Sending request to Ollama API with payload: %s", json)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.url, headers=self.headers, json=json)
                response.raise_for_status()
            logger.info("Response received successfully from Ollama API.")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Model api error: %s", str(e), exc_info=True)
            raise LLMGenerationError(f"Model API error: {str(e)}")
        except httpx.RequestError as e:
            logger.error(f"Request to model API failed: %s", str(e), exc_info=True)
            raise LLMGenerationError(f"Request to model API failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during query: %s", str(e), exc_info=True)
            raise LLMGenerationError(f"Unexpected error during query: {str(e)}")

    @log_duration
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> Dict[str, Any]:
        """
        Asynchronously generates a response from the Ollama chat model using a list of messages.

        Args:
            messages (Sequence[BaseMessage]): A list of messages for the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response.

        Raises:
            LLMGenerationError: If there is an error during the generation process.

        NOTE: Currently, streaming is not supported in this method.
        """

        logger.debug(f"Invoking Ollama model: %s with messages: %s...", self.model, messages[:1])

        print(f"Default tools: {self.default_params.get('tools', 'None')}")

        if 'tools' in self.default_params and isinstance(self.default_params['tools'], ListToolsResult):
            logger.debug("Converting MCP tools to Ollama tools...")
            self.default_params['tools'] = self.convert_mcp_tools_to_ollama_tools(self.default_params['tools'].tools)

        print(f"Default tools after conversion: {self.default_params.get('tools', 'None')}")

        ollama_messages = self.convert_lc_messages_to_ollama_messages(messages)

        print(f"Ollama messages: {ollama_messages}")

        json = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            **self.default_params
        }
        ollama_response = await self._make_ollama_request(json=json)

        print(f"Ollama response: {ollama_response} ")

        if ollama_response and 'message' in ollama_response:
            logger.debug(f"Ollama response received: {ollama_response}")
            response_content = ollama_response['message']
            logger.info("Ollama response generated successfully.")
            return {"content": response_content} 
        else:
            logger.error(f"Unexpected Ollama response format: {ollama_response}")
            raise LLMGenerationError(f"Unexpected Ollama response format: {ollama_response}")

    @staticmethod
    def convert_mcp_tools_to_ollama_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Converts a list of MCP Tool objects to OllamaTool Pydantic instances.

        Args:
            tools (List[Tool]): A list of MCP Tool objects to convert.

        Returns:
            List[Dict[str, Any]]: A list of converted OllamaTool instances.
        """
        # Prepare parameters for the OllamaTool's function
        ollama_params_properties = {}

        ollama_tools = []
        for tool in tools:
            for prop_name, prop_details in tool.inputSchema.get('properties', {}).items():
                param_prop_json_schema_extra = {}

                # Map MCP specific properties to Ollama's json_schema_extra
                if 'title' in prop_details and prop_details['title'] is not None:
                    param_prop_json_schema_extra['title'] = prop_details['title']
                if 'default' in prop_details and prop_details['default'] is not None:
                    param_prop_json_schema_extra['default'] = prop_details['default']
                if 'additionalProperties' in prop_details and prop_details['additionalProperties'] is not None:
                    param_prop_json_schema_extra['additionalProperties'] = prop_details['additionalProperties']

                ollama_params_properties[prop_name] = ParameterProperty(
                    type=prop_details['type'],
                    description=prop_details.get('description', prop_details.get('title')),
                    json_schema_extra=param_prop_json_schema_extra if param_prop_json_schema_extra else None
                )

            ollama_function_parameters = FunctionParameters(
                properties=ollama_params_properties,
                required=tool.inputSchema.get('required', []),
                additionalProperties=tool.inputSchema.get('additionalProperties') # Get from root inputSchema
            )

            ollama_function_definition = FunctionTool(
                name=tool.name,
                description=tool.description, 
                parameters=ollama_function_parameters
            )

            ollama_tool = OllamaTool(
                function=ollama_function_definition
            )

            ollama_tools.append(ollama_tool.model_dump())

        return ollama_tools

    def convert_lc_messages_to_ollama_messages(self, messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Converts a sequence of LangChain BaseMessage objects into the basic dictionary
        format expected by the Ollama API for chat messages.

        Args:
            messages (Sequence[BaseMessage]): A sequence of LangChain message objects.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an Ollama chat message.
        """
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # HumanMessage maps to Ollama's 'user' role
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                # SystemMessage maps to Ollama's 'system' role
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # AIMessage maps to Ollama's 'assistant' role
                ollama_message_dict = {
                    "role": "assistant",
                    "content": msg.content or "", # Ensure content is a string, even if empty
                }
                # If the AIMessage includes tool calls, structure them as Ollama expects.
                # The 'type: function' is an Ollama API requirement for tool calls.
                if msg.tool_calls:
                    ollama_tool_calls = []
                    for tc in msg.tool_calls:
                        ollama_tool_calls.append({
                            "type": "function", # Ollama expects this type for function calls
                            "function": {
                                "name": tc.name,
                                "arguments": tc.args # tc.args is already a dictionary from LangChain's ToolCall
                            }
                        })
                    ollama_message_dict["tool_calls"] = ollama_tool_calls
                ollama_messages.append(ollama_message_dict)
            elif isinstance(msg, ToolMessage):
                # ToolMessage maps to Ollama's 'tool' role.
                # The 'tool_call_id' is crucial for Ollama to link the tool's result
                # back to the specific tool call made by the assistant.
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content, # Tool's result is its content
                    "tool_call_id": msg.tool_call_id # Required by Ollama to match tool output
                })
            elif isinstance(msg, FunctionMessage): # Included for backward compatibility with older LangChain versions
                # FunctionMessage typically represents tool output as well
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id if hasattr(msg, 'tool_call_id') else msg.name # Fallback for tool_call_id
                })
            else:
                # Fallback for any other BaseMessage type not explicitly handled,
                # includes its string representation in the user content.
                logger.warning(f"Unsupported LangChain message type encountered: {type(msg)}. Converting to user message content.")
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