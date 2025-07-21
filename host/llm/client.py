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

        The default_params are passed to the 'chat' method of the AsyncClient.
    """

    def __init__(self, url: Optional[str] = None, headers: dict = None, model: str = None, tools: Optional[Sequence[Tool]] = None, format: Optional[BaseModel] = None, **kwargs):
        """
        Initializes the Ollama instance.

        Args:
            url (str): The API endpoint URL.
            headers (dict, optional): HTTP headers for the API request. Defaults to None.
            model (str, optional): The model to use for generation. Defaults to "phi3:mini".
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
        Under the hood, it uses the Ollama API to send the messages and receive a response.

        Args:
            messages (Sequence[BaseMessage]): A list of messages for the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response.

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
            except json.JSONDecodeError as e:
                response = raw_response.message.content

            tool_calls = self._get_tool_calls(raw_response.message)

            return {
                'content': response,
                'tool_calls': tool_calls,
                'raw_response': raw_response
            }

        except httpx.HTTPError as e:
            logger.error(f"Model API error: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise LLMGenerationError(f"Model API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request to model API failed: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Request to model API failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during query: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Unexpected error during query: {str(e)}")
        
    def _get_tool_calls(self, ollama_result) -> List[Dict[str, Any]]:
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
        Converts a list of MCP Tool objects to OllamaTool Pydantic instances.

        Args:
            tools (List[Tool]): A list of MCP Tool objects to convert.

        Returns:
            List[Dict[str, Any]]: A list of converted OllamaTool instances.
        """
        ollama_params_properties = {}

        # pprint(f"Tool: {tools.__getattribute__('tools')[0]} \n   type: {type(tools.__getattribute__('tools')[0])}")

        ollama_tools = []
        for tool in tools.__getattribute__('tools'):
            for prop_name, prop_details in tool.inputSchema.get('properties', {}).items():
                param_prop_json_schema_extra = {}

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
                additionalProperties=tool.inputSchema.get('additionalProperties') 
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

    @staticmethod
    def convert_lc_messages_to_ollama_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Converts a sequence of LangChain BaseMessage objects into the basic dictionary
        format expected by the Ollama API for chat messages, correctly handling
        tool calls as dictionaries.

        Args:
            messages (Sequence[BaseMessage]): A sequence of LangChain message objects.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an Ollama chat message.
        """
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_message_dict = {
                    "role": "assistant",
                    "content": msg.content or "", # Ensure content is a string, even if empty
                }
                if msg.tool_calls:
                    ollama_tool_calls = []
                    for tc_dict in msg.tool_calls: # Renamed variable to tc_dict for clarity
                        # Access name and args using dictionary key access
                        # as you are providing dictionaries to AIMessage's tool_calls
                        tool_name = tc_dict.get("name")
                        tool_args = tc_dict.get("args")
                        # The 'id' is in tc_dict but not needed for Ollama's 'tool_calls' in assistant message

                        if tool_name is None or tool_args is None:
                            logger.warning(f"Malformed tool call dictionary encountered in AIMessage: {tc_dict}. Skipping this tool call.")
                            continue

                        ollama_tool_calls.append({
                            "type": "function", # Ollama expects this type for function calls
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        })
                    ollama_message_dict["tool_calls"] = ollama_tool_calls
                ollama_messages.append(ollama_message_dict)
            elif isinstance(msg, ToolMessage):
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id # This ID is crucial for Ollama to link the tool output
                })
            elif isinstance(msg, FunctionMessage):
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, 'tool_call_id', None) or msg.name # Fallback if tool_call_id not direct
                })
            else:
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