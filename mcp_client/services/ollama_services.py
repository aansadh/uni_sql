import httpx
from exceptions import LLMGenerationError
from utils import log_duration
import logging
from typing import List, Dict, Any
from core.config import settings
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List
from mcp import Tool

logger = logging.getLogger(__name__)

class ParameterProperty(BaseModel):
    """
    Represents a single property within a function's parameters.

    Attributes:
        type (str): The type of the parameter (e.g., string, integer).
        description (Optional[str]): A description of the parameter.
    """
    type: str
    description: Optional[str] = None

class FunctionParameters(BaseModel):
    """
    Represents the 'parameters' object for a function tool.

    Attributes:
        type (Literal['object']): The type of the parameters object (always 'object').
        properties (Dict[str, ParameterProperty]): A dictionary of parameter properties.
        required (Optional[List[str]]): A list of required parameter names.
    """
    type: Literal['object'] = 'object'
    properties: Dict[str, ParameterProperty] = Field(default_factory=dict)
    required: Optional[List[str]] = Field(default_factory=list)

class FunctionTool(BaseModel):
    """
    Represents a function tool definition.

    Attributes:
        name (str): The name of the function tool.
        description (Optional[str]): A description of the function tool.
        parameters (FunctionParameters): The parameters for the function tool.
    """
    name: str = Field(..., description="Name of the function tool")
    description: Optional[str] = Field(None, description="Description of the function tool")
    parameters: FunctionParameters = Field(..., description="Parameters for the function tool")

class OllamaTool(BaseModel):
    """
    Represents an Ollama tool definition.

    Attributes:
        type (str): The type of the tool (e.g., 'function').
        function (FunctionTool): The function tool definition.
    """
    type: str = Field(default="function", description="Type of the tool, e.g., 'function'")
    function: FunctionTool = Field(..., description="Function tool definition")


class OllamaServices():
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
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=self.headers, json=json)
                response.raise_for_status()
            logger.info("Response received successfully from Ollama API.")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Model API error: %s - %s", e.response.status_code, e.response.text, exc_info=True)
            raise LLMGenerationError(f"Model API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request to model API failed: %s", str(e), exc_info=True)
            raise LLMGenerationError(f"Request to model API failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during query: %s", str(e), exc_info=True)
            raise LLMGenerationError(f"Unexpected error during query: {str(e)}")

    @log_duration
    async def generate_async(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generates a response from the Ollama chat model using a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of messages for the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response.

        Raises:
            LLMGenerationError: If there is an error during the generation process.
        """
        logger.debug(f"Invoking Ollama model: %s with messages: %s...", self.model, messages[:1])
        
        json = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **self.default_params
        }
        ollama_response = await self._make_ollama_request(json=json)

        if ollama_response and 'message' in ollama_response:
            logger.debug(f"Ollama response received: %s", ollama_response)
            response_content = ollama_response['message']['content']
            logger.info("Ollama response generated successfully.")
            return {"content": response_content} 
        else:
            logger.error(f"Unexpected Ollama response format: %s", ollama_response)
            raise LLMGenerationError(f"Unexpected Ollama response format: {ollama_response}")

    def convert_mcp_tool_to_ollama_tool(tool: Tool) -> OllamaTool:
        """
            Converts an MCP Tool object to an OllamaTool Pydantic instance.

            Args:
                tool (Tool): The MCP Tool object to convert.

            Returns:
                OllamaTool: The converted OllamaTool instance.
        """
        # Prepare parameters for the OllamaTool's function
        ollama_params_properties = {}
        # Access properties from the tool.inputSchema dictionary
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
                # Prefer 'description' if available, otherwise use 'title' from MCP property details
                description=prop_details.get('description', prop_details.get('title')),
                json_schema_extra=param_prop_json_schema_extra if param_prop_json_schema_extra else None
            )

        # Create the FunctionParameters for Ollama
        ollama_function_parameters = FunctionParameters(
            properties=ollama_params_properties,
            required=tool.inputSchema.get('required', []), # Use .get for robustness
            additionalProperties=tool.inputSchema.get('additionalProperties') # Get from root inputSchema
        )

        # Prepare the function definition for the OllamaTool
        ollama_function_definition = FunctionTool(
            name=tool.name,
            description=tool.description, # Access description directly from tool object
            parameters=ollama_function_parameters
        )

        # Create and return the final OllamaTool instance
        ollama_tool = OllamaTool(
            function=ollama_function_definition
        )

        return ollama_tool


if __name__ == "__main__":
    test_tool = Tool(
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
    )

    ollama_tool = convert_mcp_tool_to_ollama_tool(test_tool)
    print(ollama_tool.model_dump_json(indent=2)) 