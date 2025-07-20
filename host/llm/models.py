from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any

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

###################################################

class OllamaResponseModel(BaseModel):
    """
    Represents the response model for an Ollama tool.

    Attributes:
        tool_name (Optional[str]): The name of the tool to be called.
        arguments (Optional[Dict[str, Any]]): The arguments to pass to the tool.
        content (Optional[str]): The content of the response message.
    """
    tool_name: Optional[str] = Field(None, description="Name of the tool to be called.", alias='name')
    arguments: Optional[Dict[str, Any]] = Field(None, description="Arguments to pass to the tool.", alias='arguments')
    content: Optional[str] = Field(None, description="Content of the response message.", alias='content')
