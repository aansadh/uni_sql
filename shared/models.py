from pydantic import BaseModel, Field
from typing import Dict, Any

class MCPInputModel(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to be called.")
    arguments: Dict[str, Any] = Field(..., description="Arguments to pass to the tool")