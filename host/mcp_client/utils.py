from shared.models import MCPInputModel
import json, logging
from pydantic import ValidationError

logger = logging.getLogger(__name__)

def parse_json_string_to_mcp_input_model(json_string: str) -> MCPInputModel:
    """
    Parses a JSON string into an MCPInputModel instance.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        MCPInputModel: An instance of MCPInputModel populated with the parsed data.
    """
    try:
        data = json.loads(json_string.strip())
        return MCPInputModel(**data)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON string: %s", e)
        raise ValueError("Invalid JSON string: ", e)
    except ValidationError as e:
        logger.error("Validation error while parsing JSON string: %s", e)
        raise ValueError("Validation error in JSON data: Does not match MCPInputModel: ", e)
    
def parse_json_string_to_dict(json_string: str) -> dict:
    """
    Parses a JSON string into a dictionary.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        dict: A dictionary representation of the JSON data.
    """
    try:
        return json.loads(json_string.strip())
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON string: %s", e)
        raise ValueError("Invalid JSON string: ", e)
    
if __name__ == "__main__":
    example_json = '{"key": "value", "number": 42}'
    try:
        parsed_data = parse_json_string_to_dict(example_json)
        print("Parsed data:", parsed_data)
    except ValueError as e:
        print("Error parsing JSON:", e)

    print(MCPInputModel.model_json_schema())
    
