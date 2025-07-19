import logging
import traceback

logger = logging.getLogger(__name__)

class McpClientError(Exception):
    """
    Base class for all MCP client exceptions.
    Includes status_code and a default detail message for API responses.

    Attributes:
        message (str): The error message.
        status_code (int): The HTTP status code associated with the error.
        api_response_detail (str): A detailed message for API responses.
        original_exception (Exception): The original exception, if any.
        original_traceback (str): The traceback of the original exception, if available.
    """
    
    def __init__(self, message: str, status_code: int = 500, api_response_detail: str = "An MCP client error occurred.", original_exception: Exception = None):
        super().__init__(message)
        self.message = message 
        self.status_code = status_code 
        self.api_response_detail = api_response_detail 
        self.original_exception = original_exception
        if original_exception:
            self.original_traceback = traceback.format_exc(original_exception.__traceback__)
        else:
            self.original_traceback = None
        logging.error(f"McpClientError initialized: {self.message}, Status Code: {self.status_code}")

class LLMGenerationError(McpClientError):
    """
    Exception raised for errors in LLM generation.
    Can store additional context like the prompt and model response for detailed logging.

    Attributes:
        prompt (str): The prompt sent to the LLM.
        model_response (str): The response received from the LLM.
    """
    def __init__(self, message: str, original_exception: Exception = None,
                 prompt: str = None, model_response: str = None):
        super().__init__(
            message,
            status_code=422,
            api_response_detail="LLM generation failed.",
            original_exception=original_exception
        )
        self.prompt = prompt
        self.model_response = model_response
        logging.error(f"LLMGenerationError initialized: {self.message}, Prompt: {self.prompt}, Model Response: {self.model_response}")