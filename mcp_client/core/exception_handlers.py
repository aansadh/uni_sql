import logging, traceback
from exceptions import *
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

def _find_root_cause(exception: Exception) -> Exception:
    """
    Iteratively finds the innermost exception (root cause) in a chain.

    Args:
        exception (Exception): The exception to analyze.

    Returns:
        Exception: The root cause exception.
    """
    current_exception = exception
    while current_exception.__cause__ is not None or current_exception.__context__ is not None:
        if current_exception.__cause__ is not None:
            current_exception = current_exception.__cause__
        elif current_exception.__context__ is not None:
            current_exception = current_exception.__context__
    logger.debug("Root cause identified: %s", current_exception)
    return current_exception

def generic_exception_handler(request: Request, e: Exception) -> dict:
    """
    A generic exception handler that finds the root cause and returns a JSON-like response.

    Args:
        request (Request): The FastAPI request object.
        e (Exception): The exception to handle.

    Returns:
        dict: A JSON response containing error details.
    """
    logger.error(f"An unhandled exception occurred: {e}")
    logger.error(f"Full Traceback:\n{traceback.format_exc()}")

    root_cause = _find_root_cause(e)
    logger.error(f"Root cause of the exception: {root_cause.__class__.__name__}: {root_cause}")

    response_status_code = root_cause.status_code if isinstance(root_cause, McpClientError) else 500
    response_detail_message = root_cause.api_response_detail if isinstance(root_cause, McpClientError) else "An unexpected server error occurred. Please try again later."
    response_error_type = e.__class__.__name__ 

    if isinstance(e, McpClientError) and e.original_traceback:
        logger.error(f"Original exception traceback (from caught exception): \n{e.original_traceback}")

    response = {
        "status_code": response_status_code,
        "error_type": response_error_type,
        "detail": response_detail_message
    }

    logger.error(f"Exception details: {response}")

    return JSONResponse(
        status_code=response_status_code,
        content=response
    )