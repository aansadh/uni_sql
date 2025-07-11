from fastapi import FastAPI, Request, Depends
from services.ollama_services import OllamaServices as Ollama
from services.nl2sql_services import NL2SQLServices
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def get_ollama_services(request: Request):
    """
    Retrieve or initialize the OllamaServices instance for the application.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        Ollama: An instance of OllamaServices.
    """
    if not hasattr(request.app.state, 'ollama_services') or request.app.state.ollama_services is None:
        logger.info("Initializing OllamaServices instance.")
        request.app.state.ollama_services = Ollama(
            url=request.app.state.settings.NL2SQL_LLM_URL,
            model=request.app.state.settings.NL2SQL_LLM_MODEL,
            headers={"Content-Type": "application/json"},
            # Additional parameters can be passed here if needed
        )
    else:
        logger.debug("Using existing OllamaServices instance.")
    return request.app.state.ollama_services

def get_nl2sql_services(request: Request, ollama_services: Ollama = Depends(get_ollama_services)):
    """
    Retrieve or initialize the NL2SQLServices instance for the application.

    Args:
        request (Request): The FastAPI request object.
        ollama_services (Ollama): An instance of OllamaServices.

    Returns:
        NL2SQLServices: An instance of NL2SQLServices.
    """
    if not hasattr(request.app.state, 'nl2sql_services') or request.app.state.nl2sql_services is None:
        logger.info("Initializing NL2SQLServices instance.")
        request.app.state.nl2sql_services = NL2SQLServices(
            ollama_services=ollama_services
        )
    else:
        logger.debug("Using existing NL2SQLServices instance.")
    return request.app.state.nl2sql_services