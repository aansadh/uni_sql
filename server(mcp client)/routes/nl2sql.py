from fastapi import FastAPI, APIRouter, Depends
from core.dependencies import get_nl2sql_services
from services.nl2sql_services import NL2SQLServices
from utils import log_duration
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/nl2sql")
@log_duration
async def nl2sql_endpoint(query: str, nl2sql_services: NL2SQLServices = Depends(get_nl2sql_services)): 
    """
    Endpoint to convert natural language query to SQL using Ollama services.

    Args:
        query (str): The natural language query to convert.
        nl2sql_services (NL2SQLServices): The NL2SQL service instance.

    Returns:
        dict: The generated SQL query.
    """
    logger.info("Received request to convert query: '%s'", query)
    sql_query = await nl2sql_services.generate_sql_query(query=query)
    logger.info("Generated SQL query: '%s'", sql_query)
    return {"sql_query": sql_query}
