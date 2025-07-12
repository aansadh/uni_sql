import logging
from mcp.server.fastmcp import FastMCP
from core.config import settings
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
from core.dependencies import create_engine, get_sessionmaker
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from services.db_services import DBServices
from exceptions import QueryExecutionError, QueryValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

@dataclass
class AppContext:
    db_engine: Engine
    db_sessionmaker: sessionmaker

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manages the lifespan of the FastMCP application.

    This function initializes the database engine at the start of the application
    and disposes of it when the application shuts down.

    Args:
        app (FastMCP): The FastMCP application instance.

    Yields:
        AsyncIterator[AsyncIterator]: An asynchronous iterator for the application lifecycle.
    """
    try:
        db_engine, db_sessionmaker = create_engine(settings.DATABASE_URL)
        yield AppContext(db_engine=db_engine, db_sessionmaker=db_sessionmaker) 
    finally:
        db_engine.dispose()

kwargs = {}
if settings.SERVER_TRANSPORT == 'streamable-http':
    kwargs = {
        "stateless_http": True
    }

mcp = FastMCP(
    name="DBServer",
    lifespan=lifespan,
    description="A server for executing PostgreSQL queries and managing database interactions.",
    **kwargs
)

@mcp.tool(
    name="execute_sql_query",  
    title="Execute SQL Query",  
    description="Executes a SQL query asynchronously using provided parameters and returns the results.",
)
async def execute_sql_query(
    query: str, params: dict = {}
):
    """
    Executes a SQL query asynchronously using the provided parameters.

    Args:
        query (str): The SQL query to execute.
        params (dict): Optional parameters for the query.
        db_services (DBServices): The database services instance for executing queries.

    Returns:
        dict: A dictionary containing the query results.

    Raises:
        QueryExecutionError: If there is an error during query execution.
    """
    db_sessionmaker = get_sessionmaker(mcp=mcp) 
    async with db_sessionmaker() as session:
        db_services = DBServices(db_session=session)
        try:
            results = await db_services.execute_sql_query_async(query=query, params=params)
            return {"results": results}
        except QueryValidationError as e:
            raise QueryExecutionError(f"Query validation failed: {str(e)}") from e
        except QueryExecutionError as e:
            raise
        except Exception as e:
            raise QueryExecutionError(f"Error executing query: {str(e)}") from e
        
if __name__ == "__main__":
    """
    Main entry point for the server.
    This will start the FastMCP server.
    """
    print("Starting the MCP server...")
    mcp.run(
        transport="streamable-http"
    )