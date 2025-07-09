from mcp.server.fastmcp import FastMCP
from mcp_server.services.db_services import DBServices, get_db_services
from fastapi import Depends
from mcp_server.exceptions import QueryExecutionError, QueryValidationError

async def db_query_tools(mcp: FastMCP):
    """
    Register the db_query_tools resource with the MCP server.
    This resource allows executing SQL queries against a PostgreSQL database.
    """
    @mcp.resource
    async def execute_sql_query(
        query: str,
        params: dict = {},
        db_services: DBServices = Depends(get_db_services)
    ):
        try:
            results = await db_services.execute_sql_query_async(query=query, params=params)
            return {"results": results}
        except QueryValidationError as e:
            raise QueryExecutionError(f"Query validation failed: {str(e)}") from e
        except QueryExecutionError as e:
            raise 
        except Exception as e:
            raise QueryExecutionError(f"Error executing query: {str(e)}") from e
    
    await execute_sql_query
