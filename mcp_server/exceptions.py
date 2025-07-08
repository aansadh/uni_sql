class MCPServerError(Exception):
    """Base class for all MCP server exceptions."""
    pass

class DatabaseConnectionError(MCPServerError):
    """Exception raised for errors in the database connection."""
    def __init__(self, message: str):
        super().__init__(message)

class QueryValidationError(MCPServerError):
    """Exception raised for errors in SQL query validation."""
    def __init__(self, message: str):
        super().__init__(message)

class QueryExecutionError(MCPServerError):
    """Exception raised for errors during SQL query execution."""
    def __init__(self, message: str):
        super().__init__(message)

