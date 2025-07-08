import logging
from mcp_server.exceptions import QueryValidationError

logger = logging.getLogger(__name__)

def validate_sql_query(query: str) -> bool:
    """
    Validates an SQL query to ensure it does not contain potentially dangerous keywords.

    Args:
        query (str): The SQL query to validate.

    Returns:
        bool: True if the query is safe, otherwise raises an exception.

    Raises:
        QueryValidationError: If the query contains dangerous SQL keywords.
    """
    logger.debug("Validating SQL query: %s", query)
    malicious_keywords = ["DROP TABLE", "DELETE FROM", "ALTER TABLE", "TRUNCATE TABLE"] 
    if any(keyword in query.upper() for keyword in malicious_keywords):
        logger.warning("Query contains potentially dangerous keyword.")
        return False
    logger.info("Query validation passed.")
    return True


def is_read_only_query(query: str) -> bool:
    """
    Checks if an SQL query is read-only based on its starting keywords.

    Args:
        query (str): The SQL query to check.

    Returns:
        bool: True if the query is read-only, False otherwise.
    """
    logger.debug("Checking if query is read-only: %s", query)
    read_only_keywords = ["SELECT", "SHOW", "DESCRIBE", "EXPLAIN"]
    is_read_only = any(query.upper().startswith(keyword) for keyword in read_only_keywords)
    logger.info("Query is read-only: %s", is_read_only)
    return is_read_only