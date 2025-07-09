import logging
from exceptions import QueryValidationError, QueryExecutionError
from utils import validate_sql_query, is_read_only_query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import asyncio

logger = logging.getLogger(__name__)

class DBServices:
    """
    A class for interacting with the database using SQLAlchemy.
    """
    def __init__(self, db_session: AsyncSession):
        """
        Initialize the DbServices with a database session.

        Args:
            db_session (AsyncSession): SQLAlchemy asynchronous session.
        """
        self.db_session = db_session
        logger.info("DbServices initialized with provided database session.")

    async def execute_sql_query_async(self, query: str, params: dict = {}):
        """
        Asynchronously executes a raw SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (dict): Optional parameters for the query.

        Returns:
            list: The result of the executed query as a list of mappings.

        Raises:
            QueryValidationError: If the query validation fails.
            QueryExecutionError: If there is an error during query execution.
        """
        logger.debug("Preparing to execute query: %s with params: %s", query, params)
        try:
            self._validate_query(query)
            logger.info("Query validation passed.")
            result = await self.db_session.execute(text(query), params)
            logger.info("Query executed successfully.")
            return result.mappings().fetchall()
        except QueryValidationError as e:
            logger.error("Query validation error: %s", str(e))
            raise e
        except Exception as e:
            logger.error("Error executing query: %s", str(e), exc_info=True)
            raise QueryExecutionError(f"Error executing query: {str(e)}")

    def _validate_query(self, query: str):
        """
        Validates the SQL query to ensure it does not contain potentially dangerous keywords.

        Args:
            query (str): The SQL query to validate.

        Raises:
            QueryValidationError: If the query contains dangerous SQL keywords or is not read-only.
        """
        logger.debug("Validating query: %s", query)
        if not query.strip():
            logger.error("Query validation failed: Query cannot be empty.")
            raise QueryValidationError("Query cannot be empty.")
        
        if not validate_sql_query(query) or not is_read_only_query(query):
            logger.error("Query validation failed: Query contains potentially dangerous SQL keywords or is not read-only.")
            raise QueryValidationError("Query contains potentially dangerous SQL keywords or is not read-only.")
        logger.info("Query validation successful.")

async def main():
    """
    Main function to demonstrate the usage of DBServices.

    This function creates a database session, initializes the DBServices class,
    and executes a sample SQL query.
    """
    from core.dependencies import create_engine
    from core.config import settings
    from sqlalchemy.orm import sessionmaker

    logger.info("Starting main function.")
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine, class_=AsyncSession, autoflush=False, autocommit=False)
    session = Session()
    db_services = DBServices(db_session=session)

    query = "SELECT * FROM customers;"
    try:
        logger.info("Executing sample query: %s", query)
        result = await db_services.execute_sql_query_async(query=query)
        logger.info("Query executed successfully: %s", result)
        print("Query executed successfully:", result)
    except QueryValidationError as e:
        logger.error("Validation error: %s", str(e))
        print("Validation error:", str(e))
    except QueryExecutionError as e:
        logger.error("Execution error: %s", str(e))
        print("Execution error:", str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        print("Unexpected error:", str(e))
    finally:
        logger.info("Closing session and disposing engine.")
        await session.close()
        await engine.dispose()

if __name__ == "__main__":
    logger.info("Running main function.")
    asyncio.run(main())